"""H-dist machinery: exact trace likelihoods and the importance-sampling estimators."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "graft"))


def test_snis_and_ess_limits() -> None:
    from analyze_dist import ess, snis

    values = [1.0, 0.0, 1.0, 1.0]
    assert snis([0.3, -1.0, 2.0, 0.5], values, 0.0) == pytest.approx(0.75)
    assert snis([0.7] * 4, values, 1.0) == pytest.approx(0.75)  # equal weights
    assert snis([0.0, 50.0, 0.0, 0.0], values, 1.0) == pytest.approx(0.0, abs=1e-12)  # one dominant sample
    assert ess([0.2] * 4, 1.0) == pytest.approx(4.0)
    assert ess([0.0, 50.0, 0.0, 0.0], 1.0) == pytest.approx(1.0, abs=1e-9)
    assert ess([0.0, 50.0, 0.0, 0.0], 0.0) == 4.0


def test_per_row_values_and_beta_choice() -> None:
    from analyze_dist import choose_beta, per_row_values

    dist = {"edits": ["a:0"], "samples": 4,
            "logp": {"base": [[-10.0, -12.0, -11.0, -9.0]], "a:0": [[-10.0, -12.0, -11.0, -9.0]]},
            "hard": {"base": [[1, 0, 0, 1]], "a:0": [[1, 1, 0, 1]]},
            "soft": {"base": [[0.9, 0.1, 0.2, 0.8]], "a:0": [[0.9, 0.6, 0.2, 0.8]]},
            "fresh": {"base": [[1, 0, 1, 0]], "a:0": [[1, 1, 1, 0]]}}
    beta, medians = choose_beta(dist, 4)
    assert beta == 1.0 and medians["1.0"] == pytest.approx(4.0)  # identical likelihoods: no degeneracy
    values = per_row_values(dist, "a:0", beta)
    assert values["primary"][0] == pytest.approx(0.25)  # same traces, one read flips to correct
    assert values["snis_soft_0.0"][0] == pytest.approx(0.125)
    assert values["fresh_dist"][0] == pytest.approx(0.25)
    assert values["score_fn"][0] == pytest.approx(0.25)  # log w = 0: only the read-off term


def test_analyze_dist_runs_on_synthetic_pools(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json
    import random

    import analyze_dist

    rng = random.Random(0)
    pairs = []
    for p in range(3):
        names = [f"strategy:{i}" for i in range(6)]
        rows, samples, fresh_samples, questions = 5, 4, 2, 20
        record = {"edits": names, "task": "toy", "seed": p, "model_path": "/m/models--org--toy/snapshots/x",
                  "raw": {"base_fresh": [rng.randint(0, 1) for _ in range(rows)],
                          "base_dev": [rng.randint(0, 1) for _ in range(questions)],
                          "fresh": {n: [rng.randint(0, 1) for _ in range(rows)] for n in names},
                          "dev": {n: [rng.randint(0, 1) for _ in range(questions)] for n in names},
                          "answer_exact": {n: [rng.gauss(0, 1) for _ in range(rows)] for n in names}}}
        prompts = ["base", *names]
        dist = {"edits": names, "samples": samples,
                "logp": {n: [[rng.gauss(-50, 3) for _ in range(samples)] for _ in range(rows)] for n in prompts},
                "hard": {n: [[rng.randint(0, 1) for _ in range(samples)] for _ in range(rows)] for n in prompts},
                "soft": {n: [[rng.random() for _ in range(samples)] for _ in range(rows)] for n in prompts},
                "fresh": {n: [[rng.randint(0, 1) for _ in range(fresh_samples)] for _ in range(rows)]
                          for n in prompts}}
        rec_path, dist_path = tmp_path / f"r{p}.json", tmp_path / f"d{p}.json"
        rec_path.write_text(json.dumps(record), encoding="utf-8")
        dist_path.write_text(json.dumps(dist), encoding="utf-8")
        pairs.append(f"{rec_path}={dist_path}")
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["analyze_dist.py", *pairs, "--replicates", "20", "--json", str(out)])
    analyze_dist.main()
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert len(summary["pools"]) == 3 and "primary-random" in summary["comparisons"]
    assert isinstance(summary["h_dist_supported"], bool) and "primary" in summary["verify_all"]


def test_trace_scores_match_direct_computation() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from is_study import trace_scores

    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2)
    model = transformers.LlamaForCausalLM(config).float().eval()
    model.to = lambda *a, **k: model  # keep on CPU
    stops, tau = [3, 7], 0.7
    items = [([5, 6, 8, 9, 10, 11, 12, 13], 2, 5, True, 6),  # reasoning 8,9,10 then stop; answer 12,13
             ([5, 6, 9, 14, 15], 2, 3, False, 4)]             # truncated reasoning 9; answer 15
    got = trace_scores(model, items, tau=tau, stops=stops, batch=2)
    for (seq, lo, hi, ended, alo), (reason, answer) in zip(items, got):
        with torch.no_grad():
            logits = model(torch.tensor([seq])).logits[0].float()
        want = sum(torch.log_softmax(logits[t - 1] / tau, -1)[seq[t]].item() for t in range(lo, hi))
        if ended:
            want += math.log(sum(torch.softmax(logits[hi - 1] / tau, -1)[s].item() for s in stops))
        want_answer = sum(torch.log_softmax(logits[t - 1], -1)[seq[t]].item() for t in range(alo, len(seq)))
        assert reason == pytest.approx(want, abs=1e-4)
        assert answer == pytest.approx(want_answer, abs=1e-4)
