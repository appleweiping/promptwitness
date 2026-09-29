"""Which cheap signal predicts held-out accuracy of whole-block edits? (validity v2)

Greedy decoding is decided where a correct and an incorrect self-generated reasoning
diverge. For every train row we collect such forks: if greedy is wrong, its first
divergence from each correct sample; if greedy is right, its first divergence from each
incorrect sample. An edit's *fork margin* is CE(correct token) - CE(incorrect token)
given the shared prefix, averaged over forks (lower is better).

Predictors of an edit's dev-accuracy change, each with its measured cost:
  answer_exact / answer_patch  GReaTer's answer loss on greedy reasoning
  fork_exact / fork_patch      decision-margin change at forks
  fresh8 / fresh_all           fresh greedy accuracy on 8 / all train rows (generation)
Target: fresh accuracy change on N dev examples (never used to choose anything here).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import answer_losses, generate_batch, load_tokenizer, sample_reasonings
from promptwitness.superposed_gates import GateScorer, build_superposed, candidate_token_ids
from promptwitness.superposed_patching import patch_estimates

from fidelity_study import spearman
from run_graft import Ledger, Runner, apply, propose


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--dev", type=int, default=60)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-forks", type=int, default=3)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    started = perf_counter()
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    scorer = GateScorer(model)
    spec = graft_tasks.spec(args.task)
    splits = graft_tasks.load_splits(args.data_dir, args.task)
    rng = random.Random(args.seed)
    train = list(splits["train"])
    rng.shuffle(train)
    proposal_rows, rows = train[:2], train[2:2 + args.rows]
    dev = splits["dev"][: args.dev]
    runner = Runner(model, tokenizer, spec, Ledger(), args.max_new_tokens)
    prompt = graft_tasks.initial_prompt()
    slots = [b.block_id for b in prompt.blocks if b.editable]
    pools = {s: propose(runner, prompt, s, [r.question for r in proposal_rows], args.seed * 100 + j, args.k)
             for j, s in enumerate(slots)}
    probe = {"input": rows[0].question}
    edits = [(s, i) for s, texts in pools.items() for i, t in enumerate(texts)
             if candidate_token_ids(prompt, tokenizer, probe, s, t) is not None]
    edits += [(b.block_id, None) for b in prompt.blocks if b.editable and b.text]
    ext = runner.extractor
    cost: dict[str, float] = {}

    # --- reasoning material (shared by answer and fork objectives) ---
    t0 = perf_counter()
    base_prompts = [runner.ids(prompt, r) for r in rows]
    greedy = runner.evaluate(prompt, rows, "train_greedy")
    drafts = sample_reasonings(model, tokenizer, base_prompts, args.samples, args.max_new_tokens, args.seed)
    reads, _, _ = generate_batch(model, tokenizer, [p + d + ext for p, ds in zip(base_prompts, drafts) for d in ds],
                                 max_new_tokens=8)
    forks: list[list[tuple[list[int], int, int]]] = []  # per row: (prefix, correct token, wrong token)
    for i, row in enumerate(rows):
        g, g_ok = list(greedy["reasoning"][i]), greedy["correct"][i]
        found: dict[tuple[int, int, int], tuple[list[int], int, int]] = {}
        for j, d in enumerate(drafts[i]):
            ok = spec.correct(reads[i * args.samples + j].text, row.answer)
            if not d or ok == g_ok:
                continue
            good, bad = (g, d) if g_ok else (d, g)
            t = first_divergence(good, bad)
            if t is not None and t < len(good) and t < len(bad):
                found[(t, good[t], bad[t])] = (good[:t], good[t], bad[t])
        forks.append(list(found.values())[: args.max_forks])
    cost["reasoning_material"] = perf_counter() - t0
    answers = [tokenizer.encode(spec.target(r.answer), add_special_tokens=False) for r in rows]

    # --- answer objective (GReaTer), exact and patch ---
    answer_exact = {e: [] for e in edits}
    answer_patch = {e: [] for e in edits}
    t_exact = t_patch = 0.0
    for i, row in enumerate(rows):
        tail = list(greedy["reasoning"][i]) + ext
        t0 = perf_counter()
        pairs = [(runner.ids(prompt, row) + tail, answers[i])]
        pairs += [(runner.ids(apply(prompt, pools, e), row) + tail, answers[i]) for e in edits]
        losses, _ = answer_losses(model, pairs)
        t_exact += perf_counter() - t0
        t0 = perf_counter()
        seq = build_superposed(prompt, tokenizer, {"input": row.question}, pools, tail, answers[i], mode="exact")
        _, est = patch_estimates(scorer, seq)
        t_patch += perf_counter() - t0
        for j, e in enumerate(edits):
            answer_exact[e].append(losses[j + 1] - losses[0])
            answer_patch[e].append(est.get(e, 0.0))
    cost["answer_exact"], cost["answer_patch"] = t_exact, t_patch

    # --- fork margin, exact and patch ---
    fork_exact = {e: [] for e in edits}
    fork_patch = {e: [] for e in edits}
    t_exact = t_patch = 0.0
    for i, row in enumerate(rows):
        if not forks[i]:
            continue
        row_exact = {e: [] for e in edits}
        row_patch = {e: [] for e in edits}
        for prefix, good, bad in forks[i]:
            t0 = perf_counter()
            pairs, owners = [], []
            for owner, p in [(None, prompt)] + [(e, apply(prompt, pools, e)) for e in edits]:
                ids = runner.ids(p, row) + prefix
                pairs += [(ids, [good]), (ids, [bad])]
                owners.append(owner)
            losses, _ = answer_losses(model, pairs)
            margins = [losses[2 * k] - losses[2 * k + 1] for k in range(len(owners))]
            t_exact += perf_counter() - t0
            t0 = perf_counter()
            est_margin = patch_estimates(scorer, build_superposed(
                prompt, tokenizer, {"input": row.question}, pools, prefix, [good], mode="exact",
                contrast_ids=[bad]))[1]
            t_patch += perf_counter() - t0
            for k, e in enumerate(edits):
                row_exact[e].append(margins[k + 1] - margins[0])
                row_patch[e].append(est_margin.get(e, 0.0))
        for e in edits:
            fork_exact[e].append(statistics.mean(row_exact[e]))
            fork_patch[e].append(statistics.mean(row_patch[e]))
    cost["fork_exact"], cost["fork_patch"] = t_exact, t_patch

    # --- fresh accuracy on train rows (what the search loop's verification sees) ---
    t0 = perf_counter()
    fresh = {e: runner.evaluate(apply(prompt, pools, e), rows, "fresh")["correct"] for e in edits}
    cost["fresh_all"] = perf_counter() - t0
    cost["fresh8"] = cost["fresh_all"] * 8 / len(rows)
    base_correct = greedy["correct"]

    # --- held-out target ---
    t0 = perf_counter()
    base_dev = runner.evaluate(prompt, dev, "dev")["accuracy"]
    dev_delta = {e: runner.evaluate(apply(prompt, pools, e), dev, "dev")["accuracy"] - base_dev for e in edits}
    cost["dev_target"] = perf_counter() - t0

    predictors = {
        "answer_exact": {e: -statistics.mean(v) for e, v in answer_exact.items()},
        "answer_patch": {e: -statistics.mean(v) for e, v in answer_patch.items()},
        "fork_exact": {e: -statistics.mean(v) if v else 0.0 for e, v in fork_exact.items()},
        "fork_patch": {e: -statistics.mean(v) if v else 0.0 for e, v in fork_patch.items()},
        "fresh8": {e: statistics.mean(fresh[e][:8]) - statistics.mean(base_correct[:8]) for e in edits},
        "fresh_all": {e: statistics.mean(fresh[e]) - statistics.mean(base_correct) for e in edits},
    }
    delta = [dev_delta[e] for e in edits]
    summary: dict[str, Any] = {"base_dev_accuracy": base_dev, "edits": len(edits),
                               "dev_delta_mean": statistics.mean(delta), "dev_delta_best": max(delta),
                               "rows_with_forks": sum(bool(f) for f in forks),
                               "forks": sum(len(f) for f in forks), "cost_seconds": cost}
    for name, pred in predictors.items():
        score = [pred[e] for e in edits]
        order = sorted(range(len(edits)), key=lambda j: -score[j])
        summary[name] = {"spearman_vs_dev": spearman(score, delta),
                         "best5_dev_delta": statistics.mean(delta[j] for j in order[:5]),
                         "top1_dev_delta": delta[order[0]]}
    names = [f"{s}:{'del' if i is None else i}" for s, i in edits]
    result = {"format": "promptwitness.graft-decision/v1", "task": args.task, "model_path": args.model_path,
              "seed": args.seed, "rows": [r.example_id for r in rows], "pools": pools, "edits": names,
              "dev_delta": delta, "predictors": {k: [v[e] for e in edits] for k, v in predictors.items()},
              "forks_per_row": [len(f) for f in forks], "summary": summary,
              "wall_seconds": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
