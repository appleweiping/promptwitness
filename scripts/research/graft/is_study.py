"""H-dist kill test: score prompt edits by their effect on the distribution of reasoning.

GReaTer scores an edit P -> P' on one fixed greedy reasoning r under P. The quantity an
edit should improve is the expected accuracy J(P') = E_x E_{r ~ pi_P'(.|x)} c(P', x, r),
where c is the correctness of the answer read under P' after r. This script records, for a
validity-study pool (decision_study.py output: rows, proposal pools, edits, held-out
targets), everything needed to estimate J(P') - J(P) for every edit from samples drawn
once under the incumbent (plan amendment of 2026-10-01, H-dist):

  per row x, S reasonings r_s ~ pi_P^tau(. | x)   (temperature tau, no top-k/top-p)
  per candidate P' and sample s:
    log w_s = log pi_P'^tau(r_s | x) - log pi_P^tau(r_s | x)   exact teacher-forced ratio,
              including the probability of stopping where the sample stopped
    hard_s  = correctness of the greedy answer read under P' after r_s (+ extractor)
    soft_s  = probability of the gold answer under P' after r_s (+ extractor)
  distributional fresh evaluation: F samples per row under every candidate and the
  incumbent, with the same seed per (row, sample) for all prompts (coupled sampling).

Generation (samples, reads) goes through a vLLM generation server when ``--gen-server`` is
given (deterministic configuration: no prefix caching), else through the HF model; the
teacher-forced likelihoods always come from the HF model in bf16 with float32 softmax.
Estimators (beta-tempered self-normalized importance sampling, score function, ESS) are
computed by ``analyze_dist.py`` from the stored per-sample values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import (ANSWER_TOKENS, generate_batch, load_tokenizer, remote_sample,
                                         sampling_stops, use_remote_generation)

from run_graft import Ledger, Runner, apply


def sample_hf(model: Any, tokenizer: Any, prompts: list[list[int]], *, max_new_tokens: int, tau: float,
              stops: list[int], seed: int, batch: int = 16) -> list[tuple[list[int], bool]]:
    """Untruncated temperature sampling with the HF model (top_k=0, top_p=1)."""
    import torch

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    out: list[tuple[list[int], bool]] = []
    for begin in range(0, len(prompts), batch):
        chunk = prompts[begin:begin + batch]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(chunk):
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        torch.manual_seed(seed * 1000 + begin)
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=True, temperature=tau,
                                 top_k=0, top_p=1.0, pad_token_id=pad, eos_token_id=stops)
        for row in gen[:, width:].tolist():
            tokens: list[int] = []
            ended = False
            for token in row:
                if token in stops:
                    ended = True
                    break
                tokens.append(token)
            out.append((tokens, ended))
    return out


def sample(model: Any, tokenizer: Any, prompts: list[list[int]], *, max_new_tokens: int, tau: float,
           stops: list[int], seed: int) -> list[tuple[list[int], bool]]:
    remote = remote_sample(prompts, max_new_tokens=max_new_tokens, stop=stops, temperature=tau, top_p=1.0,
                           seed=seed, with_ended=True)
    if remote is not None:
        return [(list(tokens), bool(ended)) for tokens, ended in remote]
    return sample_hf(model, tokenizer, prompts, max_new_tokens=max_new_tokens, tau=tau, stops=stops, seed=seed)


def trace_scores(model: Any, items: list[tuple[list[int], int, int, bool, int]], *, tau: float,
                 stops: list[int], batch: int = 8) -> list[tuple[float, float]]:
    """Teacher-forced scores of (prompt + reasoning + extractor + gold answer) sequences.

    Each item is (ids, reason_start, reason_end, ended, answer_start); the gold answer runs
    to the end of ``ids``. Returns per item the tempered log-probability of the reasoning
    tokens (plus log P(stop) at its end when the sample ended there; any stop token counts)
    and the untempered log-probability of the gold answer tokens.
    """
    import torch

    device = next(model.parameters()).device
    stop_index = torch.tensor(sorted(set(stops)), device=device)
    scores: list[tuple[float, float]] = []
    for begin in range(0, len(items), batch):
        chunk = items[begin:begin + batch]
        width = max(len(item[0]) for item in chunk)
        ids = torch.zeros((len(chunk), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, item in enumerate(chunk):
            ids[row, :len(item[0])] = torch.tensor(item[0])
            mask[row, :len(item[0])] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
            for row, (seq, lo, hi, ended, alo) in enumerate(chunk):
                reason = 0.0
                if hi > lo:
                    z = torch.log_softmax(logits[row, lo - 1:hi - 1].float() / tau, dim=-1)
                    reason = float(z.gather(1, ids[row, lo:hi, None]).sum())
                if ended:
                    zs = torch.log_softmax(logits[row, hi - 1].float() / tau, dim=-1)
                    reason += float(torch.logsumexp(zs[stop_index], dim=0))
                za = torch.log_softmax(logits[row, alo - 1:len(seq) - 1].float(), dim=-1)
                answer = float(za.gather(1, ids[row, alo:len(seq), None]).sum())
                scores.append((reason, answer))
        del logits
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, help="decision_study output (pools, edits, rows, targets)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gen-server", help="vLLM generation server for samples and reads")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--fresh-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    started = perf_counter()
    if args.gen_server:
        use_remote_generation(args.gen_server)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    spec = graft_tasks.spec(record["task"])
    splits = graft_tasks.load_splits(args.data_dir, record["task"])
    by_id = {e.example_id: e for e in splits["train"]}
    rows = [by_id[i] for i in record["rows"]]
    runner = Runner(model, tokenizer, spec, Ledger(), args.max_new_tokens)
    base = graft_tasks.initial_prompt()
    if record.get("state_edit"):
        base = base.replace_block("procedure", record["state_edit"])
    pools = record["pools"]
    names = record["edits"]
    edits = [(n.split(":")[0], None if n.split(":")[1] == "del" else int(n.split(":")[1])) for n in names]
    prompts = {"base": base} | {n: apply(base, pools, e) for n, e in zip(names, edits)}
    stops = sampling_stops(model, tokenizer)
    ext = runner.extractor
    golds = [runner.target(r) for r in rows]
    cost: dict[str, float] = {}
    tokens: dict[str, int] = {}

    # 1. S reasonings per row under the incumbent (shared by every candidate).
    t0 = perf_counter()
    base_ids = [runner.ids(base, r) for r in rows]
    flat = [p for p in base_ids for _ in range(args.samples)]
    drawn = sample(model, tokenizer, flat, max_new_tokens=args.max_new_tokens, tau=args.tau, stops=stops,
                   seed=args.seed)
    traces = [drawn[i * args.samples:(i + 1) * args.samples] for i in range(len(rows))]
    cost["incumbent_samples"] = perf_counter() - t0
    print(json.dumps({"phase": "incumbent_samples", "seconds": cost["incumbent_samples"]}), flush=True)
    tokens["incumbent_samples_generated"] = sum(len(t) for t, _ in drawn)

    # 2. Teacher-forced likelihoods and soft reads under the incumbent and every candidate.
    t0 = perf_counter()
    logp: dict[str, list[list[float]]] = {}
    soft: dict[str, list[list[float]]] = {}
    forced = 0
    for name, prompt in prompts.items():
        items = []
        for i, row in enumerate(rows):
            p = runner.ids(prompt, row)
            for reasoning, ended in traces[i]:
                seq = p + reasoning + ext + golds[i]
                items.append((seq, len(p), len(p) + len(reasoning), ended, len(p) + len(reasoning) + len(ext)))
                forced += len(seq)
        scores = trace_scores(model, items, tau=args.tau, stops=stops, batch=args.batch)
        logp[name] = [[scores[i * args.samples + s][0] for s in range(args.samples)] for i in range(len(rows))]
        soft[name] = [[math.exp(scores[i * args.samples + s][1]) for s in range(args.samples)]
                      for i in range(len(rows))]
    cost["teacher_forcing"] = perf_counter() - t0
    print(json.dumps({"phase": "teacher_forcing", "seconds": cost["teacher_forcing"]}), flush=True)
    tokens["teacher_forced"] = forced

    # 3. Hard reads (greedy answer after reasoning + extractor) under every prompt.
    t0 = perf_counter()
    hard: dict[str, list[list[int]]] = {}
    for name, prompt in prompts.items():
        reads_in = [runner.ids(prompt, row) + reasoning + ext
                    for row, row_traces in zip(rows, traces) for reasoning, _ in row_traces]
        reads, _, _ = generate_batch(model, tokenizer, reads_in, max_new_tokens=ANSWER_TOKENS)
        flags = [int(spec.correct(r.text, rows[k // args.samples].answer)) for k, r in enumerate(reads)]
        hard[name] = [flags[i * args.samples:(i + 1) * args.samples] for i in range(len(rows))]
    cost["hard_reads"] = perf_counter() - t0
    print(json.dumps({"phase": "hard_reads", "seconds": cost["hard_reads"]}), flush=True)

    # 4. Distributional fresh evaluation with coupled seeds (same seed per (row, sample)).
    t0 = perf_counter()
    fresh: dict[str, list[list[int]]] = {}
    fresh_generated = 0
    for name, prompt in prompts.items():
        ids = [runner.ids(prompt, r) for r in rows]
        flat = [p for p in ids for _ in range(args.fresh_samples)]
        outs = sample(model, tokenizer, flat, max_new_tokens=args.max_new_tokens, tau=args.tau, stops=stops,
                      seed=args.seed + 1)
        fresh_generated += sum(len(t) for t, _ in outs)
        reads, _, _ = generate_batch(model, tokenizer, [p + t + ext for p, (t, _) in zip(flat, outs)],
                                     max_new_tokens=ANSWER_TOKENS)
        flags = [int(spec.correct(r.text, rows[k // args.fresh_samples].answer)) for k, r in enumerate(reads)]
        fresh[name] = [flags[i * args.fresh_samples:(i + 1) * args.fresh_samples] for i in range(len(rows))]
    cost["fresh_distributional"] = perf_counter() - t0
    tokens["fresh_distributional_generated"] = fresh_generated

    result = {
        "format": "promptwitness.graft-dist/v1", "record": str(args.record), "task": record["task"],
        "model_path": args.model_path, "seed": args.seed, "tau": args.tau, "samples": args.samples,
        "fresh_samples": args.fresh_samples, "max_new_tokens": args.max_new_tokens,
        "engine": "vllm-server" if args.gen_server else "hf", "edits": names,
        "trace_lengths": [[len(t) for t, _ in row] for row in traces],
        "trace_ended": [[int(e) for _, e in row] for row in traces],
        "logp": logp, "soft": soft, "hard": hard, "fresh": fresh,
        "cost_seconds": cost, "tokens": tokens, "wall_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cost_seconds": cost, "tokens": tokens}))


if __name__ == "__main__":
    main()
