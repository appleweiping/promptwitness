"""Scoring cost versus candidate-pool size: exact per-edit scoring vs superposed patching.

Train rows only. For K in --ks candidates per slot (prefixes of one proposal pool),
time (a) exact fixed-reasoning scoring of every edit with batched forwards and
(b) renormalized superposed patching (one forward/backward per example and chunk),
with and without gradient checkpointing, and report Spearman(patch, exact).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

from promptwitness import graft_tasks
from promptwitness.graft_runtime import load_tokenizer
from promptwitness.graft_runtime import answer_losses, answer_target, generate_batch
from promptwitness.superposed_gates import GateScorer, build_superposed, candidate_token_ids
from promptwitness.superposed_patching import patch_estimates

from fidelity_study import spearman
from run_graft import Ledger, Runner, apply, propose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", default="date_understanding")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    spec = graft_tasks.spec(args.task)
    train = graft_tasks.load_splits(args.data_dir, args.task)["train"]
    rng = random.Random(0)
    rows = rng.sample(train, args.rows)
    runner = Runner(model, tokenizer, spec, Ledger(), 384)
    prompt = graft_tasks.initial_prompt()
    slots = [b.block_id for b in prompt.blocks if b.editable]
    questions = [e.question for e in rng.sample(train, 2)]
    pools_full = {slot: propose(runner, prompt, slot, questions, 7 + j, max(args.ks))
                  for j, slot in enumerate(slots)}
    reasoning = [list(g.token_ids) for g in generate_batch(
        model, tokenizer, [runner.ids(prompt, e) for e in rows], max_new_tokens=384)[0]]
    targets = [answer_target(r, runner.extractor, runner.target(e)) for r, e in zip(reasoning, rows)]
    results = []
    for k in args.ks:
        pools = {slot: texts[:k] for slot, texts in pools_full.items()}
        edits = [(s, i) for s, texts in pools.items() for i, t in enumerate(texts)
                 if candidate_token_ids(prompt, tokenizer, {"input": rows[0].question}, s, t) is not None]
        torch.cuda.synchronize()
        started = perf_counter()
        pairs = []
        for example, target in zip(rows, targets):
            base = runner.ids(prompt, example) + list(target.tail)
            pairs.append((base, list(target.scored)))
            pairs += [(runner.ids(apply(prompt, pools, e), example) + list(target.tail), list(target.scored))
                      for e in edits]
        losses, _ = answer_losses(model, pairs, batch_size=16)
        torch.cuda.synchronize()
        exact_seconds = perf_counter() - started
        per = len(edits) + 1
        exact = {e: sum(losses[r * per + 1 + j] - losses[r * per] for r in range(len(rows))) / len(rows)
                 for j, e in enumerate(edits)}
        record = {"k": k, "edits": len(edits), "exact_seconds": exact_seconds}
        for checkpointing in (True, False):
            scorer = GateScorer(model, checkpointing=checkpointing)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = perf_counter()
            estimates = {e: 0.0 for e in edits}
            try:
                for example, target in zip(rows, targets):
                    seq = build_superposed(prompt, tokenizer, {"input": example.question}, pools,
                                           list(target.tail), list(target.scored), mode="exact")
                    _, est = patch_estimates(scorer, seq, include_deletions=False)
                    for e in edits:
                        estimates[e] += est.get(e, 0.0) / len(rows)
                torch.cuda.synchronize()
                label = "ckpt" if checkpointing else "nockpt"
                record[f"patch_{label}_seconds"] = perf_counter() - started
                record[f"patch_{label}_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                record[f"patch_{label}_tokens"] = len(seq.input_ids)
                record["spearman_patch_exact"] = spearman([estimates[e] for e in edits],
                                                          [exact[e] for e in edits])
            except torch.cuda.OutOfMemoryError:
                record[f"patch_{'ckpt' if checkpointing else 'nockpt'}_seconds"] = None
                torch.cuda.empty_cache()
        results.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"task": args.task, "model_path": args.model_path,
                                       "rows": args.rows, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
