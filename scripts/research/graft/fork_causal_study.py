"""Are self-generated reasoning forks causal decision points? (validation for H-fork)

For train rows under the initial prompt, forks are the first divergences between the
greedy reasoning and self-samples of opposite correctness (as in decision_study). For each
fork (prefix, correct-branch token, wrong-branch token) we force each token after the
prefix and sample R continuations, reading the answer with the shared reader:
    dP_fork = P(correct | prefix + good) - P(correct | prefix + bad).
Control: a uniformly random position u on the shared prefix, forcing the actual prefix
token vs the model's most likely different token there:
    dP_ctrl = P(correct | prefix[:u] + actual) - P(correct | prefix[:u] + runner_up).
H-fork's causal premise predicts mean dP_fork well above mean |dP_ctrl|.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

from promptwitness import graft_tasks
from promptwitness.graft_runtime import (
    fork_targets,
    generate_batch,
    load_tokenizer,
    sample_reasonings,
)

from run_graft import Ledger, Runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--continuations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    started = perf_counter()
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                 attn_implementation=args.attn).to("cuda").eval()
    spec = graft_tasks.spec(args.task)
    train = list(graft_tasks.load_splits(args.data_dir, args.task)["train"])
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rows = train[: args.rows]
    runner = Runner(model, tokenizer, spec, Ledger(), args.max_new_tokens)
    prompt = graft_tasks.initial_prompt()
    ext = runner.extractor
    prompts = [runner.ids(prompt, r) for r in rows]
    greedy = runner.evaluate(prompt, rows, "greedy")
    drafts = sample_reasonings(model, tokenizer, prompts, args.samples, args.max_new_tokens, args.seed)
    reads, _, _ = generate_batch(model, tokenizer, [p + d + ext for p, ds in zip(prompts, drafts) for d in ds],
                                 max_new_tokens=8)
    judged = [[spec.correct(reads[i * args.samples + j].text, r.answer) for j in range(args.samples)]
              for i, r in enumerate(rows)]
    forks = fork_targets(greedy["reasoning"], greedy["correct"], drafts, judged, max_forks=2)

    def success_rate(row: int, context: list[int], seed: int) -> float:
        """Sample continuations after a forced context and read answers."""
        budget = max(8, args.max_new_tokens - (len(context) - len(prompts[row])))
        conts = sample_reasonings(model, tokenizer, [context], args.continuations, budget, seed)[0]
        outs, _, _ = generate_batch(model, tokenizer, [context + c + ext for c in conts], max_new_tokens=8)
        return statistics.mean(spec.correct(o.text, rows[row].answer) for o in outs)

    records = []
    for k, fork in enumerate(forks):
        row, prefix = fork.row, list(fork.tail)
        base = prompts[row] + prefix
        p_good = success_rate(row, base + list(fork.scored), args.seed + 10 * k)
        p_bad = success_rate(row, base + list(fork.contrast), args.seed + 10 * k + 1)
        record = {"row": rows[row].example_id, "position": len(prefix),
                  "good": tokenizer.decode(list(fork.scored)), "bad": tokenizer.decode(list(fork.contrast)),
                  "p_good": p_good, "p_bad": p_bad}
        if prefix:
            u = rng.randrange(len(prefix))
            context = prompts[row] + prefix[:u]
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([context], device=model.device)).logits[0, -1]
            actual = prefix[u]
            ranked = torch.topk(logits.float(), 5).indices.tolist()
            runner_up = next(t for t in ranked if t != actual)
            record["control_position"] = u
            record["p_ctrl_actual"] = success_rate(row, context + [actual], args.seed + 10 * k + 2)
            record["p_ctrl_alt"] = success_rate(row, context + [runner_up], args.seed + 10 * k + 3)
        records.append(record)
        print(json.dumps(record), flush=True)
    d_fork = [r["p_good"] - r["p_bad"] for r in records]
    d_ctrl = [r["p_ctrl_actual"] - r["p_ctrl_alt"] for r in records if "p_ctrl_actual" in r]
    summary = {"rows": len(rows), "forks": len(records),
               "rows_with_forks": len({r["row"] for r in records}),
               "mean_dP_fork": statistics.mean(d_fork) if d_fork else None,
               "mean_abs_dP_ctrl": statistics.mean(abs(x) for x in d_ctrl) if d_ctrl else None,
               "frac_fork_positive": sum(x > 0 for x in d_fork) / len(d_fork) if d_fork else None}
    result = {"format": "promptwitness.graft-fork-causal/v1", "task": args.task, "model_path": args.model_path,
              "seed": args.seed, "records": records, "summary": summary,
              "wall_seconds": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
