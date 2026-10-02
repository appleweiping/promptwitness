"""Shared dev selection and one test read for search runs (vLLM reader).

Search runs made with ``run_graft.py --defer-eval`` stop after search and record their
checkpoint prompts (incumbents at evenly spaced rounds) as typed blocks. This script
evaluates every checkpoint of every run on the task's dev split with the same reader
and engine, selects the best (ties go to the later checkpoint, as in run_graft), and
reads test once for the selected prompt. All runs of one model share a single engine,
so every method in a comparison is read identically. Results are resumable and keyed
by run file name.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_reader import RemoteReader
from promptwitness.graft_runtime import Ledger, load_tokenizer, use_remote_generation
from promptwitness.graft_vllm import VllmReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--gen-server", help="read through a running graft_genserver instead of an offline engine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reroll-chunks", default="",
                        help="comma-separated sizes: re-read the selected prompt's test questions in separate requests "
                             "of this many (numerical re-rolls by batch shape; deterministic engine)")
    args = parser.parse_args()
    if args.gen_server:
        use_remote_generation(args.gen_server)

    results: dict[str, Any] = (json.loads(args.output.read_text(encoding="utf-8"))
                               if args.output.exists() else {})
    by_model: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if path.name in results or "checkpoints" not in run:
            continue
        by_model[run["model_path"]].append((path, run))
    for model_path, runs in by_model.items():
        llm = None
        if not args.gen_server:
            from vllm import LLM

            llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=0,
                      gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
                      enable_prefix_caching=False)  # cached prefixes change numerics; reads must repeat exactly
        tokenizer = load_tokenizer(model_path)
        for path, run in runs:
            started = perf_counter()
            spec = graft_tasks.spec(run["task"])
            splits = graft_tasks.load_splits(args.data_dir, run["task"])
            ledger = Ledger()
            reader = (RemoteReader(tokenizer, spec, ledger, args.max_new_tokens) if args.gen_server
                      else VllmReader(model_path, tokenizer, spec, ledger, args.max_new_tokens, llm=llm))
            prompts = [graft_tasks.prompt_from_record(c["blocks"], c["text"]) for c in run["checkpoints"]]
            dev = [reader.evaluate(p, splits["dev"], "dev") for p in prompts]
            scores = [d["accuracy"] for d in dev]
            chosen = max(range(len(prompts)), key=lambda i: (scores[i], i))
            test = reader.evaluate(prompts[chosen], splits["test"], "test")
            rerolls = []
            for chunk in [int(c) for c in args.reroll_chunks.split(",") if c.strip()]:
                rows = splits["test"]
                correct = [c for b in range(0, len(rows), chunk)
                           for c in reader.evaluate(prompts[chosen], rows[b:b + chunk], "test_reroll")["correct"]]
                rerolls.append({"chunk": chunk, "accuracy": sum(correct) / len(correct), "correct": correct})
            results[path.name] = {
                "test_rerolls": rerolls,
                "task": run["task"], "method": run["method"], "seed": run["seed"],
                "objective": run["config"].get("objective"), "accept": run["config"].get("accept"),
                "model_path": model_path, "engine": "vllm-server" if args.gen_server else "vllm",
                "checkpoint_rounds": [c["round"] for c in run["checkpoints"]],
                "dev_accuracy": scores, "selected": chosen,
                "selected_round": run["checkpoints"][chosen]["round"],
                "selected_text": prompts[chosen].document.messages[0].content,
                "test_accuracy": test["accuracy"], "test_correct": test["correct"],
                "test_truncated": test["truncated"], "search_seconds": run.get("search_seconds"),
                "search_cost": run.get("cost"), "eval_cost": ledger.phases,
                "eval_seconds": perf_counter() - started}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"run": path.name, "dev": scores, "test": test["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
