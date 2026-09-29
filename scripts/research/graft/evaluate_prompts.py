"""Evaluate fixed prompt sets with the shared two-stage reader (dev selection, one test).

Prompt sets:
  zs_cot            "Let's think step by step."
  greater_init      GReaTer's initial instruction as typed PromptWitness blocks
  greater_published GReaTer's released final-beam prompts for this model (pooled over
                    runs); up to --max-candidates are scored on dev, the best is tested
  file:<path>       a JSON {task: [prompt texts]} (e.g. transfer of optimized prompts)
Selection uses dev only; each selected prompt is evaluated once on test. With
``--engine vllm`` the same reader runs on vLLM (as for search runs in select_and_test.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness import graft_tasks
from promptwitness.graft_runtime import Ledger, load_tokenizer


def candidates_for(name: str, task: str, model_key: str, published: dict[str, Any]) -> list[Any]:
    if name == "zs_cot":
        return [graft_tasks.flat_prompt("Let's think step by step.")]
    if name == "greater_init":
        return [graft_tasks.initial_prompt(insertion_slot=False)]
    if name == "greater_published":
        return [graft_tasks.flat_prompt(t) for t in published["prompts"].get(model_key, {}).get(task, [])]
    if name.startswith("file:"):
        texts = json.loads(Path(name[5:]).read_text(encoding="utf-8")).get(task, [])
        return [graft_tasks.flat_prompt(t) for t in texts]
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-key", required=True, help="llama3 | gemma2 | qwen3")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--sets", nargs="+", default=["zs_cot", "greater_init", "greater_published"])
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    if args.engine == "vllm":
        from vllm import LLM

        from promptwitness.graft_vllm import VllmReader

        llm = LLM(model=args.model_path, tokenizer=args.model_path, dtype="bfloat16", seed=0,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
                  enable_prefix_caching=True)

        def make_reader(spec: graft_tasks.TaskSpec, ledger: Ledger) -> Any:
            return VllmReader(args.model_path, tokenizer, spec, ledger, args.max_new_tokens, llm=llm)
    else:
        import torch
        from transformers import AutoModelForCausalLM

        from run_graft import Runner

        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                                     attn_implementation=args.attn).to("cuda").eval()

        def make_reader(spec: graft_tasks.TaskSpec, ledger: Ledger) -> Any:
            return Runner(model, tokenizer, spec, ledger, args.max_new_tokens)
    published = json.loads(args.published.read_text(encoding="utf-8"))
    results: dict[str, Any] = json.loads(args.output.read_text()) if args.output.exists() else {}
    for task in args.tasks:
        spec = graft_tasks.spec(task)
        splits = graft_tasks.load_splits(args.data_dir, task)
        for name in args.sets:
            key = f"{task}/{name}"
            if key in results:
                continue
            ledger = Ledger()
            runner = make_reader(spec, ledger)
            pool = candidates_for(name, task, args.model_key, published)[: args.max_candidates]
            if not pool:
                results[key] = {"missing": True}
                continue
            started = perf_counter()
            dev = [runner.evaluate(p, splits["dev"], "dev")["accuracy"] for p in pool] if len(pool) > 1 else [None]
            chosen = max(range(len(pool)), key=lambda i: dev[i] if dev[i] is not None else 0.0)
            test = runner.evaluate(pool[chosen], splits["test"], "test")
            results[key] = {"task": task, "set": name, "candidates": len(pool), "dev": dev,
                            "selected": chosen, "text": pool[chosen].document.messages[0].content,
                            "test_accuracy": test["accuracy"], "test_correct": test["correct"],
                            "test_truncated": test["truncated"], "cost": ledger.phases, "engine": args.engine,
                            "seconds": perf_counter() - started}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"task": task, "set": name, "test": test["accuracy"],
                              "dev_best": dev[chosen]}), flush=True)


if __name__ == "__main__":
    main()
