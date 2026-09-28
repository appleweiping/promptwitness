"""Evaluate only frozen four-arm prompts on the private BBH holdout shards.

Requires all 24 task/method/seed search records before reading any holdout label.
Identical frozen prompts share one deterministic holdout generation cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness.gradient_backend import FrozenGradientBackend
from promptwitness.structured_data import TASKS, load_split
from promptwitness.structured_prompt import StructuredPrompt
from promptwitness.structured_search import initial_prompt, score_multiple_choice

METHODS = ("A", "B", "C", "D")
SEEDS = (1, 2, 3)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frozen_prompt(block_texts: dict[str, str]) -> StructuredPrompt:
    prompt = initial_prompt()
    if set(block_texts) != {block.block_id for block in prompt.blocks}:
        raise ValueError("frozen prompt has a missing or extra block")
    for block in prompt.blocks:
        if block_texts[block.block_id] != block.text:
            prompt = prompt.replace_block(block.block_id, block_texts[block.block_id])
    return prompt


def _records(
    records_dir: Path, budget_seconds: float, max_new_tokens: int, model_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    if budget_seconds <= 0:
        raise ValueError("holdout requires a positive, frozen optimization budget")
    records: dict[str, Any] = {}
    record_hashes: dict[str, str] = {}
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                key = f"{task}-{method}-seed{seed}"
                path = records_dir / f"{key}.json"
                source = path.read_bytes()
                record = json.loads(source)
                if (
                    record.get("status") != "frozen_validation_selected"
                    or record.get("method") != method
                    or record.get("task") != task
                    or record.get("seed") != seed
                    or record.get("max_new_tokens") != max_new_tokens
                    or record.get("model_path") != str(model_path)
                ):
                    raise ValueError(f"search result is not frozen for {key}")
                if method != "A" and (
                    record.get("comparison_mode") != "budget_capped"
                    or record.get("optimization_budget_seconds") != budget_seconds
                    or record.get("round_limit") != 8
                ):
                    raise ValueError(f"optimizer budget differs from frozen protocol for {key}")
                if method == "A" and record.get("round_limit") != 0:
                    raise ValueError(f"baseline A must have zero search rounds for {key}")
                _frozen_prompt(record["final_block_texts"])
                records[key] = record
                record_hashes[key] = hashlib.sha256(source).hexdigest()
    return records, record_hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records, record_hashes = _records(
        args.records_dir, args.budget_seconds, args.max_new_tokens, args.model_path
    )
    private_manifest = json.loads(
        (args.holdout_dir / "shard_manifest.json").read_text(encoding="utf-8")
    )
    holdout_hashes = {task: private_manifest["sha256"][f"{task}.holdout.jsonl"] for task in TASKS}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started = perf_counter()
    manifest_path = args.output_dir / "holdout_manifest.json"
    prior: dict[str, Any] | None = None
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("search_record_sha256") != record_hashes
            or prior.get("holdout_shard_sha256") != holdout_hashes
            or prior.get("model_path") != str(args.model_path)
            or prior.get("max_new_tokens") != args.max_new_tokens
            or prior.get("optimization_budget_seconds") != args.budget_seconds
        ):
            raise ValueError("existing holdout manifest differs from frozen search or data")
    result: dict[str, Any] = {
        "format": "promptwitness.structured-gradient-holdout/v1",
        "status": "running",
        "optimization_budget_seconds": args.budget_seconds,
        "max_new_tokens": args.max_new_tokens,
        "model_path": str(args.model_path),
        "search_record_sha256": record_hashes,
        "holdout_shard_sha256": holdout_hashes,
        "assignments": {},
        "unique_prompts": {},
        "run_attempts": (prior.get("run_attempts", 0) if prior else 0) + 1,
        "prior_process_wall_seconds": prior.get("process_wall_seconds", 0.0) if prior else 0.0,
        "unknown_prior_wall_seconds": bool(prior and "process_wall_seconds" not in prior)
        or bool(prior and prior.get("unknown_prior_wall_seconds")),
    }
    _write(manifest_path, result)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, use_fast=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda:0")
        backend = FrozenGradientBackend(model, tokenizer)
        result["model_loading_seconds"] = perf_counter() - started
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["process_wall_seconds"] = (
            result["prior_process_wall_seconds"] + perf_counter() - started
        )
        _write(manifest_path, result)
        raise
    try:
        for task in TASKS:
            rows = load_split(args.holdout_dir, args.split_manifest, task, "holdout")
            row_ids = [row.row_id for row in rows]
            if len(row_ids) != 100 or len(set(row_ids)) != 100:
                raise ValueError("holdout shard must contain exactly 100 unique rows")
            seen: dict[tuple[tuple[str, str], ...], str] = {}
            for method in METHODS:
                for seed in SEEDS:
                    key = f"{task}-{method}-seed{seed}"
                    block_texts = records[key]["final_block_texts"]
                    signature = tuple(sorted(block_texts.items()))
                    if signature not in seen:
                        seen[signature] = f"{task}-prompt-{len(seen):02d}"
                    prompt_key = seen[signature]
                    result["assignments"][key] = prompt_key
                    result["unique_prompts"][prompt_key] = block_texts
            _write(args.output_dir / "holdout_manifest.json", result)
            for signature, prompt_key in seen.items():
                prompt = _frozen_prompt(dict(signature))
                output = args.output_dir / f"{prompt_key}.json"
                if output.exists():
                    record = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        record.get("task") != task
                        or record.get("prompt_key") != prompt_key
                        or record.get("block_texts") != dict(signature)
                        or record.get("model_path") != str(args.model_path)
                        or record.get("max_new_tokens") != args.max_new_tokens
                        or record.get("holdout_shard_sha256") != holdout_hashes[task]
                        or record.get("holdout_row_ids") != row_ids
                        or len({item["row_id"] for item in record["rows"]}) != len(record["rows"])
                        or any(item["row_id"] not in row_ids for item in record["rows"])
                    ):
                        raise ValueError("holdout cache identity or row population differs")
                    if record.get("status") == "completed":
                        if [item["row_id"] for item in record["rows"]] != row_ids:
                            raise ValueError(
                                "completed holdout cache has incomplete row population"
                            )
                        continue
                    if record.get("inflight_row_id"):
                        record.setdefault("unknown_generation_attempts", []).append(
                            {"row_id": record["inflight_row_id"], "reason": "prior run interrupted"}
                        )
                        record.pop("inflight_row_id", None)
                    record["status"] = "running"
                    _write(output, record)
                else:
                    record = {
                        "format": "promptwitness.structured-gradient-holdout-prompt/v1",
                        "status": "running",
                        "task": task,
                        "prompt_key": prompt_key,
                        "block_texts": dict(signature),
                        "model_path": str(args.model_path),
                        "max_new_tokens": args.max_new_tokens,
                        "holdout_shard_sha256": holdout_hashes[task],
                        "holdout_row_ids": row_ids,
                        "rows": [],
                        "generation_attempts": 0,
                        "generation_output_tokens": 0,
                        "generation_input_tokens": 0,
                        "generation_seconds": 0.0,
                    }
                completed = {item["row_id"] for item in record["rows"]}
                for row in rows:
                    if row.row_id in completed:
                        continue
                    input_tokens = len(
                        prompt.render_tokens(tokenizer, {"input": row.question}).input_ids
                    )
                    record["generation_attempts"] += 1
                    record["generation_input_tokens"] += input_tokens
                    record["inflight_row_id"] = row.row_id
                    _write(output, record)
                    try:
                        response, output_tokens, seconds, ended, _ = backend.generate_response(
                            prompt, {"input": row.question}, max_new_tokens=args.max_new_tokens
                        )
                    except BaseException as error:
                        record["status"] = "failed"
                        record.setdefault("unknown_generation_attempts", []).append(
                            {"row_id": row.row_id, "reason": type(error).__name__}
                        )
                        record.pop("inflight_row_id", None)
                        _write(output, record)
                        raise
                    score = score_multiple_choice(
                        response,
                        row.answer,
                        max_letter="C" if task == "logical_deduction_three_objects" else "F",
                    )
                    truncated = output_tokens >= args.max_new_tokens and not ended
                    record["rows"].append(
                        {
                            "row_id": row.row_id,
                            "answer": row.answer,
                            "response": response,
                            "predicted": score.predicted,
                            "correct": score.correct and not truncated,
                            "format_valid": score.format_valid and not truncated,
                            "truncated": truncated,
                            "output_tokens": output_tokens,
                        }
                    )
                    record["generation_output_tokens"] += output_tokens
                    record["generation_seconds"] += seconds
                    record.pop("inflight_row_id", None)
                    _write(output, record)
                record["status"] = "completed"
                record["correct"] = sum(item["correct"] for item in record["rows"])
                record["format_violations"] = sum(
                    not item["format_valid"] for item in record["rows"]
                )
                record["truncated"] = sum(item["truncated"] for item in record["rows"])
                _write(output, record)
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        raise
    else:
        result["status"] = "completed"
    finally:
        result["process_wall_seconds"] = (
            result["prior_process_wall_seconds"] + perf_counter() - started
        )
        _write(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
