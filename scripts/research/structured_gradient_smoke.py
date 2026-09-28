"""Run one real BBH gradient, block rewrite, and fresh answer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness.diff import DiffOptions, MessageAlignment, compare_prompts
from promptwitness.gradient_backend import FrozenGradientBackend
from promptwitness.structured_data import TASKS, load_split
from promptwitness.structured_search import (
    initial_prompt,
    propose_block_rewrites,
    ranked_blocks,
    score_multiple_choice,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no real GPU gradient was run")
    row = load_split(args.shard_dir, args.manifest, args.task, "fit")[0]
    loaded = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)
    loading_seconds = perf_counter() - loaded
    original = initial_prompt()
    backend = FrozenGradientBackend(model, tokenizer)
    all_frozen = all(not parameter.requires_grad for parameter in model.parameters())
    first_parameter = next(model.parameters())
    first_weight = first_parameter.detach().flatten()[:1024].clone()
    started = perf_counter()
    observation = backend.observe(
        original,
        {"input": row.question},
        row.answer,
        max_reasoning_tokens=args.max_reasoning_tokens,
        finite_difference_block="reasoning",
    )
    check = observation.finite_difference
    if check is None or check["predicted"] <= 0 or check["measured"] <= 0:
        raise ValueError("finite-difference direction did not confirm the input gradient")
    if abs(check["predicted"] - check["measured"]) > max(1.0, 0.5 * check["predicted"]):
        raise ValueError("finite-difference magnitude was inconsistent with the gradient")
    parent_score = score_multiple_choice(
        observation.reasoning,
        row.answer,
        max_letter="C" if args.task == "logical_deduction_three_objects" else "F",
    )
    parent_truncated = (
        observation.generated_tokens >= args.max_reasoning_tokens
        and not observation.generation_ended
    )
    attempts: list[dict[str, Any]] = []
    for block_id in ranked_blocks(observation.block_sensitivity):
        for (
            rewrite,
            proposal_input_tokens,
            proposal_tokens,
            proposal_seconds,
        ) in propose_block_rewrites(
            model, tokenizer, original, block_id, count=args.candidates_per_block
        ):
            attempt: dict[str, Any] = {
                "block_id": block_id,
                "rewrite": rewrite,
                "proposal_tokens": proposal_tokens,
                "proposal_input_tokens": proposal_input_tokens,
                "proposal_seconds": proposal_seconds,
                "block_gradient_mean_norm": observation.block_sensitivity[block_id],
            }
            try:
                modified = original.replace_block(block_id, rewrite)
            except ValueError as error:
                attempt["status"] = "structurally_invalid"
                attempt["error"] = str(error)
                attempts.append(attempt)
                continue
            response, response_tokens, response_seconds, response_ended, _ = (
                backend.generate_response(
                    modified,
                    {"input": row.question},
                    max_new_tokens=args.max_reasoning_tokens,
                )
            )
            score = score_multiple_choice(
                response,
                row.answer,
                max_letter="C" if args.task == "logical_deduction_three_objects" else "F",
            )
            truncated = response_tokens >= args.max_reasoning_tokens and not response_ended
            diff = compare_prompts(
                original.document,
                modified.document,
                DiffOptions(message_alignment=MessageAlignment.ID),
            )
            attempt.update(
                {
                    "status": "scored",
                    "response": response,
                    "response_tokens": response_tokens,
                    "response_seconds": response_seconds,
                    "truncated": truncated,
                    "score": {
                        "correct": score.correct and not truncated,
                        "format_valid": score.format_valid and not truncated,
                        "predicted": score.predicted,
                    },
                    "diff": [
                        {"kind": change.kind.value, "path": change.path} for change in diff.changes
                    ],
                }
            )
            attempts.append(attempt)
            if (
                score.correct
                and not truncated
                and not (parent_score.correct and not parent_truncated)
            ):
                break
        if any(attempt.get("status") == "scored" for attempt in attempts):
            break
    return {
        "format": "promptwitness.structured-gradient-smoke/v1",
        "status": (
            "completed"
            if any(a.get("status") == "scored" for a in attempts)
            else "no_valid_rewrite"
        ),
        "task": args.task,
        "fit_row_id": row.row_id,
        "model_path": str(args.model_path),
        "device": args.device,
        "dtype": "bfloat16" if args.device.startswith("cuda") else "float32",
        "loading_seconds": loading_seconds,
        "experiment_seconds": perf_counter() - started,
        "all_model_parameters_frozen": all_frozen,
        "sampled_weight_unchanged": bool(
            torch.equal(first_weight, first_parameter.detach().flatten()[:1024])
        ),
        "parent": {
            "response": observation.reasoning,
            "answer": row.answer,
            "truncated": parent_truncated,
            "score": {
                "correct": parent_score.correct and not parent_truncated,
                "format_valid": parent_score.format_valid and not parent_truncated,
                "predicted": parent_score.predicted,
            },
            "conditional_answer_loss": observation.loss,
            "block_sensitivity": observation.block_sensitivity,
            "token_positions": observation.token_positions,
            "answer_tokens": observation.answer_tokens,
            "finite_difference": observation.finite_difference,
            "generation_seconds": observation.generation_seconds,
            "gradient_seconds": observation.gradient_seconds,
            "generated_tokens": observation.generated_tokens,
            "forward_tokens": observation.forward_tokens,
            "backward_tokens": observation.backward_tokens,
        },
        "attempts": attempts,
        "note": "One fit row is a mechanism check, not pilot accuracy or a method comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-reasoning-tokens", type=int, default=320)
    parser.add_argument("--candidates-per-block", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    try:
        result = run(args)
    except Exception as error:
        result = {
            "format": "promptwitness.structured-gradient-smoke/v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": perf_counter() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
