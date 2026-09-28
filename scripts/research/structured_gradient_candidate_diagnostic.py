"""Rank one fixed block-candidate set by gradient and random order, then score it.

This is a fit-only diagnostic, not holdout selection or a new optimizer run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness.gradient_backend import FrozenGradientBackend
from promptwitness.structured_data import TASKS, load_split
from promptwitness.structured_prompt import StructuredPrompt
from promptwitness.structured_search import (
    first_order_rewrite_delta,
    initial_prompt,
    propose_block_rewrites,
    ranked_blocks,
)

if __package__:
    from .structured_gradient_search import CostJournal, _observe_batch, _score_batch
else:
    from structured_gradient_search import CostJournal, _observe_batch, _score_batch


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.output.exists():
        raise ValueError("diagnostic output already exists; preserve it and use a new path")
    rows = load_split(args.shard_dir, args.manifest, args.task, "fit")
    rng = random.Random(args.seed)
    batch = tuple(rng.sample(rows, len(rows))[:4])
    started = perf_counter()
    record: dict[str, Any] = {
        "format": "promptwitness.structured-gradient-candidate-diagnostic/v1",
        "status": "running",
        "task": args.task,
        "seed": args.seed,
        "fit_row_ids": [row.row_id for row in batch],
        "max_new_tokens": args.max_new_tokens,
        "meter": {},
        "failed_operations": [],
        "candidates": [],
    }
    journal = CostJournal(record, args.output)
    journal.save()
    try:
        with journal.operation("model_load"):
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
        parent = initial_prompt()
        observations = _observe_batch(backend, parent, batch, journal, args.max_new_tokens)
        sensitivities = {
            block_id: sum(
                observation.block_sensitivity[block_id] for _, observation in observations
            )
            / len(observations)
            for block_id in observations[0][1].block_sensitivity
        }
        record["block_sensitivity"] = sensitivities
        parent_score = _score_batch(
            backend,
            parent,
            batch,
            args.task,
            journal,
            args.max_new_tokens,
            "diagnostic_parent",
            include_loss=True,
        )
        record["parent_score"] = parent_score
        journal.save()
        candidate_prompts: list[tuple[int, StructuredPrompt]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for block in parent.blocks:
            if not block.editable:
                continue
            for variant in range(2):
                with journal.operation("diagnostic_rewrite_generation") as bucket:
                    rewrite, input_tokens, output_tokens, seconds = propose_block_rewrites(
                        model,
                        tokenizer,
                        parent,
                        block.block_id,
                        count=1,
                        variant_start=variant,
                    )[0]
                    bucket["input_tokens"] = bucket.get("input_tokens", 0) + input_tokens
                    bucket["output_tokens"] = bucket.get("output_tokens", 0) + output_tokens
                    bucket["model_seconds"] = bucket.get("model_seconds", 0) + seconds
                item: dict[str, Any] = {
                    "block_id": block.block_id,
                    "variant": variant,
                    "rewrite": rewrite,
                    "proposal_input_tokens": input_tokens,
                    "proposal_output_tokens": output_tokens,
                    "proposal_seconds": seconds,
                }
                try:
                    candidate = parent.replace_block(block.block_id, rewrite)
                except ValueError as error:
                    item["rejection"] = str(error)
                    record["candidates"].append(item)
                    journal.save()
                    continue
                signature = tuple(sorted((part.block_id, part.text) for part in candidate.blocks))
                if candidate.document == parent.document or signature in seen:
                    item["rejection"] = "unchanged_or_duplicate"
                    record["candidates"].append(item)
                    journal.save()
                    continue
                seen.add(signature)
                item["predicted_delta"] = first_order_rewrite_delta(
                    model, tokenizer, parent, candidate, block.block_id, observations
                )
                index = len(record["candidates"])
                record["candidates"].append(item)
                candidate_prompts.append((index, candidate))
                journal.save()
        block_order = ranked_blocks(sensitivities, len(sensitivities))
        gradient_order = sorted(
            (index for index, _ in candidate_prompts),
            key=lambda index: (
                block_order.index(record["candidates"][index]["block_id"]),
                record["candidates"][index]["predicted_delta"] is None,
                record["candidates"][index]["predicted_delta"] or 0,
                index,
            ),
        )
        random_order = [index for index, _ in candidate_prompts]
        rng.shuffle(random_order)
        record["gradient_order"] = gradient_order
        record["random_order"] = random_order
        journal.save()
        for index, candidate in candidate_prompts:
            record["candidates"][index]["score"] = _score_batch(
                backend,
                candidate,
                batch,
                args.task,
                journal,
                args.max_new_tokens,
                "diagnostic_candidate",
                include_loss=True,
            )
            journal.save()
        record["gradient_first_index"] = gradient_order[0] if gradient_order else None
        record["random_first_index"] = random_order[0] if random_order else None
        record["status"] = "completed" if candidate_prompts else "no_valid_candidate"
    except BaseException as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        raise
    finally:
        record["process_wall_seconds"] = perf_counter() - started
        journal.save()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
