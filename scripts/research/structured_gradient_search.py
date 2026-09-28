"""Run one four-arm pilot method and seed using fit and validation shards only.

All records are private raw research artifacts. Do not commit row text or labels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from promptwitness.gradient_backend import FrozenGradientBackend, GradientObservation
from promptwitness.structured_data import TASKS, BBHRow, load_split
from promptwitness.structured_prompt import StructuredPrompt
from promptwitness.structured_search import (
    first_order_rewrite_delta,
    initial_prompt,
    propose_block_rewrites,
    ranked_blocks,
    score_multiple_choice,
)
from promptwitness.structured_token_baseline import propose_token_replacements


class BudgetExhausted(Exception):
    """A common optimization wall-time cap was reached before another operation."""


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class CostJournal:
    def __init__(self, record: dict[str, Any], path: Path) -> None:
        self.record = record
        self.path = path
        self.optimization_started: float | None = None
        self.budget_seconds = 0.0
        self.model_loaded_at: float | None = None

    def save(self) -> None:
        _write(self.path, self.record)

    def check_budget(self) -> None:
        if (
            self.optimization_started is not None
            and self.budget_seconds > 0
            and perf_counter() - self.optimization_started >= self.budget_seconds
        ):
            raise BudgetExhausted

    @contextmanager
    def operation(self, kind: str, *, input_tokens: int = 0) -> Iterator[dict[str, float]]:
        self.check_budget()
        meter: dict[str, dict[str, float]] = self.record["meter"]
        bucket = meter.setdefault(kind, {})
        bucket["attempts"] = bucket.get("attempts", 0) + 1
        bucket["input_tokens"] = bucket.get("input_tokens", 0) + input_tokens
        self.record["inflight"] = {"kind": kind, "input_tokens": input_tokens}
        self.save()
        started = perf_counter()
        try:
            yield bucket
        except BaseException as error:
            bucket["failures"] = bucket.get("failures", 0) + 1
            self.record["failed_operations"].append(
                {
                    "kind": kind,
                    "type": type(error).__name__,
                    "message": str(error),
                    "partial_cost_unknown": True,
                }
            )
            raise
        else:
            bucket["successes"] = bucket.get("successes", 0) + 1
        finally:
            bucket["wall_seconds"] = bucket.get("wall_seconds", 0) + perf_counter() - started
            self.record["inflight"] = None
            self.save()


def _add(bucket: dict[str, float], key: str, value: int | float) -> None:
    bucket[key] = bucket.get(key, 0) + value


def _prompt_texts(prompt: StructuredPrompt) -> dict[str, str]:
    return {block.block_id: block.text for block in prompt.blocks}


def _score_batch(
    backend: FrozenGradientBackend,
    prompt: StructuredPrompt,
    rows: tuple[BBHRow, ...],
    task: str,
    journal: CostJournal,
    max_new_tokens: int,
    phase: str,
    *,
    include_loss: bool,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        input_tokens = len(
            prompt.render_tokens(backend.tokenizer, {"input": row.question}).input_ids
        )
        with journal.operation(f"{phase}_generation", input_tokens=input_tokens) as bucket:
            response, output_tokens, seconds, ended, response_ids = backend.generate_response(
                prompt, {"input": row.question}, max_new_tokens=max_new_tokens
            )
            _add(bucket, "output_tokens", output_tokens)
            _add(bucket, "model_seconds", seconds)
        score = score_multiple_choice(
            response,
            row.answer,
            max_letter="C" if task == "logical_deduction_three_objects" else "F",
        )
        truncated = output_tokens >= max_new_tokens and not ended
        detail: dict[str, Any] = {
            "row_id": row.row_id,
            "answer": row.answer,
            "response": response,
            "predicted": score.predicted,
            "correct": score.correct and not truncated,
            "format_valid": score.format_valid and not truncated,
            "truncated": truncated,
            "output_tokens": output_tokens,
        }
        if include_loss:
            with journal.operation(f"{phase}_answer_loss") as bucket:
                loss, forward_tokens, seconds = backend.conditional_answer_loss(
                    prompt, {"input": row.question}, response_ids, row.answer
                )
                _add(bucket, "forward_tokens", forward_tokens)
                _add(bucket, "model_seconds", seconds)
            detail["conditional_answer_loss"] = loss
        details.append(detail)
    result = {
        "correct": sum(item["correct"] for item in details),
        "total": len(details),
        "format_violations": sum(not item["format_valid"] for item in details),
        "truncated": sum(item["truncated"] for item in details),
        "rows": details,
    }
    if include_loss:
        result["mean_conditional_answer_loss"] = sum(
            item["conditional_answer_loss"] for item in details
        ) / len(details)
    return result


def _observe_batch(
    backend: FrozenGradientBackend,
    prompt: StructuredPrompt,
    rows: tuple[BBHRow, ...],
    journal: CostJournal,
    max_new_tokens: int,
) -> tuple[tuple[str, GradientObservation], ...]:
    observations: list[tuple[str, GradientObservation]] = []
    for row in rows:
        input_tokens = len(
            prompt.render_tokens(backend.tokenizer, {"input": row.question}).input_ids
        )
        with journal.operation("gradient_observation", input_tokens=input_tokens) as bucket:
            _add(bucket, "generation_attempts", 1)
            journal.save()

            def on_stage(stage: str, tokens: int, seconds: float) -> None:
                if stage == "generation_completed":
                    _add(bucket, "generation_successes", 1)
                    _add(bucket, "generation_output_tokens", tokens)
                    _add(bucket, "generation_seconds", seconds)
                elif stage == "differentiable_forward_started":
                    _add(bucket, "forward_attempts", 1)
                elif stage == "differentiable_forward_completed":
                    _add(bucket, "forward_successes", 1)
                    _add(bucket, "forward_tokens", tokens)
                    _add(bucket, "forward_seconds", seconds)
                elif stage == "differentiable_backward_started":
                    _add(bucket, "backward_attempts", 1)
                elif stage == "differentiable_backward_completed":
                    _add(bucket, "backward_successes", 1)
                    _add(bucket, "backward_tokens", tokens)
                    _add(bucket, "backward_seconds", seconds)
                journal.save()

            result = backend.observe(
                prompt,
                {"input": row.question},
                row.answer,
                max_reasoning_tokens=max_new_tokens,
                on_stage=on_stage,
            )
        observations.append((row.question, result))
    return tuple(observations)


def _mean_sensitivity(
    observations: tuple[tuple[str, GradientObservation], ...],
) -> dict[str, float]:
    names = observations[0][1].block_sensitivity.keys()
    return {
        name: sum(observation.block_sensitivity[name] for _, observation in observations)
        / len(observations)
        for name in names
    }


def _candidate_prompts(
    method: str,
    model: Any,
    tokenizer: Any,
    prompt: StructuredPrompt,
    observations: tuple[tuple[str, GradientObservation], ...],
    round_index: int,
    rng: random.Random,
    journal: CostJournal,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    candidates: list[dict[str, Any]] = []
    if method == "B":
        with journal.operation("token_proposal_forward") as bucket:
            prior_forward_tokens = bucket.get("forward_tokens", 0)

            def on_prefix_forward(stage: str, tokens: int) -> None:
                name = (
                    "prefix_forward_attempts" if stage == "attempt" else "prefix_forward_successes"
                )
                _add(bucket, name, 1)
                if stage == "success":
                    _add(bucket, "forward_tokens", tokens)
                journal.save()

            token_candidates, forward_tokens = propose_token_replacements(
                model,
                tokenizer,
                prompt,
                observations,
                round_index,
                topk=11,
                shortlist=6,
                on_prefix_forward=on_prefix_forward,
            )
            if forward_tokens != bucket.get("forward_tokens", 0) - prior_forward_tokens:
                raise ValueError("token proposal forward accounting mismatch")
        for proposal in token_candidates:
            candidates.append(
                {
                    "prompt": proposal.prompt,
                    "block_id": proposal.block_id,
                    "predicted_delta": proposal.predicted_delta,
                    "token_position": proposal.position,
                    "old_token": proposal.old_token,
                    "new_token": proposal.new_token,
                }
            )
        return candidates, tuple(dict.fromkeys(proposal.block_id for proposal in token_candidates))

    editable = [block.block_id for block in prompt.blocks if block.editable]
    if method == "D":
        selected = ranked_blocks(_mean_sensitivity(observations), 2)
    else:
        selected = tuple(rng.sample(editable, k=min(2, len(editable))))
    for block_id in selected:
        for variant_index in range(3):
            with journal.operation("block_rewrite_generation") as bucket:
                (rewrite, input_tokens, output_tokens, seconds) = propose_block_rewrites(
                    model, tokenizer, prompt, block_id, count=1, variant_start=variant_index
                )[0]
                _add(bucket, "input_tokens", input_tokens)
                _add(bucket, "output_tokens", output_tokens)
                _add(bucket, "model_seconds", seconds)
            item: dict[str, Any] = {
                "block_id": block_id,
                "rewrite": rewrite,
                "proposal_tokens": output_tokens,
                "proposal_seconds": seconds,
            }
            try:
                candidate = prompt.replace_block(block_id, rewrite)
            except ValueError as error:
                item["rejection"] = f"structural: {error}"
                candidates.append(item)
                continue
            if candidate.document.messages == prompt.document.messages:
                item["rejection"] = "unchanged"
                candidates.append(item)
                continue
            item["prompt"] = candidate
            item["predicted_delta"] = (
                first_order_rewrite_delta(
                    model, tokenizer, prompt, candidate, block_id, observations
                )
                if method == "D"
                else None
            )
            candidates.append(item)
    if method == "C":
        rng.shuffle(candidates)
    else:
        candidates.sort(
            key=lambda item: (item.get("predicted_delta") is None, item.get("predicted_delta") or 0)
        )
    return candidates, selected


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
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
    return model, tokenizer


def _run_loaded(args: argparse.Namespace, record: dict[str, Any], journal: CostJournal) -> None:
    fit_rows = load_split(args.shard_dir, args.manifest, args.task, "fit")
    validation_rows = load_split(args.shard_dir, args.manifest, args.task, "validation")
    rng = random.Random(args.seed)
    fit_rows = tuple(rng.sample(fit_rows, len(fit_rows)))
    with journal.operation("model_load"):
        model, tokenizer = _load_model(args)
    backend = FrozenGradientBackend(model, tokenizer)
    journal.model_loaded_at = perf_counter()
    parent = initial_prompt()
    initial = parent
    journal.optimization_started = perf_counter()
    journal.budget_seconds = args.budget_seconds
    for round_index in range(args.rounds if args.method != "A" else 0):
        batch = fit_rows[4 * round_index : 4 * (round_index + 1)]
        stage = "parent_score"
        try:
            journal.check_budget()
            parent_score = _score_batch(
                backend,
                parent,
                batch,
                args.task,
                journal,
                args.max_new_tokens,
                "fit_parent",
                include_loss=True,
            )
            stage = "gradient_observation"
            observations = _observe_batch(backend, parent, batch, journal, args.max_new_tokens)
            sensitivities = _mean_sensitivity(observations)
            stage = "candidate_proposal"
            proposals, selected = _candidate_prompts(
                args.method, model, tokenizer, parent, observations, round_index, rng, journal
            )
            best = parent
            best_score = parent_score["correct"]
            best_loss = parent_score["mean_conditional_answer_loss"]
            seen: set[tuple[tuple[str, str], ...]] = set()
            candidate_rows: list[dict[str, Any]] = [
                {
                    **{key: value for key, value in proposal.items() if key != "prompt"},
                    "structurally_valid": proposal.get("prompt") is not None,
                }
                for proposal in proposals
            ]
            budget_hit = False
            stage = "candidate_score"
            for proposal_index, proposal in enumerate(proposals):
                entry = candidate_rows[proposal_index]
                candidate = proposal.get("prompt")
                if candidate is None:
                    continue
                signature = tuple(sorted(_prompt_texts(candidate).items()))
                if signature in seen:
                    entry["rejection"] = "duplicate"
                    continue
                seen.add(signature)
                try:
                    score = _score_batch(
                        backend,
                        candidate,
                        batch,
                        args.task,
                        journal,
                        args.max_new_tokens,
                        "fit_candidate",
                        include_loss=True,
                    )
                except BudgetExhausted:
                    budget_hit = True
                    break
                entry["score"] = score
                entry["block_texts"] = _prompt_texts(candidate)
                if args.method == "B":
                    if score["mean_conditional_answer_loss"] < best_loss:
                        best = candidate
                        best_loss = score["mean_conditional_answer_loss"]
                elif score["correct"] > best_score:
                    best = candidate
                    best_score = score["correct"]
        except BudgetExhausted:
            record["budget_stop_before_round_completion"] = round_index + 1
            record["incomplete_stage"] = stage
            journal.save()
            break
        accepted = best is not parent
        if accepted:
            parent = best
        record["rounds"].append(
            {
                "round": round_index + 1,
                "fit_row_ids": [row.row_id for row in batch],
                "parent_score": parent_score,
                "conditional_losses": [item.loss for _, item in observations],
                "block_sensitivity": sensitivities,
                "selected_blocks": selected,
                "proposals": candidate_rows,
                "accepted": accepted,
                "budget_stopped_candidates": budget_hit,
                "survivor_block_texts": _prompt_texts(parent),
            }
        )
        journal.save()
        if budget_hit:
            record["budget_stop_before_round_completion"] = round_index + 1
            record["incomplete_stage"] = "candidate_score"
            journal.save()
            break
    if journal.optimization_started is not None:
        record["optimization_seconds"] = perf_counter() - journal.optimization_started
        record["budget_overrun_seconds"] = (
            max(0.0, record["optimization_seconds"] - args.budget_seconds)
            if args.budget_seconds > 0
            else None
        )
    journal.optimization_started = None
    # Two fixed validation candidates for every optimizer: initial and final.
    candidates = [initial] if args.method == "A" else [initial, parent]
    validation: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        score = _score_batch(
            backend,
            candidate,
            validation_rows,
            args.task,
            journal,
            args.max_new_tokens,
            "validation",
            include_loss=False,
        )
        validation.append(
            {"checkpoint": index, "block_texts": _prompt_texts(candidate), "score": score}
        )
        record["validation"] = validation
        journal.save()
    best_validation = max(
        validation, key=lambda item: (item["score"]["correct"], -item["checkpoint"])
    )
    record["selected_checkpoint"] = best_validation["checkpoint"]
    record["final_block_texts"] = best_validation["block_texts"]
    record["final_validation_correct"] = best_validation["score"]["correct"]
    record["status"] = "frozen_validation_selected"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds < 0 or args.rounds > 8 or args.max_new_tokens < 1 or args.budget_seconds < 0:
        raise ValueError("expected 0-8 rounds, positive generation limit and nonnegative budget")
    started = perf_counter()
    record: dict[str, Any] = {
        "format": "promptwitness.structured-gradient-search/v2",
        "status": "starting",
        "method": args.method,
        "task": args.task,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "model_dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "fit_batch_size": 4,
        "round_limit": args.rounds,
        "optimization_budget_seconds": args.budget_seconds,
        "comparison_mode": "calibration" if args.budget_seconds == 0 else "budget_capped",
        "budget_semantics": (
            "soft wall cap checked before each model operation; one operation may overrun"
        ),
        "meter": {},
        "failed_operations": [],
        "rounds": [],
    }
    journal = CostJournal(record, args.output)
    journal.save()
    try:
        record["status"] = "running"
        _run_loaded(args, record, journal)
    except BaseException as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        raise
    finally:
        if journal.optimization_started is not None:
            record["optimization_seconds"] = perf_counter() - journal.optimization_started
            record["budget_overrun_seconds"] = (
                max(0.0, record["optimization_seconds"] - args.budget_seconds)
                if args.budget_seconds > 0
                else None
            )
        record["process_wall_seconds"] = perf_counter() - started
        if journal.model_loaded_at is not None:
            record["gpu_residency_wall_seconds"] = perf_counter() - journal.model_loaded_at
        record["inflight"] = None
        journal.save()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--budget-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
