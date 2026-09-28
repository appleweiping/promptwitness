"""Produce a descriptive four-arm table and one-page English pilot note.

Reads frozen search and holdout sidecars; does not call a model or tune prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from promptwitness.structured_data import TASKS

METHODS = ("A", "B", "C", "D")
SEEDS = (1, 2, 3)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _meter_sum(meter: dict[str, Any], field: str) -> float:
    return sum(float(bucket.get(field, 0)) for bucket in meter.values())


def _row(search: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    validation = search["validation"]
    selected = validation[search["selected_checkpoint"]]
    rounds = search["rounds"]
    proposals = [proposal for round_record in rounds for proposal in round_record["proposals"]]
    accepted = sum(bool(round_record["accepted"]) for round_record in rounds)
    scored = sum("score" in proposal for proposal in proposals)
    structural_valid = sum(bool(proposal.get("structurally_valid")) for proposal in proposals)
    accepted_pairs = [
        (round_record["parent_score"], proposal["score"])
        for round_record in rounds
        if round_record["accepted"]
        for proposal in round_record["proposals"]
        if proposal.get("block_texts") == round_record["survivor_block_texts"]
        and "score" in proposal
    ]
    if len(accepted_pairs) != accepted:
        raise ValueError("accepted survivor lacks exactly one scored candidate")
    meter = search["meter"]
    generation_attempts = sum(
        float(bucket.get("attempts", 0))
        for kind, bucket in meter.items()
        if kind.endswith("_generation")
    ) + float(meter.get("gradient_observation", {}).get("generation_attempts", 0))
    generated_tokens = sum(
        float(bucket.get("output_tokens", 0))
        for kind, bucket in meter.items()
        if kind.endswith("_generation")
    ) + float(meter.get("gradient_observation", {}).get("generation_output_tokens", 0))
    validation_truncated = sum(item["score"]["truncated"] for item in validation)
    fit_truncated = sum(
        round_record["parent_score"]["truncated"]
        + sum(
            proposal["score"]["truncated"]
            for proposal in round_record["proposals"]
            if "score" in proposal
        )
        for round_record in rounds
    )
    return {
        "task": search["task"],
        "method": search["method"],
        "seed": search["seed"],
        "initial_validation_correct_32": validation[0]["score"]["correct"],
        "final_validation_correct_32": selected["score"]["correct"],
        "holdout_correct_100": holdout["correct"],
        "validation_format_violations_32": selected["score"]["format_violations"],
        "holdout_format_violations_100": holdout["format_violations"],
        "accepted_edits": accepted,
        "accepted_loss_improvements": sum(
            candidate["mean_conditional_answer_loss"] < parent["mean_conditional_answer_loss"]
            for parent, candidate in accepted_pairs
        ),
        "accepted_task_improvements": sum(
            candidate["correct"] > parent["correct"] for parent, candidate in accepted_pairs
        ),
        "accepted_task_regressions": sum(
            candidate["correct"] < parent["correct"] for parent, candidate in accepted_pairs
        ),
        "completed_rounds": len(rounds),
        "accepted_edit_fraction": accepted / len(rounds) if rounds else 0.0,
        "retained_round_scored_candidates": scored,
        "retained_round_structurally_valid_proposals": structural_valid,
        "retained_round_proposals": len(proposals),
        "retained_round_structural_valid_fraction": (
            structural_valid / len(proposals)
            if proposals and not search.get("incomplete_stage")
            else None
        ),
        "search_generation_attempts": generation_attempts,
        "search_generation_output_tokens": generated_tokens,
        "gradient_forward_tokens": meter.get("gradient_observation", {}).get("forward_tokens", 0),
        "gradient_backward_tokens": meter.get("gradient_observation", {}).get("backward_tokens", 0),
        "token_proposal_forward_tokens": meter.get("token_proposal_forward", {}).get(
            "forward_tokens", 0
        ),
        "optimization_wall_seconds": search.get("optimization_seconds", 0),
        "gpu_residency_wall_seconds": search["gpu_residency_wall_seconds"],
        "model_operation_wall_seconds": _meter_sum(meter, "model_seconds")
        + meter.get("gradient_observation", {}).get("generation_seconds", 0)
        + meter.get("gradient_observation", {}).get("forward_seconds", 0)
        + meter.get("gradient_observation", {}).get("backward_seconds", 0)
        + meter.get("token_proposal_forward", {}).get("wall_seconds", 0),
        "failed_operations": len(search["failed_operations"]),
        "recorded_search_truncations_lower_bound": fit_truncated + validation_truncated,
        "incomplete_stage": search.get("incomplete_stage"),
        "holdout_truncations": holdout["truncated"],
        "holdout_generation_attempts_shared": holdout["generation_attempts"],
    }


def _average(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def run(args: argparse.Namespace) -> None:
    manifest = _read(args.holdout_dir / "holdout_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("holdout evaluation is not complete")
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                key = f"{task}-{method}-seed{seed}"
                search_path = args.records_dir / f"{key}.json"
                search = _read(search_path)
                holdout_key = manifest["assignments"][key]
                holdout = _read(args.holdout_dir / f"{holdout_key}.json")
                if (
                    search["status"] != "frozen_validation_selected"
                    or holdout["status"] != "completed"
                ):
                    raise ValueError(f"incomplete result for {key}")
                if (
                    hashlib.sha256(search_path.read_bytes()).hexdigest()
                    != manifest["search_record_sha256"][key]
                    or holdout["task"] != task
                    or holdout["prompt_key"] != holdout_key
                    or holdout["block_texts"] != search["final_block_texts"]
                    or holdout["model_path"] != search["model_path"]
                    or holdout["max_new_tokens"] != search["max_new_tokens"]
                    or holdout["holdout_shard_sha256"] != manifest["holdout_shard_sha256"][task]
                    or [item["row_id"] for item in holdout["rows"]] != holdout["holdout_row_ids"]
                    or len(holdout["rows"]) != 100
                ):
                    raise ValueError(f"search and holdout identities differ for {key}")
                rows.append(_row(search, holdout))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Structured-gradient BBH development pilot",
        "",
        "Descriptive results from two tasks, three search seeds, and frozen held-out prompts.",
        "Scores are counts: validation /32, holdout /100. No significance or SOTA claim.",
        "GPU residency is model-loaded process wall time, including CPU work; "
        "it is not kernel time.",
        "Identical frozen prompts share a single deterministic holdout generation cache.",
        "Proposal counts cover retained rounds; an incomplete stage is listed in the CSV. "
        "Search truncations are a recorded lower bound when a round stops mid-stage.",
        "",
        "| Task | Arm | Seed | Initial val | Final val | Holdout | Accepted/rounds | "
        "Recorded valid/scored/proposed | Val/Holdout format violations | "
        "Search gen calls/tokens | "
        "Gradient F/B tokens | GPU residency s | Failures | Truncations search≥/holdout |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['seed']} | "
            f"{row['initial_validation_correct_32']} | {row['final_validation_correct_32']} | "
            f"{row['holdout_correct_100']} | {row['accepted_edits']}/{row['completed_rounds']} | "
            f"{row['retained_round_structurally_valid_proposals']}/"
            f"{row['retained_round_scored_candidates']}/"
            f"{row['retained_round_proposals']} | "
            f"{row['validation_format_violations_32']}/{row['holdout_format_violations_100']} | "
            f"{int(row['search_generation_attempts'])}/"
            f"{int(row['search_generation_output_tokens'])} | "
            f"{int(row['gradient_forward_tokens'])}/{int(row['gradient_backward_tokens'])} | "
            f"{row['gpu_residency_wall_seconds']:.1f} | {row['failed_operations']} | "
            f"{row['recorded_search_truncations_lower_bound']}/"
            f"{row['holdout_truncations']} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    arm_rows = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    d = arm_rows["D"]
    c = arm_rows["C"]
    b = arm_rows["B"]
    note = [
        "# Preliminary note for Mihai: structured gradient prompt search",
        "",
        "This development pilot adapts GReaTer-style answer-loss input gradients to "
        "whole semantic prompt blocks. The model was frozen; the sampled reasoning "
        "was fixed for the differentiable forward. Arm B was an adapted token-level "
        "baseline; C chose blocks randomly while discarding shadow gradients; "
        "D selected blocks by mean mapped token-gradient norm.",
        "",
        f"Across two BBH tasks and three search seeds, mean held-out correct /100 was "
        f"B {_average(b, 'holdout_correct_100'):.1f}, "
        f"C {_average(c, 'holdout_correct_100'):.1f}, and "
        f"D {_average(d, 'holdout_correct_100'):.1f}. "
        f"Mean accepted edits per run were C {_average(c, 'accepted_edits'):.2f} "
        f"and D {_average(d, 'accepted_edits'):.2f}. "
        f"Among accepted edits, task-score improvements numbered "
        f"B {sum(row['accepted_task_improvements'] for row in b)}, "
        f"C {sum(row['accepted_task_improvements'] for row in c)}, and "
        f"D {sum(row['accepted_task_improvements'] for row in d)}; "
        f"B task-score regressions numbered "
        f"{sum(row['accepted_task_regressions'] for row in b)}. "
        f"Mean optimization wall time was B {_average(b, 'optimization_wall_seconds'):.1f}s, "
        f"C {_average(c, 'optimization_wall_seconds'):.1f}s, and "
        f"D {_average(d, 'optimization_wall_seconds'):.1f}s.",
        "",
        "These are small-sample descriptive comparisons. The time cap was soft and "
        "actual consumed time and token work are in the accompanying table. "
        "The pilot does not establish a general advantage, statistical significance, "
        "or an original GReaTer reproduction. Examine accepted-step loss/accuracy "
        "trajectories and the fixed-candidate diagnostic before a larger study.",
    ]
    args.english.parent.mkdir(parents=True, exist_ok=True)
    args.english.write_text("\n".join(note) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
