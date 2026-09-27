"""Validate the final bounded cost calibration and retain all prior consumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from reproduce.run_native_context_probe import validate_requests
from reproduce.summarize_cost_envelope import finite_nonnegative


def ledger_rows(path: Path) -> tuple[dict, dict]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        runs = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT model,revision,requests_sha,elapsed,cold_start FROM runs"
            )
        }
        attempts = {
            (row[0], row[1]): row[2:]
            for row in connection.execute("SELECT model,id,role,status FROM attempts")
        }
    return runs, attempts


def summarize(
    ledger: Path,
    archived_ledger: Path,
    records: list[Path],
    plan_path: Path,
    requests_path: Path,
    prior_summary_path: Path,
    *,
    completed_model_boundary_only: bool = False,
) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
    validate_requests(requests)
    if (
        hashlib.sha256(requests_path.read_bytes()).hexdigest() != plan["request_file_sha256"]
        or hashlib.sha256(archived_ledger.read_bytes()).hexdigest() != plan["prior_ledger_sha256"]
        or hashlib.sha256(prior_summary_path.read_bytes()).hexdigest()
        != plan["prior_summary_sha256"]
        or prior["ledger_sha256"] != plan["prior_ledger_sha256"]
    ):
        raise ValueError("changed frozen provenance")
    previous_runs, previous_attempts = ledger_rows(archived_ledger)
    runs, attempts = ledger_rows(ledger)
    if any(runs.get(key) != value for key, value in previous_runs.items()) or any(
        attempts.get(key) != value for key, value in previous_attempts.items()
    ):
        raise ValueError("past ledger consumption changed")
    if len(previous_attempts) != prior["completed_requests_all_g0_phases"]:
        raise ValueError("prior attempt total changed")
    if not math.isclose(
        sum(finite_nonnegative(row[2], "prior elapsed") for row in previous_runs.values()) / 3600,
        prior["allocated_gpu_hours_all_g0_phases"],
        rel_tol=1e-12,
    ):
        raise ValueError("prior time total changed")
    models = {row["id"] + "@" + plan["probe_id"]: row for row in plan["models"]}
    missing_models = []
    if completed_model_boundary_only:
        missing_models = [metadata["id"] for model, metadata in models.items() if model not in runs]
        models = {model: metadata for model, metadata in models.items() if model in runs}
        if not models:
            raise ValueError("no completed model boundary available")
    if set(runs) != set(previous_runs) | set(models):
        raise ValueError("unexpected or missing model run")
    raw = [
        json.loads(line)
        for path in records
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    keys = [(row["ledger_model"], row["id"]) for row in raw]
    expected_keys = {(model, row["id"]) for model in models for row in requests}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("duplicate or missing calibrated output")
    if set(attempts) != set(previous_attempts) | expected_keys:
        raise ValueError("uncharged or unreported attempt")
    totals = {
        "requests": prior["completed_requests_all_g0_phases"],
        "input_tokens": prior["input_tokens_all_g0_phases"],
        "output_tokens": prior["output_tokens_all_g0_phases"],
        "allocated_gpu_hours": prior["allocated_gpu_hours_all_g0_phases"],
    }
    request_by_id = {row["id"]: row for row in requests}
    summarized_models = []
    for model, metadata in models.items():
        revision, request_sha, elapsed, cold = runs[model]
        finite_nonnegative(elapsed, "elapsed")
        finite_nonnegative(cold, "cold")
        if revision != metadata["revision"] or request_sha != plan["request_file_sha256"]:
            raise ValueError("model run provenance changed")
        rows = [row for row in raw if row["ledger_model"] == model]
        for row in rows:
            request = request_by_id[row["id"]]
            if (
                attempts[model, row["id"]] != (request["role"], "completed")
                or row["model"] != metadata["id"]
                or row["revision"] != revision
                or row["requests_sha256"] != request_sha
                or row["script_sha256"] != plan["script_sha256"]
                or row["status"] != "completed"
                or row["scoring_executed"] is not False
                or any(
                    row[key] != request.get(key)
                    for key in (
                        "family",
                        "role",
                        "profile",
                        "max_new_tokens",
                        "repeat_of",
                    )
                )
            ):
                raise ValueError("calibration provenance or label access changed")
            if (
                type(row["batch_size"]) is not int
                or type(row["cpu_threads"]) is not int
                or type(row["max_input_tokens"]) is not int
                or row["batch_size"] != 1
                or row["cpu_threads"] != 4
                or row["torch_version"] != plan["runtime"]["torch"]
                or row["max_input_tokens"] != 32768
                or row["disable_compile"] is not True
                or row["deterministic_algorithms"] is not True
                or row["cublas_workspace_config"] != ":4096:8"
                or row["decoding"] != {"do_sample": False, "enable_thinking": False}
            ):
                raise ValueError("frozen execution differs")
            for key in ("input_tokens", "output_tokens"):
                if type(row[key]) is not int or not 0 <= row[key] <= (
                    32768 if key == "input_tokens" else request["max_new_tokens"]
                ):
                    raise ValueError("invalid actual token accounting")
            for key in ("batch_elapsed_seconds", "allocated_seconds_share"):
                finite_nonnegative(row[key], key)
            if row["batch_elapsed_seconds"] != row["allocated_seconds_share"]:
                raise ValueError("single-unit time accounting differs")
        warm = sum(row["allocated_seconds_share"] for row in rows)
        overhead = elapsed - cold - warm
        if overhead < -1e-6:
            raise ValueError("time consumption undercounted")
        original_rows = {row["id"]: row for row in rows if row["repeat_of"] is None}
        repeat_checks = []
        for row in rows:
            if row["repeat_of"] is not None:
                original = original_rows[row["repeat_of"]]
                repeat_checks.append(
                    {
                        "id": row["id"],
                        "repeat_of": row["repeat_of"],
                        "identical_text_and_token_counts": all(
                            row[key] == original[key]
                            for key in (
                                "output",
                                "input_tokens",
                                "output_tokens",
                            )
                        ),
                    }
                )
        profile_counts = Counter(
            (row["family"], row["role"], row["profile"]) for row in original_rows.values()
        )
        profiles = []
        for key, count in sorted(profile_counts.items()):
            group = [
                r for r in original_rows.values() if (r["family"], r["role"], r["profile"]) == key
            ]
            profiles.append(
                {
                    "family": key[0],
                    "role": key[1],
                    "profile": key[2],
                    "requests": count,
                    "mean_input_tokens": sum(r["input_tokens"] for r in group) / count,
                    "mean_output_tokens": sum(r["output_tokens"] for r in group) / count,
                    "mean_warm_allocated_seconds": sum(r["allocated_seconds_share"] for r in group)
                    / count
                    + max(0, overhead) / 70,
                    "output_cap_reached": sum(
                        r["output_tokens"] == r["max_new_tokens"] for r in group
                    ),
                }
            )
        summarized_models.append(
            {
                "model": metadata["id"],
                "revision": revision,
                "requests": len(rows),
                "elapsed_seconds": elapsed,
                "cold_start_seconds": cold,
                "other_warm_seconds": max(0, overhead),
                "profiles": profiles,
                "repeat_checks": repeat_checks,
                "sampled_repeat_consistency": all(
                    row["identical_text_and_token_counts"] for row in repeat_checks
                ),
            }
        )
        totals["requests"] += len(rows)
        totals["input_tokens"] += sum(row["input_tokens"] for row in rows)
        totals["output_tokens"] += sum(row["output_tokens"] for row in rows)
        totals["allocated_gpu_hours"] += elapsed / 3600
    planned_missing_calls = 70 * len(missing_models)
    if totals["requests"] + planned_missing_calls != plan["aggregate_g0_attempt_ceiling"]:
        raise ValueError("fixed aggregate accounting changed")
    return {
        "format": "promptwitness.delta.native-context-summary/v1",
        "status": "COST_MEASUREMENT_NOT_SCIENTIFIC_EFFECTS_OR_G0_APPROVAL",
        "consumed_all_g0_phases": totals,
        "calibration_complete": not missing_models,
        "remaining_model_costs_not_measured": missing_models,
        "remaining_planned_calls_not_executed": planned_missing_calls,
        "models_native_context_phase": summarized_models,
        "prior_startup_failures_retained": prior["startup_failures_all_g0_phases"],
        "real_inference_failures": 0,
        "task_accuracy_or_method_effects_measured": False,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "raw_files": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in records
        ],
        "limitations": [
            "Four fixed repeats per model are a diagnostic, not proof of a deterministic "
            "response table or cross-device stability.",
            "Valid fit-side demonstrations are a cost fixture, not the native shuffled "
            "bootstrap/teacher sequence.",
            "The two typed proposer inputs use the official formatter with static training-side "
            "summaries; full native proposal workflow costs remain unverified.",
            "No method benefits, held-out task scores, final prompt selection or scientific "
            "confidence intervals are measured here.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--archived-ledger", type=Path, required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-model-boundary-only", action="store_true")
    args = parser.parse_args()
    result = summarize(
        args.ledger,
        args.archived_ledger,
        args.records,
        args.plan,
        args.requests,
        args.prior_summary,
        completed_model_boundary_only=args.completed_model_boundary_only,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["consumed_all_g0_phases"], indent=2))
