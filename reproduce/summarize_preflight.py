"""Cross-check cost-probe records against their durable attempt ledger (no scoring)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def summarize(
    ledger_path: Path,
    record_paths: list[Path],
    plan_path: Path,
    *,
    prior_plan_path: Path | None = None,
    startup_failure_paths: list[Path] | None = None,
) -> dict:
    """Reject incomplete, duplicate or mismatched evidence instead of imputing costs."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    phase = plan.get("probe_id")
    expected_models = {
        row["id"] + (f"@{phase}" if phase else ""): row["revision"] for row in plan["models"]
    }
    expected_plans = {model: plan for model in expected_models}
    if phase:
        if prior_plan_path is None:
            raise ValueError("repair summaries must retain the original plan and consumption")
        if hashlib.sha256(prior_plan_path.read_bytes()).hexdigest() != plan["source_plan_sha256"]:
            raise ValueError("repair source plan hash disagrees with the frozen plan")
        prior = json.loads(prior_plan_path.read_text(encoding="utf-8"))
        prior_models = {row["id"]: row["revision"] for row in prior["models"]}
        repair_models = {row["id"]: row["revision"] for row in plan["models"]}
        if (
            prior_models != repair_models
            or prior["request_file_sha256"] != plan["request_file_sha256"]
        ):
            raise ValueError("repair cannot silently change models or requests")
        for row in prior["models"]:
            expected_models[row["id"]] = row["revision"]
            expected_plans[row["id"]] = prior
    elif prior_plan_path is not None:
        raise ValueError("prior plan is only meaningful for an explicitly named repair")
    connection = sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        runs = connection.execute(
            "SELECT model,revision,requests_sha,elapsed,cold_start FROM runs"
        ).fetchall()
        attempts = connection.execute("SELECT model,id,role,status FROM attempts").fetchall()
    finally:
        connection.close()
    startup_evidence = {}
    for path in startup_failure_paths or []:
        failure = json.loads(path.read_text(encoding="utf-8"))
        archived = failure["archive_run"]
        original = failure["ledger_run"]
        if (
            original not in expected_models
            or not archived.startswith(original + ":startup-failed")
            or failure["model_requests_attempted"] != 0
            or archived in startup_evidence
        ):
            raise ValueError("invalid explicit startup-failure evidence")
        startup_evidence[archived] = (original, path)
    complete_runs = []
    startup_costs = []
    for row in runs:
        if row[0] not in startup_evidence:
            complete_runs.append(row)
            continue
        original, path = startup_evidence[row[0]]
        if (
            row[1] != expected_models[original]
            or row[2] != expected_plans[original]["request_file_sha256"]
            or row[3] is None
            or not math.isfinite(row[3])
            or row[3] < 0
            or row[4] is not None
            or any(attempt[0] == row[0] for attempt in attempts)
        ):
            raise ValueError("startup failure is not a finished zero-request pre-load failure")
        startup_costs.append(
            {
                "ledger_model": row[0],
                "elapsed_seconds": row[3],
                "allocated_gpu_hours": row[3] / 3600,
                "model_requests": 0,
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    charged_count = 100 * len(expected_models)
    if (
        len(complete_runs) != len(expected_models)
        or len(attempts) != charged_count
        or len(startup_costs) != len(startup_evidence)
    ):
        raise ValueError("expected complete planned model runs and all charged attempts")
    if any(row[3] != "completed" for row in attempts):
        raise ValueError("failed/reserved attempts require explicit failure cost accounting")
    by_attempt = {(model, item_id): (role, status) for model, item_id, role, status in attempts}
    by_run = {row[0]: row for row in complete_runs}
    if set(by_run) != set(expected_models) or any(
        row[1] != expected_models[row[0]] or row[2] != expected_plans[row[0]]["request_file_sha256"]
        for row in complete_runs
    ):
        raise ValueError("ledger disagrees with the frozen model/input plan")
    records = []
    manifest = []
    for path in record_paths:
        payload = path.read_bytes()
        records.extend(json.loads(line) for line in payload.decode("utf-8").splitlines())
        manifest.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    keys = [(row.get("ledger_model", row["model"]), row["id"]) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != set(by_attempt):
        raise ValueError("duplicate, missing or uncharged records")
    groups = defaultdict(list)
    for row in records:
        ledger_model = row.get("ledger_model", row["model"])
        key = ledger_model, row["id"]
        run = by_run[ledger_model]
        if (
            by_attempt[key] != (row["role"], row["status"])
            or row["status"] != "completed"
            or row["revision"] != run[1]
            or row["requests_sha256"] != run[2]
            or row["scoring_executed"] is not False
            or row["model"] != ledger_model.split("@", 1)[0]
        ):
            raise ValueError("record provenance disagrees with ledger or is not cost-only")
        if "@" in ledger_model and (
            row.get("probe_id") != phase
            or row.get("script_sha256") != plan["script_sha256"]
            or row.get("torch_version") != plan["runtime"]["torch"]
            or row.get("cpu_threads") != plan["cpu_threads"]
            or row["batch_size"] > plan["batch_size"]
        ):
            raise ValueError("repair record disagrees with the frozen runtime")
        for field in ("input_tokens", "output_tokens", "max_new_tokens", "batch_size"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"invalid token/count field: {field}")
        if row["batch_size"] == 0 or row["output_tokens"] > row["max_new_tokens"]:
            raise ValueError("invalid batch/output bounds")
        seconds = row["allocated_seconds_share"]
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("missing/invalid elapsed-time evidence")
        groups[ledger_model, row["family"], row["role"]].append(row)
    for model in expected_models:
        expected_counts = {
            (model, "bfcl", "task"): 32,
            (model, "hotpotqa", "task"): 32,
            (model, "instruction_following", "task"): 32,
            (model, "bfcl", "proposer"): 2,
            (model, "hotpotqa", "proposer"): 1,
            (model, "instruction_following", "proposer"): 1,
        }
        actual_counts = {key: len(value) for key, value in groups.items() if key[0] == model}
        if actual_counts != expected_counts:
            raise ValueError("task/proposer family counts disagree with the frozen plan")
    model_summaries = []
    for model, revision, request_sha, elapsed, cold in sorted(complete_runs):
        if (
            elapsed is None
            or cold is None
            or not 0 <= cold <= elapsed
            or not math.isfinite(elapsed)
        ):
            raise ValueError("unfinished or invalid run timing")
        model_records = [row for row in records if row.get("ledger_model", row["model"]) == model]
        generated_seconds = sum(row["allocated_seconds_share"] for row in model_records)
        overhead = elapsed - cold - generated_seconds
        if overhead < -1e-6 or len(model_records) != 100:
            raise ValueError("inconsistent timing or model request count")
        role_summaries = []
        for (group_model, family, role), rows in sorted(groups.items()):
            if group_model != model:
                continue
            count = len(rows)
            totals = {
                field: sum(row[field] for row in rows)
                for field in ("input_tokens", "output_tokens", "allocated_seconds_share")
            }
            role_summaries.append(
                {
                    "family": family,
                    "role": role,
                    "requests": count,
                    "totals": totals,
                    "mean_input_tokens": totals["input_tokens"] / count,
                    "mean_output_tokens": totals["output_tokens"] / count,
                    "mean_allocated_generation_seconds": totals["allocated_seconds_share"] / count,
                    "mean_allocated_warm_seconds": (
                        totals["allocated_seconds_share"] / count + max(0, overhead) / 100
                    ),
                    "output_cap_reached_requests": sum(
                        row["output_tokens"] == row["max_new_tokens"] for row in rows
                    ),
                }
            )
        model_summaries.append(
            {
                "model": model.split("@", 1)[0],
                "ledger_model": model,
                "probe_id": model.split("@", 1)[1] if "@" in model else None,
                "revision": revision,
                "request_file_sha256": request_sha,
                "requests": 100,
                "failures": 0,
                "elapsed_seconds": elapsed,
                "cold_start_seconds": cold,
                "warm_generation_seconds": generated_seconds,
                "other_warm_allocated_seconds": max(0, overhead),
                "allocated_gpu_hours": elapsed / 3600,
                "roles": role_summaries,
            }
        )
    return {
        "format": "promptwitness.delta.cost-probe-summary/v1",
        "purpose": "G0 resource measurement; not method effectiveness",
        "charged_attempts": charged_count,
        "completed_requests": charged_count,
        "startup_failure_count": len(startup_costs),
        "failures": 0,
        "retries": 0,
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
        "allocated_gpu_hours": sum(row[3] for row in runs) / 3600,
        "startup_failures": startup_costs,
        "models": model_summaries,
        "raw_record_files": manifest,
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prior_plan_sha256": (
            hashlib.sha256(prior_plan_path.read_bytes()).hexdigest() if prior_plan_path else None
        ),
        "latency_scope": "batch elapsed shares; not individual request latency",
        "peak_gpu_memory_mib": None,
        "missing_peak_memory_action": (
            "unavailable; contemporaneous GPU samples are not peak measurements"
        ),
        "limitations": [
            "BFCL probe contains simple Python only; retain diversified protocol categories",
            "Only four training-context proposer measurements per model",
            "No accuracy, eligibility, cost-reduction or optimizer-benefit measurement",
            "Output truncation and slower outlier batches are retained, not filtered",
            "Two main models measured; third-model transfer latency is unmeasured",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path)
    parser.add_argument("--startup-failure", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.ledger,
        args.records,
        args.plan,
        prior_plan_path=args.prior_plan,
        startup_failure_paths=args.startup_failure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps({"completed_requests": result["completed_requests"], "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
