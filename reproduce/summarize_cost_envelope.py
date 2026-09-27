"""Validate all G0 phases and summarize cost profiles, never task correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def finite_nonnegative(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("invalid nonnegative cost field: " + field)
    return value


def summarize(
    ledger_path: Path,
    record_paths: list[Path],
    plan_path: Path,
    prior_summary_path: Path,
    startup_failure_paths: list[Path],
) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    expected_models = {
        row["id"] + "@" + plan["probe_id"]: row["revision"] for row in plan["models"]
    }
    if prior["charged_attempts"] != plan["prior_charged_requests"] or not math.isclose(
        prior["allocated_gpu_hours"], plan["prior_allocated_gpu_hours"], rel_tol=1e-12
    ):
        raise ValueError("prior costs disagree with the frozen supplement plan")
    with sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        runs = {
            row[0]: row
            for row in connection.execute(
                "SELECT model,revision,requests_sha,elapsed,cold_start FROM runs"
            )
        }
        attempts = list(connection.execute("SELECT model,id,role,status FROM attempts"))
    failures = {}
    for path in startup_failure_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row["ledger_run"] not in expected_models
            or row["model_requests_attempted"] != 0
            or not row["archive_run"].startswith(row["ledger_run"] + ":startup-failed")
        ):
            raise ValueError("invalid additional startup evidence")
        if row["archive_run"] in failures:
            raise ValueError("duplicate startup evidence")
        failures[row["archive_run"]] = {
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_run": row["ledger_run"],
        }
    prior_models = {row["ledger_model"]: row for row in prior["models"]}
    prior_failures = {row["ledger_model"]: row for row in prior["startup_failures"]}
    if set(runs) != set(expected_models) | set(prior_models) | set(prior_failures) | set(failures):
        raise ValueError("unexpected, missing or unreported ledger run")
    for key, previous in prior_models.items():
        row = runs[key]
        if (
            row[1] != previous["revision"]
            or row[2] != previous["request_file_sha256"]
            or row[3] != previous["elapsed_seconds"]
            or row[4] != previous["cold_start_seconds"]
        ):
            raise ValueError("past run changed after archival summary")
    for key, previous in prior_failures.items():
        row = runs[key]
        if row[3] != previous["elapsed_seconds"] or row[4] is not None:
            raise ValueError("past startup failure changed")
    for key, metadata in failures.items():
        row = runs[key]
        if (
            row[1] != expected_models[metadata["original_run"]]
            or row[2] != plan["request_file_sha256"]
            or row[4] is not None
            or any(item[0] == key for item in attempts)
        ):
            raise ValueError("startup failure contains inference or changed provenance")
        metadata["elapsed_seconds"] = finite_nonnegative(row[3], "startup elapsed")
    if len(attempts) != plan["aggregate_g0_attempt_ceiling"] or any(
        row[3] != "completed" for row in attempts
    ):
        raise ValueError("all planned completed attempts are required; no dropped failures")
    counts = Counter(row[0] for row in attempts)
    if any(counts[key] != previous["requests"] for key, previous in prior_models.items()):
        raise ValueError("past attempt accounting changed")
    if any(counts[key] != plan["requests_per_model"] for key in expected_models):
        raise ValueError("incomplete envelope model")
    records = []
    artifacts = []
    for path in record_paths:
        payload = path.read_bytes()
        records.extend(json.loads(line) for line in payload.decode("utf-8").splitlines())
        artifacts.append(
            {
                "file": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    expected_keys = {
        (model, item_id): (role, status)
        for model, item_id, role, status in attempts
        if model in expected_models
    }
    keys = [(row["ledger_model"], row["id"]) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys):
        raise ValueError("duplicate, missing or uncharged envelope records")
    groups = defaultdict(list)
    for row in records:
        key = row["ledger_model"], row["id"]
        role, status = expected_keys[key]
        if (
            status != "completed"
            or row["status"] != status
            or row["role"] != role
            or row["model"] != key[0].split("@", 1)[0]
            or row["revision"] != expected_models[key[0]]
            or row["requests_sha256"] != plan["request_file_sha256"]
            or row["script_sha256"] != plan["script_sha256"]
            or row["scoring_executed"] is not False
        ):
            raise ValueError("invalid envelope provenance or label access")
        if (
            row["torch_version"] != plan["runtime"]["torch"]
            or row["cpu_threads"] != plan["cpu_threads"]
            or row["max_input_tokens"] != plan["max_input_tokens"]
            or row["disable_compile"] is not True
            or row["proposers_batch_one"] is not True
        ):
            raise ValueError("runtime differs from the frozen plan")
        cap = plan["max_new_tokens"]["proposer" if role == "proposer" else row["family"]]
        max_batch = (
            plan["proposer_batch_size"]
            if role == "proposer"
            else plan["task_stress_batch_size"]
            if row["profile"] == "eight_demo_stress_proxy"
            else plan["task_base_batch_size"]
        )
        for field in ("input_tokens", "output_tokens", "batch_size", "max_new_tokens"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("invalid token/count field")
        if (
            row["max_new_tokens"] != cap
            or row["output_tokens"] > cap
            or row["input_tokens"] > plan["max_input_tokens"]
            or not 1 <= row["batch_size"] <= max_batch
        ):
            raise ValueError("request exceeds frozen bounds")
        finite_nonnegative(row["allocated_seconds_share"], "generation share")
        finite_nonnegative(row["batch_elapsed_seconds"], "batch elapsed")
        if not math.isclose(
            row["batch_elapsed_seconds"] / row["batch_size"],
            row["allocated_seconds_share"],
            rel_tol=1e-10,
        ):
            raise ValueError("batch accounting does not sum")
        groups[key[0], row["family"], role, row["profile"]].append(row)
    models = []
    total_input = prior["input_tokens"]
    total_output = prior["output_tokens"]
    total_seconds = 3600 * prior["allocated_gpu_hours"] + sum(
        row["elapsed_seconds"] for row in failures.values()
    )
    for model, revision in expected_models.items():
        run = runs[model]
        if run[1] != revision or run[2] != plan["request_file_sha256"]:
            raise ValueError("model run provenance mismatch")
        elapsed = finite_nonnegative(run[3], "model elapsed")
        cold = finite_nonnegative(run[4], "cold start")
        warm_generation = sum(
            row["allocated_seconds_share"] for row in records if row["ledger_model"] == model
        )
        overhead = elapsed - cold - warm_generation
        if overhead < -1e-6:
            raise ValueError("elapsed time undercounts generation")
        profiles = []
        for (_, family, role, profile), rows in sorted(groups.items()):
            if rows[0]["ledger_model"] != model:
                continue
            expected_count = plan["per_model_profiles"][
                "serial_proposer_proxy"
                if role == "proposer"
                else "task_"
                + ("eight_demo_stress_proxy" if profile == "eight_demo_stress_proxy" else "base")
            ][family]
            if len(rows) != expected_count:
                raise ValueError("frozen family/profile coverage changed")
            profile_input = sum(row["input_tokens"] for row in rows)
            profile_output = sum(row["output_tokens"] for row in rows)
            profile_seconds = sum(row["allocated_seconds_share"] for row in rows)
            total_input += profile_input
            total_output += profile_output
            profiles.append(
                {
                    "family": family,
                    "role": role,
                    "profile": profile,
                    "requests": len(rows),
                    "mean_input_tokens": profile_input / len(rows),
                    "mean_output_tokens": profile_output / len(rows),
                    "mean_warm_allocated_seconds": profile_seconds / len(rows)
                    + max(0, overhead) / 100,
                    "max_output_tokens": max(row["output_tokens"] for row in rows),
                    "output_cap_reached": sum(
                        row["output_tokens"] == row["max_new_tokens"] for row in rows
                    ),
                    "generation_allocated_seconds": profile_seconds,
                }
            )
        if len(profiles) != 9:
            raise ValueError("missing family/profile measurement")
        models.append(
            {
                "model": model.split("@", 1)[0],
                "ledger_model": model,
                "revision": revision,
                "requests": 100,
                "elapsed_seconds": elapsed,
                "cold_start_seconds": cold,
                "other_warm_seconds": max(0, overhead),
                "profiles": profiles,
            }
        )
        total_seconds += elapsed
    return {
        "format": "promptwitness.delta.cost-envelope-summary/v1",
        "completed_requests_all_g0_phases": len(attempts),
        "real_model_failures": 0,
        "startup_failures_all_g0_phases": [
            *prior["startup_failures"],
            *[{"ledger_model": key, **metadata} for key, metadata in failures.items()],
        ],
        "input_tokens_all_g0_phases": total_input,
        "output_tokens_all_g0_phases": total_output,
        "allocated_gpu_hours_all_g0_phases": total_seconds / 3600,
        "models_envelope_phase": models,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prior_summary_sha256": hashlib.sha256(prior_summary_path.read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "raw_files": artifacts,
        "scores_or_effectiveness_measured": False,
        "limitations": plan["proxy_limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--startup-failure", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.ledger, args.records, args.plan, args.prior_summary, args.startup_failure
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "completed_requests_all_g0_phases",
                    "input_tokens_all_g0_phases",
                    "output_tokens_all_g0_phases",
                    "allocated_gpu_hours_all_g0_phases",
                )
            },
            indent=2,
        )
    )
