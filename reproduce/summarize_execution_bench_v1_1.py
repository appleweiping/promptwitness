"""Audit external cost evidence; publish no raw response or private connection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from promptwitness.incremental.budget import allocated_gpu_seconds
from promptwitness.incremental.sampling import digest


def summarize(root: Path) -> dict:
    summary_path = root / "bench-v1_1.json"
    raw_path = root / "responses-v1_1.jsonl"
    ledger_path = root / "resources-v1_1.sqlite"
    requests_path = root / "execution-requests-v1_1.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    requests = {
        r["id"]: r
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line))
    }
    with sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        spec = json.loads(
            connection.execute("SELECT specification FROM resource_spec").fetchone()[0]
        )
        calls = {row[0]: row[1:] for row in connection.execute("SELECT * FROM model_calls")}
        allocations = connection.execute(
            "SELECT gpu_uuid,start,end FROM gpu_allocations"
        ).fetchall()
    if (
        len(rows) != 128
        or len(calls) != 128
        or len(requests) != 32
        or any(end is None for _, _, end in allocations)
    ):
        raise ValueError("incomplete execution, calls or unreleased allocation")
    if len({r["call_id"] for r in rows}) != 128:
        raise ValueError("duplicate physical attempts")
    configurations = {result["config"]: result for result in summary["configs"]}
    vectors = {}
    for row in rows:
        stage, identity, status, inputs, outputs, known = calls[row["call_id"]]
        config = configurations[row["config"]]
        config_index = int(row["config"].split("-")[1])
        if (stage, status, known, inputs, outputs) != (
            "unblocking_v1_1",
            "completed",
            1,
            row["input_tokens"],
            row["output_tokens"],
        ) or identity != digest(
            {
                "execution": summary["identity"],
                "batch_size": config["batch_size"],
                "request": requests[row["id"]],
                "independent_replicate": config_index,
            }
        ):
            raise ValueError("raw response, complete request identity or ledger changed")
        vectors.setdefault(row["config"], {})[row["id"]] = (
            hashlib.sha256(row["output"].encode("utf-8")).hexdigest(),
            inputs,
            outputs,
        )
    if any(set(vector) != set(requests) for vector in vectors.values()):
        raise ValueError("configuration did not evaluate the same complete workload")
    comparisons = []
    baseline = summary["configs"][0]
    for result in summary["configs"][1:]:
        matched = sum(
            vectors[result["config"]][unit] == vectors[baseline["config"]][unit]
            for unit in requests
        )
        comparisons.append(
            {
                "config": result["config"],
                "same_text_and_token_counts_as_batch_1": matched,
                "different": 32 - matched,
                "speedup_vs_batch_1": baseline["elapsed_seconds"] / result["elapsed_seconds"],
            }
        )
    best = min(summary["configs"][:3], key=lambda result: result["elapsed_seconds"])
    repeated = summary["configs"][3]
    if best["batch_size"] != repeated["batch_size"]:
        raise ValueError("best complete configuration was not repeated")
    repeat_matched = sum(
        vectors[best["config"]][unit] == vectors[repeated["config"]][unit] for unit in requests
    )
    hours = allocated_gpu_seconds(allocations) / 3600
    stage = {
        "calls": len(calls),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "gpu_hours": hours,
    }
    total = {key: spec["history"][key] + value for key, value in stage.items()}
    return {
        "format": "promptwitness.delta.execution-bench/v1.1",
        "status": "COMPLETED_COST_ONLY",
        "configurations_tested": 3,
        "best_repeated_once": True,
        "fixed_complete_requests_per_configuration": 32,
        "workload_profiles": dict(
            Counter(row["family"] + ":" + row["profile"] for row in requests.values())
        ),
        "execution_identity": summary["identity"],
        "cold_start_seconds": summary["cold_start_seconds"],
        "configurations": summary["configs"],
        "output_comparisons": comparisons,
        "best_repeat_same_text_and_token_counts": repeat_matched,
        "physical_stage_usage": stage,
        "physical_cross_version_usage": total,
        "failed_calls": 0,
        "unknown_cost_calls": 0,
        "new_startup_failures": 0,
        "output_cap_hits": sum(r["cap_hit"] for r in rows),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (summary_path, raw_path, ledger_path, requests_path)
        },
        "allocation_intervals": [
            {"gpu_uuid": device, "start_unix": start, "end_unix": end}
            for device, start, end in allocations
        ],
        "accounting": (
            "supervisor reservation through child exit; interval union, "
            "not summed overlapping request latency"
        ),
        "scientific_scores_measured": False,
        "native_optimizer_savings_measured": False,
        "online_batch_invariance_validated": False,
        "limits": [
            "same-workload repeat is diagnostic, not a universal deterministic proof",
            "batch composition changes observed outputs; "
            "an adaptive online finite-table certificate is not admitted",
            "no prefix backend cache; tokenizer/rendered reuse only",
            "input length quantiles, not a proven full-campaign cost distribution; "
            "retain v1 long-output stress evidence",
            "BFCL accepted native demos and complete proposer workflow remain unmeasured",
            "Olmo, Mistral, scoring, pilot and full cold-switch schedules remain unpriced",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "physical_stage_usage",
                    "physical_cross_version_usage",
                    "output_comparisons",
                    "best_repeat_same_text_and_token_counts",
                )
            },
            indent=2,
        )
    )
