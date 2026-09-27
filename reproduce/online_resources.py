"""Continue verified closed physical accounting; never import an authored ledger.

The historical database stays read-only. Unknown failed token costs remain
their conservative reservations, not fabricated actual zeros. The new stage
uses the user's total ceilings, without an additional development quota.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from promptwitness.incremental.budget import (
    ResourceLedger,
    ResourceLimit,
    allocated_gpu_seconds,
)

GPU = "GPU-18ce5833-27a4-ac4b-f952-49cc52045714"
CEILING = ResourceLimit(800000, 2000000000, 200000000, 1000)
STAGE = "research_continuation"


def closed_history(source: Path, expected_sha256: str) -> dict:
    """Price an independently verified immutable actual ResourceLedger copy.

    Caller must verify the copy against the original registered physical DB.
    A pending call/allocation is not reinterpreted as closed or free. This
    function cannot establish that a cooperating caller supplied a real ledger;
    never point it at the explicitly separate AUTHORED test databases.
    """
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("verified physical ledger copy changed")
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as database:
        specs = database.execute("SELECT specification FROM resource_spec").fetchall()
        if len(specs) != 1:
            raise ValueError("one original physical resource specification required")
        spec = json.loads(specs[0][0])
        if spec["gpu_uuid"] != GPU or spec["global_limit"] != asdict(CEILING):
            raise ValueError("original assigned device or total ceiling differs")
        calls = database.execute(
            "SELECT status,input_tokens,output_tokens,tokens_known FROM model_calls"
        ).fetchall()
        intervals = database.execute("SELECT gpu_uuid,start,end FROM gpu_allocations").fetchall()
        if any(r[0] == "reserved" for r in calls) or any(r[2] is None for r in intervals):
            raise ValueError("unresolved original call/allocation; inspect, do not reset")
        if any(
            r[0] not in {"completed", "failed", "failed_unknown_cost", "overrun"} for r in calls
        ):
            raise ValueError("unknown original physical call status")
        if any(r[0] != GPU for r in intervals):
            raise ValueError("original allocation uses an unassigned device")
        base = spec["history"]
        usage = {
            "calls": base["calls"] + len(calls),
            "input_tokens": base["input_tokens"] + sum(r[1] for r in calls),
            "output_tokens": base["output_tokens"] + sum(r[2] for r in calls),
            "gpu_hours": base["gpu_hours"] + allocated_gpu_seconds(intervals) / 3600,
        }
        ResourceLimit(**usage)
    if source.read_bytes() != raw:
        raise ValueError("historical ledger mutated during read")
    return {
        "format": "promptwitness.resource-continuation-history/v1",
        "source_ledger_sha256": expected_sha256,
        "original_historical_digest": spec["history_digest"],
        "usage": usage,
        "original_local_calls_with_reserved_or_unknown_token_cost": sum(1 - r[3] for r in calls),
        "historical_unknown_cost_count": "NOT_ENCODED_IN_ORIGINAL_BASE_SUMMARY",
        "unknown_costs_are_conservative_not_known_actual_tokens": True,
    }


def continue_ledger(path: Path, history: dict) -> ResourceLedger:
    if history.get("format") != "promptwitness.resource-continuation-history/v1":
        raise ValueError("verified continuous physical history required")
    return ResourceLedger(
        path,
        historical_usage=history["usage"],
        historical_digest=history["source_ledger_sha256"],
        global_limit=CEILING,
        stage_limits={STAGE: CEILING},
        gpu_uuid=GPU,
    )
