"""Cross-version physical model accounting and per-device interval unions."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .journal import BudgetExhausted
from .statistics import integer


@dataclass(frozen=True, slots=True)
class ResourceLimit:
    calls: int
    input_tokens: int
    output_tokens: int
    gpu_hours: float

    def __post_init__(self) -> None:
        for field in ("calls", "input_tokens", "output_tokens"):
            integer(getattr(self, field), field)
        if (
            isinstance(self.gpu_hours, bool)
            or not math.isfinite(self.gpu_hours)
            or self.gpu_hours < 0
        ):
            raise ValueError("finite nonnegative GPU allocation limit required")


def allocated_gpu_seconds(intervals: Iterable[tuple[str, float, float]]) -> float:
    """Union overlapping reservations per UUID, then sum across UUIDs."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for device, start, end in intervals:
        if (
            not device
            or any(isinstance(v, bool) or not math.isfinite(v) for v in (start, end))
            or not 0 <= start <= end
        ):
            raise ValueError("invalid allocated interval")
        grouped.setdefault(device, []).append((start, end))
    total = 0.0
    for rows in grouped.values():
        end = -1.0
        for left, right in sorted(rows):
            total += max(0.0, right - max(left, end))
            end = max(end, right)
    return total


class ResourceLedger:
    """Physical calls reserve token caps before initiation; failures stay charged.

    Import only a separately verified immutable historical summary and its hash.
    The original database is never overwritten. Unknown failure tokens retain
    their reservations; they are disclosed as conservative accounting, not known
    actual token counts. GPU time counts cold/idle/exit reservation windows.
    """

    def __init__(
        self,
        path: Path,
        *,
        historical_usage: Mapping[str, Any],
        historical_digest: str,
        global_limit: ResourceLimit,
        stage_limits: Mapping[str, ResourceLimit],
        gpu_uuid: str,
    ) -> None:
        expected_usage = {"calls", "input_tokens", "output_tokens", "gpu_hours"}
        if (
            set(historical_usage) != expected_usage
            or len(historical_digest) != 64
            or not gpu_uuid.startswith("GPU-")
        ):
            raise ValueError("verified history and assigned GPU UUID required")
        ResourceLimit(**dict(historical_usage))
        if not stage_limits or any(not stage for stage in stage_limits):
            raise ValueError("fixed named stage grants required")
        specification: dict[str, Any] = {
            "history": dict(historical_usage),
            "history_digest": historical_digest,
            "global_limit": asdict(global_limit),
            "stages": {k: asdict(v) for k, v in stage_limits.items()},
            "gpu_uuid": gpu_uuid,
        }
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS resource_spec (
              singleton INTEGER PRIMARY KEY CHECK(singleton=1), specification TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_calls (
              call_id TEXT PRIMARY KEY, stage TEXT NOT NULL, request_digest TEXT NOT NULL,
              status TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
              tokens_known INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS gpu_allocations (
              allocation_id TEXT PRIMARY KEY, stage TEXT NOT NULL, gpu_uuid TEXT NOT NULL,
              start REAL NOT NULL, end REAL);
        """)
        expected = json.dumps(specification, sort_keys=True, allow_nan=False)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO resource_spec VALUES (1,?)", (expected,))
            existing = self.connection.execute(
                "SELECT specification FROM resource_spec WHERE singleton=1"
            ).fetchone()[0]
            if expected != existing:
                raise ValueError("resource history, device or grant changed on recovery")
        self.specification = specification

    def close(self) -> None:
        self.connection.close()

    def usage(self, stage: str | None = None, *, now: float | None = None) -> dict[str, Any]:
        if stage is not None and stage not in self.specification["stages"]:
            raise ValueError("unknown stages cannot be priced as zero")
        tick = time.time() if now is None else now
        params = (stage, stage)
        row = self.connection.execute(
            "SELECT count(*),coalesce(sum(input_tokens),0),coalesce(sum(output_tokens),0),"
            "coalesce(sum(1-tokens_known),0) FROM model_calls WHERE (? IS NULL OR stage=?)",
            params,
        ).fetchone()
        intervals = self.connection.execute(
            "SELECT gpu_uuid,start,end FROM gpu_allocations WHERE (? IS NULL OR stage=?)", params
        ).fetchall()
        seconds = allocated_gpu_seconds(
            (device, start, tick if end is None else end) for device, start, end in intervals
        )
        base = (
            self.specification["history"]
            if stage is None
            else {"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0}
        )
        return {
            "calls": base["calls"] + row[0],
            "input_tokens": base["input_tokens"] + row[1],
            "output_tokens": base["output_tokens"] + row[2],
            "gpu_hours": base["gpu_hours"] + seconds / 3600,
            "calls_with_reserved_or_unknown_token_cost": row[3],
        }

    def _check(
        self,
        stage: str,
        *,
        add_input: int = 0,
        add_output: int = 0,
        add_calls: int = 0,
        now: float | None = None,
    ) -> None:
        if stage not in self.specification["stages"]:
            raise ValueError("unregistered resource stage")
        for named, limit in (
            (None, self.specification["global_limit"]),
            (stage, self.specification["stages"][stage]),
        ):
            usage = self.usage(named, now=now)
            if (
                usage["calls"] + add_calls > limit["calls"]
                or usage["input_tokens"] + add_input > limit["input_tokens"]
                or usage["output_tokens"] + add_output > limit["output_tokens"]
                or usage["gpu_hours"] >= limit["gpu_hours"]
            ):
                raise BudgetExhausted("cross-version or stage resource grant exhausted")

    def reserve_call(
        self,
        call_id: str,
        stage: str,
        request_digest: str,
        *,
        input_cap: int,
        output_cap: int,
        now: float | None = None,
    ) -> None:
        integer(input_cap, "input_cap")
        integer(output_cap, "output_cap")
        if not call_id or len(request_digest) != 64:
            raise ValueError("unique attempt and complete request digest required")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._check(stage, add_input=input_cap, add_output=output_cap, add_calls=1, now=now)
            self.connection.execute(
                "INSERT INTO model_calls VALUES (?,?,?,?,?,?,0)",
                (call_id, stage, request_digest, "reserved", input_cap, output_cap),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def settle_call(
        self, call_id: str, *, succeeded: bool, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        if type(succeeded) is not bool or (input_tokens is None) != (output_tokens is None):
            raise ValueError("explicit status and both token counts required")
        if succeeded and input_tokens is None:
            raise ValueError("completed calls require actual token counts")
        with self.connection:
            row = self.connection.execute(
                "SELECT status,input_tokens,output_tokens FROM model_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
            if row is None or row[0] != "reserved":
                raise ValueError("missing or already resolved call reservation")
            if input_tokens is not None and output_tokens is not None:
                integer(input_tokens, "input_tokens")
                integer(output_tokens, "output_tokens")
                if input_tokens > row[1] or output_tokens > row[2]:
                    # Retain incurred overruns before stopping; no fake clipping.
                    self.connection.execute(
                        "UPDATE model_calls SET status='overrun',input_tokens=?,"
                        "output_tokens=?,tokens_known=1 WHERE call_id=?",
                        (input_tokens, output_tokens, call_id),
                    )
                    self.connection.commit()
                    raise BudgetExhausted(
                        "incurred tokens exceeded preauthorized caps; retained and stopped"
                    )
                self.connection.execute(
                    "UPDATE model_calls SET status=?,input_tokens=?,output_tokens=?,"
                    "tokens_known=1 WHERE call_id=?",
                    ("completed" if succeeded else "failed", input_tokens, output_tokens, call_id),
                )
            else:
                self.connection.execute(
                    "UPDATE model_calls SET status='failed_unknown_cost' WHERE call_id=?",
                    (call_id,),
                )

    def begin_gpu(self, allocation_id: str, stage: str, gpu_uuid: str, *, start: float) -> None:
        if gpu_uuid != self.specification["gpu_uuid"] or not allocation_id:
            raise ValueError("only the assigned GPU UUID may be allocated")
        allocated_gpu_seconds([(gpu_uuid, start, start)])
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._check(stage, now=start)
            self.connection.execute(
                "INSERT INTO gpu_allocations VALUES (?,?,?,?,NULL)",
                (allocation_id, stage, gpu_uuid, start),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def end_gpu(self, allocation_id: str, *, end: float) -> None:
        with self.connection:
            row = self.connection.execute(
                "SELECT gpu_uuid,start,end FROM gpu_allocations WHERE allocation_id=?",
                (allocation_id,),
            ).fetchone()
            if row is None or row[2] is not None:
                raise ValueError("missing or closed GPU allocation")
            allocated_gpu_seconds([(row[0], row[1], end)])
            self.connection.execute(
                "UPDATE gpu_allocations SET end=? WHERE allocation_id=?", (end, allocation_id)
            )
