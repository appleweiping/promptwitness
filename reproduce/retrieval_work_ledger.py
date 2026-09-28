"""Durable physical attempt receipts for retrieval-side CPU operations.

This ledger supplements, never replaces, the historical generator token/GPU
ResourceLedger. A reserved operation survives process loss as unresolved cost;
no zero-cost or automatic replay is inferred from an absent result. The caller
still owns the frozen source, encoder, gallery and complete role allocation.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
STAGES = frozenset({"shared", "fit", "search", "selection"})
KINDS = frozenset(
    {"encoder_load", "image_encode", "text_encode", "rank_callback", "restricted_score"}
)


class RetrievalWorkLedger:
    """Single-writer attempt ledger; unknown work remains unknown on recovery."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, timeout=10)
        self._active_attempts: set[str] = set()
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS retrieval_work ("
            "attempt_id TEXT PRIMARY KEY, stage TEXT NOT NULL, kind TEXT NOT NULL, "
            "status TEXT NOT NULL, started_unix REAL NOT NULL, wall_seconds REAL, "
            "forward_calls INTEGER)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def unresolved(self) -> tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT attempt_id FROM retrieval_work WHERE status='reserved' ORDER BY attempt_id"
        ).fetchall()
        return tuple(row[0] for row in rows)

    def run(
        self,
        attempt_id: str,
        stage: str,
        kind: str,
        operation: Callable[[], T],
        *,
        forward_counter: Callable[[], int] | None = None,
    ) -> T:
        """Reserve before execution; settle failure with any observed forwards.

        A counter is read before and after the operation when supplied. If it
        cannot be read reliably, forward_calls stays NULL, not zero. An outer
        callback receipt cannot certify that its nested model work was metered.
        """
        if not attempt_id or stage not in STAGES or kind not in KINDS:
            raise ValueError("nonempty attempt and registered retrieval stage/kind required")
        if set(self.unresolved()) - self._active_attempts:
            raise ValueError("unresolved retrieval operation; inspect before further work")
        before = self._counter(forward_counter)
        with self.connection:
            try:
                self.connection.execute(
                    "INSERT INTO retrieval_work VALUES (?,?,?,?,?,?,?)",
                    (attempt_id, stage, kind, "reserved", time.time(), None, None),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("retrieval operation attempt cannot be replayed") from error
        started = time.monotonic()
        self._active_attempts.add(attempt_id)
        try:
            try:
                result = operation()
            except BaseException:
                self._settle(
                    attempt_id, "failed", time.monotonic() - started, before, forward_counter
                )
                raise
            self._settle(
                attempt_id, "completed", time.monotonic() - started, before, forward_counter
            )
            return result
        finally:
            self._active_attempts.remove(attempt_id)

    @staticmethod
    def _counter(counter: Callable[[], int] | None) -> int | None:
        if counter is None:
            return None
        try:
            value = counter()
        except Exception:
            return None
        return value if type(value) is int and value >= 0 else None

    def _settle(
        self,
        attempt_id: str,
        status: str,
        elapsed: float,
        before: int | None,
        counter: Callable[[], int] | None,
    ) -> None:
        after = self._counter(counter)
        forwards = (
            after - before if before is not None and after is not None and after >= before else None
        )
        measured_elapsed = elapsed if math.isfinite(elapsed) and elapsed >= 0 else None
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE retrieval_work SET status=?,wall_seconds=?,forward_calls=? "
                "WHERE attempt_id=? AND status='reserved'",
                (status, measured_elapsed, forwards, attempt_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("missing retrieval operation reservation")

    def summary(self) -> dict[str, int | float]:
        """Return counts and an overlapping operation-time sum, not CPU/GPU allocation.

        Nested rank callbacks include child encoder time, so their durations
        cannot be added into a lower bound on real elapsed or allocated time.
        NULL cost observations remain explicit in the unmeasured counts.
        """
        row = self.connection.execute(
            "SELECT count(*), "
            "coalesce(sum(CASE WHEN status='completed' THEN 1 ELSE 0 END),0), "
            "coalesce(sum(CASE WHEN status='failed' THEN 1 ELSE 0 END),0), "
            "coalesce(sum(CASE WHEN status='reserved' THEN 1 ELSE 0 END),0), "
            "coalesce(sum(CASE WHEN forward_calls IS NULL THEN 1 ELSE 0 END),0), "
            "coalesce(sum(CASE WHEN wall_seconds IS NULL THEN 1 ELSE 0 END),0), "
            "coalesce(sum(forward_calls),0),coalesce(sum(wall_seconds),0) "
            "FROM retrieval_work"
        ).fetchone()
        return {
            "attempts": row[0],
            "completed": row[1],
            "failed": row[2],
            "unresolved": row[3],
            "forward_count_unmeasured_attempts": row[4],
            "wall_count_unmeasured_attempts": row[5],
            "known_forward_calls": row[6],
            "settled_operation_wall_seconds_sum_nonadditive": row[7],
        }


class MeteredClipEncoder:
    """Persist actual single-item CLIP forward attempts around an owned encoder."""

    def __init__(self, encoder: Any, ledger: RetrievalWorkLedger, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError("registered retrieval stage required")
        self.encoder, self.ledger, self.stage = encoder, ledger, stage

    def _forward_count(self, name: str) -> int:
        value = self.encoder.costs[name]
        if type(value) is not int or value < 0:
            raise ValueError("encoder forward counter is not a nonnegative integer")
        return value

    def encode_image(self, attempt_id: str, image: object) -> object:
        return self.ledger.run(
            attempt_id,
            self.stage,
            "image_encode",
            lambda: self.encoder.encode_image(image),
            forward_counter=lambda: self._forward_count("image_forward_calls"),
        )

    def encode_text(self, attempt_id: str, value: str) -> object:
        return self.ledger.run(
            attempt_id,
            self.stage,
            "text_encode",
            lambda: self.encoder.encode_text(value),
            forward_counter=lambda: self._forward_count("text_forward_calls"),
        )
