"""Atomic candidate slots and charged observations survive crashes and resumes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .sampling import AuditPlan, digest
from .statistics import integer


class BudgetExhausted(ValueError):
    """A fixed logical query/slot budget cannot be extended during recovery."""


class AuditJournal:
    """Logical audit episodes, not a replacement for physical model/GPU accounting.

    Every run is charged its full observed reference vector even if physically
    reused. Reservations, including unresolved/failed attempts, consume budget.
    A recovered process never silently replays an in-flight request or replans.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        reference_digest: str,
        execution_digest: str,
        reference_episodes: int,
        max_episodes: int = 2048,
    ) -> None:
        if not run_id or len(reference_digest) != 64 or len(execution_digest) != 64:
            raise ValueError("run and complete reference/execution identity required")
        integer(reference_episodes, "reference_episodes", 1)
        integer(max_episodes, "max_episodes", reference_episodes)
        self.run_id = run_id
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS delta_runs (
              run_id TEXT PRIMARY KEY, reference_digest TEXT NOT NULL,
              execution_digest TEXT NOT NULL, reference_episodes INTEGER NOT NULL,
              max_episodes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS delta_candidates (
              run_id TEXT NOT NULL, candidate_digest TEXT NOT NULL, slot INTEGER NOT NULL,
              plan_sha TEXT NOT NULL, plan_json TEXT NOT NULL,
              PRIMARY KEY(run_id,candidate_digest), UNIQUE(run_id,slot),
              FOREIGN KEY(run_id) REFERENCES delta_runs(run_id));
            CREATE TABLE IF NOT EXISTS delta_queries (
              run_id TEXT NOT NULL, candidate_digest TEXT NOT NULL, unit_id TEXT NOT NULL,
              attempt INTEGER NOT NULL, status TEXT NOT NULL, score INTEGER, error_type TEXT,
              PRIMARY KEY(run_id,candidate_digest,unit_id,attempt),
              FOREIGN KEY(run_id,candidate_digest)
                REFERENCES delta_candidates(run_id,candidate_digest));
            CREATE TABLE IF NOT EXISTS delta_events (
              event_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, candidate_digest TEXT NOT NULL,
              payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS delta_outcome_sources (
              run_id TEXT NOT NULL, candidate_digest TEXT NOT NULL, source_digest TEXT NOT NULL,
              PRIMARY KEY(run_id,candidate_digest));
        """)
        self.connection.commit()
        expected = (reference_digest, execution_digest, reference_episodes, max_episodes)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO delta_runs VALUES (?,?,?,?,?)", (run_id, *expected)
            )
            existing = self.connection.execute(
                "SELECT reference_digest,execution_digest,reference_episodes,max_episodes "
                "FROM delta_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if existing != expected:
                raise ValueError("run identity or budget changed on recovery")

    def close(self) -> None:
        self.connection.close()

    def bind_outcome_source(self, plan: AuditPlan, source_digest: str) -> None:
        """A recovered recorded-table audit cannot substitute a different oracle."""
        if self.load_plan(plan.candidate_digest) != plan or len(source_digest) != 64:
            raise ValueError("freeze candidate before binding an oracle source")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO delta_outcome_sources VALUES (?,?,?)",
                (self.run_id, plan.candidate_digest, source_digest),
            )
            existing = self.connection.execute(
                "SELECT source_digest FROM delta_outcome_sources "
                "WHERE run_id=? AND candidate_digest=?",
                (self.run_id, plan.candidate_digest),
            ).fetchone()[0]
            if existing != source_digest:
                raise ValueError("recorded outcome source changed on recovery")

    def load_plan(self, candidate_digest: str) -> AuditPlan | None:
        row = self.connection.execute(
            "SELECT plan_sha,plan_json FROM delta_candidates WHERE run_id=? AND candidate_digest=?",
            (self.run_id, candidate_digest),
        ).fetchone()
        if row is None:
            return None
        plan = AuditPlan.from_dict(json.loads(row[1]))
        if plan.sha256 != row[0] or plan.candidate_digest != candidate_digest:
            raise ValueError("persisted plan integrity failure")
        return plan

    def freeze(self, plan: AuditPlan) -> int:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            run = self.connection.execute(
                "SELECT reference_digest,execution_digest,reference_episodes "
                "FROM delta_runs WHERE run_id=?",
                (self.run_id,),
            ).fetchone()
            if run != (plan.reference_digest, plan.execution_digest, plan.population):
                raise ValueError("plan and fixed run reference/execution differ")
            existing = self.connection.execute(
                "SELECT slot,plan_sha FROM delta_candidates WHERE run_id=? AND candidate_digest=?",
                (self.run_id, plan.candidate_digest),
            ).fetchone()
            if existing:
                if existing[1] != plan.sha256:
                    raise ValueError("candidate replan/resampling on recovery is forbidden")
                slot = int(existing[0])
            else:
                slot = int(
                    self.connection.execute(
                        "SELECT count(*) FROM delta_candidates WHERE run_id=?", (self.run_id,)
                    ).fetchone()[0]
                )
                if slot >= 32:
                    raise BudgetExhausted("all 32 candidate error-budget slots are reserved")
                self.connection.execute(
                    "INSERT INTO delta_candidates VALUES (?,?,?,?,?)",
                    (
                        self.run_id,
                        plan.candidate_digest,
                        slot,
                        plan.sha256,
                        json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False),
                    ),
                )
            self.connection.commit()
            return slot
        except BaseException:
            self.connection.rollback()
            raise

    def consumed_episodes(self) -> int:
        reference = self.connection.execute(
            "SELECT reference_episodes FROM delta_runs WHERE run_id=?", (self.run_id,)
        ).fetchone()[0]
        attempts = self.connection.execute(
            "SELECT count(*) FROM delta_queries WHERE run_id=?", (self.run_id,)
        ).fetchone()[0]
        return int(reference + attempts)

    def reserve(self, plan: AuditPlan, unit_id: str, *, max_unit_attempts: int = 1) -> int:
        integer(max_unit_attempts, "max_unit_attempts", 1)
        if max_unit_attempts > 3:
            raise ValueError("at most two additional unit retries are allowed")
        persisted = self.load_plan(plan.candidate_digest)
        if (
            persisted is None
            or persisted.sha256 != plan.sha256
            or unit_id not in {x for s in plan.strata for x in s.permutation}
        ):
            raise ValueError("query outside the previously frozen candidate plan")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            previous = self.connection.execute(
                "SELECT attempt,status FROM delta_queries WHERE run_id=? "
                "AND candidate_digest=? AND unit_id=? ORDER BY attempt",
                (self.run_id, plan.candidate_digest, unit_id),
            ).fetchall()
            if any(status in ("completed", "reserved") for _, status in previous):
                raise ValueError("completed or unresolved request cannot be replayed")
            if len(previous) >= max_unit_attempts:
                raise BudgetExhausted("unit attempt limit exhausted")
            limit = self.connection.execute(
                "SELECT max_episodes FROM delta_runs WHERE run_id=?", (self.run_id,)
            ).fetchone()[0]
            if self.consumed_episodes() >= limit:
                raise BudgetExhausted("fixed run evaluation-episode budget exhausted")
            attempt = len(previous)
            self.connection.execute(
                "INSERT INTO delta_queries VALUES (?,?,?,?,?,NULL,NULL)",
                (self.run_id, plan.candidate_digest, unit_id, attempt, "reserved"),
            )
            self.connection.commit()
            return attempt
        except BaseException:
            self.connection.rollback()
            raise

    def settle(
        self,
        plan: AuditPlan,
        unit_id: str,
        attempt: int,
        *,
        score: int | None,
        error_type: str | None = None,
    ) -> None:
        if score is not None and (type(score) is not int or score not in (0, 1)):
            raise ValueError("missing/ambiguous/nonbinary outcomes cannot become observed scores")
        if (score is None) == (error_type is None):
            raise ValueError("one completed score or failed-attempt error type required")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE delta_queries SET status=?,score=?,error_type=? WHERE run_id=? "
                "AND candidate_digest=? AND unit_id=? AND attempt=? AND status='reserved'",
                (
                    "failed" if score is None else "completed",
                    score,
                    error_type,
                    self.run_id,
                    plan.candidate_digest,
                    unit_id,
                    attempt,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("missing or already resolved reservation")

    def observations(self, plan: AuditPlan) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT unit_id,score FROM delta_queries WHERE run_id=? "
            "AND candidate_digest=? AND status='completed'",
            (self.run_id, plan.candidate_digest),
        ).fetchall()
        allowed = {x for s in plan.strata for x in s.permutation}
        if any(
            unit not in allowed or type(score) is not int or score not in (0, 1)
            for unit, score in rows
        ) or len({unit for unit, _ in rows}) != len(rows):
            raise ValueError("observation-store integrity failure")
        return dict(rows)

    def has_unresolved(self, plan: AuditPlan) -> bool:
        return bool(
            self.connection.execute(
                "SELECT 1 FROM delta_queries WHERE run_id=? "
                "AND candidate_digest=? AND status='reserved' LIMIT 1",
                (self.run_id, plan.candidate_digest),
            ).fetchone()
        )

    def record(self, candidate_digest: str, result: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO delta_events(run_id,candidate_digest,payload) VALUES (?,?,?)",
                (
                    self.run_id,
                    candidate_digest,
                    json.dumps(result, sort_keys=True, allow_nan=False),
                ),
            )

    def report(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "consumed_logical_evaluation_episodes": self.consumed_episodes(),
            "reserved_candidate_slots": self.connection.execute(
                "SELECT count(*) FROM delta_candidates WHERE run_id=?", (self.run_id,)
            ).fetchone()[0],
            "events": [
                json.loads(row[0])
                for row in self.connection.execute(
                    "SELECT payload FROM delta_events WHERE run_id=? ORDER BY event_id",
                    (self.run_id,),
                )
            ],
            "physical_model_calls_or_gpu_costs_inferred": False,
            "run_configuration_digest": digest(
                self.connection.execute(
                    "SELECT * FROM delta_runs WHERE run_id=?", (self.run_id,)
                ).fetchone()
            ),
        }
