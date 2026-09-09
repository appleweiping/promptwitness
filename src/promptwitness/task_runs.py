"""Configuration-bound durable task execution with explicit retry semantics."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import _freeze_json, _thaw_json
from .providers import OpenAICompatibleProvider
from .sessions import OpenAISessionProvider, _identity, _json, _load, _positive
from .task_data import FAMILIES, TaskSuitePlan, digest
from .task_scores import METRIC_CAPABILITIES, response_text, score_task


class TaskRunConflict(ValueError):
    """A stale task revision or different run configuration was rejected."""


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """A gold-free, immutable provider request bound to model and generation settings."""

    item_id: str
    provider_identity: Mapping[str, Any]
    messages: tuple[Mapping[str, str], ...]
    generation: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("provider_identity", "messages", "generation"):
            object.__setattr__(self, name, _freeze_json(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "provider_identity": _thaw_json(self.provider_identity),
            "messages": _thaw_json(self.messages),
            "generation": _thaw_json(self.generation),
        }

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


class OpenAITaskProvider:
    """Use the existing environment-credential transport for full task requests."""

    def __init__(self, provider: OpenAICompatibleProvider) -> None:
        self.provider = provider

    @property
    def identity(self) -> Mapping[str, Any]:
        return OpenAISessionProvider(self.provider).identity

    def __call__(self, request: TaskRequest) -> Any:
        if _json(request.provider_identity) != _json(self.identity):
            raise TaskRunConflict("provider settings changed after reservation")
        value = request.to_dict()
        return self.provider.complete(value["messages"], generation=value["generation"])


class TaskRunStore:
    """One SQLite database for one exact suite plan and one model configuration.

    A committed running reservation precedes each external call. Failed and
    interrupted tasks are never retried implicitly. Explicit retry invalidates
    the old revision, so late results from the previous worker cannot overwrite
    the new attempt. This is not an external exactly-once guarantee.
    """

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        existing = target.exists() and target.stat().st_size > 0
        if not existing and any(
            Path(str(target) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")
        ):
            raise ValueError("refusing to create a database beside pre-existing SQLite sidecars")
        self.connection = sqlite3.connect(str(path), isolation_level=None, timeout=10)
        self.connection.row_factory = sqlite3.Row
        try:
            if not self._pristine():
                self._validate_database()
                return
            # A second initializer may have committed after our first read.
            # Recheck identity under the same writer lock as every schema change.
            self.connection.execute("BEGIN IMMEDIATE")
            if self._pristine():
                self._initialize()
            else:
                self._validate_database()
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            self.connection.close()
            raise

    def _pristine(self) -> bool:
        return (
            self.connection.execute("PRAGMA application_id").fetchone()[0] == 0
            and self.connection.execute("PRAGMA user_version").fetchone()[0] == 0
            and self.connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        )

    def _initialize(self) -> None:
        for statement in (
            "PRAGMA application_id=1347900499",
            "PRAGMA user_version=1",
            """CREATE TABLE task_run (
                id INTEGER PRIMARY KEY CHECK(id=1), plan TEXT NOT NULL,
                plan_digest TEXT NOT NULL, identity TEXT NOT NULL
            )""",
            """CREATE TABLE task_state (
                item_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                revision INTEGER NOT NULL, attempts INTEGER NOT NULL,
                result TEXT, error TEXT
            )""",
            """CREATE TABLE task_event (
                item_id TEXT NOT NULL, revision INTEGER NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(item_id, revision)
            )""",
            """CREATE TRIGGER task_event_no_update
                BEFORE UPDATE ON task_event BEGIN
                SELECT RAISE(ABORT, 'append-only task events'); END""",
            """CREATE TRIGGER task_event_no_delete
                BEFORE DELETE ON task_event BEGIN
                SELECT RAISE(ABORT, 'append-only task events'); END""",
        ):
            self.connection.execute(statement)

    def _validate_database(self) -> None:
        if (
            self.connection.execute("PRAGMA application_id").fetchone()[0] != 1347900499
            or self.connection.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("not a supported PromptWitness task database")
        tables = {
            row[0]
            for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if tables != {"task_run", "task_state", "task_event"}:
            raise ValueError("task database schema mismatch")
        schemas = {
            "task_run": ["id", "plan", "plan_digest", "identity"],
            "task_state": ["item_id", "status", "revision", "attempts", "result", "error"],
            "task_event": ["item_id", "revision", "kind", "payload"],
        }
        for name, columns in schemas.items():
            actual = [row[1] for row in self.connection.execute(f"PRAGMA table_info({name})")]
            if actual != columns:
                raise ValueError("task database columns mismatch")
        triggers = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        if triggers != {"task_event_no_update", "task_event_no_delete"}:
            raise ValueError("task database append-only guards mismatch")

    def __enter__(self) -> TaskRunStore:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def bind(self, plan: TaskSuitePlan, identity: Mapping[str, Any]) -> None:
        """Initialize, or verify exact inputs/configuration/model before resuming."""
        plan = TaskSuitePlan(plan.payload)
        encoded_identity = _json(_identity(identity))
        with self._transaction():
            row = self.connection.execute("SELECT * FROM task_run WHERE id=1").fetchone()
            if row is not None:
                if (
                    row["plan_digest"] != plan.digest
                    or row["plan"] != _json(plan.payload)
                    or row["identity"] != encoded_identity
                ):
                    raise TaskRunConflict(
                        "run inputs, suite configuration, counter, or provider identity changed"
                    )
            else:
                self.connection.execute(
                    "INSERT INTO task_run VALUES (1,?,?,?)",
                    (_json(plan.payload), plan.digest, encoded_identity),
                )
                for item in plan.payload["items"]:
                    status = "skipped" if item["skip_reason"] else "ready"
                    self.connection.execute(
                        "INSERT INTO task_state VALUES (?,?,0,0,NULL,NULL)", (item["id"], status)
                    )
        self.snapshot()

    def _transaction(self) -> _Transaction:
        return _Transaction(self.connection)

    def _bound(self) -> tuple[dict[str, Any], dict[str, Any]]:
        row = self.connection.execute("SELECT * FROM task_run WHERE id=1").fetchone()
        if row is None:
            raise ValueError("task database has not been bound to a suite")
        plan = _load(row["plan"])
        if digest(plan) != row["plan_digest"]:
            raise ValueError("stored plan checksum mismatch")
        TaskSuitePlan(plan)
        return plan, _identity(_load(row["identity"]))

    def snapshot(self) -> dict[str, Any]:
        """Read a consistent report; recompute saved metrics from pinned inputs."""
        self.connection.execute("BEGIN")
        try:
            plan, identity = self._bound()
            states = {
                row["item_id"]: dict(row)
                for row in self.connection.execute("SELECT * FROM task_state")
            }
            if set(states) != {item["id"] for item in plan["items"]}:
                raise ValueError("stored task inventory differs from the bound plan")
            results = []
            for item in plan["items"]:
                state = states[item["id"]]
                if state["status"] not in {"ready", "running", "failed", "succeeded", "skipped"}:
                    raise ValueError("invalid stored task status")
                result = _load(state["result"]) if state["result"] else None
                if result is not None:
                    if state["status"] != "succeeded" or _json(result.get("score")) != _json(
                        score_task(item, result["prediction"])
                    ):
                        raise ValueError("stored result does not match pinned scoring inputs")
                    if result["request_digest"] != self._request(item, plan, identity).digest:
                        raise ValueError("stored result request identity mismatch")
                elif state["status"] == "succeeded":
                    raise ValueError("successful task has no result")
                if (state["status"] == "skipped") != bool(item["skip_reason"]):
                    raise ValueError("stored skip status differs from the plan")
                self._verify_events(state, item, plan, identity, result)
                results.append(
                    {
                        "item_id": item["id"],
                        "task_id": item["task_id"],
                        "record_id": item["record_id"],
                        "family": item["family"],
                        "budget": item["budget"],
                        "input_units": item["input_units"],
                        "status": state["status"],
                        "revision": state["revision"],
                        "attempts": state["attempts"],
                        "skip_reason": item["skip_reason"],
                        "error": state["error"],
                        "result": result,
                    }
                )
            report = _report(plan, identity, results)
        finally:
            self.connection.rollback()
        return report

    def _verify_events(
        self,
        state: Mapping[str, Any],
        item: Mapping[str, Any],
        plan: Mapping[str, Any],
        identity: Mapping[str, Any],
        result: Any,
    ) -> None:
        status = "skipped" if item["skip_reason"] else "ready"
        revision = attempts = 0
        error_type = None
        for event in self.connection.execute(
            "SELECT * FROM task_event WHERE item_id=? ORDER BY revision", (item["id"],)
        ):
            revision += 1
            if event["revision"] != revision:
                raise ValueError("task event revisions are not contiguous")
            payload = _load(event["payload"])
            kind = event["kind"]
            if kind == "reserved" and status == "ready":
                if payload != {"request_digest": self._request(item, plan, identity).digest}:
                    raise ValueError("reserved request checksum mismatch")
                status, attempts, error_type = "running", attempts + 1, None
            elif kind == "completed" and status == "running":
                if payload != {"result_digest": digest(result)}:
                    raise ValueError("completed result checksum mismatch")
                status = "succeeded"
            elif kind == "failed" and status == "running":
                if set(payload) != {"error_type"} or not isinstance(payload["error_type"], str):
                    raise ValueError("invalid failure event")
                status, error_type = "failed", payload["error_type"]
            elif kind == "retry_authorized" and status in {"running", "failed"} and payload == {}:
                status, error_type = "ready", None
            else:
                raise ValueError("invalid task event transition")
        if (status, revision, attempts, error_type) != (
            state["status"],
            state["revision"],
            state["attempts"],
            state["error"],
        ):
            raise ValueError("task state differs from its event history")

    @staticmethod
    def _request(
        item: Mapping[str, Any], plan: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> TaskRequest:
        return TaskRequest(item["id"], identity, tuple(item["messages"]), plan["generation"])

    def _item(self, item_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        plan, identity = self._bound()
        item = next((item for item in plan["items"] if item["id"] == item_id), None)
        if item is None:
            raise ValueError("unknown task item")
        return item, plan, identity

    def _state(self, item_id: str, expected_revision: int, allowed: set[str]) -> sqlite3.Row:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        row: sqlite3.Row | None = self.connection.execute(
            "SELECT * FROM task_state WHERE item_id=?", (item_id,)
        ).fetchone()
        if row is None or row["revision"] != expected_revision:
            raise TaskRunConflict("stale or unknown task revision")
        if row["status"] not in allowed:
            raise TaskRunConflict(f"task is {row['status']}, expected {sorted(allowed)}")
        return row

    def _event(self, item_id: str, revision: int, kind: str, payload: Mapping[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO task_event VALUES (?,?,?,?)", (item_id, revision, kind, _json(payload))
        )

    def reserve(self, item_id: str, *, expected_revision: int) -> tuple[TaskRequest, int]:
        with self._transaction():
            state = self._state(item_id, expected_revision, {"ready"})
            item, plan, identity = self._item(item_id)
            request = self._request(item, plan, identity)
            revision = state["revision"] + 1
            self.connection.execute(
                "UPDATE task_state SET status='running', revision=?, "
                "attempts=attempts+1, error=NULL WHERE item_id=?",
                (revision, item_id),
            )
            self._event(item_id, revision, "reserved", {"request_digest": request.digest})
        return request, revision

    def complete(self, item_id: str, response: Any, *, expected_revision: int) -> None:
        prediction = response_text(response)
        if len(prediction.encode("utf-8")) > 1024 * 1024:
            raise ValueError("task prediction exceeds 1 MiB")
        with self._transaction():
            self._state(item_id, expected_revision, {"running"})
            item, plan, identity = self._item(item_id)
            result = {
                "prediction": prediction,
                "score": score_task(item, prediction),
                "request_digest": self._request(item, plan, identity).digest,
            }
            self.connection.execute(
                "UPDATE task_state SET status='succeeded', revision=?, result=? WHERE item_id=?",
                (expected_revision + 1, _json(result), item_id),
            )
            self._event(
                item_id, expected_revision + 1, "completed", {"result_digest": digest(result)}
            )

    def fail(self, item_id: str, error: Exception, *, expected_revision: int) -> None:
        with self._transaction():
            self._state(item_id, expected_revision, {"running"})
            error_type = type(error).__name__
            self.connection.execute(
                "UPDATE task_state SET status='failed', revision=?, error=? WHERE item_id=?",
                (expected_revision + 1, error_type, item_id),
            )
            self._event(item_id, expected_revision + 1, "failed", {"error_type": error_type})

    def retry(self, item_id: str, *, expected_revision: int) -> None:
        """Authorize another invocation after a failure or an interrupted reservation.

        For a running item, first ensure the previous worker has stopped and
        accept that a remote call may already have happened. No call occurs here.
        """
        with self._transaction():
            self._state(item_id, expected_revision, {"failed", "running"})
            self.connection.execute(
                "UPDATE task_state SET status='ready', revision=?, error=NULL WHERE item_id=?",
                (expected_revision + 1, item_id),
            )
            self._event(item_id, expected_revision + 1, "retry_authorized", {})

    def run(
        self,
        plan: TaskSuitePlan,
        provider: Callable[[TaskRequest], Any],
        *,
        identity: Mapping[str, Any] | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        """Run only ready items; resume reuses successful outputs without calls."""
        if not callable(provider):
            raise TypeError("provider must be callable")
        exposed = getattr(provider, "identity", None)
        active = _identity(identity if identity is not None else exposed)
        if exposed is not None and _json(active) != _json(exposed):
            raise TaskRunConflict("explicit provider identity differs from the provider")
        if max_cases is not None:
            _positive(max_cases, "max_cases")
        self.bind(plan, active)
        used = 0
        for state in self.snapshot()["results"]:
            if state["status"] != "ready" or (max_cases is not None and used >= max_cases):
                continue
            if exposed is not None and _json(getattr(provider, "identity", None)) != _json(active):
                raise TaskRunConflict("provider identity changed during the run")
            request, revision = self.reserve(state["item_id"], expected_revision=state["revision"])
            used += 1
            try:
                response = provider(request)
                self.complete(state["item_id"], response, expected_revision=revision)
            except TaskRunConflict:
                raise
            except Exception as error:
                self.fail(state["item_id"], error, expected_revision=revision)
        return self.snapshot()


class _Transaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def __exit__(self, error_type: Any, *_args: Any) -> None:
        if error_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    scored = [
        row["result"]["score"]["primary_score"]
        for row in rows
        if row["result"] and row["result"]["score"]["primary_score"] is not None
    ]
    units = [row["input_units"] for row in rows]
    return {
        "planned": len(rows),
        **{
            status: counts[status]
            for status in ("ready", "running", "failed", "succeeded", "skipped")
        },
        "scored": len(scored),
        "score_coverage": len(scored) / len(rows) if rows else None,
        "mean_primary_score": sum(scored) / len(scored) if scored else None,
        "execution_complete": bool(rows) and counts["succeeded"] == len(rows),
        "scoring_complete": bool(rows) and len(scored) == len(rows),
        "actual_input_units": {
            "minimum": min(units) if units else None,
            "maximum": max(units) if units else None,
            "mean": sum(units) / len(units) if units else None,
        },
    }


def _report(
    plan: Mapping[str, Any], identity: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    family = {
        name: {
            **_coverage([row for row in rows if row["family"] == name]),
            "configured": any(task["family"] == name for task in plan["tasks"]),
            "metric_capabilities": _load(_json(METRIC_CAPABILITIES[name])),
        }
        for name in FAMILIES
    }
    # Never average differently defined family metrics into a purported benchmark score.
    overall = _coverage(rows)
    overall.pop("mean_primary_score")
    return {
        "format": "promptwitness.task-report/v1",
        "suite_id": plan["suite_id"],
        "plan_digest": digest(plan),
        "provider_identity": identity,
        "counter": plan["counter"],
        "generation": plan["generation"],
        "coverage": overall,
        "seven_family_execution_complete": all(
            group["execution_complete"] for group in family.values()
        ),
        "seven_family_scoring_complete": all(
            group["scoring_complete"] for group in family.values()
        ),
        "tasks": plan["tasks"],
        "by_family": family,
        "by_budget_family": {
            str(budget): {
                name: _coverage(
                    [row for row in rows if row["family"] == name and row["budget"] == budget]
                )
                for name in FAMILIES
            }
            for budget in plan["budgets"]
        },
        "results": rows,
    }
