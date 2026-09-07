"""Task-oriented benchmark cases and replay evaluation primitives."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One task instance; expected output is never part of the prompt digest."""

    case_id: str
    task: str
    prompt: str
    expected: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("task", self.task),
            ("prompt", self.prompt),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.expected, str):
            raise TypeError("expected must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def prompt_digest(self) -> str:
        """Return a stable digest that excludes the gold answer."""

        return _digest(
            {
                "case_id": self.case_id,
                "task": self.task,
                "prompt": self.prompt,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """Named collection of benchmark cases with optional task metadata."""

    name: str
    cases: tuple[BenchmarkCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("task name must be a non-empty string")
        if not self.cases:
            raise ValueError("benchmark task must contain at least one case")
        if not all(isinstance(case, BenchmarkCase) for case in self.cases):
            raise TypeError("benchmark task cases must be BenchmarkCase values")
        if any(case.task != self.name for case in self.cases):
            raise ValueError("benchmark case task must match its suite task name")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("benchmark task case IDs must be unique")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("task metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Versioned multi-task benchmark collection."""

    suite_id: str
    tasks: tuple[BenchmarkTask, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.suite_id, str) or not self.suite_id.strip():
            raise ValueError("suite_id must be a non-empty string")
        if not self.tasks:
            raise ValueError("benchmark suite must contain at least one task")
        if not all(isinstance(task, BenchmarkTask) for task in self.tasks):
            raise TypeError("suite tasks must be BenchmarkTask values")
        if len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("benchmark suite task names must be unique")
        case_ids = [case.case_id for task in self.tasks for case in task.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("benchmark suite case IDs must be globally unique")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("suite metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(case for task in self.tasks for case in task.cases)

    @property
    def digest(self) -> str:
        """Digest task structure and prompts without embedding gold answers."""

        return _digest(
            {
                "suite_id": self.suite_id,
                "metadata": dict(self.metadata),
                "tasks": [
                    {
                        "name": task.name,
                        "metadata": dict(task.metadata),
                        "cases": [
                            {
                                "case_id": case.case_id,
                                "task": case.task,
                                "prompt_digest": case.prompt_digest,
                            }
                            for case in task.cases
                        ],
                    }
                    for task in self.tasks
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.benchmark-suite.v1",
            "suite_id": self.suite_id,
            "metadata": dict(self.metadata),
            "tasks": [
                {
                    "name": task.name,
                    "metadata": dict(task.metadata),
                    "cases": [
                        {
                            "case_id": case.case_id,
                            "task": case.task,
                            "prompt": case.prompt,
                            "expected": case.expected,
                            "metadata": dict(case.metadata),
                        }
                        for case in task.cases
                    ],
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One replay outcome retaining failure details."""

    case_id: str
    task: str
    prompt_digest: str
    prediction: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Ordered task results with aggregate and per-task diagnostics."""

    results: tuple[BenchmarkResult, ...]

    @property
    def attempted(self) -> int:
        return sum(item.correct is not None for item in self.results)

    @property
    def failed(self) -> int:
        return sum(item.error is not None for item in self.results)

    @property
    def accuracy(self) -> float | None:
        return _accuracy(self.results)

    def by_task(self) -> Mapping[str, Mapping[str, Any]]:
        buckets: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for result in self.results:
            buckets[result.task].append(result)
        return MappingProxyType(
            {
                task: {
                    "cases": len(rows),
                    "attempted": sum(item.correct is not None for item in rows),
                    "failed": sum(item.error is not None for item in rows),
                    "accuracy": _accuracy(rows),
                }
                for task, rows in sorted(buckets.items())
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": len(self.results),
            "attempted": self.attempted,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "by_task": dict(self.by_task()),
            "results": [
                {
                    "case_id": item.case_id,
                    "task": item.task,
                    "prompt_digest": item.prompt_digest,
                    "prediction": item.prediction,
                    "correct": item.correct,
                    "error": item.error,
                }
                for item in self.results
            ],
        }


def load_benchmark_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load strict JSON or JSONL benchmark cases with unique IDs."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            values = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load benchmark cases: {error}") from error
    if not isinstance(values, list):
        raise ValueError("benchmark root must be an array")
    cases: list[BenchmarkCase] = []
    for index, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise ValueError(f"benchmark case {index} must be an object")
        unknown = set(value) - {"case_id", "task", "prompt", "expected", "metadata"}
        if unknown:
            raise ValueError(
                f"benchmark case {index} has unknown fields: {', '.join(sorted(unknown))}"
            )
        try:
            cases.append(
                BenchmarkCase(
                    str(value["case_id"]),
                    str(value["task"]),
                    str(value["prompt"]),
                    value["expected"],
                    value.get("metadata", {}),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid benchmark case {index}: {error}") from error
    if not cases:
        raise ValueError("benchmark must contain at least one case")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("benchmark case IDs must be unique")
    return tuple(cases)


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load a strict multi-task benchmark suite from JSON."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load benchmark suite: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("benchmark suite root must be an object")
    required = {"format", "suite_id", "tasks"}
    unknown = set(value) - (required | {"metadata"})
    if unknown:
        raise ValueError(f"benchmark suite has unknown fields: {', '.join(sorted(unknown))}")
    if value.get("format") != "promptwitness.benchmark-suite.v1":
        raise ValueError("unsupported benchmark suite format")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("benchmark suite tasks must be an array")
    tasks: list[BenchmarkTask] = []
    for position, raw_task in enumerate(raw_tasks, 1):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"benchmark suite task {position} must be an object")
        task_unknown = set(raw_task) - {"name", "metadata", "cases"}
        if task_unknown:
            raise ValueError(
                f"benchmark suite task {position} has unknown fields: "
                f"{', '.join(sorted(task_unknown))}"
            )
        name = raw_task.get("name")
        raw_cases = raw_task.get("cases")
        if not isinstance(name, str) or not isinstance(raw_cases, list):
            raise ValueError(f"benchmark suite task {position} requires name and cases")
        cases: list[BenchmarkCase] = []
        for case_position, raw_case in enumerate(raw_cases, 1):
            if not isinstance(raw_case, Mapping):
                raise ValueError(
                    f"benchmark suite task {position} case {case_position} must be an object"
                )
            try:
                cases.append(
                    BenchmarkCase(
                        raw_case["case_id"],
                        raw_case["task"],
                        raw_case["prompt"],
                        raw_case["expected"],
                        raw_case.get("metadata", {}),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid benchmark suite task {position} case {case_position}: {error}"
                ) from error
        try:
            tasks.append(BenchmarkTask(name, tuple(cases), raw_task.get("metadata", {})))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid benchmark suite task {position}: {error}") from error
    try:
        return BenchmarkSuite(value["suite_id"], tuple(tasks), value.get("metadata", {}))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid benchmark suite: {error}") from error


def evaluate_benchmark(
    cases: Iterable[BenchmarkCase],
    answerer: Callable[[BenchmarkCase], str],
    *,
    strict: bool = False,
) -> BenchmarkReport:
    """Replay an answerer over task cases while retaining per-case failures."""

    if not callable(answerer):
        raise TypeError("answerer must be callable")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one benchmark case is required")
    results: list[BenchmarkResult] = []
    for case in materialized:
        try:
            prediction = answerer(case)
            if not isinstance(prediction, str):
                raise TypeError("answerer must return a string")
            results.append(
                BenchmarkResult(
                    case.case_id,
                    case.task,
                    case.prompt_digest,
                    prediction,
                    prediction.strip().casefold() == case.expected.strip().casefold(),
                )
            )
        except Exception as error:
            if strict:
                raise ValueError(f"benchmark case {case.case_id!r} failed: {error}") from error
            results.append(
                BenchmarkResult(
                    case.case_id,
                    case.task,
                    case.prompt_digest,
                    None,
                    None,
                    f"{type(error).__name__}: {error}",
                )
            )
    return BenchmarkReport(tuple(results))


def _accuracy(results: Iterable[BenchmarkResult]) -> float | None:
    rows = tuple(results)
    attempted = sum(item.correct is not None for item in rows)
    return None if not attempted else sum(item.correct is True for item in rows) / attempted
