"""Actual-score bridges for pinned GEPA/DSPy interfaces; never fabricated vectors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ActualScoreVector:
    unit_ids: tuple[str, ...]
    scores: tuple[int, ...]


def complete_survivor_scores(
    unit_ids: tuple[str, ...], observed: Mapping[str, int], evaluate: Callable[[str], int]
) -> ActualScoreVector:
    """Complete a survivor before native selection; remaining cost is not free."""
    if not unit_ids or len(set(unit_ids)) != len(unit_ids) or set(observed) - set(unit_ids):
        raise ValueError("missing/duplicate/foreign native score identities")
    values = []
    for unit in unit_ids:
        score = observed[unit] if unit in observed else evaluate(unit)
        if type(score) is not int or score not in (0, 1):
            raise ValueError("no native survivor vector from unknown/predicted/nonbinary outcomes")
        values.append(score)
    return ActualScoreVector(unit_ids, tuple(values))


class GEPAEvaluationInterface(Protocol):
    def evaluate(
        self, batch: list[Any], candidate: dict[str, str], capture_traces: bool = False
    ) -> Any: ...


def evaluate_gepa_native(
    adapter: GEPAEvaluationInterface,
    batch: list[Any],
    candidate: dict[str, str],
    *,
    capture_traces: bool = False,
) -> Any:
    """Pinned GEPA EvaluationBatch contract; no partial result sent to its selector.

    This boundary leaves native selection unchanged. An early-rejection engine
    hook is NOT implemented by this bridge and must be labeled adapted when added.
    """
    result = adapter.evaluate(batch, candidate, capture_traces=capture_traces)
    if (
        len(result.outputs) != len(batch)
        or len(result.scores) != len(batch)
        or any(output is None for output in result.outputs)
    ):
        raise ValueError("native EvaluationBatch has incomplete actual outputs/scores")
    if capture_traces and (result.trajectories is None or len(result.trajectories) != len(batch)):
        raise ValueError("native reflective trajectories incomplete")
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in result.scores
    ):
        raise ValueError("native score is missing/non-finite")
    return result


class DSPyEvaluationInterface(Protocol):
    def __call__(self, program: Any, **kwargs: Any) -> Any: ...


def evaluate_dspy_native(
    evaluate: DSPyEvaluationInterface, program: Any, devset: Sequence[Any]
) -> Any:
    """Pinned DSPy Evaluate returns every actual per-example result.

    Its default failure_score=0 silently fabricates a wrong-answer score for an
    execution failure. Require an explicit NaN sentinel and max_errors=1 before
    invoking the evaluator, without mutating a caller's optimizer configuration.
    Native failures propagate or the nonfinite sentinel is rejected below.
    """
    failure_score = getattr(evaluate, "failure_score", None)
    max_errors = getattr(evaluate, "max_errors", None)
    if (
        not isinstance(failure_score, float)
        or not math.isnan(failure_score)
        or type(max_errors) is not int
        or max_errors != 1
    ):
        raise ValueError("DSPy requires failure_score=NaN and max_errors=1; no fabricated zero")
    result = evaluate(program, devset=list(devset))
    if not hasattr(result, "results") or len(result.results) != len(devset):
        raise ValueError("DSPy full-evaluation result vector is incomplete")
    if any(
        len(row) != 3
        or isinstance(row[2], bool)
        or not isinstance(row[2], (int, float))
        or not math.isfinite(row[2])
        for row in result.results
    ):
        raise ValueError("DSPy outcomes are missing or non-finite")
    return result
