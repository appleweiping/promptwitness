"""Deterministic long-context needle evaluation primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .matrix import RenderedScenario
from .models import Message


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class LongContextCase:
    """One context with a known needle at a measured relative position."""

    case_id: str
    context: tuple[str, ...]
    needle: str
    query: str
    expected: str
    needle_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if (
            not isinstance(self.context, tuple)
            or not self.context
            or not all(isinstance(item, str) and item for item in self.context)
        ):
            raise ValueError("context must be a non-empty tuple of non-empty strings")
        for name, value in (
            ("needle", self.needle),
            ("query", self.query),
            ("expected", self.expected),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.needle_index, bool) or not isinstance(self.needle_index, int):
            raise TypeError("needle_index must be an integer")
        if not 0 <= self.needle_index < len(self.context):
            raise ValueError("needle_index is outside context")
        if self.needle not in self.context[self.needle_index]:
            raise ValueError("needle must occur in context[needle_index]")

    @property
    def depth(self) -> int:
        """Number of context blocks supplied to the evaluator."""

        return len(self.context)

    @property
    def position(self) -> float:
        """Needle position normalized to the inclusive [0, 1] interval."""

        return self.needle_index / max(1, self.depth - 1)

    @property
    def prompt(self) -> str:
        """Render a stable prompt without leaking the expected answer."""

        context = "\n".join(f"[{index}] {item}" for index, item in enumerate(self.context))
        return f"Context:\n{context}\n\nQuestion: {self.query}\nAnswer:"

    def digest(self) -> str:
        """Return a stable case digest for result joins."""

        return _digest(
            {
                "case_id": self.case_id,
                "context": self.context,
                "needle": self.needle,
                "query": self.query,
                "expected": self.expected,
                "needle_index": self.needle_index,
            }
        )


@dataclass(frozen=True, slots=True)
class LongContextResult:
    """One provider outcome, retaining failure details instead of hiding them."""

    case_id: str
    case_digest: str
    depth: int
    position: float
    prediction: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LongContextReport:
    """Aggregated long-context accuracy and position/depth diagnostics."""

    results: tuple[LongContextResult, ...]

    @property
    def attempted(self) -> int:
        return sum(result.correct is not None for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.error is not None for result in self.results)

    @property
    def accuracy(self) -> float | None:
        """Return accuracy over attempted cases, or None when all failed."""

        if not self.attempted:
            return None
        return sum(result.correct is True for result in self.results) / self.attempted

    @property
    def by_depth(self) -> Mapping[int, float | None]:
        """Return accuracy buckets keyed by context depth."""

        depths = sorted({result.depth for result in self.results})
        return MappingProxyType(
            {
                depth: _accuracy(result for result in self.results if result.depth == depth)
                for depth in depths
            }
        )

    @property
    def by_position(self) -> Mapping[str, float | None]:
        """Return front/middle/back accuracy buckets."""

        buckets: dict[str, list[LongContextResult]] = {"front": [], "middle": [], "back": []}
        for result in self.results:
            bucket = (
                "front"
                if result.position < 1 / 3
                else "back"
                if result.position > 2 / 3
                else "middle"
            )
            buckets[bucket].append(result)
        return MappingProxyType({name: _accuracy(items) for name, items in buckets.items()})

    def to_dict(self) -> dict[str, Any]:
        """Render an audit-friendly JSON-compatible report."""

        return {
            "cases": len(self.results),
            "attempted": self.attempted,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "by_depth": {str(key): value for key, value in self.by_depth.items()},
            "by_position": dict(self.by_position),
            "results": [
                {
                    "case_id": result.case_id,
                    "case_digest": result.case_digest,
                    "depth": result.depth,
                    "position": result.position,
                    "prediction": result.prediction,
                    "correct": result.correct,
                    "error": result.error,
                }
                for result in self.results
            ],
        }


def evaluate_long_context(
    cases: Iterable[LongContextCase],
    answerer: Callable[[LongContextCase], str],
    *,
    strict: bool = False,
) -> LongContextReport:
    """Evaluate a provider over long-context cases with optional fail-fast mode."""

    if not callable(answerer):
        raise TypeError("answerer must be callable")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one long-context case is required")
    results: list[LongContextResult] = []
    for case in materialized:
        try:
            prediction = answerer(case)
            if not isinstance(prediction, str):
                raise TypeError("answerer must return a string")
            normalized_prediction = prediction.strip().casefold()
            normalized_expected = case.expected.strip().casefold()
            results.append(
                LongContextResult(
                    case.case_id,
                    case.digest(),
                    case.depth,
                    case.position,
                    prediction,
                    normalized_prediction == normalized_expected,
                )
            )
        except Exception as error:
            if strict:
                raise ValueError(f"long-context case {case.case_id!r} failed: {error}") from error
            results.append(
                LongContextResult(
                    case.case_id,
                    case.digest(),
                    case.depth,
                    case.position,
                    None,
                    None,
                    f"{type(error).__name__}: {error}",
                )
            )
    return LongContextReport(tuple(results))


def make_provider_answerer(
    provider: Callable[[RenderedScenario], Any],
) -> Callable[[LongContextCase], str]:
    """Adapt an OpenAI-compatible provider callback to long-context cases.

    The adapter sends only the rendered prompt as a user message and extracts
    the common ``choices[0].message.content`` or ``output_text`` response
    shapes. The expected answer is never included in the provider payload or
    prompt digest, so a trace can be audited without leaking the gold label.
    """

    if not callable(provider):
        raise TypeError("provider must be callable")

    def answer(case: LongContextCase) -> str:
        if not isinstance(case, LongContextCase):
            raise TypeError("case must be a LongContextCase")
        prompt_digest = _digest({"case_id": case.case_id, "prompt": case.prompt})
        row = RenderedScenario(
            case.case_id,
            "promptwitness.long-context/v1",
            (Message("user", case.prompt),),
            (),
            prompt_digest,
            (),
        )
        response = provider(row)
        if isinstance(response, str):
            return response
        if not isinstance(response, Mapping):
            raise ValueError("provider response must be a string or JSON object")
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        raise ValueError("provider response does not contain text content")

    return answer


def make_needle_cases(
    contexts: Sequence[Sequence[str]],
    *,
    needle: str,
    query: str,
    expected: str,
    seed: str | int = "0",
) -> tuple[LongContextCase, ...]:
    """Insert one needle per context at deterministic front/middle/back positions."""

    if not contexts:
        raise ValueError("at least one context is required")
    if not all(contexts):
        raise ValueError("contexts must not be empty")
    cases: list[LongContextCase] = []
    for index, raw_context in enumerate(contexts):
        blocks = tuple(raw_context)
        digest = hashlib.sha256(f"promptwitness-needle-v1\0{seed}\0{index}".encode()).digest()
        needle_index = int.from_bytes(digest[:8], "big") % len(blocks)
        updated = list(blocks)
        updated[needle_index] = f"{updated[needle_index]}\nNEEDLE: {needle}"
        cases.append(
            LongContextCase(str(index), tuple(updated), needle, query, expected, needle_index)
        )
    return tuple(cases)


def _accuracy(results: Iterable[LongContextResult]) -> float | None:
    materialized = tuple(results)
    attempted = sum(result.correct is not None for result in materialized)
    return (
        None
        if not attempted
        else sum(result.correct is True for result in materialized) / attempted
    )
