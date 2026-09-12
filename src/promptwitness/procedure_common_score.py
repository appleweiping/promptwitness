"""Bounded, inert parsing primitives for the versioned procedural scorers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class ProcedureScoreLimitError(ValueError):
    """Scoring was not attempted/completed within the declared resource policy."""


@dataclass(frozen=True, slots=True)
class ProcedureScoreLimits:
    """Hard-bounded aggregate admission, not a CPU-time or process-RSS guarantee."""

    max_prediction_bytes: int = 1024 * 1024
    max_case_bytes: int = 16 * 1024 * 1024
    max_line_bytes: int = 64 * 1024
    max_lines: int = 20_000
    max_work_items: int = 500_000

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_prediction_bytes", 4 * 1024 * 1024),
            ("max_case_bytes", 32 * 1024 * 1024),
            ("max_line_bytes", 256 * 1024),
            ("max_lines", 100_000),
            ("max_work_items", 2_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer from 1 to {ceiling}")

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "max_prediction_bytes",
                "max_case_bytes",
                "max_line_bytes",
                "max_lines",
                "max_work_items",
            )
        }


def utf8_size(value: str, maximum: int) -> int:
    """Check before allocating a whole UTF-8 encoding; reject lone surrogates."""
    if len(value) > maximum:
        raise ProcedureScoreLimitError("procedural scoring text byte limit exceeded")
    size = 0
    try:
        for offset in range(0, len(value), 4096):
            size += len(value[offset : offset + 4096].encode("utf-8"))
            if size > maximum:
                raise ProcedureScoreLimitError("procedural scoring text byte limit exceeded")
    except UnicodeEncodeError as error:
        raise ValueError("procedural scoring text must be valid Unicode") from error
    return size


class ScoreWork:
    """One call's combined source/response nodes, parsed units, and text lines."""

    def __init__(self, limits: ProcedureScoreLimits) -> None:
        self.limits = limits
        self.items = 0
        self.line_count = 0
        self.source_bytes = 0

    def use(self, amount: int) -> None:
        if amount > self.limits.max_work_items - self.items:
            raise ProcedureScoreLimitError("procedural scoring aggregate work limit exceeded")
        self.items += amount

    def source(self, value: Any) -> None:
        # ProcedureCase is an already validated immutable JSON tree. Pending
        # children count toward admission before the stack is expanded.
        stack = [value]
        while stack:
            current = stack.pop()
            self.use(1)
            if isinstance(current, str):
                self.source_bytes += utf8_size(
                    current, self.limits.max_case_bytes - self.source_bytes
                )
            elif isinstance(current, Mapping):
                self.use(len(current))  # keys are also work and byte-bearing data
                for key in current:
                    self.source_bytes += utf8_size(
                        key, self.limits.max_case_bytes - self.source_bytes
                    )
                if len(current) + len(stack) > self.limits.max_work_items - self.items:
                    raise ProcedureScoreLimitError(
                        "procedural scoring aggregate work limit exceeded"
                    )
                stack.extend(current.values())
            elif isinstance(current, (list, tuple)):
                if len(current) + len(stack) > self.limits.max_work_items - self.items:
                    raise ProcedureScoreLimitError(
                        "procedural scoring aggregate work limit exceeded"
                    )
                stack.extend(current)

    def lines(self, text: str) -> list[str]:
        # Only CRLF is a newline alias; Unicode separators remain actual data.
        text = text.replace("\r\n", "\n")
        count = text.count("\n") + 1
        if count > self.limits.max_lines - self.line_count:
            raise ProcedureScoreLimitError("procedural scoring aggregate line limit exceeded")
        self.line_count += count
        self.use(count)
        lines = text.split("\n")
        for line in lines:
            utf8_size(line, self.limits.max_line_bytes)
        return lines


def tagged(text: str, tag: str) -> str | None:
    """Extract exactly one correctly ordered pair, never an unclosed prefix."""
    opening, closing = f"<{tag}>", f"</{tag}>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        return None
    start, end = text.index(opening) + len(opening), text.index(closing)
    if end < start:
        return None
    return text[start:end].strip()


def prefix_count(predicted: list[Any], expected: list[Any]) -> int:
    count = 0
    for actual, gold in zip(predicted, expected, strict=False):
        if actual != gold:
            break
        count += 1
    return count


def prefix_fraction(predicted: list[Any], expected: list[Any]) -> float:
    return prefix_count(predicted, expected) / len(expected) if expected else float(not predicted)


def result(
    metrics: dict[str, float],
    primary: str | None,
    processed: Any,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    unsupported: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "metrics": metrics,
        "primary_metric": primary,
        "primary_score": metrics[primary] if primary else None,
        "score_status": "scored" if primary else "unsupported",
        "unsupported_metrics": dict(unsupported or {}),
        "processed": processed,
        "diagnostics": dict(diagnostics or {}),
    }
