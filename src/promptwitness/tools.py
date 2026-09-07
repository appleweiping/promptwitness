"""Bounded local tool-call dispatch for provider responses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

ToolHandler = Callable[[Mapping[str, Any]], Any]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """One redacted tool result with digests for request and response."""

    call_id: str
    name: str
    arguments_digest: str
    result: Any = None
    result_digest: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.result_digest is not None


@dataclass(frozen=True, slots=True)
class ToolBatch:
    """Ordered outcomes for one provider response."""

    outcomes: tuple[ToolOutcome, ...]

    @property
    def complete(self) -> bool:
        return all(item.succeeded for item in self.outcomes)

    def messages(self) -> tuple[dict[str, Any], ...]:
        """Return provider-compatible tool messages without request arguments."""

        return tuple(
            {
                "role": "tool",
                "tool_call_id": outcome.call_id,
                "name": outcome.name,
                "content": json.dumps(
                    outcome.result if outcome.succeeded else {"error": outcome.error},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            for outcome in self.outcomes
        )


class ToolDispatcher:
    """Dispatch declared tool calls under deterministic resource limits."""

    def __init__(
        self,
        handlers: Mapping[str, ToolHandler],
        *,
        max_calls: int = 8,
        max_result_bytes: int = 1_048_576,
    ) -> None:
        if not isinstance(handlers, Mapping) or not all(
            isinstance(name, str) and name.strip() and callable(handler)
            for name, handler in handlers.items()
        ):
            raise TypeError("handlers must map non-empty names to callables")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if isinstance(max_result_bytes, bool) or not isinstance(max_result_bytes, int):
            raise TypeError("max_result_bytes must be an integer")
        if max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive")
        self.handlers = dict(handlers)
        self.max_calls = max_calls
        self.max_result_bytes = max_result_bytes

    def dispatch(self, response: Mapping[str, Any]) -> ToolBatch:
        """Execute response ``tool_calls`` and retain per-call failures."""

        if not isinstance(response, Mapping):
            raise TypeError("provider response must be an object")
        raw_calls = response.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be an array")
        if len(raw_calls) > self.max_calls:
            raise ValueError(f"tool call count exceeds {self.max_calls}")
        outcomes: list[ToolOutcome] = []
        for index, raw in enumerate(raw_calls):
            call_id = str(raw.get("id", index)) if isinstance(raw, Mapping) else str(index)
            name = str(raw.get("name", "")) if isinstance(raw, Mapping) else ""
            arguments: Any = raw.get("arguments", {}) if isinstance(raw, Mapping) else {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    outcomes.append(
                        ToolOutcome(
                            call_id,
                            name,
                            _digest(arguments),
                            error=f"invalid JSON arguments: {error}",
                        )
                    )
                    continue
            if not isinstance(arguments, Mapping):
                outcomes.append(
                    ToolOutcome(
                        call_id, name, _digest(arguments), error="tool arguments must be an object"
                    )
                )
                continue
            arguments_digest = _digest(dict(arguments))
            handler = self.handlers.get(name)
            if handler is None:
                outcomes.append(
                    ToolOutcome(call_id, name, arguments_digest, error=f"unknown tool {name!r}")
                )
                continue
            try:
                result = handler(dict(arguments))
                encoded = json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                if len(encoded) > self.max_result_bytes:
                    raise ValueError(f"tool result exceeds {self.max_result_bytes} bytes")
                outcomes.append(
                    ToolOutcome(
                        call_id, name, arguments_digest, result, hashlib.sha256(encoded).hexdigest()
                    )
                )
            except Exception as error:  # tool failures are isolated per call
                outcomes.append(
                    ToolOutcome(
                        call_id, name, arguments_digest, error=f"{type(error).__name__}: {error}"
                    )
                )
        return ToolBatch(tuple(outcomes))
