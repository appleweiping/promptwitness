"""Provider wrappers for replayable, secret-safe execution traces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matrix import RenderedScenario


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One provider-boundary event without secret request headers."""

    kind: str
    scenario_id: str
    prompt_digest: str
    payload_digest: str


@dataclass(frozen=True, slots=True)
class ProviderTrace:
    """Authenticated event inventory for one provider call."""

    scenario_id: str
    prompt_digest: str
    output: Any
    output_digest: str
    events: tuple[TraceEvent, ...]

    def digest(self) -> str:
        """Return a stable digest over output and event metadata."""
        payload = {
            "scenario_id": self.scenario_id,
            "prompt_digest": self.prompt_digest,
            "output": self.output,
            "output_digest": self.output_digest,
            "events": [
                {
                    "kind": event.kind,
                    "scenario_id": event.scenario_id,
                    "prompt_digest": event.prompt_digest,
                    "payload_digest": event.payload_digest,
                }
                for event in self.events
            ],
        }
        return _digest(payload)


class ReplayProvider:
    """Deterministic provider keyed by rendered prompt digest."""

    def __init__(self, responses: Mapping[str, Any]) -> None:
        if not isinstance(responses, Mapping) or not all(isinstance(key, str) for key in responses):
            raise TypeError("responses must map prompt digests to JSON values")
        self._responses = dict(responses)

    def __call__(self, row: RenderedScenario) -> Any:
        """Return the recorded output or fail explicitly for an unseen prompt."""
        if row.digest not in self._responses:
            raise KeyError(f"no replay response for prompt digest {row.digest}")
        return self._responses[row.digest]


class TraceRecorder:
    """Wrap a provider and retain authenticated request/response event traces."""

    def __init__(self, provider: Callable[[RenderedScenario], Any]) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._provider = provider
        self._traces: list[ProviderTrace] = []

    @property
    def traces(self) -> tuple[ProviderTrace, ...]:
        """Return traces in invocation order."""
        return tuple(self._traces)

    def __call__(self, row: RenderedScenario) -> Any:
        """Record request, optional tool calls, and response without raw secrets."""
        request_digest = _digest(
            {
                "prompt_digest": row.digest,
                "messages": [
                    {"role": message.role, "name": message.name, "content": message.content}
                    for message in row.messages
                ],
            }
        )
        events = [TraceEvent("request", row.scenario_id, row.digest, request_digest)]
        output = self._provider(row)
        output_digest = _digest(output)
        if isinstance(output, Mapping) and isinstance(output.get("tool_calls"), list):
            for tool_call in output["tool_calls"]:
                events.append(
                    TraceEvent("tool_call", row.scenario_id, row.digest, _digest(tool_call))
                )
        events.append(TraceEvent("response", row.scenario_id, row.digest, output_digest))
        trace = ProviderTrace(row.scenario_id, row.digest, output, output_digest, tuple(events))
        self._traces.append(trace)
        return output

    def save(self, path: str | Path) -> None:
        """Persist traces with a per-trace authentication digest."""
        payload = {
            "format": "promptwitness.provider-trace/v1",
            "traces": [
                {
                    "scenario_id": trace.scenario_id,
                    "prompt_digest": trace.prompt_digest,
                    "output": trace.output,
                    "output_digest": trace.output_digest,
                    "events": [
                        {
                            "kind": event.kind,
                            "scenario_id": event.scenario_id,
                            "prompt_digest": event.prompt_digest,
                            "payload_digest": event.payload_digest,
                        }
                        for event in trace.events
                    ],
                    "digest": trace.digest(),
                }
                for trace in self._traces
            ],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
