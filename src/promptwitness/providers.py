"""Provider wrappers for replayable, secret-safe execution traces."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matrix import RenderedScenario
from .models import message_content_to_wire


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


class OpenAICompatibleProvider:
    """Minimal JSON provider for OpenAI-compatible chat endpoints.

    The API key is read from an environment variable for each call and is never
    included in returned output or trace objects. The provider returns the
    decoded response object so callers can preserve usage and model metadata.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key_env: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.strip()
        ):
            raise ValueError("api_key_env must be a non-empty string or None")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string or None")
        if headers is not None and not all(
            isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
        ):
            raise TypeError("headers must map strings to strings")
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = float(timeout)
        self.headers = dict(headers or {})

    def __call__(self, row: RenderedScenario) -> Any:
        """Send a non-streaming chat request and return its JSON response."""
        return self.complete(
            [
                {
                    "role": message.role,
                    **({"name": message.name} if message.name else {}),
                    "content": message_content_to_wire(message),
                }
                for message in row.messages
            ]
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        generation: Mapping[str, Any] | None = None,
    ) -> Any:
        """Send chat history, including correlated tool calls and tool results.

        Session callers validate lifecycle and tool contracts before this
        transport boundary. API keys remain environment-only as in ``__call__``.
        """
        if not messages or not all(isinstance(message, Mapping) for message in messages):
            raise ValueError("messages must be a non-empty sequence of objects")
        if not all(isinstance(tool, Mapping) for tool in tools):
            raise ValueError("tools must be a sequence of objects")
        body: dict[str, Any] = {"messages": [dict(message) for message in messages]}
        if generation is not None:
            # Local import keeps the existing provider/session modules acyclic.
            from .task_data import generation_settings

            body.update(generation_settings(generation))
        if tools:
            body["tools"] = [dict(tool) for tool in tools]
        if self.model is not None:
            body["model"] = self.model
        request_headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key_env is not None:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise ValueError(f"environment variable {self.api_key_env!r} is not set")
            request_headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec B310 - endpoint scheme is restricted above
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise ValueError(f"provider returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ValueError(f"provider request failed: {type(error).__name__}") from error
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("provider returned invalid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("provider response must be a JSON object")
        return dict(payload)


class OpenAICompatibleStreamingProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for newline-delimited server-sent events.

    ``stream`` yields decoded JSON chunks in wire order and stops at the
    standard ``data: [DONE]`` sentinel. Calling the provider aggregates text
    deltas into the same small response shape used by replay fixtures while
    retaining the raw chunks under ``stream_chunks`` for audit consumers.
    The implementation deliberately uses only the standard library so a
    provider can be exercised in tests without an SDK or model dependency.
    """

    def stream(self, row: RenderedScenario) -> Iterator[Mapping[str, Any]]:
        """Yield one JSON object for each non-empty SSE data event."""
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": message.role,
                    **({"name": message.name} if message.name else {}),
                    "content": message_content_to_wire(message),
                }
                for message in row.messages
            ],
            "stream": True,
        }
        if self.model is not None:
            body["model"] = self.model
        request_headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key_env is not None:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise ValueError(f"environment variable {self.api_key_env!r} is not set")
            request_headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec B310 - endpoint scheme is restricted above
                for raw_line in response:
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeError as error:
                        raise ValueError("provider stream contained invalid UTF-8") from error
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise ValueError("provider stream contained invalid JSON") from error
                    if not isinstance(payload, Mapping):
                        raise ValueError("provider stream chunks must be JSON objects")
                    yield dict(payload)
        except urllib.error.HTTPError as error:
            raise ValueError(f"provider returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ValueError(f"provider request failed: {type(error).__name__}") from error

    def __call__(self, row: RenderedScenario) -> dict[str, Any]:
        """Aggregate streamed text deltas and retain a redacted chunk audit."""
        chunks = tuple(self.stream(row))
        text_parts: list[str] = []
        role: str | None = None
        for chunk in chunks:
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                if isinstance(delta.get("role"), str):
                    role = delta["role"]
                content = delta.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
        return {
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role or "assistant", "content": "".join(text_parts)},
                    "finish_reason": "stop",
                }
            ],
            "stream_chunks": list(chunks),
        }


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
                    {
                        "role": message.role,
                        "name": message.name,
                        "content": message_content_to_wire(message),
                    }
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


def load_traces(path: str | Path) -> tuple[ProviderTrace, ...]:
    """Load and authenticate a saved provider-trace artifact."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read provider traces: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "promptwitness.provider-trace/v1"
    ):
        raise ValueError("unsupported provider-trace format")
    raw_traces = payload.get("traces")
    if not isinstance(raw_traces, list):
        raise ValueError("provider traces must contain an array")
    traces: list[ProviderTrace] = []
    for index, raw in enumerate(raw_traces, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"provider trace {index} must be an object")
        events_value = raw.get("events")
        if not isinstance(events_value, list):
            raise ValueError(f"provider trace {index} events must be an array")
        try:
            events = tuple(
                TraceEvent(
                    str(event["kind"]),
                    str(event["scenario_id"]),
                    str(event["prompt_digest"]),
                    str(event["payload_digest"]),
                )
                for event in events_value
            )
            trace = ProviderTrace(
                str(raw["scenario_id"]),
                str(raw["prompt_digest"]),
                raw["output"],
                str(raw["output_digest"]),
                events,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid provider trace {index}") from error
        if trace.output_digest != _digest(trace.output) or raw.get("digest") != trace.digest():
            raise ValueError(f"provider trace {index} digest mismatch")
        traces.append(trace)
    return tuple(traces)
