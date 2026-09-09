"""Provider wrappers for replayable, secret-safe execution traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._provider_http import (
    MAX_BODY_BYTES,
    TRANSPORT_VERSION,
    checked_headers,
    endpoint_parts,
    exchange,
    json_object,
    response_chunks,
    timeout_value,
)
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

    The API key is read from an environment variable for each call; request
    authorization headers are not copied into traces. Raw upstream output is
    not content-redacted and may itself contain sensitive data. The provider
    returns the decoded object to preserve usage, finish reasons and metadata.
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
        endpoint_parts(endpoint)
        timeout_value(timeout)
        if api_key_env is not None and (
            not isinstance(api_key_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", api_key_env) is None
        ):
            raise ValueError("api_key_env must be a non-empty string or None")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string or None")
        checked_headers(headers)
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = float(timeout)
        self.headers = dict(headers or {})

    @property
    def transport_version(self) -> str:
        """Fixed bounds/routing contract included in durable caller identities."""
        return TRANSPORT_VERSION

    def _request(self, body: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        headers = checked_headers(self.headers)
        request_headers = {
            key: value for key, value in headers.items() if key.lower() != "content-type"
        }
        request_headers["Content-Type"] = "application/json"
        if self.api_key_env is not None:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise ValueError("provider credential environment variable is not set")
            if len(token) > 4096 or any(not 33 <= ord(char) <= 126 for char in token):
                raise ValueError("provider credential is invalid")
            request_headers = {
                key: value
                for key, value in request_headers.items()
                if key.lower() != "authorization"
            }
            request_headers["Authorization"] = f"Bearer {token}"
        try:
            encoded = bytearray()
            for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(body):
                encoded.extend(chunk.encode("utf-8"))
                if len(encoded) > MAX_BODY_BYTES:
                    raise ValueError
            return bytes(encoded), request_headers
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ValueError("provider request is invalid or exceeds byte limit") from None

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
        encoded, request_headers = self._request(body)
        with exchange(self.endpoint, self.timeout, request_headers, encoded) as response:
            raw = b"".join(response_chunks(response))
        return json_object(raw)


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
        encoded, request_headers = self._request(body)
        with exchange(
            self.endpoint, self.timeout, request_headers, encoded, streaming=True
        ) as response:
            pending = bytearray()
            done = False
            for chunk in response_chunks(response):
                if done:
                    continue  # Still verify the HTTP body/trailer framing after the SSE sentinel.
                pending.extend(chunk)
                while (end := pending.find(b"\n")) >= 0:
                    raw_line = bytes(pending[:end])
                    del pending[: end + 1]
                    if self._stream_done(raw_line):
                        done = True
                        pending.clear()
                        break
                    payload = self._stream_object(raw_line)
                    if payload is not None:
                        yield payload
            if pending and not done and not self._stream_done(bytes(pending)):
                payload = self._stream_object(bytes(pending))
                if payload is not None:
                    yield payload

    @staticmethod
    def _stream_done(line: bytes) -> bool:
        value = line.strip()
        return value.startswith(b"data:") and value[5:].strip() == b"[DONE]"

    @staticmethod
    def _stream_object(raw: bytes) -> dict[str, Any] | None:
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeError:
            raise ValueError("provider stream contained invalid UTF-8") from None
        if not line.startswith("data:"):
            return None
        return json_object(line[5:].strip().encode("utf-8"))

    def __call__(self, row: RenderedScenario) -> dict[str, Any]:
        """Aggregate one text choice without inventing a successful finish."""
        chunks = tuple(self.stream(row))
        text_parts: list[str] = []
        role: str | None = None
        finish_reason: str | None = None
        choice_index: int | None = None
        for chunk in chunks:
            if any(
                chunk.get(key) for key in ("error", "tool_calls", "function_call", "refusal")
            ) or ("status" in chunk and chunk["status"] != "completed"):
                raise ValueError("provider stream cannot aggregate errors or incomplete status")
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                if "choices" in chunk:
                    raise ValueError("provider stream choices must be an array")
                continue
            if len(choices) > 1:
                raise ValueError("provider stream aggregation requires one choice")
            for choice in choices:
                if not isinstance(choice, Mapping):
                    raise ValueError("provider stream choice must be an object")
                index = choice.get("index", 0)
                if type(index) is not int or index < 0 or choice_index not in (None, index):
                    raise ValueError("provider stream aggregation requires one choice")
                choice_index = index
                delta = choice.get("delta")
                if finish_reason is not None and delta:
                    raise ValueError("provider stream has content after its terminal choice")
                reason = choice.get("finish_reason")
                if reason is not None:
                    if not isinstance(reason, str) or finish_reason is not None:
                        raise ValueError("provider stream terminal reason is ambiguous")
                    finish_reason = reason
                if choice.get("message"):
                    raise ValueError("provider stream aggregation requires text deltas")
                if not isinstance(delta, Mapping):
                    if delta is not None:
                        raise ValueError("provider stream delta must be an object")
                    continue
                if delta.get("tool_calls") or delta.get("function_call") or delta.get("refusal"):
                    raise ValueError(
                        "provider stream aggregation requires text without tools or refusal"
                    )
                if "role" in delta:
                    if delta["role"] != "assistant":
                        raise ValueError("provider stream aggregation requires an assistant role")
                    role = "assistant"
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError("provider stream aggregation requires string content")
                if isinstance(content, str):
                    text_parts.append(content)
        return {
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role or "assistant", "content": "".join(text_parts)},
                    "finish_reason": finish_reason,
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
