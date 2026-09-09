"""Durable conversation sessions with validated, replayable state transitions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .adapters import prompt_to_dict
from .invocations import validate_tool_arguments
from .matrix import Scenario, render_matrix
from .models import PromptDocument, Severity, _freeze_json, _thaw_json, message_content_to_wire
from .parser import _finite_float, _reject_non_finite, _unique_object, parse_prompt
from .providers import OpenAICompatibleProvider
from .schemas import resolve_local_refs
from .tools import ToolHandler
from .validation import validate_prompt

_ZERO = "0" * 64


def _json(value: Any) -> str:
    return json.dumps(
        _thaw_json(_freeze_json(value, "session data")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_non_finite,
        parse_float=_finite_float,
    )


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("provider identity must be a non-empty object")
    _text(value.get("provider"), "provider identity.provider")
    result: dict[str, Any] = _load(_json(value))
    return result


class SessionConflict(ValueError):
    """The expected session revision is stale; no event was committed."""


class SessionTransitionError(ValueError):
    """An event cannot follow the current conversation state."""


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One immutable event, linked to its predecessor by a content checksum."""

    session_id: str
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    previous_digest: str
    digest: str

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _positive(self.sequence, "sequence")
        _text(self.kind, "event kind")
        if not isinstance(self.payload, Mapping):
            raise ValueError("event payload must be an object")
        object.__setattr__(self, "payload", _freeze_json(self.payload, "event payload"))
        expected = hashlib.sha256(_json(self.to_dict(include_digest=False)).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("session event checksum mismatch")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": _thaw_json(self.payload),
            "previous_digest": self.previous_digest,
        }
        if include_digest:
            result["digest"] = self.digest
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SessionEvent:
        if not isinstance(raw, Mapping) or set(raw) != {
            "session_id",
            "sequence",
            "kind",
            "payload",
            "previous_digest",
            "digest",
        }:
            raise ValueError("invalid session event fields")
        return cls(**dict(raw))


@dataclass(frozen=True, slots=True)
class SessionRequest:
    """A reserved model request, including the complete current tool history."""

    session_id: str
    request_id: str
    provider_identity: Mapping[str, Any]
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_identity",
            _freeze_json(_identity(self.provider_identity), "provider identity"),
        )
        object.__setattr__(self, "messages", _freeze_json(self.messages, "messages"))
        object.__setattr__(self, "tools", _freeze_json(self.tools, "tools"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "provider_identity": _thaw_json(self.provider_identity),
            "messages": _thaw_json(self.messages),
            "tools": _thaw_json(self.tools),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionView:
    """Validated snapshot reconstructed entirely from the event journal."""

    session_id: str
    revision: int
    digest: str
    status: str
    turns: int
    max_turns: int
    messages: tuple[Mapping[str, Any], ...]
    pending: Mapping[str, Any] | None
    reason: str | None
    provider_identity: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _freeze_json(self.messages, "messages"))
        object.__setattr__(self, "pending", _freeze_json(self.pending, "pending"))
        object.__setattr__(
            self, "provider_identity", _freeze_json(self.provider_identity, "provider identity")
        )

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "revision": self.revision,
            "digest": self.digest,
            "status": self.status,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "message_count": len(self.messages),
            "pending": _thaw_json(self.pending),
            "reason": self.reason,
            "provider_identity": _thaw_json(self.provider_identity),
        }
        if include_messages:
            result["messages"] = _thaw_json(self.messages)
        return result


@dataclass
class _State:
    session_id: str = ""
    revision: int = 0
    digest: str = _ZERO
    document: PromptDocument | None = None
    status: str = "new"
    turns: int = 0
    max_turns: int = 0
    max_tool_calls: int = 0
    max_event_bytes: int = 0
    max_context_bytes: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    seen_calls: set[str] = field(default_factory=set)
    pending: dict[str, Any] | None = None
    reason: str | None = None
    provider_identity: dict[str, Any] | None = None

    def view(self) -> SessionView:
        return SessionView(
            self.session_id,
            self.revision,
            self.digest,
            self.status,
            self.turns,
            self.max_turns,
            tuple(self.messages),
            self.pending,
            self.reason,
            self.provider_identity,
        )

    def request(self, request_id: str, identity: Mapping[str, Any]) -> SessionRequest:
        if self.document is None:
            raise SessionTransitionError("session has not been created")
        tools = tuple(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: resolve_local_refs(schema)
                            for name, schema in tool.parameters.items()
                        },
                        "required": list(tool.required),
                        "additionalProperties": False,
                    },
                },
            }
            for tool in self.document.tools
        )
        active = _identity(identity)
        if self.provider_identity is not None and _json(active) != _json(self.provider_identity):
            raise SessionTransitionError(
                "provider identity differs from the pinned session provider"
            )
        request = SessionRequest(self.session_id, request_id, active, tuple(self.messages), tools)
        if len(_json(request.to_dict()).encode()) > self.max_context_bytes:
            raise ValueError("session request exceeds max_context_bytes")
        return request


def _assistant(raw: Any, state: _State) -> dict[str, Any]:
    if isinstance(raw, Mapping) and "choices" in raw:
        choices = raw["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("session response must contain exactly one choice")
        raw = choices[0].get("message")
    if not isinstance(raw, Mapping) or raw.get("role", "assistant") != "assistant":
        raise ValueError("provider response must be an assistant message")
    content = raw.get("content")
    calls = raw.get("tool_calls", [])
    if content is not None and not isinstance(content, str):
        raise ValueError("session assistant content must be text or null")
    if not isinstance(calls, list) or len(calls) > state.max_tool_calls:
        raise ValueError("tool_calls must be an array within max_tool_calls")
    if not calls and (content is None or not content.strip()):
        raise ValueError("assistant response requires text or tool calls")
    normalized: list[dict[str, Any]] = []
    identifiers = set(state.seen_calls)
    assert state.document is not None
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("tool call requires id, type, and function")
        identifier = _text(call["id"], "tool call id")
        if identifier in identifiers or call["type"] != "function":
            raise ValueError("tool call ID must be unique and type must be function")
        identifiers.add(identifier)
        function = call["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ValueError("tool function requires name and arguments")
        name = _text(function["name"], "tool name")
        tool = state.document.tool_map().get(name)
        if tool is None:
            raise ValueError(f"undeclared session tool {name!r}")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = _load(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must decode to an object")
        validation = validate_tool_arguments(tool, arguments)
        if not validation.valid:
            raise ValueError(f"invalid arguments for {name!r}: {validation.to_dict()['issues']}")
        normalized.append(
            {
                "id": identifier,
                "type": "function",
                "function": {"name": name, "arguments": _json(arguments)},
            }
        )
    return {
        "role": "assistant",
        "content": content,
        **({"tool_calls": normalized} if calls else {}),
    }


def _require(state: _State, *statuses: str) -> None:
    if state.status not in statuses:
        raise SessionTransitionError(f"operation is invalid while session is {state.status}")


def _apply(state: _State, kind: str, payload: dict[str, Any]) -> None:
    if kind == "created":
        _require(state, "new")
        if set(payload) != {
            "session_id",
            "prompt",
            "max_turns",
            "max_tool_calls",
            "max_event_bytes",
            "max_context_bytes",
        }:
            raise ValueError("invalid session creation fields")
        state.session_id = _text(payload["session_id"], "session_id")
        for name in ("max_turns", "max_tool_calls", "max_event_bytes", "max_context_bytes"):
            setattr(state, name, _positive(payload[name], name))
        state.document = parse_prompt(payload["prompt"])
        if not state.document.messages or any(
            item.role not in {"system", "developer", "assistant", "user"}
            for item in state.document.messages
        ):
            raise ValueError("initial prompt requires messages with supported conversation roles")
        state.messages = [
            {
                "role": item.role,
                "content": message_content_to_wire(item),
                **({"name": item.name} if item.name else {}),
            }
            for item in state.document.messages
        ]
        state.status = "ready_provider" if state.messages[-1]["role"] == "user" else "waiting_user"
    elif kind == "user":
        _require(state, "waiting_user")
        if set(payload) != {"content"}:
            raise ValueError("user event requires content")
        state.messages.append({"role": "user", "content": _text(payload["content"], "content")})
        state.status = "ready_provider"
    elif kind == "provider_started":
        _require(state, "ready_provider")
        expected_id = f"request-{state.revision + 1}"
        identity = _identity(payload.get("provider_identity"))
        if payload != {
            "request_id": expected_id,
            "request_digest": state.request(expected_id, identity).digest,
            "provider_identity": identity,
        }:
            raise ValueError("provider reservation does not match session request")
        state.provider_identity = identity
        state.pending = {"kind": "provider", **payload}
        state.status = "pending_provider"
    elif kind == "assistant":
        _require(state, "pending_provider")
        assert state.pending is not None
        if (
            set(payload) != {"request_id", "response"}
            or payload["request_id"] != state.pending["request_id"]
        ):
            raise SessionTransitionError("response request_id does not match pending provider")
        message = _assistant(payload["response"], state)
        state.messages.append(message)
        state.calls = message.get("tool_calls", []).copy()
        state.seen_calls.update(call["id"] for call in state.calls)
        state.pending = None
        state.turns += 1
        if state.turns >= state.max_turns:
            state.status, state.reason = "finished", "turn_limit"
            state.calls.clear()
        else:
            state.status = "ready_tool" if state.calls else "waiting_user"
    elif kind == "tool_started":
        _require(state, "ready_tool")
        call = state.calls[0]
        if payload != {"call_id": call["id"]}:
            raise SessionTransitionError("tool reservation does not match next call")
        state.pending = {"kind": "tool", "call_id": call["id"], "name": call["function"]["name"]}
        state.status = "pending_tool"
    elif kind == "tool_result":
        _require(state, "pending_tool")
        assert state.pending is not None
        if set(payload) != {"call_id", "result"} or payload["call_id"] != state.pending["call_id"]:
            raise SessionTransitionError("result call_id does not match pending tool")
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": payload["call_id"],
                "name": state.pending["name"],
                "content": _json(payload["result"]),
            }
        )
        state.calls.pop(0)
        state.pending = None
        state.status = "ready_tool" if state.calls else "ready_provider"
    elif kind == "operation_failed":
        _require(state, "pending_provider", "pending_tool")
        if set(payload) != {"error_type"}:
            raise ValueError("failure event requires error_type")
        assert state.pending is not None
        state.pending["error_type"] = _text(payload["error_type"], "error_type")
    elif kind == "finished":
        if state.status in {"new", "finished"}:
            raise SessionTransitionError("cannot finish a new or finished session")
        if set(payload) != {"reason"}:
            raise ValueError("finish event requires reason")
        state.reason = _text(payload["reason"], "reason")
        state.status = "finished"
    else:
        raise ValueError(f"unknown session event kind {kind!r}")
    if len(_json(payload).encode()) > state.max_event_bytes:
        raise ValueError("session event exceeds max_event_bytes")


def _replay(events: Iterable[SessionEvent], expected_digest: str | None = None) -> _State:
    state = _State()
    for event in events:
        if not isinstance(event, SessionEvent):
            raise TypeError("events must contain SessionEvent values")
        if event.sequence != state.revision + 1 or event.previous_digest != state.digest:
            raise ValueError("session event chain is discontinuous")
        if state.session_id and event.session_id != state.session_id:
            raise ValueError("event session_id changed")
        _apply(state, event.kind, _thaw_json(event.payload))
        if state.session_id != event.session_id:
            raise ValueError("creation session_id differs from event")
        state.revision, state.digest = event.sequence, event.digest
    if state.revision == 0:
        raise ValueError("session event journal is empty")
    if expected_digest is not None and state.digest != expected_digest:
        raise ValueError("session head checksum mismatch")
    return state


def replay_session(
    events: Iterable[SessionEvent], *, expected_digest: str | None = None
) -> SessionView:
    """Rebuild and verify session state without invoking models or tools."""
    return _replay(events, expected_digest).view()


class SessionJournal:
    """SQLite append-only event store with transactional optimistic concurrency.

    External calls run after their reservation commits. A crash leaves a pending
    operation for explicit recovery. The journal never retries uncertain tool effects.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS session_heads (
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_events (
                session_id TEXT NOT NULL REFERENCES session_heads(session_id),
                sequence INTEGER NOT NULL, event TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence)
            );
            CREATE TRIGGER IF NOT EXISTS immutable_session_events_update
            BEFORE UPDATE ON session_events BEGIN
                SELECT RAISE(ABORT, 'session events are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS immutable_session_events_delete
            BEFORE DELETE ON session_events BEGIN
                SELECT RAISE(ABORT, 'session events are append-only');
            END;
        """)

    def __enter__(self) -> SessionJournal:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def events(self, session_id: str) -> tuple[SessionEvent, ...]:
        _text(session_id, "session_id")
        rows = self._connection.execute(
            "SELECT event FROM session_events WHERE session_id = ? ORDER BY sequence", (session_id,)
        ).fetchall()
        return tuple(SessionEvent.from_dict(_load(row[0])) for row in rows)

    def _state(self, session_id: str) -> _State:
        head = self._connection.execute(
            "SELECT revision, digest FROM session_heads WHERE session_id = ?", (session_id,)
        ).fetchone()
        if head is None:
            raise ValueError(f"unknown session {session_id!r}")
        state = _replay(self.events(session_id), head[1])
        if state.revision != head[0] or state.session_id != session_id:
            raise ValueError("session head revision or identity mismatch")
        return state

    def read(self, session_id: str) -> SessionView:
        """Read one consistent, verified snapshot."""
        return self._read_state(session_id).view()

    def _read_state(self, session_id: str) -> _State:
        self._connection.execute("BEGIN")
        try:
            return self._state(session_id)
        finally:
            self._connection.rollback()

    def _append(
        self, session_id: str, kind: str, payload: Mapping[str, Any], expected_revision: int
    ) -> SessionView:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if expected_revision == 0:
                if self._connection.execute(
                    "SELECT 1 FROM session_heads WHERE session_id = ?", (session_id,)
                ).fetchone():
                    raise SessionConflict("session already exists")
                state = _State()
            else:
                state = self._state(session_id)
            if state.revision != expected_revision:
                raise SessionConflict(
                    f"expected revision {expected_revision}, found {state.revision}"
                )
            copied = _load(_json(payload))
            previous_digest = state.digest
            _apply(state, kind, copied)
            if state.session_id != session_id:
                raise ValueError("session identity mismatch")
            raw = {
                "session_id": session_id,
                "sequence": expected_revision + 1,
                "kind": kind,
                "payload": copied,
                "previous_digest": previous_digest,
            }
            digest = hashlib.sha256(_json(raw).encode()).hexdigest()
            event = SessionEvent.from_dict({**raw, "digest": digest})
            if expected_revision == 0:
                self._connection.execute(
                    "INSERT INTO session_heads VALUES (?, ?, ?)",
                    (session_id, event.sequence, digest),
                )
            else:
                self._connection.execute(
                    "UPDATE session_heads SET revision = ?, digest = ? WHERE session_id = ?",
                    (event.sequence, digest, session_id),
                )
            self._connection.execute(
                "INSERT INTO session_events VALUES (?, ?, ?)",
                (session_id, event.sequence, _json(event.to_dict())),
            )
            self._connection.commit()
            state.revision, state.digest = event.sequence, digest
            return state.view()
        except BaseException:
            self._connection.rollback()
            raise

    def create(
        self,
        session_id: str,
        document: PromptDocument,
        *,
        values: Mapping[str, Any] | None = None,
        max_turns: int = 32,
        max_tool_calls: int = 8,
        max_event_bytes: int = 1_048_576,
        max_context_bytes: int = 4_194_304,
    ) -> SessionView:
        """Render and validate a pinned prompt before recording session creation."""
        _text(session_id, "session_id")
        expanded = replace(
            document,
            tools=tuple(
                replace(
                    tool,
                    parameters={
                        name: resolve_local_refs(schema) for name, schema in tool.parameters.items()
                    },
                )
                for tool in document.tools
            ),
        )
        report = validate_prompt(expanded)
        if any(finding.severity is Severity.BREAKING for finding in report.findings):
            raise ValueError("session prompt has breaking validation findings")
        try:
            (rendered,) = render_matrix(document, (Scenario(session_id, values or {}),))
        except KeyError as error:
            raise ValueError(str(error)) from error
        pinned = replace(document, messages=rendered.messages)
        return self._append(
            session_id,
            "created",
            {
                "session_id": session_id,
                "prompt": prompt_to_dict(pinned),
                "max_turns": max_turns,
                "max_tool_calls": max_tool_calls,
                "max_event_bytes": max_event_bytes,
                "max_context_bytes": max_context_bytes,
            },
            0,
        )

    def add_user(self, session_id: str, content: str, *, expected_revision: int) -> SessionView:
        return self._append(session_id, "user", {"content": content}, expected_revision)

    def reserve_provider(
        self, session_id: str, *, identity: Mapping[str, Any], expected_revision: int
    ) -> SessionRequest:
        state = self._read_state(session_id)
        request = state.request(f"request-{expected_revision + 1}", identity)
        self._append(
            session_id,
            "provider_started",
            {
                "request_id": request.request_id,
                "request_digest": request.digest,
                "provider_identity": _thaw_json(request.provider_identity),
            },
            expected_revision,
        )
        return request

    def complete_provider(
        self, session_id: str, request_id: str, response: Any, *, expected_revision: int
    ) -> SessionView:
        return self._append(
            session_id,
            "assistant",
            {
                "request_id": request_id,
                "response": response,
            },
            expected_revision,
        )

    def reserve_tool(self, session_id: str, *, expected_revision: int) -> dict[str, Any]:
        state = self._read_state(session_id)
        _require(state, "ready_tool")
        call = state.calls[0]
        self._append(session_id, "tool_started", {"call_id": call["id"]}, expected_revision)
        return {
            "call_id": call["id"],
            "name": call["function"]["name"],
            "arguments": _load(call["function"]["arguments"]),
        }

    def complete_tool(
        self, session_id: str, call_id: str, result: Any, *, expected_revision: int
    ) -> SessionView:
        return self._append(
            session_id,
            "tool_result",
            {
                "call_id": call_id,
                "result": result,
            },
            expected_revision,
        )

    def record_failure(
        self, session_id: str, error: Exception, *, expected_revision: int
    ) -> SessionView:
        """Record only the error type; retain the reservation for explicit recovery."""
        return self._append(
            session_id,
            "operation_failed",
            {
                "error_type": type(error).__name__,
            },
            expected_revision,
        )

    def finish(
        self, session_id: str, reason: str = "user_finished", *, expected_revision: int
    ) -> SessionView:
        return self._append(session_id, "finished", {"reason": reason}, expected_revision)


class OpenAISessionProvider:
    """Use the existing OpenAI-compatible transport with complete session history."""

    def __init__(self, provider: OpenAICompatibleProvider) -> None:
        self.provider = provider

    @property
    def identity(self) -> Mapping[str, Any]:
        """Pin non-secret routing/settings; credentials themselves are never stored."""
        # Custom transport headers can select a tenant or deployment. Hash the
        # configuration rather than exposing potentially private header values.
        return {
            "provider": "openai-compatible",
            "transport": self.provider.transport_version,
            "model": self.provider.model,
            "endpoint_sha256": hashlib.sha256(self.provider.endpoint.encode()).hexdigest(),
            "headers_sha256": hashlib.sha256(_json(self.provider.headers).encode()).hexdigest(),
            "api_key_env": self.provider.api_key_env,
            "timeout": self.provider.timeout,
        }

    def __call__(self, request: SessionRequest) -> Any:
        if _json(_identity(request.provider_identity)) != _json(self.identity):
            raise SessionTransitionError("provider settings changed after request reservation")
        body = request.to_dict()
        return self.provider.complete(body["messages"], tools=body["tools"])


class SessionReplayProvider:
    """Answer only exact recorded requests; never invoke an external provider."""

    def __init__(self, events: Iterable[SessionEvent]) -> None:
        recorded = tuple(events)
        state = _replay(recorded)
        self.identity: Mapping[str, Any] | None = state.provider_identity
        requests: dict[str, str] = {}
        self._responses: dict[str, Any] = {}
        for event in recorded:
            payload = _thaw_json(event.payload)
            if event.kind == "provider_started":
                requests[payload["request_id"]] = payload["request_digest"]
            elif event.kind == "assistant":
                self._responses[requests[payload["request_id"]]] = payload["response"]

    def __call__(self, request: SessionRequest) -> Any:
        if request.digest not in self._responses:
            raise ValueError("no recorded response for this exact session request")
        return _load(_json(self._responses[request.digest]))


class SessionRunner:
    """Advance a session until input, a missing handler, a failure, or its turn limit."""

    def __init__(self, journal: SessionJournal) -> None:
        self.journal = journal

    def run(
        self,
        session_id: str,
        provider: Callable[[SessionRequest], Any],
        *,
        handlers: Mapping[str, ToolHandler] | None = None,
        identity: Mapping[str, Any] | None = None,
    ) -> SessionView:
        if not callable(provider):
            raise TypeError("provider must be callable")
        exposed_identity = getattr(provider, "identity", None)
        if (
            identity is not None
            and exposed_identity is not None
            and _json(_identity(identity)) != _json(_identity(exposed_identity))
        ):
            raise ValueError("explicit identity differs from provider settings")
        active_identity = _identity(identity if identity is not None else exposed_identity)
        active_handlers = dict(handlers or {})
        if not all(
            isinstance(name, str) and callable(handler) for name, handler in active_handlers.items()
        ):
            raise TypeError("handlers must map names to callables")
        while True:
            state = self.journal.read(session_id)
            if state.provider_identity is not None and _json(active_identity) != _json(
                state.provider_identity
            ):
                raise SessionTransitionError(
                    "provider identity differs from the pinned session provider"
                )
            if state.status == "ready_provider":
                request = self.journal.reserve_provider(
                    session_id, identity=active_identity, expected_revision=state.revision
                )
                revision = state.revision + 1
                try:
                    response = provider(request)
                    self.journal.complete_provider(
                        session_id, request.request_id, response, expected_revision=revision
                    )
                except SessionConflict:
                    raise
                except Exception as error:
                    return self.journal.record_failure(
                        session_id, error, expected_revision=revision
                    )
            elif state.status == "ready_tool":
                internal = self.journal._read_state(session_id)
                name = internal.calls[0]["function"]["name"]
                if name not in active_handlers:
                    return state
                call = self.journal.reserve_tool(session_id, expected_revision=state.revision)
                revision = state.revision + 1
                try:
                    result = active_handlers[name](call["arguments"])
                    self.journal.complete_tool(
                        session_id, call["call_id"], result, expected_revision=revision
                    )
                except SessionConflict:
                    raise
                except Exception as error:
                    return self.journal.record_failure(
                        session_id, error, expected_revision=revision
                    )
            else:
                return state
