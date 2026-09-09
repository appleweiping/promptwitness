"""CLI commands for durable sessions and explicit interrupted-call recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .parser import load_prompt
from .providers import OpenAICompatibleProvider
from .sessions import (
    OpenAISessionProvider,
    SessionEvent,
    SessionJournal,
    SessionReplayProvider,
    SessionRunner,
    _load,
    replay_session,
)


def add_session_commands(subparsers: Any) -> None:
    """Register lifecycle commands with the existing public argument parser."""
    session = subparsers.add_parser("session", help="create, resume, and inspect durable sessions")
    commands = session.add_subparsers(dest="session_command", required=True)
    for name, help_text in (
        ("create", "create a session with a pinned rendered prompt"),
        ("show", "verify and inspect current state"),
        ("user", "append a user message"),
        ("run", "run a provider until user input or a pending tool"),
        ("respond", "record an assistant response or recover a pending request"),
        ("tool-result", "record a supplied tool result without executing a tool"),
        ("finish", "explicitly stop a session"),
        ("export", "export verified events for offline replay"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("database", type=Path)
        command.add_argument("session_id")
        command.set_defaults(handler=run_session_command)
        if name in {"user", "respond", "tool-result", "finish"}:
            command.add_argument("--revision", type=int, required=True)
        if name == "create":
            command.add_argument("prompt", type=Path)
            command.add_argument("--values", type=Path)
            command.add_argument("--max-turns", type=int, default=32)
            command.add_argument("--max-tool-calls", type=int, default=8)
        elif name == "user":
            command.add_argument("text", help="UTF-8 file containing the user message", type=Path)
        elif name == "respond":
            command.add_argument("response", type=Path)
        elif name == "tool-result":
            command.add_argument("call_id")
            command.add_argument("result", type=Path)
        elif name == "finish":
            command.add_argument("--reason", default="user_finished")
        elif name == "show":
            command.add_argument("--messages", action="store_true")
        elif name == "run":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--endpoint")
            source.add_argument("--replay-events", type=Path)
            command.add_argument("--model")
            command.add_argument("--api-key-env")
        elif name == "export":
            command.add_argument("output", type=Path)
    replay = commands.add_parser("replay", help="reconstruct state from exported events offline")
    replay.add_argument("events", type=Path)
    replay.add_argument("--messages", action="store_true")
    replay.set_defaults(handler=run_session_command)


def _events(path: Path) -> tuple[tuple[SessionEvent, ...], str]:
    payload = _load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"format", "head_digest", "events"}:
        raise ValueError("invalid session event export fields")
    if payload["format"] != "promptwitness.session/v1" or not isinstance(payload["events"], list):
        raise ValueError("unsupported session event export")
    if not isinstance(payload["head_digest"], str):
        raise ValueError("head_digest must be a string")
    events = tuple(SessionEvent.from_dict(event) for event in payload["events"])
    replay_session(events, expected_digest=payload["head_digest"])
    return events, payload["head_digest"]


def run_session_command(arguments: argparse.Namespace) -> int:
    """Run a bounded session command; print metadata unless content is requested."""
    action = arguments.session_command
    if action == "replay":
        events, digest = _events(arguments.events)
        view = replay_session(events, expected_digest=digest)
    else:
        if action != "create" and not arguments.database.is_file():
            raise ValueError("session database does not exist")
        for name in ("prompt", "values", "text", "response", "result", "replay_events", "output"):
            value = getattr(arguments, name, None)
            if value is not None and (
                value.resolve() == arguments.database.resolve()
                or (
                    value.exists()
                    and arguments.database.exists()
                    and value.samefile(arguments.database)
                )
            ):
                raise ValueError("session database must be distinct from input and export paths")
        with SessionJournal(arguments.database) as journal:
            identifier = arguments.session_id
            if action == "create":
                values = (
                    _load(arguments.values.read_text(encoding="utf-8")) if arguments.values else {}
                )
                if not isinstance(values, dict):
                    raise ValueError("session values must be an object")
                view = journal.create(
                    identifier,
                    load_prompt(arguments.prompt),
                    values=values,
                    max_turns=arguments.max_turns,
                    max_tool_calls=arguments.max_tool_calls,
                )
            elif action == "user":
                view = journal.add_user(
                    identifier,
                    arguments.text.read_text(encoding="utf-8"),
                    expected_revision=arguments.revision,
                )
            elif action == "finish":
                view = journal.finish(
                    identifier, arguments.reason, expected_revision=arguments.revision
                )
            elif action == "respond":
                response = _load(arguments.response.read_text(encoding="utf-8"))
                before = journal.read(identifier)
                revision = arguments.revision
                if before.revision != revision:
                    raise ValueError("session revision does not match --revision")
                if before.status == "ready_provider":
                    request_id = journal.reserve_provider(
                        identifier,
                        identity=before.provider_identity or {"provider": "manual"},
                        expected_revision=revision,
                    ).request_id
                    revision += 1
                elif before.status == "pending_provider" and before.pending is not None:
                    request_id = before.pending["request_id"]
                else:
                    raise ValueError("session is not waiting for a provider response")
                view = journal.complete_provider(
                    identifier, request_id, response, expected_revision=revision
                )
            elif action == "tool-result":
                result = _load(arguments.result.read_text(encoding="utf-8"))
                before = journal.read(identifier)
                revision = arguments.revision
                if before.revision != revision:
                    raise ValueError("session revision does not match --revision")
                if before.status == "ready_tool":
                    state = journal._read_state(identifier)
                    if state.calls[0]["id"] != arguments.call_id:
                        raise ValueError("call_id does not match next tool")
                    journal.reserve_tool(identifier, expected_revision=revision)
                    revision += 1
                view = journal.complete_tool(
                    identifier, arguments.call_id, result, expected_revision=revision
                )
            elif action == "run":
                provider: Any
                if arguments.replay_events is not None:
                    events, _digest = _events(arguments.replay_events)
                    provider = SessionReplayProvider(events)
                else:
                    provider = OpenAISessionProvider(
                        OpenAICompatibleProvider(
                            arguments.endpoint,
                            model=arguments.model,
                            api_key_env=arguments.api_key_env,
                        )
                    )
                view = SessionRunner(journal).run(identifier, provider)
            else:
                view = journal.read(identifier)
                if action == "export":
                    events = journal.events(identifier)
                    # Verify against the earlier snapshot to detect concurrent changes.
                    replay_session(events, expected_digest=view.digest)
                    payload = {
                        "format": "promptwitness.session/v1",
                        "head_digest": view.digest,
                        "events": [event.to_dict() for event in events],
                    }
                    with arguments.output.open("x", encoding="utf-8") as destination:
                        destination.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(view.to_dict(include_messages=getattr(arguments, "messages", False)), indent=2)
    )
    return 2 if view.status.startswith("pending_") else 0
