from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from promptwitness import (
    Message,
    OpenAICompatibleProvider,
    OpenAISessionProvider,
    PromptDocument,
    SessionConflict,
    SessionEvent,
    SessionReplayProvider,
    SessionTransitionError,
    ToolSpec,
    replay_session,
)
from promptwitness import (
    SessionJournal as Journal,
)
from promptwitness import (
    SessionRunner as Runner,
)
from promptwitness.cli import main
from promptwitness.sessions import _json

LOCAL_IDENTITY = {"provider": "local-test", "version": "1"}


class SessionJournal(Journal):
    def reserve_provider(self, session_id, *, expected_revision, identity=LOCAL_IDENTITY):
        return super().reserve_provider(
            session_id, expected_revision=expected_revision, identity=identity
        )


class SessionRunner(Runner):
    def run(self, session_id, provider, *, handlers=None, identity=None):
        return super().run(
            session_id,
            provider,
            handlers=handlers,
            identity=identity
            if identity is not None
            else getattr(provider, "identity", LOCAL_IDENTITY),
        )


def prompt(*, user=False):
    messages = (Message("system", "Use the lookup tool before answering {{ name }}."),)
    if user:
        messages += (Message("user", "Where is order 17?"),)
    return PromptDocument(
        "support",
        messages,
        (
            ToolSpec(
                "lookup", "Read an order", {"order": {"type": "integer", "minimum": 1}}, ("order",)
            ),
        ),
    )


def tool_reply(identifier="lookup-1", *, arguments=None, name="lookup"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"order": 17} if arguments is None else arguments),
                },
            }
        ],
    }


def create(journal, *, user=True, **limits):
    return journal.create("s", prompt(user=user), values={"name": "Ada"}, **limits)


def reserve_reply(journal, reply):
    state = journal.read("s")
    request = journal.reserve_provider("s", expected_revision=state.revision)
    return journal.complete_provider(
        "s", request.request_id, reply, expected_revision=state.revision + 1
    )


def test_complete_tool_feedback_conversation_survives_restart_and_replays(tmp_path):
    database = tmp_path / "session.sqlite"
    observed = []
    with SessionJournal(database) as journal:
        initial = create(journal, user=False)
        assert initial.status == "waiting_user"
        ready = journal.add_user("s", "Where is order 17?", expected_revision=initial.revision)
        assert ready.status == "ready_provider"

        def first(request):
            observed.append(request.to_dict())
            return {"choices": [{"message": tool_reply()}], "usage": {"total_tokens": 23}}

        state = SessionRunner(journal).run("s", first)
        assert state.status == "ready_tool"
        assert state.turns == 1
        assert observed[0]["messages"][-1] == {"role": "user", "content": "Where is order 17?"}
        assert observed[0]["tools"][0]["function"]["parameters"]["required"] == ["order"]
        original_events = journal.events("s")
    called = []
    with SessionJournal(database) as journal:
        assert journal.read("s").digest == state.digest

        def lookup(arguments):
            called.append(arguments)
            return {"status": "shipped"}

        def answer(request):
            observed.append(request.to_dict())
            assert request.messages[-1]["tool_call_id"] == "lookup-1"
            assert json.loads(request.messages[-1]["content"]) == {"status": "shipped"}
            assert request.messages[-2]["tool_calls"][0]["id"] == "lookup-1"
            return {"role": "assistant", "content": "Your order shipped."}

        state = SessionRunner(journal).run("s", answer, handlers={"lookup": lookup})
        assert state.status == "waiting_user"
        assert called == [{"order": 17}]
        assert journal.events("s")[: len(original_events)] == original_events
        journal.add_user("s", "Thank you", expected_revision=state.revision)
        state = SessionRunner(journal).run("s", lambda request: {"content": "You're welcome."})
        assert state.turns == 3
        finished = journal.finish("s", expected_revision=state.revision)
        assert finished.status == "finished"
        events = journal.events("s")
        assert replay_session(events, expected_digest=finished.digest) == finished
        assert (
            replay_session(tuple(SessionEvent.from_dict(e.to_dict()) for e in events)) == finished
        )
    # A second journal reproduces the exact request sequence without network calls.
    with SessionJournal(tmp_path / "replay.sqlite") as journal:
        create(journal, user=False)
        journal.add_user("s", "Where is order 17?", expected_revision=1)
        replay = SessionReplayProvider(events)
        state = SessionRunner(journal).run(
            "s", replay, handlers={"lookup": lambda _: {"status": "shipped"}}
        )
        journal.add_user("s", "Thank you", expected_revision=state.revision)
        state = SessionRunner(journal).run("s", replay)
        assert journal.finish("s", expected_revision=state.revision) == finished


def test_tool_crash_reserves_effect_and_requires_explicit_recovery(tmp_path):
    path = tmp_path / "journal.sqlite"
    calls = []
    with SessionJournal(path) as journal:
        create(journal)

        def failure(arguments):
            calls.append(arguments)
            raise RuntimeError("secret must not be stored")

        view = SessionRunner(journal).run("s", lambda _: tool_reply(), handlers={"lookup": failure})
        assert view.status == "pending_tool"
        assert view.pending["error_type"] == "RuntimeError"
        assert "secret" not in _json([e.to_dict() for e in journal.events("s")])
    with SessionJournal(path) as journal:
        assert (
            SessionRunner(journal).run(
                "s", lambda _: pytest.fail("provider invoked"), handlers={"lookup": failure}
            )
            == view
        )
        assert len(calls) == 1
        recovered = journal.complete_tool(
            "s", "lookup-1", {"status": "verified separately"}, expected_revision=view.revision
        )
        assert recovered.status == "ready_provider"
        done = SessionRunner(journal).run("s", lambda _: {"content": "Done"})
        assert done.status == "waiting_user"


def test_provider_crash_and_invalid_response_require_explicit_recovery(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)

        def crash(_):
            raise TimeoutError("private provider body")

        view = SessionRunner(journal).run("s", crash)
        assert view.status == "pending_provider"
        assert view.pending["error_type"] == "TimeoutError"
        assert SessionRunner(journal).run("s", crash) == view
        recovered = journal.complete_provider(
            "s",
            view.pending["request_id"],
            {"content": "Recovered"},
            expected_revision=view.revision,
        )
        assert recovered.status == "waiting_user"
        journal.add_user("s", "Again", expected_revision=recovered.revision)
        invalid = SessionRunner(journal).run("s", lambda _: {"role": "user", "content": "wrong"})
        assert invalid.pending["error_type"] == "ValueError"
        assert (
            journal.finish("s", "abandoned", expected_revision=invalid.revision).reason
            == "abandoned"
        )


def test_process_interruption_between_reservation_and_result_does_not_retry(tmp_path):
    path = tmp_path / "journal.db"
    with SessionJournal(path) as journal:
        create(journal)
        request = journal.reserve_provider("s", expected_revision=1)
    with SessionJournal(path) as journal:
        state = SessionRunner(journal).run(
            "s", lambda _: pytest.fail("must not repeat pending request")
        )
        assert state.pending["request_id"] == request.request_id
        journal.complete_provider(
            "s", request.request_id, tool_reply(), expected_revision=state.revision
        )
        call = journal.reserve_tool("s", expected_revision=state.revision + 1)
        assert call == {"call_id": "lookup-1", "name": "lookup", "arguments": {"order": 17}}
    with SessionJournal(path) as journal:
        state = SessionRunner(journal).run(
            "s",
            lambda _: pytest.fail("pending"),
            handlers={"lookup": lambda _: pytest.fail("pending")},
        )
        assert state.status == "pending_tool"
        with pytest.raises(SessionTransitionError):
            journal.add_user("s", "new", expected_revision=state.revision)


def test_stale_revision_and_second_worker_do_not_duplicate_side_effects(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as first, SessionJournal(path) as second:
        view = create(first)
        with pytest.raises(SessionConflict):
            create(second)
        request = first.reserve_provider("s", expected_revision=view.revision)
        with pytest.raises(SessionConflict):
            second.reserve_provider("s", expected_revision=view.revision)
        assert second.read("s").revision == view.revision + 1
        with pytest.raises(SessionTransitionError):
            second.complete_provider("s", "other", {"content": "wrong"}, expected_revision=2)
        first.complete_provider("s", request.request_id, {"content": "ok"}, expected_revision=2)
        with pytest.raises(SessionConflict):
            second.complete_provider(
                "s", request.request_id, {"content": "duplicate"}, expected_revision=2
            )


def test_multiple_tool_results_keep_declared_order_and_stop_at_turn_limit(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal, max_turns=2)
        first = tool_reply("a")
        first["tool_calls"] += tool_reply("b", arguments={"order": 18})["tool_calls"]
        called = []

        def provider(request):
            if request.messages[-1]["role"] == "user":
                return first
            assert [message["tool_call_id"] for message in request.messages[-2:]] == ["a", "b"]
            return tool_reply("never-run")

        state = SessionRunner(journal).run(
            "s", provider, handlers={"lookup": lambda args: called.append(args["order"])}
        )
        assert state.status == "finished" and state.reason == "turn_limit"
        assert state.turns == 2 and called == [17, 18]
        assert state.messages[-1]["tool_calls"][0]["id"] == "never-run"
        with pytest.raises(SessionTransitionError):
            journal.add_user("s", "continue", expected_revision=state.revision)


@pytest.mark.parametrize(
    "response",
    [
        None,
        "text",
        [],
        {},
        {"role": "tool", "content": "x"},
        {"content": 123},
        {"content": " "},
        {"content": []},
        {"choices": []},
        {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
        {"choices": [None]},
        {"choices": [{}]},
        {"content": "x", "tool_calls": {}},
        {"tool_calls": [None]},
        {"tool_calls": [{"id": "x", "type": "function"}]},
        tool_reply(name="unknown"),
        tool_reply(arguments={}),
        tool_reply(arguments={"order": 0}),
        tool_reply(arguments={"order": "17"}),
        tool_reply(arguments={"order": 17, "extra": 1}),
        tool_reply(arguments=[]),
    ],
)
def test_invalid_response_is_atomic_before_any_tools_run(tmp_path, response):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        request = journal.reserve_provider("s", expected_revision=1)
        pending = journal.read("s")
        with pytest.raises((TypeError, ValueError)):
            journal.complete_provider("s", request.request_id, response, expected_revision=2)
        assert journal.read("s") == pending


@pytest.mark.parametrize(
    "change",
    ["duplicate", "empty_id", "bad_type", "bad_function", "bad_json", "duplicate_json_key", "nan"],
)
def test_malformed_function_calls_cannot_bypass_contracts(tmp_path, change):
    reply = tool_reply()
    call = reply["tool_calls"][0]
    if change == "duplicate":
        reply["tool_calls"].append(call.copy())
    elif change == "empty_id":
        call["id"] = " "
    elif change == "bad_type":
        call["type"] = "shell"
    elif change == "bad_function":
        call["function"] = []
    elif change == "bad_json":
        call["function"]["arguments"] = "{"
    elif change == "duplicate_json_key":
        call["function"]["arguments"] = '{"order":17,"order":18}'
    elif change == "nan":
        call["function"]["arguments"] = '{"order":NaN}'
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        state = SessionRunner(journal).run(
            "s", lambda _: reply, handlers={"lookup": lambda _: pytest.fail("must not dispatch")}
        )
        assert state.status == "pending_provider"


def test_call_ids_cannot_be_reused_across_assistant_turns(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        state = SessionRunner(journal).run(
            "s", lambda _: tool_reply(), handlers={"lookup": lambda _: "ok"}
        )
        assert state.status == "pending_provider"
        assert state.turns == 1
        assert len([event for event in journal.events("s") if event.kind == "tool_result"]) == 1


@pytest.mark.parametrize(
    "limit", ["max_turns", "max_tool_calls", "max_event_bytes", "max_context_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_invalid_limits_cannot_create_session(tmp_path, limit, value):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        with pytest.raises(ValueError, match="positive integer"):
            create(journal, **{limit: value})
        assert journal.events("s") == ()


def test_input_prompt_contract_and_resource_limits(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        with pytest.raises(ValueError):
            journal.create("s", prompt())  # missing render value
        with pytest.raises(ValueError):
            journal.create("s", PromptDocument("empty", ()))
        with pytest.raises(ValueError):
            journal.create("s", PromptDocument("tool", (Message("tool", "hi"),)))
        with pytest.raises(ValueError, match="max_event_bytes"):
            create(journal, max_event_bytes=2)
        create(journal, max_context_bytes=10)
        with pytest.raises(ValueError, match="max_context_bytes"):
            journal.reserve_provider("s", expected_revision=1)
        assert journal.read("s").revision == 1
    with SessionJournal(tmp_path / "limits.db") as journal:
        create(journal, max_tool_calls=1, max_event_bytes=1000)
        reply = tool_reply("a")
        reply["tool_calls"] += tool_reply("b")["tool_calls"]
        state = SessionRunner(journal).run("s", lambda _: reply)
        assert state.status == "pending_provider"
        with pytest.raises(ValueError, match="max_event_bytes"):
            journal.complete_provider(
                "s",
                state.pending["request_id"],
                {"content": "x" * 1001},
                expected_revision=state.revision,
            )


def test_invalid_tool_result_retains_reservation(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        state = SessionRunner(journal).run(
            "s", lambda _: tool_reply(), handlers={"lookup": lambda _: float("nan")}
        )
        assert state.status == "pending_tool"
        with pytest.raises(SessionTransitionError):
            journal.complete_tool("s", "different", "ok", expected_revision=state.revision)
        with pytest.raises(TypeError):
            journal.complete_tool(
                "s", "lookup-1", {1: "bad JSON key"}, expected_revision=state.revision
            )
        assert journal.read("s") == state


def test_session_wire_schema_expands_parameter_local_references(tmp_path):
    from promptwitness import resolve_local_refs

    document = PromptDocument(
        "search",
        (Message("user", "Search"),),
        tools=(
            ToolSpec(
                "search",
                "Search",
                {
                    "query": {
                        "type": "string",
                        "$ref": "#/$defs/query",
                        "$defs": {"query": {"type": "string", "minLength": 2}},
                    }
                },
                ("query",),
            ),
        ),
    )
    with SessionJournal(tmp_path / "sessions.db") as journal:
        journal.create("s", document)
        request = journal.reserve_provider("s", expected_revision=1)
        wire = request.to_dict()["tools"][0]["function"]["parameters"]
        assert resolve_local_refs(wire)["properties"]["query"]["minLength"] == 2


def test_provider_model_identity_is_pinned_before_execution_and_replay(tmp_path):
    invoked = []

    class OfflineTransport(OpenAICompatibleProvider):
        def complete(self, messages, *, tools=()):
            invoked.append(self.model)
            return {"content": self.model}

    provider = OpenAISessionProvider(
        OfflineTransport("http://localhost/chat?private=secret", model="A")
    )
    with SessionJournal(tmp_path / "journal.db") as journal:
        create(journal)
        request = journal.reserve_provider("s", expected_revision=1, identity=provider.identity)
        assert request.to_dict()["provider_identity"]["model"] == "A"
        provider.provider.model = "B"
        with pytest.raises(SessionTransitionError, match="settings changed"):
            provider(request)
        assert invoked == []
        provider.provider.model = "A"
        view = journal.complete_provider(
            "s", request.request_id, provider(request), expected_revision=2
        )
        assert invoked == ["A"]
        replay = SessionReplayProvider(journal.events("s"))
        assert replay(request) == {"content": "A"}
        modified = replace(request, provider_identity={**provider.identity, "model": "B"})
        assert request.digest != modified.digest
        with pytest.raises(ValueError, match="exact session request"):
            replay(modified)
        journal.add_user("s", "next", expected_revision=view.revision)
        provider.provider.model = "B"
        with pytest.raises(SessionTransitionError, match="pinned"):
            SessionRunner(journal).run("s", provider)
        assert invoked == ["A"]
        assert journal.read("s").revision == 4
        assert "secret" not in _json([event.to_dict() for event in journal.events("s")])
        with pytest.raises(ValueError, match="explicit identity"):
            Runner(journal).run("s", provider, identity={"provider": "claimed-other"})


def test_raw_provider_requires_explicit_identity_and_invalid_identity_does_not_reserve(tmp_path):
    with SessionJournal(tmp_path / "journal.db") as journal:
        create(journal)
        for identity in (None, {}, {"provider": ""}, {"version": 1}, []):
            with pytest.raises(ValueError, match="identity"):
                Runner(journal).run("s", lambda _: {"content": "test"}, identity=identity)
        assert journal.read("s").revision == 1
        state = Runner(journal).run(
            "s", lambda _: {"content": "test"}, identity={"provider": "named-local", "version": 1}
        )
        journal.add_user("s", "Again", expected_revision=state.revision)
        with pytest.raises(SessionTransitionError, match="pinned"):
            Runner(journal).run(
                "s",
                lambda _: pytest.fail("must not run"),
                identity={"provider": "named-local", "version": True},
            )


def test_changed_provider_cannot_execute_a_pending_tool_before_rejection(tmp_path):
    with SessionJournal(tmp_path / "journal.db") as journal:
        create(journal)
        view = SessionRunner(journal).run("s", lambda _: tool_reply())
        assert view.status == "ready_tool"
        effects = []
        with pytest.raises(SessionTransitionError, match="pinned"):
            Runner(journal).run(
                "s",
                lambda _: {"content": "other provider"},
                identity={"provider": "changed"},
                handlers={"lookup": lambda args: effects.append(args)},
            )
        assert effects == []
        assert journal.read("s") == view


def test_session_contract_rejects_frozen_ref_and_boolean_enum_bypasses_before_effects(tmp_path):
    schema = {
        "n": {
            "type": "integer",
            "allOf": [{"$ref": "#/$defs/positive"}],
            "$defs": {"positive": {"minimum": 1}},
        },
        "flag": {"type": "boolean", "enum": [1]},
    }
    document = PromptDocument(
        "strict", (Message("user", "Run"),), tools=(ToolSpec("lookup", "Validate", schema),)
    )
    with SessionJournal(tmp_path / "journal.db") as journal:
        journal.create("s", document)
        state = SessionRunner(journal).run(
            "s",
            lambda _: tool_reply(arguments={"n": 0, "flag": True}),
            handlers={"lookup": lambda _: pytest.fail("contract bypass executed")},
        )
        assert state.status == "pending_provider"
        assert all(event.kind != "tool_started" for event in journal.events("s"))


def test_cli_rejects_database_hardlink_alias_and_corruption(tmp_path, capsys):
    database = tmp_path / "session.db"
    with SessionJournal(database) as journal:
        create(journal)
    alias = tmp_path / "alias.json"
    alias.hardlink_to(database)
    before = database.read_bytes()
    assert main(["session", "export", str(database), "s", str(alias)]) == 1
    assert "distinct" in capsys.readouterr().err
    assert database.read_bytes() == before
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite database")
    assert main(["session", "show", str(bad), "s"]) == 1
    assert "database" in capsys.readouterr().err


def test_replay_rejects_tampering_reordering_foreign_events_and_missing_head(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        reserve_reply(journal, {"content": "done"})
        events = journal.events("s")
    with pytest.raises(ValueError, match="checksum"):
        SessionEvent.from_dict({**events[-1].to_dict(), "kind": "finished"})
    with pytest.raises(ValueError, match="fields"):
        SessionEvent.from_dict({})
    with pytest.raises(ValueError, match="discontinuous"):
        replay_session(events[::-1])
    with pytest.raises(ValueError, match="head checksum"):
        replay_session(events[:-1], expected_digest=events[-1].digest)
    with pytest.raises(ValueError, match="empty"):
        replay_session(())
    with pytest.raises(TypeError):
        replay_session((events[0], {}))
    with pytest.raises(ValueError, match="session_id changed"):
        raw = {**events[1].to_dict(include_digest=False), "session_id": "different"}
        event = SessionEvent.from_dict(
            {**raw, "digest": hashlib.sha256(_json(raw).encode()).hexdigest()}
        )
        replay_session((events[0], event))
    raw = {**events[0].to_dict(include_digest=False), "session_id": "different"}
    event = SessionEvent.from_dict(
        {**raw, "digest": hashlib.sha256(_json(raw).encode()).hexdigest()}
    )
    with pytest.raises(ValueError, match="creation session_id"):
        replay_session((event,))


def test_database_prohibits_event_mutation_and_verifies_anchored_head(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        create(journal)
        with sqlite3.connect(path) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute("DELETE FROM session_events")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute("UPDATE session_events SET event = '{}' ")
            connection.execute("UPDATE session_heads SET revision = 12")
        with pytest.raises(ValueError, match="head revision"):
            journal.read("s")
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE session_heads SET revision = 1, digest = 'bad'")
        with pytest.raises(ValueError, match="head checksum"):
            journal.read("s")


def test_session_views_are_deeply_immutable_and_unknown_replay_fails(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        create(journal)
        request = journal.reserve_provider("s", expected_revision=1)
        with pytest.raises(TypeError):
            request.tools[0]["function"]["name"] = "mutated"
        with pytest.raises(TypeError):
            journal.read("s").pending["request_id"] = "mutated"
        journal.complete_provider("s", request.request_id, {"content": "done"}, expected_revision=2)
        provider = SessionReplayProvider(journal.events("s"))
        first = provider(request)
        first["content"] = "mutated"
        assert provider(request)["content"] == "done"
        with pytest.raises(ValueError, match="exact session request"):
            provider(replace(request, request_id="different"))
        with pytest.raises(ValueError, match="unknown session"):
            journal.read("missing")
        with pytest.raises(ValueError):
            journal.events("")
        for revision in (True, -1, "1"):
            with pytest.raises(ValueError, match="expected_revision"):
                journal.finish("s", expected_revision=revision)
        with pytest.raises(TypeError):
            SessionRunner(journal).run("s", None)
        with pytest.raises(TypeError):
            SessionRunner(journal).run("s", lambda _: None, handlers={"x": 1})
        done = journal.finish("s", expected_revision=3)
        with pytest.raises(SessionTransitionError):
            journal.finish("s", expected_revision=done.revision)


def test_openai_session_transport_sends_tool_contract_and_correlated_history(tmp_path, capsys):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = (
                tool_reply() if len(received) == 1 else {"role": "assistant", "content": "shipped"}
            )
            encoded = json.dumps({"choices": [{"message": reply}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleProvider(
            f"http://127.0.0.1:{server.server_port}/chat", model="local-test"
        )
        with SessionJournal(tmp_path / "sessions.db") as journal:
            create(journal)
            view = SessionRunner(journal).run(
                "s", OpenAISessionProvider(provider), handlers={"lookup": lambda _: "shipped"}
            )
            assert view.status == "waiting_user"
        assert received[0]["tools"][0]["function"]["name"] == "lookup"
        assert received[1]["messages"][-1]["tool_call_id"] == "lookup-1"
        assert (
            received[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == '{"order":17}'
        )
        assert received[0]["model"] == "local-test"
        other = tmp_path / "cli.db"
        with SessionJournal(other) as journal:
            create(journal)
        assert (
            main(
                [
                    "session",
                    "run",
                    str(other),
                    "s",
                    "--endpoint",
                    provider.endpoint,
                    "--model",
                    "cli-test",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["provider_identity"]["model"] == "cli-test"
        with pytest.raises(ValueError, match="messages"):
            provider.complete([])
        with pytest.raises(ValueError, match="tools"):
            provider.complete([{"role": "user", "content": "hi"}], tools=[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cli_complete_offline_tool_conversation_export_and_replay(tmp_path, capsys):
    from promptwitness import prompt_to_dict

    path = tmp_path / "sessions.db"
    document = tmp_path / "prompt.json"
    document.write_text(json.dumps(prompt_to_dict(prompt(user=False))))
    values = tmp_path / "values.json"
    values.write_text('{"name":"Ada"}')
    message = tmp_path / "user.txt"
    message.write_text("Where is order 17?")
    response = tmp_path / "response.json"
    response.write_text(json.dumps(tool_reply()))
    result = tmp_path / "result.json"
    result.write_text('{"status":"shipped"}')

    def cli(*parts):
        code = main(["session", *[str(part) for part in parts]])
        captured = capsys.readouterr()
        return code, json.loads(captured.out) if captured.out else captured.err

    assert cli("create", path, "s", document, "--values", values)[0] == 0
    code, state = cli("user", path, "s", message, "--revision", 1)
    assert code == 0 and state["revision"] == 2
    code, state = cli("respond", path, "s", response, "--revision", 2)
    assert code == 0 and state["status"] == "ready_tool"
    assert cli("tool-result", path, "s", "wrong", result, "--revision", 4)[0] == 1
    code, state = cli("tool-result", path, "s", "lookup-1", result, "--revision", 4)
    assert code == 0 and state["revision"] == 6
    response.write_text('{"content":"Your order shipped."}')
    assert cli("respond", path, "s", response, "--revision", 6)[0] == 0
    code, state = cli("show", path, "s", "--messages")
    assert code == 0 and state["messages"][-1]["content"] == "Your order shipped."
    exported = tmp_path / "events.json"
    assert cli("export", path, "s", exported)[0] == 0
    assert cli("export", path, "s", exported)[0] == 1  # exclusive creation
    code, replayed = cli("replay", exported, "--messages")
    assert code == 0 and replayed == state
    other = tmp_path / "replay.db"
    cli("create", other, "s", document, "--values", values)
    cli("user", other, "s", message, "--revision", 1)
    assert cli("run", other, "s", "--replay-events", exported)[1]["status"] == "ready_tool"
    cli("tool-result", other, "s", "lookup-1", result, "--revision", 4)
    assert cli("run", other, "s", "--replay-events", exported)[1]["digest"] == state["digest"]
    assert cli("finish", path, "s", "--revision", 8)[1]["status"] == "finished"
    assert cli("respond", path, "s", response, "--revision", 9)[0] == 1
    assert cli("show", tmp_path / "missing.db", "s")[0] == 1
    assert not (tmp_path / "missing.db").exists()
    assert cli("export", path, "s", path)[0] == 1
    assert cli("user", path, "s", message, "--revision", 1)[0] == 1
    assert cli("respond", path, "s", response, "--revision", 1)[0] == 1
    assert cli("tool-result", path, "s", "x", result, "--revision", 1)[0] == 1


def test_cli_recovery_pending_provider_and_tool(tmp_path, capsys):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        create(journal)
        journal.reserve_provider("s", expected_revision=1)
    response = tmp_path / "response.json"
    response.write_text(json.dumps(tool_reply()))
    assert main(["session", "show", str(path), "s"]) == 2
    capsys.readouterr()
    assert main(["session", "respond", str(path), "s", str(response), "--revision", "2"]) == 0
    capsys.readouterr()
    with SessionJournal(path) as journal:
        journal.reserve_tool("s", expected_revision=3)
    result = tmp_path / "result.json"
    result.write_text("17")
    assert (
        main(["session", "tool-result", str(path), "s", "lookup-1", str(result), "--revision", "4"])
        == 0
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"format": 1, "head_digest": "x", "events": []},
        {"format": "promptwitness.session/v1", "head_digest": 1, "events": []},
        {"format": "promptwitness.session/v1", "head_digest": "x", "events": {}},
    ],
)
def test_cli_replay_rejects_ambiguous_exports(tmp_path, capsys, payload):
    path = tmp_path / "events.json"
    path.write_text(json.dumps(payload))
    assert main(["session", "replay", str(path)]) == 1
    assert "promptwitness:" in capsys.readouterr().err
