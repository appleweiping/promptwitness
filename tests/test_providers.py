from __future__ import annotations

from promptwitness import ReplayProvider, Scenario, TraceRecorder, load_prompt, render_matrix


def test_replay_and_trace_recorder_are_digest_stable(tmp_path) -> None:
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    provider = TraceRecorder(lambda _row: {"text": "ok", "tool_calls": [{"name": "lookup"}]})
    assert provider(row)["text"] == "ok"
    assert [event.kind for event in provider.traces[0].events] == [
        "request",
        "tool_call",
        "response",
    ]
    path = tmp_path / "traces.json"
    provider.save(path)
    assert "promptwitness.provider-trace/v1" in path.read_text(encoding="utf-8")
    assert ReplayProvider({row.digest: {"text": "ok"}})(row) == {"text": "ok"}


def test_replay_provider_fails_for_unknown_digest() -> None:
    document = load_prompt("examples/before.json")
    (row,) = render_matrix(document, (Scenario("one", {"customer_name": "Ada", "order_id": "1"}),))
    try:
        ReplayProvider({})(row)
    except KeyError as error:
        assert "no replay response" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing replay response was accepted")
