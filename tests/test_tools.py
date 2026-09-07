from __future__ import annotations

import pytest

from promptwitness import ToolDispatcher


def test_tool_dispatcher_is_ordered_and_returns_provider_messages() -> None:
    dispatcher = ToolDispatcher({"lookup": lambda args: {"value": args["key"]}})
    batch = dispatcher.dispatch(
        {
            "tool_calls": [
                {"id": "one", "name": "lookup", "arguments": '{"key":"Ada"}'},
                {"id": "two", "name": "missing", "arguments": {}},
            ]
        }
    )
    assert not batch.complete
    assert batch.outcomes[0].succeeded
    assert batch.outcomes[1].error == "unknown tool 'missing'"
    assert batch.messages()[0]["content"] == '{"value":"Ada"}'
    assert batch.messages()[1]["content"] == '{"error":"unknown tool \'missing\'"}'


def test_tool_dispatcher_isolates_bad_arguments_and_bounds_results() -> None:
    dispatcher = ToolDispatcher({"echo": lambda args: args}, max_result_bytes=8)
    batch = dispatcher.dispatch(
        {
            "tool_calls": [
                {"id": "bad-json", "name": "echo", "arguments": "{"},
                {"id": "bad-shape", "name": "echo", "arguments": []},
                {"id": "large", "name": "echo", "arguments": {"long": "0123456789"}},
            ]
        }
    )
    assert all(not outcome.succeeded for outcome in batch.outcomes)
    assert "invalid JSON" in (batch.outcomes[0].error or "")
    assert "must be an object" in (batch.outcomes[1].error or "")
    assert "exceeds" in (batch.outcomes[2].error or "")
    with pytest.raises(ValueError, match="exceeds"):
        ToolDispatcher({"echo": lambda _: {}}, max_calls=1).dispatch({"tool_calls": [{}, {}]})
