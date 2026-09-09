"""Offline tool feedback and process-resume example; no network calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from promptwitness import (
    Message,
    PromptDocument,
    SessionJournal,
    SessionRequest,
    SessionRunner,
    ToolSpec,
    replay_session,
)


def provider(request: SessionRequest) -> dict:
    if request.messages[-1]["role"] == "user":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "order-17",
                    "type": "function",
                    "function": {"name": "lookup_order", "arguments": '{"order":17}'},
                }
            ],
        }
    result = json.loads(request.messages[-1]["content"])
    return {"role": "assistant", "content": f"Your order is {result['status']}."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    if args.database.exists():
        parser.error("choose a new database path for the example")
    document = PromptDocument(
        "order-support",
        (
            Message("system", "Use lookup_order to answer order questions."),
            Message("user", "Where is order 17?"),
        ),
        tools=(
            ToolSpec("lookup_order", "Read an order", {"order": {"type": "integer"}}, ("order",)),
        ),
    )
    with SessionJournal(args.database) as journal:
        journal.create("demo", document, max_turns=4)
        waiting = SessionRunner(journal).run(
            "demo", provider, identity={"provider": "offline-order-demo", "version": 1}
        )
        assert waiting.status == "ready_tool"
    with SessionJournal(args.database) as journal:
        state = SessionRunner(journal).run(
            "demo",
            provider,
            handlers={"lookup_order": lambda _: {"status": "shipped"}},
            identity={"provider": "offline-order-demo", "version": 1},
        )
        assert state.messages[-1]["content"] == "Your order is shipped."
        assert replay_session(journal.events("demo"), expected_digest=state.digest) == state
        print(json.dumps(state.to_dict(include_messages=True), indent=2))


if __name__ == "__main__":
    main()
