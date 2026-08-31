"""Deterministic workload benchmark for PromptWitness message alignment."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from statistics import median
from typing import TypeVar

from promptwitness import DiffOptions, Message, MessageAlignment, PromptDocument, compare_prompts

T = TypeVar("T")


def measure(operation: Callable[[], T], repeats: int) -> tuple[float, T]:
    durations: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - started)
    assert result is not None
    return median(durations), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.messages < 1 or args.repeats < 1:
        parser.error("messages and repeats must be positive")

    before_messages = tuple(
        Message("user" if index % 2 == 0 else "assistant", f"message {index}")
        for index in range(args.messages)
    )
    after_messages = list(before_messages)
    after_messages.insert(args.messages // 2, Message("system", "inserted policy"))
    for index in range(10, len(after_messages), 50):
        original = after_messages[index]
        after_messages[index] = Message(original.role, original.content + " revised")
    before = PromptDocument("before", before_messages)
    after = PromptDocument("after", tuple(after_messages))

    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    for alignment in MessageAlignment:
        duration, report = measure(
            lambda mode=alignment: compare_prompts(
                before, after, DiffOptions(message_alignment=mode)
            ),
            args.repeats,
        )
        timings[alignment.value] = duration
        counts[alignment.value] = len(report.changes)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "environment": {
                    "python": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "platform": platform.platform(),
                },
                "workload": {"messages": args.messages, "repeats": args.repeats},
                "median_seconds": timings,
                "change_counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
