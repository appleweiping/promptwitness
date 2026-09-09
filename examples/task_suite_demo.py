"""Offline integration example, not real-data or long-context model validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from promptwitness import TaskRequest, TaskRunStore, load_task_suite


class ExampleProvider:
    """Small deterministic canned responses; never reads the gold labels."""

    def __init__(self) -> None:
        self.identity = {"provider": "checked-in-example", "model": "canned-responses/v1"}
        self.calls = 0

    def __call__(self, request: TaskRequest) -> Any:
        self.calls += 1
        instruction = request.messages[0]["content"]
        if "JSON array of strings" in instruction:
            return '["Ada"]'
        if "Rank all" in instruction:
            return '["d1", "d2"]'
        if "classification" in instruction:
            return "question"
        if "cite" in instruction:
            return "Ada owns the blue file [d1]."
        if "Summarize" in instruction:
            return "Grace took over Ada's blue file while Ada was on vacation."
        if "complete context" in instruction:
            return "Ada went on vacation and needed Grace to handle urgent requests."
        return "Ada"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.database.exists() or args.report.exists():
        raise ValueError("example output files must not already exist")
    if any(
        Path(str(args.database) + suffix).resolve() == args.report.resolve()
        for suffix in ("", "-journal", "-wal", "-shm")
    ):
        raise ValueError("report must not alias the database or its sidecars")
    plan = load_task_suite(Path(__file__).parent / "task_suite" / "suite.json")
    provider = ExampleProvider()
    with TaskRunStore(args.database) as store:
        first = store.run(plan, provider, max_cases=3)
        assert first["coverage"]["succeeded"] == 3
    with TaskRunStore(args.database) as store:
        report = store.run(plan, provider)
        assert report["coverage"]["succeeded"] == 7
        assert report["coverage"]["skipped"] == 7
        assert report["coverage"]["scored"] == 4
        assert not report["seven_family_scoring_complete"]
        assert store.run(plan, provider) == report
        assert provider.calls == 7
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "kind": "checked-in-example",
                "provider_calls": provider.calls,
                "plan_digest": plan.digest,
                "coverage": report["coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
