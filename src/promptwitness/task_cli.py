"""CLI for local task planning, real provider runs, and explicit recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .providers import OpenAICompatibleProvider
from .sessions import _load
from .task_data import load_task_suite
from .task_runs import OpenAITaskProvider, TaskRunStore


def add_task_commands(subparsers: Any) -> None:
    suite = subparsers.add_parser(
        "task-suite", help="plan and run durable long-context task suites"
    )
    commands = suite.add_subparsers(dest="task_command", required=True)
    plan = commands.add_parser(
        "plan", help="validate local inputs and preview budget coverage; no model call"
    )
    plan.add_argument("suite", type=Path)
    plan.add_argument(
        "--include-inputs", action="store_true", help="include full prompts and gold labels"
    )
    plan.set_defaults(handler=run_task_command)
    for name in ("run", "show", "retry", "respond"):
        command = commands.add_parser(name)
        command.add_argument("database", type=Path)
        command.set_defaults(handler=run_task_command)
        if name == "run":
            command.add_argument("suite", type=Path)
            command.add_argument("--endpoint", required=True)
            command.add_argument("--model", required=True)
            command.add_argument("--api-key-env")
            command.add_argument("--timeout", type=float, default=60.0)
            command.add_argument("--max-cases", type=int)
        elif name in {"retry", "respond"}:
            command.add_argument("item_id")
            command.add_argument("--revision", type=int, required=True)
            if name == "respond":
                command.add_argument("response", type=Path)


def _distinct(database: Path, source: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        target = Path(str(database) + suffix)
        if target.resolve() == source.resolve() or (
            target.exists() and source.exists() and target.samefile(source)
        ):
            raise ValueError("task database and SQLite sidecars must not alias an input file")


def run_task_command(args: argparse.Namespace) -> int:
    """Print JSON; exit two means missing invocations or unsupported primary scores."""
    if args.task_command == "plan":
        plan = load_task_suite(args.suite)
        payload = plan.to_dict()
        if not args.include_inputs:
            payload["items"] = [
                {key: value for key, value in item.items() if key not in {"record", "messages"}}
                for item in payload["items"]
            ]
        print(
            json.dumps({"plan_digest": plan.digest, "plan": payload}, ensure_ascii=False, indent=2)
        )
        return 0
    if args.task_command == "run":
        _distinct(args.database, args.suite)
        manifest = _load(args.suite.read_text(encoding="utf-8"))
        plan = load_task_suite(args.suite)
        for task in manifest["tasks"]:
            _distinct(args.database, args.suite.resolve().parent / task["path"])
        provider = OpenAITaskProvider(
            OpenAICompatibleProvider(
                args.endpoint, model=args.model, api_key_env=args.api_key_env, timeout=args.timeout
            )
        )
        with TaskRunStore(args.database) as store:
            result = store.run(plan, provider, max_cases=args.max_cases)
    else:
        if not args.database.is_file():
            raise ValueError("task database does not exist")
        if args.task_command == "respond":
            _distinct(args.database, args.response)
            response = _load(args.response.read_text(encoding="utf-8"))
        with TaskRunStore(args.database) as store:
            if args.task_command == "retry":
                store.retry(args.item_id, expected_revision=args.revision)
            elif args.task_command == "respond":
                store.complete(args.item_id, response, expected_revision=args.revision)
            result = store.snapshot()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return (
        0
        if result["coverage"]["execution_complete"] and result["coverage"]["scoring_complete"]
        else 2
    )
