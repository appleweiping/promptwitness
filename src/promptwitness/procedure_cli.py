"""Explicit local procedure planning, execution and durable recovery commands."""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .procedure_json_data import canonical_bytes, integer, load_json, sequence, string
from .procedure_plan import ProcedureSuitePlan, load_procedure_suite
from .providers import OpenAICompatibleProvider
from .task_runs import OpenAITaskProvider, TaskRunStore
from .task_scores import response_text

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PREDICTION_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
REPORT_FORMAT = "promptwitness.procedure-report/v1"
_MUTATIONS = {"run", "retry", "respond"}


def add_procedure_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "procedure-suite", help="plan and run six-family local procedure suites"
    )
    commands = parser.add_subparsers(dest="procedure_command", required=True)
    for name in ("plan", "run", "show", "retry", "respond"):
        command = commands.add_parser(name)
        command.set_defaults(handler=run_procedure_command)
        command.add_argument(
            "--include-inputs",
            action="store_true",
            help="include private plan inputs/gold, or saved predictions for run reports",
        )
        if name == "plan":
            command.add_argument("suite", type=Path)
            continue
        command.add_argument("database", type=Path)
        if name == "run":
            command.add_argument("suite", type=Path)
            command.add_argument("--endpoint", required=True)
            command.add_argument("--model", required=True)
            command.add_argument("--api-key-env")
            command.add_argument("--timeout", type=float, default=60.0)
            command.add_argument("--max-cases", type=int)
        elif name in ("retry", "respond"):
            command.add_argument("item_id")
            command.add_argument("--revision", type=int, required=True)
            if name == "respond":
                command.add_argument("response", type=Path)


def _same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def _database_paths(database: Path) -> tuple[Path, ...]:
    return tuple(
        Path(str(base) + suffix)
        for base in (database.absolute(), database.resolve())
        for suffix in ("", "-wal", "-shm", "-journal")
    )


def _distinct(database: Path, source: Path) -> None:
    if any(_same_file(path, source) for path in _database_paths(database)):
        raise ValueError("procedure database aliases a protected input")


def _outside_root(database: Path, root: Path) -> None:
    if not root.is_dir():
        raise ValueError("procedure data root is unavailable")
    for path in _database_paths(database):
        if path.absolute().is_relative_to(root.absolute()) or path.resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError("procedure database must be outside the LongProc source directory")


def _read(path: Path, maximum: int) -> bytes:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("procedure input must be a regular local file")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("procedure input is not an admitted regular file")
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("procedure input exceeds byte limit")
    return raw


def _plan(path: Path, database: Path | None = None) -> ProcedureSuitePlan:
    if database is not None:
        _distinct(database, path)
    raw = _read(path, 1024 * 1024)
    manifest = load_json(raw, limit=1024 * 1024)
    if not isinstance(manifest, Mapping):
        raise ValueError("procedure manifest must be an object")
    tasks = sequence(manifest.get("tasks"), minimum=1, maximum=256)
    directory = path.resolve().parent
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("procedure task must be an object")
        source = directory / string(task.get("path"))
        if database is not None:
            if task.get("adapter") == "longproc":
                _outside_root(database, source)
            else:
                _distinct(database, source)
    plan = load_procedure_suite(path)
    if plan.payload["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("procedure manifest changed during validation")
    # Inventory comes from the validated loader, not arbitrary manifest paths.
    # This catches external hard links to referenced LongProc files as well.
    if database is not None:
        by_id = {task["id"]: task for task in plan.payload["tasks"]}
        for task in tasks:
            if task["adapter"] == "longproc":
                root = directory / task["path"]
                _outside_root(database, root)
                for source in by_id[task["id"]]["source"]["files"]:
                    _distinct(database, root / source["path"])
    return plan


def _plan_view(plan: ProcedureSuitePlan, *, content: bool) -> dict[str, Any]:
    payload = plan.to_dict()
    if not content:
        payload["items"] = [
            {key: value for key, value in item.items() if key not in {"messages", "record"}}
            for item in payload["items"]
        ]
    return {
        "format": "promptwitness.procedure-plan-view/v1",
        "private_data": True,
        "includes_content": content,
        "plan_digest": plan.digest,
        "plan": payload,
    }


def _report_view(report: dict[str, Any], *, content: bool) -> dict[str, Any]:
    if report.get("format") != REPORT_FORMAT:
        raise ValueError("database is not bound to a procedure suite")
    if content:
        return {"private_data": True, "includes_content": True, **report}
    result = dict(report)
    result["results"] = [
        {
            **row,
            "result": (
                {
                    key: value
                    for key, value in row["result"].items()
                    if key not in {"prediction", "score"}
                }
                | {
                    "score": {
                        key: value
                        for key, value in row["result"]["score"].items()
                        if key != "processed"
                    }
                }
                if row["result"] is not None
                else None
            ),
        }
        for row in report["results"]
    ]
    return {"private_data": True, "includes_content": False, **result}


def _status(report: Mapping[str, Any]) -> int:
    coverage = report["coverage"]
    return 0 if coverage["execution_complete"] and coverage["scoring_complete"] else 2


def _existing_database(database: Path) -> None:
    if not database.is_file() or database.stat().st_size == 0:
        raise ValueError("procedure database must already exist and be nonempty")
    # TaskRunStore initializes pristine SQLite databases. Recovery/read commands
    # must first admit an already bound task database using a read-only connection.
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != 1347900499
            or connection.execute("PRAGMA user_version").fetchone()[0] != 1
            or connection.execute("SELECT 1 FROM task_run WHERE id=1").fetchone() is None
        ):
            raise ValueError("procedure database is not an existing bound task store")
    finally:
        connection.close()


def _execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    action = args.procedure_command
    if action == "plan":
        plan = _plan(args.suite)
        return _plan_view(plan, content=args.include_inputs), 0
    if action == "run":
        if args.max_cases is not None:
            integer(args.max_cases, minimum=1)
        plan = _plan(args.suite, args.database)
        provider = OpenAITaskProvider(
            OpenAICompatibleProvider(
                args.endpoint, model=args.model, api_key_env=args.api_key_env, timeout=args.timeout
            )
        )
        with TaskRunStore(args.database) as store:
            report = store.run(plan, provider, max_cases=args.max_cases)
    else:
        _existing_database(args.database)
        if action in ("retry", "respond"):
            integer(args.revision)
            string(args.item_id)
        response = None
        if action == "respond":
            _distinct(args.database, args.response)
            response = load_json(_read(args.response, MAX_RESPONSE_BYTES), limit=MAX_RESPONSE_BYTES)
            prediction = response_text(response)
            if (
                len(prediction) > MAX_PREDICTION_BYTES
                or len(prediction.encode("utf-8")) > MAX_PREDICTION_BYTES
            ):
                raise ValueError("procedure prediction exceeds 4 MiB")
        with TaskRunStore(args.database) as store:
            report = store.snapshot()
            if report.get("format") != REPORT_FORMAT:
                raise ValueError("database is not bound to a procedure suite")
            if action == "retry":
                store.retry(args.item_id, expected_revision=args.revision)
            elif action == "respond":
                store.complete(args.item_id, response, expected_revision=args.revision)
            if action in ("retry", "respond"):
                report = store.snapshot()
    return _report_view(report, content=args.include_inputs), _status(report)


def _silence(stream: Any) -> None:
    try:
        descriptor = stream.fileno()
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except (OSError, ValueError, AttributeError):
        return


def _diagnostic(message: str) -> None:
    try:
        if sys.stderr is None:
            return
        sys.stderr.write("promptwitness procedure-suite: " + message + "\n")
        sys.stderr.flush()
    except (OSError, ValueError, AttributeError):
        _silence(sys.stderr)


def _publish(result: Mapping[str, Any]) -> None:
    raw = canonical_bytes(result, limit=MAX_OUTPUT_BYTES, node_limit=8_000_000) + b"\n"
    binary = getattr(sys.stdout, "buffer", None)
    if binary is not None:
        written = binary.write(raw)
        if type(written) is not int or written != len(raw):
            raise OSError("procedure stdout short write")
        binary.flush()
    else:
        text = raw.decode("utf-8")
        written = sys.stdout.write(text)
        if type(written) is not int or written != len(text):
            raise OSError("procedure stdout short write")
        sys.stdout.flush()


def run_procedure_command(args: argparse.Namespace) -> int:
    """Keep input failures private and never turn publication loss into a retry."""
    try:
        result, status = _execute(args)
    except (ValueError, OSError, sqlite3.Error, KeyError):
        suffix = (
            "; inspect stored revisions before retrying; earlier work may have committed"
            if args.procedure_command in _MUTATIONS
            else ""
        )
        _diagnostic("command could not complete" + suffix)
        return 1
    try:
        _publish(result)
    except (ValueError, OSError, AttributeError):
        _silence(sys.stdout)
        _diagnostic(
            "operation completed, but its report could not be delivered; "
            "inspect stored revisions before deciding whether to retry"
        )
        return status if args.procedure_command in _MUTATIONS else 1
    return status
