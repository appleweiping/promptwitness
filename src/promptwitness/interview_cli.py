"""Explicit interview lifecycle commands; no implicit network execution."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from .interview_journal import InterviewJournal, verify_interview_export
from .interview_memory import BM25MemoryIndex, SourceHead
from .interview_models import (
    MAX_CONTRACT_BYTES,
    InterviewContractError,
    InterviewPlan,
    _array,
    _closed,
    _identifier,
    contract_json,
    load_interview_json,
)
from .interview_stages import _provider_identity
from .interview_state import (
    InterviewCommand,
    InterviewState,
    current_memory_snapshot,
    interview_report,
    schedule,
)
from .interviews import InterviewRunner, OpenAIInterviewProvider, create_interview
from .providers import OpenAICompatibleProvider

_INPUTS = ("plan", "roles", "prior_heads", "answer", "query_file", "events", "trusted_head")
_MUTATIONS = {"start", "run", "retry", "answer", "skip", "proposal", "finish"}


def add_interview_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser("interview", help="explicit evidence-linked interviews")
    actions = parser.add_subparsers(dest="interview_command", required=True)
    for name in (
        "start",
        "run",
        "retry",
        "answer",
        "skip",
        "proposal",
        "finish",
        "show",
        "report",
        "recall",
        "export",
    ):
        action = actions.add_parser(name)
        action.add_argument("database", type=Path)
        action.add_argument("interview_id")
        action.set_defaults(handler=run_interview_command)
        if name in ("start", "run", "retry"):
            action.add_argument(
                "--roles",
                type=Path,
                required=True,
                help="explicit nonsecret provider configuration JSON",
            )
        if name in ("run", "retry"):
            action.add_argument("--operation-id", required=True)
        elif name == "start":
            action.add_argument("plan", type=Path)
            action.add_argument("--participant-id", required=True)
            action.add_argument("--command-id", required=True)
            action.add_argument("--prior-heads", type=Path)
        if name in ("answer", "skip", "proposal", "finish"):
            action.add_argument("--revision", type=int, required=True)
            action.add_argument("--digest", required=True)
            action.add_argument("--command-id", required=True)
        if name in ("answer", "skip"):
            action.add_argument("--question-id", required=True)
        if name == "answer":
            action.add_argument("answer", type=Path, help="exact UTF-8 participant answer file")
            action.add_argument("--answer-id", required=True)
        elif name == "skip":
            action.add_argument("--scope", choices=("criterion", "topic"), required=True)
        elif name == "proposal":
            action.add_argument("--proposal-id", required=True)
            action.add_argument("--decision", choices=("accepted", "rejected"), required=True)
        elif name == "finish":
            action.add_argument(
                "--reason",
                choices=(
                    "participant_stopped",
                    "failed",
                    "agenda_completed",
                    "unresolved",
                    "budget_exhausted",
                ),
                required=True,
            )
        elif name == "show":
            action.add_argument(
                "--content",
                action="store_true",
                help="include private transcript, prompts and evidence",
            )
        elif name == "recall":
            action.add_argument("--query-file", type=Path, required=True)
        elif name == "export":
            action.add_argument(
                "output", type=Path, help="new private output file; never overwritten"
            )
    replay = actions.add_parser("replay")
    replay.add_argument("events", type=Path)
    replay.add_argument(
        "--trusted-head", type=Path, required=True, help="independently trusted SourceHead JSON"
    )
    replay.add_argument("--content", action="store_true")
    replay.set_defaults(handler=run_interview_command)


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def _paths(args: argparse.Namespace) -> None:
    inputs = [getattr(args, name) for name in _INPUTS if getattr(args, name, None) is not None]
    database = getattr(args, "database", None)
    protected: list[Path] = []
    if database is not None:
        # Protect both lexical and resolved sidecars, including nonexistent ones.
        for base in (database.absolute(), database.resolve()):
            protected.extend(
                Path(str(base) + suffix) for suffix in ("", "-wal", "-shm", "-journal")
            )
    for index, source in enumerate(inputs):
        if not source.is_file():
            raise InterviewContractError("input must be an existing regular file")
        if any(_same_file(source, other) for other in (*protected, *inputs[:index])):
            raise InterviewContractError(
                "input paths must be independent of database and other inputs"
            )
    output = getattr(args, "output", None)
    if output is not None:
        if any(_same_file(output, source) for source in (*protected, *inputs)):
            raise InterviewContractError("export output aliases a protected file")
        if output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise InterviewContractError("export requires a new file in an existing directory")


def _read(path: Path, *, maximum: int = MAX_CONTRACT_BYTES) -> str:
    with path.open("rb") as source:
        data = source.read(maximum + 1)
    if len(data) > maximum:
        raise InterviewContractError("input exceeds its byte limit")
    try:
        return data.decode("utf-8")
    except UnicodeError:
        raise InterviewContractError("input is not strict UTF-8") from None


def _providers(path: Path) -> dict[str, OpenAIInterviewProvider]:
    data = _closed(
        load_interview_json(_read(path, maximum=131072)),
        {"analyst", "questioner"},
        "provider roles",
    )
    providers = {}
    for role, raw in data.items():
        item = _closed(
            raw,
            {"endpoint", "model", "api_key_env", "timeout", "headers"},
            "provider configuration",
        )
        _identifier(item["model"], "model")
        headers = item["headers"]
        if not isinstance(headers, dict) or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or any(
                marker in name.casefold()
                for marker in ("authorization", "cookie", "api-key", "apikey", "token", "secret")
            )
            for name, value in headers.items()
        ):
            raise InterviewContractError(
                "headers require string values and noncredential names; use api_key_env"
            )
        if item["api_key_env"] is not None and (
            not isinstance(item["api_key_env"], str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", item["api_key_env"]) is None
        ):
            raise InterviewContractError("invalid credential environment-variable name")
        provider = OpenAIInterviewProvider(OpenAICompatibleProvider(**item))
        _provider_identity(provider.identity)
        providers[role] = provider
    return providers


def _summary(state: InterviewState) -> dict[str, Any]:
    action = schedule(state)
    question = next(
        (item for item in state.questions if item.question_id == action.question_id), None
    )
    return {
        "format": "promptwitness.interview-cli/v1",
        "private_data": True,
        "interview_id": state.interview_id,
        "participant_id": state.participant_id,
        "head": state.head.to_dict(),
        "next_action": action.to_dict(),
        "finished_reason": state.finished_reason,
        "question": question.to_dict() if question else None,
        "participant_turns": state.participant_turns,
        "stage_reservations": state.stage_reservations,
        "stage_completions": state.stage_completions,
        "error_code": state.pending.error_code if state.pending else None,
        "pending_proposals": [
            {"proposal_id": item.proposal.proposal_id, "topic": item.proposal.topic.to_dict()}
            for item in state.proposals
            if item.status == "pending"
        ],
    }


def _write_export(path: Path, value: Any) -> None:
    encoded = contract_json(value).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".interview-export-", dir=path.parent)
    source = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        # A hard-link installation is atomic and fails if the requested name
        # already exists on Windows and POSIX. Never replace a concurrent file.
        os.link(source, path)
    finally:
        source.unlink(missing_ok=True)


def _execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _paths(args)
    action = args.interview_command
    if action == "replay":
        head = SourceHead.from_dict(load_interview_json(_read(args.trusted_head, maximum=16384)))
        state = verify_interview_export(_read(args.events), expected_head=head)
        result = (
            {"private_data": True, "state": state.to_dict()} if args.content else _summary(state)
        )
        return result, 0
    providers = _providers(args.roles) if action in ("start", "run", "retry") else None
    # Validate all creation inputs before allowing database initialization.
    plan = InterviewPlan.from_json(_read(args.plan)) if action == "start" else None
    if plan is not None and (
        plan.analyst_prompt_version != "analysis/v1"
        or plan.questioner_prompt_version != "question/v1"
    ):
        raise InterviewContractError("plan requires the supported interview prompt versions")
    heads = (
        tuple(
            SourceHead.from_dict(item)
            for item in _array(
                load_interview_json(_read(args.prior_heads)), "prior heads", maximum=127
            )
        )
        if getattr(args, "prior_heads", None)
        else ()
    )
    text = _read(args.answer, maximum=1024 * 1024) if action == "answer" else None
    query = _read(args.query_file, maximum=1024 * 1024) if action == "recall" else None
    if action == "start":
        _identifier(args.interview_id, "interview ID")
        _identifier(args.participant_id, "participant ID")
        _identifier(args.command_id, "command ID")
    with InterviewJournal(args.database, create=action == "start") as journal:
        if action == "start":
            assert providers is not None and plan is not None
            state = create_interview(
                journal,
                args.interview_id,
                args.participant_id,
                plan,
                command_id=args.command_id,
                analyst_identity=providers["analyst"].identity,
                questioner_identity=providers["questioner"].identity,
                prior_heads=heads,
            )
        elif action in ("run", "retry"):
            assert providers is not None
            runner = InterviewRunner(
                journal, analyst=providers["analyst"], questioner=providers["questioner"]
            )
            state = (runner.run_until_input if action == "run" else runner.retry_pending)(
                args.interview_id, operation_id=args.operation_id
            )
        elif action in ("answer", "skip", "proposal", "finish"):
            if action == "answer":
                kind, payload = (
                    "answer_committed",
                    {"question_id": args.question_id, "answer_id": args.answer_id, "text": text},
                )
            elif action == "skip":
                kind, payload = (
                    "question_skipped",
                    {"question_id": args.question_id, "scope": args.scope},
                )
            elif action == "proposal":
                kind, payload = (
                    "proposal_decided",
                    {"proposal_id": args.proposal_id, "decision": args.decision},
                )
            else:
                kind, payload = "finished", {"reason": args.reason}
            state = journal.execute(
                InterviewCommand(
                    args.interview_id, args.command_id, args.revision, args.digest, kind, payload
                )
            )
        else:
            state = journal.read(args.interview_id)
        if action == "export":
            exported = journal.export(args.interview_id)
            _write_export(args.output, exported)
            return {
                "format": "promptwitness.interview-export-receipt/v1",
                "private_data": True,
                "head": exported["head"],
            }, 0
        if action == "report":
            return {"private_data": True, "report": interview_report(state)}, 0
        if action == "recall":
            assert query is not None
            retrieval = BM25MemoryIndex(
                current_memory_snapshot(state), state.plan.retrieval
            ).search(query)
            return {
                "private_data": True,
                "head": state.head.to_dict(),
                "recall": retrieval.to_dict(),
            }, 0
        if action == "show" and args.content:
            return {"private_data": True, "state": state.to_dict()}, 0
        return _summary(state), 2 if schedule(state).kind in (
            "await_retry",
            "await_completion",
        ) else 0


def _silence_failed_stream(stream: Any) -> None:
    # CPython flushes stdout/stderr again during interpreter shutdown. Redirect
    # the real descriptor after a failed write, preventing a second flush from
    # changing an already completed command to exit 120. In-memory host streams
    # have no descriptor and need no OS-level redirection.
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
        print("promptwitness interview: " + message, file=sys.stderr, flush=True)
    except (OSError, UnicodeError):
        _silence_failed_stream(sys.stderr)


def run_interview_command(args: argparse.Namespace) -> int:
    """Print controlled API views; output failure never undoes a committed write."""
    try:
        result, status = _execute(args)
        content = contract_json(result)
    except (ValueError, OSError, sqlite3.Error, KeyError):
        suffix = (
            "; inspect the current head before retrying, as earlier steps may have committed"
            if args.interview_command in _MUTATIONS
            else ""
        )
        if args.interview_command == "export":
            suffix = (
                "; an export artifact or private temporary file may already exist; "
                "inspect the destination directory before retrying"
            )
        _diagnostic("command could not complete" + suffix)
        return 1
    try:
        binary = getattr(sys.stdout, "buffer", None)
        if binary is not None:
            binary.write((content + "\n").encode("utf-8"))
            binary.flush()
        else:
            print(content, flush=True)
    except (OSError, UnicodeError):
        _diagnostic(
            "command completed, but stdout could not be delivered; "
            "inspect the current head or export artifact"
        )
        _silence_failed_stream(sys.stdout)
        return status
    return status
