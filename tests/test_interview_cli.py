"""Real subprocess/loopback workflow, explicit permissions and alias protection."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from promptwitness.cli import build_parser, main
from promptwitness.interview_cli import run_interview_command
from promptwitness.interview_journal import InterviewJournal
from promptwitness.interview_models import InterviewCriterion, InterviewPlan, InterviewTopic


def original_plan() -> InterviewPlan:
    return InterviewPlan(
        "work",
        "1",
        "Understand workshop work.",
        (
            InterviewTopic(
                "building", "Building sensor", (InterviewCriterion("build", "Describe building."),)
            ),
            InterviewTopic(
                "testing", "Testing sensor", (InterviewCriterion("test", "Describe testing."),)
            ),
        ),
    )


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def files(path: Path, endpoint: str = "http://127.0.0.1:1/chat") -> tuple[Path, Path, Path]:
    plan = write_json(path / "plan.json", original_plan().to_dict())
    provider = {
        "endpoint": endpoint,
        "model": "authored-local-fixture",
        "api_key_env": None,
        "timeout": 2,
        "headers": {"X-Request-ID": "nonsecret-fixture"},
    }
    roles = write_json(path / "roles.json", {"analyst": provider, "questioner": provider})
    return path / "journal.db", plan, roles


def invoke(*args: Any) -> int:
    parsed = build_parser().parse_args(["interview", *(str(item) for item in args)])
    return run_interview_command(parsed)


def child(
    path: Path, *args: Any, status: int = 0, extra_env: dict[str, str] | None = None
) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "promptwitness", "interview", *(str(item) for item in args)],
        cwd=path,
        env={**os.environ, **(extra_env or {}), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == status, (result.stdout, result.stderr)
    if result.stdout:
        return json.loads(result.stdout)
    assert "could not complete" in result.stderr
    return {}


def start_args(database: Path, plan: Path, roles: Path, *, interview_id: str = "one") -> list[Any]:
    return [
        "start",
        database,
        interview_id,
        plan,
        "--participant-id",
        "person",
        "--roles",
        roles,
        "--command-id",
        "create",
    ]


def cas(view: dict[str, Any], command_id: str) -> list[Any]:
    return [
        "--revision",
        view["head"]["revision"],
        "--digest",
        view["head"]["digest"],
        "--command-id",
        command_id,
    ]


@contextmanager
def local_provider():
    requests: list[dict[str, Any]] = []
    mode: dict[str, Any] = {"finish": "stop", "propose": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            payload = json.loads(body["messages"][1]["content"])
            context = payload["context"]
            target = {name: context["target"][name] for name in ("topic_id", "criterion_id")}
            requests.append(
                {"body": body, "raw": raw, "authorization": self.headers.get("Authorization")}
            )
            if context["analysis_answer_id"] is None:
                result = {
                    "format": "promptwitness.question-response/v1",
                    "binding_digest": payload["binding_digest"],
                    "target": target,
                    "text": f"Describe {target['criterion_id']} sensor work?",
                    "memory_ids": [item["memory"]["id"] for item in payload["memories"]],
                }
            else:
                answer = next(
                    item
                    for item in context["source_answers"]
                    if item["answer_id"] == context["analysis_answer_id"]
                )
                result = {
                    "format": "promptwitness.analysis-response/v1",
                    "binding_digest": payload["binding_digest"],
                    "evidence": [
                        {
                            "id": "observed",
                            "answer_id": answer["answer_id"],
                            "start": 0,
                            "end": len(answer["text"]),
                            "quote": answer["text"],
                        }
                    ],
                    "assessments": [
                        {
                            **target,
                            "status": "covered",
                            "evidence_ids": ["observed"],
                            "rationale": "Authored fixture assessment, not model quality.",
                        }
                    ],
                    "memories": [
                        {
                            "links": [target],
                            "evidence_ids": ["observed"],
                            "summary": {
                                "kind": "model_proposed",
                                "text": "sensor work fixture summary",
                            },
                        }
                    ],
                    "proposal": {
                        "topic": {
                            "id": "next",
                            "description": "Future sensor work",
                            "criteria": [{"id": "future", "description": "Discuss future work."}],
                        },
                        "parent_topic_id": target["topic_id"],
                        "evidence_ids": ["observed"],
                    }
                    if mode["propose"]
                    else None,
                }
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(result, ensure_ascii=False),
                        },
                        "finish_reason": mode["finish"],
                    }
                ],
                "authored_fixture": True,
            }
            data = json.dumps(response, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/chat", requests, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_real_cli_complete_reopen_recall_export_and_trusted_replay(tmp_path: Path) -> None:
    with local_provider() as (endpoint, requests, _mode):
        database, plan, roles = files(tmp_path, endpoint)
        config = json.loads(roles.read_text())
        config["analyst"]["api_key_env"] = "PW_CLI_UNUSED_ANALYST_KEY"
        config["questioner"]["api_key_env"] = "PW_CLI_QUESTION_KEY"
        write_json(roles, config)
        view = child(tmp_path, *start_args(database, plan, roles))
        assert view["head"]["revision"] == 1 and not requests
        view = child(
            tmp_path,
            "run",
            database,
            "one",
            "--roles",
            roles,
            "--operation-id",
            "first",
            extra_env={"PW_CLI_QUESTION_KEY": "question-fixture-secret"},
        )
        assert view["next_action"]["kind"] == "await_answer" and len(requests) == 1
        assert requests[0]["authorization"] == "Bearer question-fixture-secret"
        assert "answers" not in view and "provider_identities" not in view
        answer_file = tmp_path / "answer.txt"
        answer_file.write_bytes("sensor work 😀e\u0301\r\nExact bytes.".encode())
        answered = child(
            tmp_path,
            "answer",
            database,
            "one",
            answer_file,
            "--question-id",
            view["question"]["question_id"],
            "--answer-id",
            "build-answer",
            *cas(view, "answer-build"),
        )
        assert answered["head"]["revision"] == 4
        # A second process and different environment supplies the previously
        # unused analyst credential only when that stage is actually executed.
        env = {
            "PW_CLI_QUESTION_KEY": "question-fixture-secret",
            "PW_CLI_UNUSED_ANALYST_KEY": "analyst-fixture-secret",
        }
        view = child(
            tmp_path,
            "run",
            database,
            "one",
            "--roles",
            roles,
            "--operation-id",
            "second",
            extra_env=env,
        )
        assert len(requests) == 3 and view["question"]["target"]["criterion_id"] == "test"
        assert requests[1]["authorization"] == "Bearer analyst-fixture-secret"
        answer_file.write_text("sensor testing passed fixture measurements.", encoding="utf-8")
        child(
            tmp_path,
            "answer",
            database,
            "one",
            answer_file,
            "--question-id",
            view["question"]["question_id"],
            "--answer-id",
            "test-answer",
            *cas(view, "answer-test"),
        )
        view = child(
            tmp_path,
            "run",
            database,
            "one",
            "--roles",
            roles,
            "--operation-id",
            "complete",
            extra_env=env,
        )
        assert view["finished_reason"] == "agenda_completed" and len(requests) == 4
        report = child(tmp_path, "report", database, "one")
        assert (
            report["private_data"]
            and report["report"]["assessed_coverage"]["required"]["covered"] == 2
        )
        query = tmp_path / "query.txt"
        query.write_text("sensor work", encoding="utf-8")
        recall = child(tmp_path, "recall", database, "one", "--query-file", query)
        assert recall["recall"]["selected"] == 2
        export = tmp_path / "private-export.json"
        receipt = child(tmp_path, "export", database, "one", export)
        assert receipt["head"] == view["head"]
        trusted = write_json(tmp_path / "trusted.json", view["head"])
        replayed = child(tmp_path, "replay", export, "--trusted-head", trusted, "--content")
        assert replayed["state"]["answers"][0]["text"] == "sensor work 😀e\u0301\r\nExact bytes."
        assert "question-fixture-secret" not in export.read_text(encoding="utf-8")
        assert "analyst-fixture-secret" not in export.read_text(encoding="utf-8")
        assert not list(tmp_path.glob(".interview-export-*"))
        # Prior heads are explicit, verified against this same database.
        heads = write_json(tmp_path / "prior.json", [view["head"]])
        new = child(
            tmp_path, *start_args(database, plan, roles, interview_id="two"), "--prior-heads", heads
        )
        assert new["interview_id"] == "two"
        second = child(
            tmp_path,
            "run",
            database,
            "two",
            "--roles",
            roles,
            "--operation-id",
            "recall-start",
            extra_env=env,
        )
        assert second["question"]["memory_ids"] and len(requests) == 5


def test_real_cli_failure_resume_zero_calls_and_explicit_retry(tmp_path: Path) -> None:
    with local_provider() as (endpoint, requests, mode):
        database, plan, roles = files(tmp_path, endpoint)
        child(tmp_path, *start_args(database, plan, roles))
        mode["finish"] = "length"
        failed = child(
            tmp_path, "run", database, "one", "--roles", roles, "--operation-id", "first", status=2
        )
        assert failed["next_action"]["kind"] == "await_retry" and len(requests) == 1
        same = child(
            tmp_path, "run", database, "one", "--roles", roles, "--operation-id", "resume", status=2
        )
        assert same["head"] == failed["head"] and len(requests) == 1
        mode["finish"] = "stop"
        retried = child(
            tmp_path, "retry", database, "one", "--roles", roles, "--operation-id", "authorize-once"
        )
        assert retried["next_action"]["kind"] == "await_answer" and len(requests) == 2
        assert retried["stage_reservations"] == 2
        child(
            tmp_path,
            "retry",
            database,
            "one",
            "--roles",
            roles,
            "--operation-id",
            "authorize-once",
            status=1,
        )
        assert len(requests) == 2


def test_skip_and_explicit_finish_use_exact_cas_in_real_cli(tmp_path: Path) -> None:
    with local_provider() as (endpoint, requests, _mode):
        database, plan, roles = files(tmp_path, endpoint)
        child(tmp_path, *start_args(database, plan, roles))
        view = child(
            tmp_path, "run", database, "one", "--roles", roles, "--operation-id", "question"
        )
        skipped = child(
            tmp_path,
            "skip",
            database,
            "one",
            "--question-id",
            view["question"]["question_id"],
            "--scope",
            "topic",
            *cas(view, "skip"),
        )
        assert skipped["participant_turns"] == 1
        child(
            tmp_path,
            "finish",
            database,
            "one",
            "--reason",
            "participant_stopped",
            *cas(view, "stale-finish"),
            status=1,
        )
        stopped = child(
            tmp_path,
            "finish",
            database,
            "one",
            "--reason",
            "participant_stopped",
            *cas(skipped, "stop"),
        )
        assert stopped["finished_reason"] == "participant_stopped" and len(requests) == 1
        assert (
            child(tmp_path, "show", database, "one", "--content")["state"]["skips"][0]["scope"]
            == "topic"
        )


@pytest.mark.parametrize("action", ["show", "report", "export", "run", "retry", "recall"])
def test_read_or_resume_never_creates_missing_database(
    tmp_path: Path, action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    database, _plan, roles = files(tmp_path)
    extra: list[Any] = []
    if action in ("run", "retry"):
        extra = ["--roles", roles, "--operation-id", "none"]
    elif action == "export":
        extra = [tmp_path / "out.json"]
    elif action == "recall":
        query = tmp_path / "query.txt"
        query.write_text("query")
        extra = ["--query-file", query]
    assert invoke(action, database, "one", *extra) == 1
    assert not database.exists() and not (tmp_path / "out.json").exists()
    captured = capsys.readouterr()
    assert not captured.out and "could not complete" in captured.err


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "credential_header",
        "raw_secret",
        "bool_timeout",
        "bad_env",
        "duplicate_json",
        "bad_utf8",
        "future_prompt",
        "header_integer",
        "header_null",
        "header_list",
    ],
)
def test_invalid_configuration_fails_before_creating_database(
    tmp_path: Path, mutation: str, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    values = json.loads(roles.read_text())
    if mutation == "unknown":
        values["analyst"]["plugin"] = "arbitrary-code"
    elif mutation == "credential_header":
        values["analyst"]["headers"] = {"Authorization": "secret-fixture-marker"}
    elif mutation == "raw_secret":
        values["analyst"]["api_key"] = "secret-fixture-marker"
    elif mutation == "bool_timeout":
        values["analyst"]["timeout"] = True
    elif mutation == "bad_env":
        values["analyst"]["api_key_env"] = "secret fixture marker"
    elif mutation.startswith("header_"):
        values["analyst"]["headers"] = {
            "X-Request-ID": {"header_integer": 123, "header_null": None, "header_list": []}[
                mutation
            ]
        }
    write_json(roles, values)
    if mutation == "duplicate_json":
        roles.write_text('{"analyst":{},"analyst":{},"questioner":{}}')
    elif mutation == "bad_utf8":
        roles.write_bytes(b"\xff")
    elif mutation == "future_prompt":
        write_json(plan, replace(original_plan(), questioner_prompt_version="future").to_dict())
    assert invoke(*start_args(database, plan, roles)) == 1
    assert not database.exists()
    captured = capsys.readouterr()
    assert "secret" not in captured.err and "arbitrary-code" not in captured.err


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_database_and_sqlite_sidecars_cannot_be_used_as_inputs(
    tmp_path: Path, suffix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    protected = Path(str(database) + suffix)
    protected.write_bytes(roles.read_bytes())
    before = protected.read_bytes()
    assert invoke(*start_args(database, plan, protected)) == 1
    assert protected.read_bytes() == before
    if suffix:
        assert not database.exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_export_cannot_target_database_or_sidecar_even_when_absent(
    tmp_path: Path, suffix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    assert invoke(*start_args(database, plan, roles)) == 0
    target = Path(str(database) + suffix)
    before = database.read_bytes()
    assert invoke("export", database, "one", target) == 1
    assert database.read_bytes() == before


def test_hardlink_and_symlink_aliases_preserve_original_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    assert invoke(*start_args(database, plan, roles)) == 0
    alias = tmp_path / "linked.json"
    os.link(database, alias)
    before = database.read_bytes()
    assert (
        invoke(
            "answer",
            database,
            "one",
            alias,
            "--question-id",
            "q",
            "--answer-id",
            "a",
            "--revision",
            1,
            "--digest",
            "a" * 64,
            "--command-id",
            "x",
        )
        == 1
    )
    assert invoke("export", database, "one", alias) == 1
    assert database.read_bytes() == alias.read_bytes() == before
    alias.unlink()
    try:
        alias.symlink_to(database)
    except OSError:
        pytest.skip("symbolic links require privileges on this Windows host")
    assert invoke("export", database, "one", alias) == 1
    assert database.read_bytes() == before


def test_export_does_not_overwrite_existing_or_racing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    assert invoke(*start_args(database, plan, roles)) == 0
    output = tmp_path / "export.json"
    output.write_text("existing user artifact")
    assert invoke("export", database, "one", output) == 1
    assert output.read_text() == "existing user artifact"
    output.unlink()
    real_link = os.link

    def raced(source: Any, destination: Any, **kwargs: Any) -> None:
        Path(destination).write_text("concurrent user artifact")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", raced)
    assert invoke("export", database, "one", output) == 1
    assert output.read_text() == "concurrent user artifact"
    assert not list(tmp_path.glob(".interview-export-*"))


def test_export_cleanup_failure_discloses_possible_published_private_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    assert invoke(*start_args(database, plan, roles)) == 0
    capsys.readouterr()
    output = tmp_path / "export.json"
    real_unlink = Path.unlink

    def failed_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.startswith(".interview-export-"):
            raise OSError("private cleanup exception text")
        real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", failed_cleanup)
        assert invoke("export", database, "one", output) == 1
    captured = capsys.readouterr()
    assert "artifact or private temporary file may already exist" in captured.err
    assert "private cleanup exception text" not in captured.err
    assert not captured.out
    temporary = list(tmp_path.glob(".interview-export-*"))
    assert len(temporary) == 1
    assert output.samefile(temporary[0])
    assert json.loads(output.read_text(encoding="utf-8"))["head"]["revision"] == 1
    temporary[0].unlink()


def test_replay_demands_independent_correct_trusted_head_and_does_not_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    assert invoke(*start_args(database, plan, roles)) == 0
    output = tmp_path / "events.json"
    assert invoke("export", database, "one", output) == 0
    value = json.loads(output.read_text())
    wrong = write_json(tmp_path / "head.json", {**value["head"], "digest": "0" * 64})
    assert invoke("replay", output, "--trusted-head", wrong) == 1
    assert invoke("replay", output, "--trusted-head", output) == 1
    alias = tmp_path / "head-alias.json"
    os.link(output, alias)
    assert invoke("replay", output, "--trusted-head", alias) == 1
    write_json(wrong, value["head"])
    before = database.read_bytes()
    assert invoke("replay", output, "--trusted-head", wrong) == 0
    assert database.read_bytes() == before


def test_broken_stdout_after_success_does_not_claim_transaction_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)

    class Broken(io.StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("closed consumer")

    monkeypatch.setattr(sys, "stdout", Broken())
    assert invoke(*start_args(database, plan, roles)) == 0
    with InterviewJournal(database) as journal:
        assert journal.read("one").revision == 1
    assert "command completed" in capsys.readouterr().err


def test_input_caps_and_utf8_failure_are_controlled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, plan, roles = files(tmp_path)
    roles.write_bytes(b" " * 131073)
    assert invoke(*start_args(database, plan, roles)) == 1
    assert not database.exists()
    assert "byte" not in capsys.readouterr().err  # no raw error string disclosure


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_embedded_cli_full_recovery_and_explicit_proposal_decision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decision: str
) -> None:
    def call(*args: Any, status: int = 0) -> dict[str, Any]:
        assert invoke(*args) == status
        captured = capsys.readouterr()
        assert not captured.err
        return json.loads(captured.out)

    with local_provider() as (endpoint, requests, mode):
        database, plan, roles = files(tmp_path, endpoint)
        call(*start_args(database, plan, roles))
        mode["finish"] = "length"
        failed = call("run", database, "one", "--roles", roles, "--operation-id", "first", status=2)
        assert failed["error_code"] == "invalid_response"
        mode["finish"] = "stop"
        view = call("retry", database, "one", "--roles", roles, "--operation-id", "retry")
        answer = tmp_path / "answer.txt"
        answer.write_bytes("sensor work 🛰️\r\n".encode())
        call(
            "answer",
            database,
            "one",
            answer,
            "--question-id",
            view["question"]["question_id"],
            "--answer-id",
            "a",
            *cas(view, "answer"),
        )
        mode["propose"] = True
        view = call("run", database, "one", "--roles", roles, "--operation-id", "analyze")
        assert view["next_action"]["kind"] == "await_review"
        call(
            "proposal",
            database,
            "one",
            "--proposal-id",
            view["pending_proposals"][0]["proposal_id"],
            "--decision",
            decision,
            *cas(view, "review"),
        )
        view = call("run", database, "one", "--roles", roles, "--operation-id", "next-question")
        skipped = call(
            "skip",
            database,
            "one",
            "--question-id",
            view["question"]["question_id"],
            "--scope",
            "topic",
            *cas(view, "skip"),
        )
        finished = call(
            "finish", database, "one", "--reason", "participant_stopped", *cas(skipped, "stop")
        )
        assert len(requests) == 4
        state = call("show", database, "one", "--content")["state"]
        assert state["proposals"][0]["status"] == decision
        assert len(state["memories"]) == 1
        report = call("report", database, "one")["report"]
        assert report["assessed_coverage"]["required"]["total"] == 2
        query = tmp_path / "query.txt"
        query.write_text("sensor work")
        assert call("recall", database, "one", "--query-file", query)["recall"]["selected"] == 1
        export = tmp_path / "export.json"
        assert call("export", database, "one", export)["head"] == finished["head"]
        head = write_json(tmp_path / "head.json", finished["head"])
        assert call("replay", export, "--trusted-head", head)["head"] == finished["head"]
        assert call("replay", export, "--trusted-head", head, "--content")["state"] == state


@pytest.mark.parametrize("error", [OSError, UnicodeEncodeError])
def test_nonpipe_stdout_failure_also_reports_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: type[Exception],
) -> None:
    database, plan, roles = files(tmp_path)

    class Broken(io.StringIO):
        def write(self, value: str) -> int:
            if error is OSError:
                raise OSError("disk full fixture")
            raise UnicodeEncodeError("ascii", "😀", 0, 1, "fixture")

    monkeypatch.setattr(sys, "stdout", Broken())
    assert invoke(*start_args(database, plan, roles)) == 0
    with InterviewJournal(database) as journal:
        assert journal.read("one").revision == 1
    captured = capsys.readouterr()
    assert "command completed" in captured.err and "could not complete" not in captured.err


def test_both_closed_output_streams_cannot_undo_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, plan, roles = files(tmp_path)

    class Broken(io.StringIO):
        def write(self, value: str) -> int:
            raise BrokenPipeError("closed")

    monkeypatch.setattr(sys, "stdout", Broken())
    monkeypatch.setattr(sys, "stderr", Broken())
    assert invoke(*start_args(database, plan, roles)) == 0
    with InterviewJournal(database) as journal:
        assert journal.read("one").revision == 1


def test_actual_closed_stdout_consumer_keeps_success_exit_and_committed_head(
    tmp_path: Path,
) -> None:
    database, plan, roles = files(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "promptwitness",
            "interview",
            *(str(item) for item in start_args(database, plan, roles)),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert process.stdout is not None
    process.stdout.close()
    process.stdout = None
    try:
        _out, error = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 0, error.decode("utf-8", errors="replace")
    assert b"command completed" in error
    assert b"Exception ignored" not in error
    with InterviewJournal(database) as journal:
        assert journal.read("one").revision == 1


def test_missing_nonregular_input_is_rejected_before_new_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, _plan, roles = files(tmp_path)
    assert invoke(*start_args(database, tmp_path / "does-not-exist", roles)) == 1
    assert invoke(*start_args(database, tmp_path, roles)) == 1
    assert not database.exists()


@pytest.mark.parametrize("close_stderr", [False, True])
def test_closed_host_text_streams_do_not_hide_committed_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    close_stderr: bool,
) -> None:
    database, plan, roles = files(tmp_path)
    closed = io.StringIO()
    closed.close()
    with monkeypatch.context() as patch:
        # Exercise Python 3.14's real color probe even on Windows without a VT
        # console. Production must not mutate process environment or streams.
        for name in ("PYTHON_COLORS", "NO_COLOR", "FORCE_COLOR", "TERM"):
            patch.delenv(name, raising=False)
        if sys.version_info >= (3, 14) and sys.platform == "win32":
            import nt

            patch.setattr(nt, "_supports_virtual_terminal", lambda: True)
        patch.setattr(sys, "stdout", closed)
        if close_stderr:
            patch.setattr(sys, "stderr", closed)
        assert invoke(*start_args(database, plan, roles)) == 0
    with InterviewJournal(database) as journal:
        assert journal.read("one").revision == 1
    captured = capsys.readouterr()
    assert not captured.out
    if not close_stderr:
        assert "command completed" in captured.err


def test_all_cli_parsers_use_plain_help_without_color_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("PYTHON_COLORS", "1")
    parser = build_parser()
    pending = [parser]
    visited = 0
    while pending:
        current = pending.pop()
        help_text = current.format_help()
        assert "usage:" in help_text and "\x1b[" not in help_text
        visited += 1
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                pending.extend(action.choices.values())
    assert visited > 20  # Includes nested interview, session and task commands.


@pytest.mark.parametrize("prefix", [(), ("interview",), ("interview", "start"), ("session",)])
def test_normal_cli_help_preserves_stdout_and_exit_zero(
    prefix: tuple[str, ...], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    with pytest.raises(SystemExit) as finished:
        main([*prefix, "--help"])
    assert finished.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out and "\x1b[" not in captured.out
    assert not captured.err


@pytest.mark.parametrize("closed_stdout", [False, True])
def test_invalid_cli_arguments_preserve_stderr_and_exit_one(
    closed_stdout: bool, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        for name in ("PYTHON_COLORS", "NO_COLOR", "FORCE_COLOR", "TERM"):
            patch.delenv(name, raising=False)
        if sys.version_info >= (3, 14) and sys.platform == "win32":
            import nt

            patch.setattr(nt, "_supports_virtual_terminal", lambda: True)
        if closed_stdout:
            output = io.StringIO()
            output.close()
            patch.setattr(sys, "stdout", output)
        with pytest.raises(SystemExit) as failed:
            main(["interview", "start"])
    assert failed.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "usage:" in captured.err and "error:" in captured.err
    assert "\x1b[" not in captured.err
