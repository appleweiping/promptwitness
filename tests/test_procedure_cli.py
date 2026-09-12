"""Real CLI processes and local HTTP fixtures, never external model requests."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock

import pytest

from promptwitness.cli import build_parser, main
from promptwitness.procedure_cli import run_procedure_command
from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_plan import load_procedure_suite
from promptwitness.task_runs import TaskRunStore

ANSWER = "<Solution>\n1 + 2 = 3\n3 + 3 = 6\n6 + 4 = 10\n</Solution>"
PRIVATE = "PRIVATE-GOLD-CANARY-秘密"


def case(family="countdown", identifier="authored-1"):
    if family == "pseudo_to_code":
        inputs = {"pseudocode_lines": ["print an integer"]}
        reference = {"code_lines": [PRIVATE], "testcases": [[["1"], ["1"]]]}
    elif family == "html_to_tsv":
        inputs = {
            "html": "<table>PRIVATE-INPUT-CANARY-é</table>",
            "header": ["name"],
            "task_topic": "Name",
            "task_description": "Extract the name",
            "filtering_instruction": "",
            "website_id": "authored",
        }
        reference = {"tsv": "name\n" + PRIVATE}
    elif family == "tom_tracking":
        inputs = {
            "story_components": "An agent and an object.",
            "story": "A sees a key.",
            "question": "Where does A believe the key is?",
        }
        reference = {"solution": "- " + PRIVATE, "answer": [PRIVATE], "trace": ["- " + PRIVATE]}
    else:
        inputs = {
            "numbers": [1, 2, 3, 4],
            "target": 10,
            "min_intermediate": 1,
            "max_intermediate": 2000,
        }
        reference = {
            "solution": ["1+2=3", "3+3=6", "6+4=10"],
            "solution_text": ANSWER,
            "demonstration": PRIVATE,
            "search_steps": 0.5,
            "num_search_tokens": 20,
        }
    return ProcedureCase(
        family,
        identifier,
        "0.5k",
        inputs,
        reference,
        {"kind": "authored", "label": "independent CLI fixture"},
    )


def suite(tmp_path, *, family="countdown", cases=1, budget=1_000_000, tasks=1):
    records = tmp_path / "cases.json"
    records.write_text(
        json.dumps([case(family, f"case-{i}").to_dict() for i in range(cases)], ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "promptwitness.procedure-suite/v1",
                "suite_id": "authored-suite",
                "budgets": [budget],
                "generation": {"max_tokens": 128},
                "tasks": [
                    {
                        "id": f"task-{i}",
                        "adapter": "cases",
                        "path": records.name,
                        "generation": {"max_tokens": 128 + i},
                    }
                    for i in range(tasks)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def cli(*arguments, env=None):
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "ascii"  # binary publication must still be UTF-8.
    if env:
        environment.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "promptwitness", "procedure-suite", *map(str, arguments)],
        capture_output=True,
        timeout=30,
        env=environment,
    )
    return result.returncode, result.stdout.decode("utf-8"), result.stderr.decode("utf-8")


def envelope(answer=ANSWER, *, finish="stop"):
    return {
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": answer}}]
    }


@contextmanager
def peer(responses):
    seen = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.connection.settimeout(3)
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append((dict(self.headers), json.loads(body)))
                response = responses[min(len(seen) - 1, len(responses) - 1)]
                if response is None:
                    self.close_connection = True
                    return
                data = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError as error:
                errors.append(type(error).__name__)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/chat", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(4)
        assert not thread.is_alive()
        assert not errors


def reserve_database(tmp_path):
    manifest = suite(tmp_path)
    plan = load_procedure_suite(manifest)
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        store.bind(plan, {"provider": "authored-offline-fixture"})
        item = plan.payload["items"][0]["id"]
        _, revision = store.reserve(item, expected_revision=0)
    return database, item, revision


def test_actual_plan_process_is_private_by_default_and_utf8_when_explicit(tmp_path):
    manifest = suite(tmp_path, family="html_to_tsv")
    code, output, errors = cli("plan", manifest)
    assert code == 0 and not errors
    data = json.loads(output)
    assert data["private_data"] is True and data["includes_content"] is False
    assert "PRIVATE-" not in output
    assert "record" not in data["plan"]["items"][0]
    assert "messages" not in data["plan"]["items"][0]
    code, output, errors = cli("plan", manifest, "--include-inputs")
    assert code == 0 and not errors
    assert PRIVATE in output and "PRIVATE-INPUT-CANARY-é" in output
    assert not (tmp_path / "run.sqlite").exists()


def test_actual_run_reopen_resume_show_and_per_item_generation(tmp_path):
    manifest = suite(tmp_path, tasks=2)
    database = tmp_path / "run.sqlite"
    with peer([envelope()]) as (endpoint, seen):
        arguments = [
            "run",
            database,
            manifest,
            "--endpoint",
            endpoint,
            "--model",
            "local-fixture",
            "--api-key-env",
            "PROCEDURE_FIXTURE_KEY",
        ]
        env = {"PROCEDURE_FIXTURE_KEY": "secret-only-in-request"}
        code, output, errors = cli(*arguments, "--max-cases", 1, env=env)
        assert code == 2 and not errors
        first = json.loads(output)
        assert first["coverage"]["succeeded"] == 1
        assert first["coverage"]["ready"] == 1
        assert len(seen) == 1
        code, output, errors = cli(*arguments, env=env)
        assert code == 0 and not errors
        assert len(seen) == 2
        report = json.loads(output)
        assert report["format"] == "promptwitness.procedure-report/v1"
        assert report["coverage"]["execution_complete"]
        assert all(row["result"]["score"]["primary_score"] == 1 for row in report["results"])
        assert all("prediction" not in row["result"] for row in report["results"])
        assert [body["max_tokens"] for _, body in seen] == [128, 129]
        assert all(PRIVATE not in json.dumps(body) for _, body in seen)
        assert seen[0][0]["Authorization"] == "Bearer secret-only-in-request"
        assert "secret-only-in-request" not in output
        assert cli(*arguments, env=env)[0] == 0
        assert len(seen) == 2
    code, output, errors = cli("show", database, "--include-inputs")
    assert code == 0 and not errors
    assert json.loads(output)["results"][0]["result"]["prediction"] == ANSWER


@pytest.mark.parametrize("first_response", [envelope(finish="length"), None])
def test_actual_failed_call_requires_explicit_retry_and_new_attempt(tmp_path, first_response):
    manifest = suite(tmp_path)
    database = tmp_path / "run.sqlite"
    with peer([first_response, envelope()]) as (endpoint, seen):
        args = ["run", database, manifest, "--endpoint", endpoint, "--model", "fixture"]
        code, output, _ = cli(*args)
        assert code == 2
        row = json.loads(output)["results"][0]
        assert row["status"] == "failed" and row["attempts"] == 1
        assert cli(*args)[0] == 2 and len(seen) == 1
        code, output, _ = cli("retry", database, row["item_id"], "--revision", row["revision"])
        assert code == 2
        ready = json.loads(output)["results"][0]
        assert ready["status"] == "ready" and ready["attempts"] == 1
        assert len(seen) == 1
        assert cli("retry", database, row["item_id"], "--revision", row["revision"])[0] == 1
        code, output, _ = cli(*args)
        assert code == 0 and len(seen) == 2
        assert json.loads(output)["results"][0]["attempts"] == 2


def test_manual_response_uses_exact_reservation_without_a_provider(tmp_path):
    database, item, revision = reserve_database(tmp_path)
    response = tmp_path / "response.json"
    response.write_text(json.dumps(envelope()), encoding="utf-8")
    code, output, errors = cli("respond", database, item, "--revision", revision, response)
    assert code == 0 and not errors
    assert json.loads(output)["results"][0]["attempts"] == 1
    code, _, errors = cli("respond", database, item, "--revision", revision, response)
    assert code == 1 and "inspect stored revisions" in errors
    with TaskRunStore(database) as store:
        assert store.snapshot()["results"][0]["revision"] == revision + 1


@pytest.mark.parametrize("family,budget", [("pseudo_to_code", 1_000_000), ("countdown", 1)])
def test_unsupported_primary_or_skipped_work_is_not_completion(tmp_path, family, budget):
    manifest = suite(tmp_path, family=family, budget=budget)
    with peer([envelope("```cpp\nint main() {}\n```")]) as (endpoint, seen):
        code, output, _ = cli(
            "run", tmp_path / "run.sqlite", manifest, "--endpoint", endpoint, "--model", "fixture"
        )
        assert code == 2
        report = json.loads(output)
        assert not report["coverage"]["scoring_complete"]
        assert len(seen) == (1 if family == "pseudo_to_code" else 0)


@pytest.mark.parametrize("raw", [b"not-json-PRIVATE", b'{"tasks":[],"tasks":[]}', b"\xff", b"[]"])
def test_invalid_manifest_never_creates_database_and_diagnostics_are_static(
    tmp_path, raw, monkeypatch, capsys
):
    import promptwitness.procedure_cli as module

    manifest = tmp_path / "PRIVATE-PATH.json"
    manifest.write_bytes(raw)
    database = tmp_path / "new.sqlite"
    store = Mock(side_effect=AssertionError("database must not be opened"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    assert (
        main(
            [
                "procedure-suite",
                "run",
                str(database),
                str(manifest),
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
            ]
        )
        == 1
    )
    assert not database.exists() and store.call_count == 0
    assert "PRIVATE" not in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_sqlite_paths_cannot_alias_typed_source_even_before_store_open(
    tmp_path, suffix, monkeypatch
):
    import promptwitness.procedure_cli as module

    manifest = suite(tmp_path)
    original = (tmp_path / "cases.json").read_bytes()
    database = tmp_path / "run.sqlite"
    os.link(tmp_path / "cases.json", Path(str(database) + suffix))
    store = Mock(side_effect=AssertionError("database must not be opened"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    assert (
        main(
            [
                "procedure-suite",
                "run",
                str(database),
                str(manifest),
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
            ]
        )
        == 1
    )
    assert store.call_count == 0
    assert (tmp_path / "cases.json").read_bytes() == original


def test_database_under_longproc_root_rejected_before_data_loading(tmp_path, monkeypatch):
    import promptwitness.procedure_cli as module

    root = tmp_path / "data"
    root.mkdir()
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "promptwitness.procedure-suite/v1",
                "suite_id": "one",
                "budgets": [100],
                "tasks": [
                    {"id": "t", "adapter": "longproc", "path": "data", "dataset": "countdown_0.5k"}
                ],
            }
        ),
        encoding="utf-8",
    )
    loader = Mock(side_effect=AssertionError("unsafe paths must precede loading"))
    monkeypatch.setattr(module, "load_procedure_suite", loader)
    assert (
        main(
            [
                "procedure-suite",
                "run",
                str(root / "run.sqlite"),
                str(manifest),
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
            ]
        )
        == 1
    )
    assert loader.call_count == 0


@pytest.mark.parametrize("command", ["show", "retry", "respond"])
def test_read_and_recovery_commands_never_create_missing_or_empty_database(tmp_path, command):
    database = tmp_path / "run.sqlite"
    response = tmp_path / "response.json"
    response.write_text('"answer"', encoding="utf-8")
    args = [command, database]
    if command != "show":
        args += ["item", "--revision", "1"]
    if command == "respond":
        args += [response]
    assert cli(*args)[0] == 1
    assert not database.exists()
    database.touch()
    assert cli(*args)[0] == 1
    assert database.read_bytes() == b""


@pytest.mark.parametrize("response", ["duplicate", "too_large", "incomplete", "non_utf8"])
def test_invalid_response_is_fully_rejected_before_store_mutation(tmp_path, response, monkeypatch):
    import promptwitness.procedure_cli as module

    database, item, revision = reserve_database(tmp_path)
    path = tmp_path / "PRIVATE-response.json"
    raw = {
        "duplicate": b'{"status":"completed","status":"failed","output_text":"PRIVATE"}',
        "too_large": b'"' + b"x" * 100 + b'"',
        "incomplete": json.dumps(envelope("PRIVATE", finish="length")).encode(),
        "non_utf8": b"\xff",
    }[response]
    path.write_bytes(raw)
    if response == "too_large":
        monkeypatch.setattr(module, "MAX_PREDICTION_BYTES", 16)
    store = Mock(side_effect=AssertionError("invalid response opened store"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    assert (
        main(
            [
                "procedure-suite",
                "respond",
                str(database),
                item,
                "--revision",
                str(revision),
                str(path),
            ]
        )
        == 1
    )
    assert store.call_count == 0
    with TaskRunStore(database) as actual:
        assert actual.snapshot()["results"][0]["status"] == "running"


def test_respond_raw_file_cap_precedes_json_parse(tmp_path, monkeypatch):
    import promptwitness.procedure_cli as module

    database, item, revision = reserve_database(tmp_path)
    response = tmp_path / "response.json"
    response.write_bytes(b"x" * 65)
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 64)
    parser = Mock(side_effect=AssertionError("oversize file parsed"))
    monkeypatch.setattr(module, "load_json", parser)
    assert (
        main(
            [
                "procedure-suite",
                "respond",
                str(database),
                item,
                "--revision",
                str(revision),
                str(response),
            ]
        )
        == 1
    )
    assert parser.call_count == 0


@pytest.mark.parametrize("mode", ["short_text", "none_text", "closed", "missing", "short_binary"])
def test_completed_mutation_keeps_status_if_host_output_unavailable(monkeypatch, capsys, mode):
    import promptwitness.procedure_cli as module

    args = build_parser().parse_args(
        ["procedure-suite", "retry", "run.sqlite", "item", "--revision", "1"]
    )
    monkeypatch.setattr(module, "_execute", lambda args: ({"completed": True}, 2))

    class Short(io.StringIO):
        def write(self, value):
            return None if mode == "none_text" else max(0, len(value) - 1)

    if mode == "closed":
        stream = io.StringIO()
        stream.close()
    elif mode == "missing":
        stream = None
    elif mode == "short_binary":
        stream = Mock(spec=["buffer"], buffer=Mock(write=Mock(return_value=0)))
    else:
        stream = Short()
    monkeypatch.setattr(sys, "stdout", stream)
    assert run_procedure_command(args) == 2
    assert "operation completed" in capsys.readouterr().err


@pytest.mark.parametrize("status", [0, 2])
def test_real_closed_pipe_does_not_replace_completed_status_with_120(status):
    script = (
        "import argparse,sys; import promptwitness.procedure_cli as m; "
        f"m._execute=lambda _:({{'ok':True}},{status}); "
        "sys.stdin.readline(); "
        "sys.exit(m.run_procedure_command(argparse.Namespace(procedure_command='run')))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin and process.stdout and process.stderr
    try:
        process.stdout.close()
        process.stdin.write(b"go\n")
        process.stdin.flush()
        process.stdin.close()
        assert process.wait(timeout=15) == status
        error = process.stderr.read().decode()
        assert "operation completed" in error
        assert "Exception ignored" not in error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def test_help_and_invalid_arguments_preserve_parent_cli_contract():
    code, output, errors = cli("--help")
    assert code == 0 and "respond" in output and not errors
    assert cli("run", "one.sqlite", "suite.json")[0] == 1


def test_nonempty_pristine_sqlite_read_is_not_implicitly_initialized(tmp_path):
    database = tmp_path / "unrelated.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("VACUUM")
    connection.close()
    assert database.stat().st_size > 0
    before = database.read_bytes()
    assert cli("show", database)[0] == 1
    assert database.read_bytes() == before


def test_legacy_task_database_rejected_before_retry_or_manual_response(tmp_path):
    from promptwitness.task_data import load_task_suite

    record = tmp_path / "legacy.json"
    record.write_text(
        json.dumps([{"id": "one", "question": "Who?", "context": "Ada.", "answers": ["Ada"]}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "legacy-suite.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "promptwitness.task-suite/v1",
                "suite_id": "legacy",
                "budgets": [10000],
                "tasks": [{"id": "old", "family": "rag", "path": "legacy.json"}],
            }
        ),
        encoding="utf-8",
    )
    plan = load_task_suite(manifest)
    database = tmp_path / "legacy.sqlite"
    with TaskRunStore(database) as store:
        store.bind(plan, {"provider": "offline-fixture"})
        item = plan.payload["items"][0]["id"]
        _, revision = store.reserve(item, expected_revision=0)
    before = database.read_bytes()
    response = tmp_path / "response.json"
    response.write_text('"Ada"', encoding="utf-8")
    assert cli("show", database)[0] == 1
    assert cli("retry", database, item, "--revision", revision)[0] == 1
    assert cli("respond", database, item, "--revision", revision, response)[0] == 1
    assert database.read_bytes() == before


def test_changed_manifest_during_loading_rejected_before_database(tmp_path, monkeypatch):
    import promptwitness.procedure_cli as module

    manifest = suite(tmp_path)
    original = load_procedure_suite(manifest)
    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["seed"] = "new"
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(module, "load_procedure_suite", lambda path: original)
    store = Mock(side_effect=AssertionError("changed manifest opened database"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    assert (
        main(
            [
                "procedure-suite",
                "run",
                str(tmp_path / "db.sqlite"),
                str(manifest),
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
            ]
        )
        == 1
    )
    assert not store.called


def test_invalid_endpoint_and_invocation_limit_precede_database_creation(tmp_path, monkeypatch):
    import promptwitness.procedure_cli as module

    manifest = suite(tmp_path)
    store = Mock(side_effect=AssertionError("invalid provider opened database"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    args = ["procedure-suite", "run", str(tmp_path / "db.sqlite"), str(manifest)]
    assert main([*args, "--endpoint", "file:///PRIVATE", "--model", "fixture"]) == 1
    assert (
        main(
            [
                *args,
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
                "--max-cases",
                "0",
            ]
        )
        == 1
    )
    assert not store.called


@pytest.mark.parametrize(
    "family,answer", [("html_to_tsv", "name\n" + PRIVATE), ("tom_tracking", "- " + PRIVATE)]
)
def test_processed_prediction_is_also_hidden_until_explicit_private_view(tmp_path, family, answer):
    manifest = suite(tmp_path, family=family)
    database = tmp_path / "run.sqlite"
    with peer([envelope(answer)]) as (endpoint, seen):
        code, output, errors = cli(
            "run", database, manifest, "--endpoint", endpoint, "--model", "fixture"
        )
        assert code == 0 and not errors
        assert len(seen) == 1
        assert PRIVATE not in output
        score = json.loads(output)["results"][0]["result"]["score"]
        assert score["primary_score"] == 1 and "processed" not in score
    code, output, _ = cli("show", database)
    assert code == 0 and PRIVATE not in output
    code, output, _ = cli("show", database, "--include-inputs")
    assert code == 0 and PRIVATE in output


def test_manual_late_response_rejected_after_retry_invalidates_old_revision(tmp_path):
    database, item, old_revision = reserve_database(tmp_path)
    response = tmp_path / "response.json"
    response.write_text(json.dumps(envelope()), encoding="utf-8")
    assert cli("retry", database, item, "--revision", old_revision)[0] == 2
    with TaskRunStore(database) as store:
        row = store.snapshot()["results"][0]
        _, new_revision = store.reserve(item, expected_revision=row["revision"])
    assert cli("respond", database, item, "--revision", old_revision, response)[0] == 1
    code, output, _ = cli("respond", database, item, "--revision", new_revision, response)
    assert code == 0
    assert json.loads(output)["results"][0]["attempts"] == 2


def test_external_database_hardlink_to_validated_longproc_source_is_rejected(tmp_path, monkeypatch):
    import promptwitness.procedure_cli as module

    root = tmp_path / "data"
    directory = root / "countdown"
    directory.mkdir(parents=True)
    native = {
        "nums": [1, 2, 3, 4],
        "target": 10,
        "solution": ["1+2=3", "3+3=6", "6+4=10"],
        "solution_text": ANSWER,
        "demonstration": PRIVATE,
        "search_steps": 0.5,
        "num_search_tokens": 50,
    }
    data_file = directory / "countdown_0.5k.json"
    data_file.write_text(json.dumps([native]), encoding="utf-8")
    (directory / "prompts.yaml").write_text("inert", encoding="utf-8")
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "promptwitness.procedure-suite/v1",
                "suite_id": "one",
                "budgets": [10000],
                "tasks": [
                    {"id": "t", "adapter": "longproc", "path": "data", "dataset": "countdown_0.5k"}
                ],
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "outside.sqlite"
    os.link(data_file, database)
    before = data_file.read_bytes()
    store = Mock(side_effect=AssertionError("hard-linked source opened as DB"))
    monkeypatch.setattr(module, "TaskRunStore", store)
    assert (
        main(
            [
                "procedure-suite",
                "run",
                str(database),
                str(manifest),
                "--endpoint",
                "http://127.0.0.1:9/chat",
                "--model",
                "fixture",
            ]
        )
        == 1
    )
    assert not store.called and data_file.read_bytes() == before


def test_report_encoding_limit_after_commit_does_not_reclassify_status(monkeypatch, capsys):
    import promptwitness.procedure_cli as module

    args = build_parser().parse_args(
        ["procedure-suite", "retry", "run.sqlite", "item", "--revision", "1"]
    )
    monkeypatch.setattr(module, "_execute", lambda _: ({"summary": "large"}, 2))
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 4)
    assert run_procedure_command(args) == 2
    captured = capsys.readouterr()
    assert not captured.out and "operation completed" in captured.err


@pytest.mark.parametrize("stderr_mode", ["none", "closed"])
def test_unavailable_diagnostic_stream_does_not_escape_controlled_input_error(
    monkeypatch, stderr_mode
):
    import promptwitness.procedure_cli as module

    args = build_parser().parse_args(["procedure-suite", "show", "run.sqlite"])
    error = Mock(side_effect=ValueError(PRIVATE))
    monkeypatch.setattr(module, "_execute", error)
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(sys, "stderr", None if stderr_mode == "none" else stream)
    assert run_procedure_command(args) == 1


def test_plan_output_failure_is_not_a_successful_delivered_preview(monkeypatch):
    import promptwitness.procedure_cli as module

    args = build_parser().parse_args(["procedure-suite", "plan", "suite.json"])
    monkeypatch.setattr(module, "_execute", lambda _: ({"validated": True}, 0))
    monkeypatch.setattr(sys, "stdout", None)
    assert run_procedure_command(args) == 1


def test_embedded_main_lifecycle_uses_real_provider_and_private_views(tmp_path, capsys):
    manifest = suite(tmp_path, family="html_to_tsv")
    database = tmp_path / "embedded.sqlite"
    assert main(["procedure-suite", "plan", str(manifest)]) == 0
    assert PRIVATE not in capsys.readouterr().out
    assert main(["procedure-suite", "plan", str(manifest), "--include-inputs"]) == 0
    assert PRIVATE in capsys.readouterr().out
    with peer([None, envelope("name\n" + PRIVATE)]) as (endpoint, seen):
        args = [
            "procedure-suite",
            "run",
            str(database),
            str(manifest),
            "--endpoint",
            endpoint,
            "--model",
            "embedded-fixture",
            "--max-cases",
            "1",
        ]
        assert main(args) == 2
        row = json.loads(capsys.readouterr().out)["results"][0]
        assert row["status"] == "failed"
        assert (
            main(
                [
                    "procedure-suite",
                    "retry",
                    str(database),
                    row["item_id"],
                    "--revision",
                    str(row["revision"]),
                ]
            )
            == 2
        )
        assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "ready"
        assert main(args) == 0
        assert PRIVATE not in capsys.readouterr().out
        assert len(seen) == 2
    assert main(["procedure-suite", "show", str(database)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert "processed" not in summary["results"][0]["result"]["score"]
    assert main(["procedure-suite", "show", str(database), "--include-inputs"]) == 0
    assert PRIVATE in capsys.readouterr().out


def test_embedded_manual_response_commits_even_with_no_stdout(tmp_path, monkeypatch, capsys):
    database, item, revision = reserve_database(tmp_path)
    response = tmp_path / "response.json"
    response.write_text(json.dumps(envelope()), encoding="utf-8")
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", None)
        assert (
            main(
                [
                    "procedure-suite",
                    "respond",
                    str(database),
                    item,
                    "--revision",
                    str(revision),
                    str(response),
                ]
            )
            == 0
        )
    assert "operation completed" in capsys.readouterr().err
    with TaskRunStore(database) as store:
        snapshot = store.snapshot()
        assert snapshot["coverage"]["execution_complete"]
        assert snapshot["results"][0]["revision"] == revision + 1
