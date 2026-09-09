"""Task lifecycle tests use hand-authored inputs and offline or loopback providers."""

import json
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from promptwitness import (
    FAMILIES,
    OpenAICompatibleProvider,
    OpenAITaskProvider,
    TaskRunConflict,
    TaskRunStore,
    TaskSuitePlan,
    load_task_suite,
    score_task,
)
from promptwitness.cli import main
from promptwitness.sessions import _json
from promptwitness.task_data import generation_settings, normalize_record
from promptwitness.task_scores import response_text


def record(family="rag", name="one"):
    value = {
        "id": name,
        "question": "Who owns the blue file?",
        "context": "Ada owns the blue file.",
        "answers": ["Ada"],
    }
    if family in {"rerank", "cite"}:
        value.pop("context")
        value["documents"] = [
            {"id": "d1", "text": "Ada owns the blue file."},
            {"id": "d2", "text": "Weather is clear."},
        ]
        if family == "rerank":
            value["relevance"] = {"d1": 3, "d2": 0}
    if family == "icl":
        value.pop("context")
        value["answers"] = ["question"]
        value["labels"] = ["question", "statement"]
        value["examples"] = [
            {"id": "e1", "text": "Where is the book?", "label": "question"},
            {"id": "e2", "text": "This is a pen.", "label": "statement"},
        ]
    return value


def manifest(tmp_path, families=("rag",), budgets=(2000,), count=1):
    tasks = []
    for family in families:
        target = tmp_path / f"{family}.json"
        target.write_text(
            json.dumps([record(family, str(index)) for index in range(count)]), encoding="utf-8"
        )
        tasks.append({"id": family, "family": family, "path": target.name})
    value = {
        "format": "promptwitness.task-suite/v1",
        "suite_id": "test-suite",
        "budgets": list(budgets),
        "seed": "stable",
        "generation": {"max_tokens": 64, "temperature": 0, "seed": 4},
        "tasks": tasks,
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class Offline:
    def __init__(self):
        self.identity = {"provider": "offline-test", "model": "fixture-v1"}
        self.requests = []
        self.fail = False

    def __call__(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("SECRET_TOKEN must not persist")
        instruction = request.messages[0]["content"]
        if "JSON array of strings" in instruction:
            return '["Ada"]'
        if "Rank all" in instruction:
            return '["d1", "d2"]'
        if "classification" in instruction:
            return "question"
        return "Ada [d1]" if "cite" in instruction else "Ada"


def test_complete_seven_family_workflow_resume_and_coverage(tmp_path):
    path = manifest(tmp_path, FAMILIES, budgets=(1, 2000, 4000))
    plan = load_task_suite(path)
    provider = Offline()
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        report = store.run(plan, provider, max_cases=3)
        assert report["coverage"]["succeeded"] == 3
        assert report["coverage"]["skipped"] == 7
    with TaskRunStore(database) as store:
        report = store.run(load_task_suite(path), provider)
        calls = len(provider.requests)
        assert calls == 14
        assert store.run(plan, provider) == report
        assert len(provider.requests) == calls
    assert report["coverage"]["planned"] == 21
    assert report["coverage"]["scored"] == 8
    assert not report["coverage"]["execution_complete"]
    assert not report["coverage"]["scoring_complete"]
    assert "mean_primary_score" not in report["coverage"]
    assert set(report["by_family"]) == set(FAMILIES)
    assert report["by_budget_family"]["1"]["rag"]["score_coverage"] == 0
    for family in ("longqa", "summ", "cite"):
        group = report["by_family"][family]
        assert group["mean_primary_score"] is None
        assert group["metric_capabilities"]["unsupported"]
    assert all(request.generation["max_tokens"] == 64 for request in provider.requests)
    assert all(
        "answers" not in request.to_dict() and '"relevance":' not in _json(request.to_dict())
        for request in provider.requests
    )


def test_failed_requires_explicit_retry_and_secret_not_persisted(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    provider = Offline()
    provider.fail = True
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        failed = store.run(plan, provider)["results"][0]
        assert failed["status"] == "failed" and failed["error"] == "RuntimeError"
        provider.fail = False
        assert store.run(plan, provider)["coverage"]["failed"] == 1
        assert len(provider.requests) == 1
        store.retry(failed["item_id"], expected_revision=failed["revision"])
        result = store.run(plan, provider)["results"][0]
        assert result["attempts"] == 2 and result["status"] == "succeeded"
        with pytest.raises(TaskRunConflict):
            store.retry(result["item_id"], expected_revision=result["revision"])
    assert b"SECRET_TOKEN" not in database.read_bytes()


def test_interrupted_reservation_never_repeats_and_late_worker_rejected(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    provider = Offline()
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        store.bind(plan, provider.identity)
        item = store.snapshot()["results"][0]
        request, revision = store.reserve(item["item_id"], expected_revision=0)
        assert request.digest
    with TaskRunStore(database) as store:
        report = store.run(plan, provider)
        assert report["coverage"]["running"] == 1 and not provider.requests
        store.retry(item["item_id"], expected_revision=revision)
        with pytest.raises(TaskRunConflict):
            store.complete(item["item_id"], "late answer", expected_revision=revision)
        assert store.run(plan, provider)["coverage"]["scored"] == 1


def test_reservation_is_committed_before_callback_and_stale_connections(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as first, TaskRunStore(database) as second:
        first.bind(plan, {"provider": "callback"})
        item = first.snapshot()["results"][0]

        def callback(_request):
            assert second.snapshot()["results"][0]["status"] == "running"
            with pytest.raises(TaskRunConflict):
                second.reserve(item["item_id"], expected_revision=0)
            return "Ada"

        assert first.run(plan, callback, identity={"provider": "callback"})["coverage"][
            "scoring_complete"
        ]


@pytest.mark.parametrize(
    "shift", ["model", "gold", "context", "budget", "generation", "seed", "counter", "source_bytes"]
)
def test_resumption_binds_every_input_and_model_setting(tmp_path, shift):
    path = manifest(tmp_path)
    plan = load_task_suite(path)
    provider = Offline()
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.bind(plan, provider.identity)
        value = json.loads(path.read_text())
        if shift == "model":
            provider.identity["model"] = "other"
        elif shift in {"gold", "context"}:
            target = tmp_path / "rag.json"
            records = json.loads(target.read_text())
            records[0]["answers" if shift == "gold" else "context"] = (
                ["Grace"] if shift == "gold" else "Grace owns the blue file."
            )
            target.write_text(json.dumps(records))
        elif shift == "source_bytes":
            target = tmp_path / "rag.json"
            target.write_text(target.read_text() + "\n")
        elif shift != "counter":
            if shift == "budget":
                value["budgets"] = [2001]
            elif shift == "generation":
                value["generation"]["max_tokens"] = 65
            else:
                value["seed"] = "changed"
            path.write_text(json.dumps(value))
        new = (
            load_task_suite(
                path,
                counter=lambda messages: 100,
                counter_identity={"unit": "model_tokens", "implementation": "fixture-counter/v2"},
            )
            if shift == "counter"
            else load_task_suite(path)
        )
        with pytest.raises(TaskRunConflict, match="changed"):
            store.run(new, provider)
        assert not provider.requests


def test_identity_mutation_between_calls_cannot_dispatch_next_request(tmp_path):
    plan = load_task_suite(manifest(tmp_path, count=2))
    provider = Offline()

    class Mutable(Offline):
        def __call__(self, request):
            answer = super().__call__(request)
            self.identity["model"] = "other"
            return answer

    provider = Mutable()
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        with pytest.raises(TaskRunConflict, match="during"):
            store.run(plan, provider)
        assert len(provider.requests) == 1
        assert store.snapshot()["coverage"]["ready"] == 1


def test_exact_full_prompt_byte_budget_and_custom_counter_identity(tmp_path):
    path = manifest(tmp_path)
    first = load_task_suite(path)
    item = first.to_dict()["items"][0]
    assert item["input_units"] == len(_json(item["messages"]).encode())
    value = json.loads(path.read_text())
    value["budgets"] = [item["input_units"] - 1, item["input_units"]]
    path.write_text(json.dumps(value))
    assert [item["skip_reason"] for item in load_task_suite(path).payload["items"]] == [
        "input_budget_exceeded",
        None,
    ]
    seen = []

    def counter(messages):
        seen.append(messages)
        return 17

    custom = load_task_suite(
        path,
        counter=counter,
        counter_identity={"unit": "model_tokens", "implementation": "test-chat-template/v1"},
    )
    assert seen[0][0]["role"] == "system" and seen[0][1]["role"] == "user"
    assert custom.payload["items"][0]["input_units"] == 17
    assert custom.payload["counter"]["unit"] == "model_tokens"
    with pytest.raises(TypeError):
        custom.payload["generation"]["max_tokens"] = 1


def test_deterministic_sampling_does_not_drop_denominator(tmp_path):
    path = manifest(tmp_path, count=4)
    value = json.loads(path.read_text())
    value["tasks"][0]["limit"] = 2
    path.write_text(json.dumps(value))
    first, second = load_task_suite(path), load_task_suite(path)
    assert first.digest == second.digest
    assert first.payload["tasks"][0]["available"] == 4
    assert first.payload["tasks"][0]["sampled_out"] == 2
    assert len(first.payload["items"]) == 2


@pytest.mark.parametrize(
    "family,answer,score",
    [
        ("recall", '["Ada"]', 1),
        ("recall", '["ADAM"]', 0),
        ("recall", "Ada", 0),
        ("recall", '["Ada","Ada"]', 0),
        ("recall", "[true]", 0),
        ("rag", "Answer: The Ada!", 1),
        ("rag", "Adam", 0),
        ("icl", "question", 1),
        ("icl", "Question", 0),
        ("rerank", '["d1","d2"]', 1),
        ("rerank", '["d1"]', 0),
        ("rerank", '["d1","d1"]', 0),
        ("rerank", '["d1","other"]', 0),
    ],
)
def test_family_specific_primary_scores(family, answer, score):
    item = {
        "family": family,
        "record": normalize_record(record(family), family, "native"),
        "ranking_k": 10,
    }
    assert score_task(item, answer)["primary_score"] == score


def test_ranking_formula_and_unimplemented_judges():
    item = {
        "family": "rerank",
        "record": normalize_record(record("rerank"), "rerank", "native"),
        "ranking_k": 10,
    }
    metrics = score_task(item, '["d2","d1"]')["metrics"]
    assert metrics["ndcg_at_k"] == pytest.approx(1 / math.log2(3))
    assert metrics["mrr_at_k"] == 0.5
    for family in ("longqa", "summ", "cite"):
        item = {
            "family": family,
            "record": normalize_record(record(family), family, "native"),
            "ranking_k": 10,
        }
        result = score_task(item, "Ada [d1] [unknown]")
        assert result["primary_score"] is None and result["unsupported_metrics"]
        if family == "cite":
            assert result["metrics"]["citation_id_valid_fraction"] == 0.5


@pytest.mark.parametrize(
    "adapter,family,raw",
    [
        (
            "kilt",
            "rag",
            {
                "id": "q",
                "question": "Who?",
                "answers": ["Ada"],
                "ctxs": [{"title": "Title", "text": "Ada owns it."}],
            },
        ),
        (
            "msmarco",
            "rerank",
            {"qid": "q", "query": "Who?", "ctxs": [{"id": "d1", "text": "Ada", "label": 1}]},
        ),
        ("processed", "recall", {"id": "q", "question": "Who?", "context": "Ada", "answer": "Ada"}),
    ],
)
def test_local_record_adapters(adapter, family, raw, tmp_path):
    path = manifest(tmp_path, (family,))
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    value = json.loads(path.read_text())
    value["tasks"][0].update(path=source.name, adapter=adapter)
    path.write_text(json.dumps(value))
    assert load_task_suite(path).payload["items"][0]["record_id"] == "q"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": 1, "messages": []},
        {"max_tokens": 1, "temperature": float("nan")},
        {"max_tokens": 1, "top_p": 2},
        {"max_tokens": 1, "seed": False},
    ],
)
def test_invalid_generation_never_overrides_request(raw):
    with pytest.raises(ValueError):
        generation_settings(raw)


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "family",
        "duplicate_task",
        "duplicate_budget",
        "bad_seed",
        "empty_tasks",
        "duplicate_records",
        "bad_adapter",
        "bad_record",
        "missing_file",
    ],
)
def test_reject_invalid_suite_before_any_model_call(tmp_path, change):
    path = manifest(tmp_path)
    value = json.loads(path.read_text())
    if change == "unknown":
        value["other"] = 1
    elif change == "family":
        value["tasks"][0]["family"] = "other"
    elif change == "duplicate_task":
        value["tasks"] *= 2
    elif change == "duplicate_budget":
        value["budgets"] *= 2
    elif change == "bad_seed":
        value["seed"] = True
    elif change == "empty_tasks":
        value["tasks"] = []
    elif change == "duplicate_records":
        (tmp_path / "rag.json").write_text(json.dumps([record(), record()]))
    elif change == "bad_record":
        (tmp_path / "rag.json").write_text("[true]")
    elif change == "bad_adapter":
        value["tasks"][0]["adapter"] = "unsupported"
    else:
        value["tasks"][0]["path"] = "missing.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, OSError)):
        load_task_suite(path)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "Ada"}}]},
        {"choices": [{"message": {"content": "Ada", "tool_calls": [{}]}}]},
        {"choices": [{}, {}]},
    ],
)
def test_bad_or_incomplete_provider_outputs_are_failures(value):
    with pytest.raises(ValueError):
        response_text(value)


def test_plan_inventory_and_result_tampering_are_detected(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.run(plan, Offline())
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("UPDATE task_event SET kind='bad'")
        store.connection.execute("UPDATE task_state SET attempts=99")
        with pytest.raises(ValueError, match="event history"):
            store.snapshot()


def test_run_requires_provider_identity_and_valid_limits(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        with pytest.raises(ValueError, match="identity"):
            store.run(plan, lambda request: "Ada")
        with pytest.raises(TaskRunConflict):
            store.run(plan, Offline(), identity={"provider": "other"})
        with pytest.raises(ValueError):
            store.run(plan, Offline(), max_cases=0)
        with pytest.raises(ValueError, match="not been bound"):
            store.snapshot()


def test_loopback_cli_sends_exact_model_and_generation_then_reuses(tmp_path, capsys):
    path = manifest(tmp_path)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": "Ada"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        database = tmp_path / "run.sqlite"
        args = [
            "task-suite",
            "run",
            str(database),
            str(path),
            "--endpoint",
            f"http://127.0.0.1:{server.server_port}/chat",
            "--model",
            "loopback",
        ]
        assert main(args) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["coverage"]["scored"] == 1
        assert main(args) == 0
        capsys.readouterr()
        assert len(received) == 1
        assert received[0]["model"] == "loopback" and received[0]["max_tokens"] == 64
        assert received[0]["temperature"] == 0 and received[0]["seed"] == 4
        assert "answers" not in received[0] and len(received[0]["messages"]) == 2
        assert main(["task-suite", "show", str(database)]) == 0
        capsys.readouterr()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_cli_plan_no_gold_default_and_recovery_revision(tmp_path, capsys):
    path = manifest(tmp_path)
    assert main(["task-suite", "plan", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "record" not in result["plan"]["items"][0]
    assert main(["task-suite", "plan", str(path), "--include-inputs"]) == 0
    assert "record" in json.loads(capsys.readouterr().out)["plan"]["items"][0]
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        store.bind(load_task_suite(path), {"provider": "manual"})
        item = store.snapshot()["results"][0]
        store.reserve(item["item_id"], expected_revision=0)
    assert main(["task-suite", "show", str(database)]) == 2
    capsys.readouterr()
    response = tmp_path / "response.json"
    response.write_text('"Ada"')
    assert (
        main(
            [
                "task-suite",
                "respond",
                str(database),
                item["item_id"],
                str(response),
                "--revision",
                "1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["task-suite", "retry", str(database), item["item_id"], "--revision", "0"]) == 1


def test_provider_transport_rechecks_identity_and_rejects_generation_override(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    provider = OpenAITaskProvider(
        OpenAICompatibleProvider(
            "http://localhost:1", model="model", headers={"x-route": "PRIVATE"}
        )
    )
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.bind(plan, provider.identity)
        item = store.snapshot()["results"][0]
        request, _revision = store.reserve(item["item_id"], expected_revision=0)
        assert "PRIVATE" not in _json(request.to_dict())
        provider.provider.model = "changed"
        with pytest.raises(TaskRunConflict):
            provider(request)
    with pytest.raises(ValueError, match="generation"):
        provider.provider.complete(
            [{"role": "user", "content": "test"}], generation={"max_tokens": 1, "model": "other"}
        )


@pytest.mark.parametrize(
    "value",
    [
        {"choices": [{"finish_reason": None, "message": {"content": "partial"}}]},
        {"choices": [{"finish_reason": "unknown", "message": {"content": "partial"}}]},
        {"choices": [{"finish_reason": "function_call", "message": {"content": "partial"}}]},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "partial", "function_call": {"name": "tool"}},
                }
            ]
        },
        {"output_text": "partial", "status": "incomplete"},
        {"output_text": "partial"},
        {"output_text": "partial", "status": "completed", "error": {"message": "bad"}},
        {"output_text": "partial", "status": "completed", "tool_calls": [{}]},
        {"output_text": "partial", "status": "completed", "output": [{"type": "function_call"}]},
    ],
)
def test_incomplete_or_tool_metadata_never_counts_as_success(value):
    with pytest.raises(ValueError):
        response_text(value)


def test_tiny_positive_relevance_has_stable_ndcg():
    value = record("rerank")
    value["relevance"] = {"d1": 1e-20, "d2": 0}
    item = {
        "family": "rerank",
        "record": normalize_record(value, "rerank", "native"),
        "ranking_k": 1,
    }
    assert score_task(item, '["d1", "d2"]')["primary_score"] == 1
    assert response_text({"output_text": "answer", "status": "completed"}) == "answer"


def test_smallest_subnormal_relevance_still_receives_correct_rank_discount():
    value = record("rerank")
    value["relevance"] = {"d1": 5e-324, "d2": 0}
    item = {
        "family": "rerank",
        "record": normalize_record(value, "rerank", "native"),
        "ranking_k": 2,
    }
    assert score_task(item, '["d2", "d1"]')["primary_score"] == pytest.approx(1 / math.log2(3))


def test_unrelated_database_is_rejected_before_any_schema_mutation(tmp_path):
    path = tmp_path / "user.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE important_data (value TEXT)")
        connection.execute("INSERT INTO important_data VALUES ('keep')")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="not a supported"):
        TaskRunStore(path)
    assert path.read_bytes() == original
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM important_data").fetchone() == ("keep",)


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_cli_protects_input_and_sqlite_sidecar_aliases(tmp_path, capsys, suffix, hardlink):
    path = manifest(tmp_path)
    value = json.loads(path.read_text())
    database = tmp_path / "protected.sqlite"
    sidecar = Path(str(database) + suffix)
    source = tmp_path / "rag.json"
    if hardlink:
        sidecar.hardlink_to(source)
    else:
        sidecar.write_bytes(source.read_bytes())
        value["tasks"][0]["path"] = sidecar.name
        path.write_text(json.dumps(value))
    original = sidecar.read_bytes()
    args = [
        "task-suite",
        "run",
        str(database),
        str(path),
        "--endpoint",
        "http://localhost:1",
        "--model",
        "unused",
    ]
    assert main(args) == 1
    assert "alias" in capsys.readouterr().err
    assert sidecar.read_bytes() == original


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_jsonl_unicode_line_separators_inside_strings_are_not_record_boundaries(
    tmp_path, separator
):
    path = manifest(tmp_path)
    value = json.loads(path.read_text())
    source = tmp_path / "data.jsonl"
    item = record()
    item["context"] = "Ada" + separator + "owns it."
    source.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    value["tasks"][0]["path"] = source.name
    path.write_text(json.dumps(value))
    assert load_task_suite(path).payload["items"][0]["record"]["context"] == item["context"]


@pytest.mark.parametrize(
    "field",
    [
        "family",
        "gold_prompt",
        "units",
        "skip",
        "budget",
        "record",
        "duplicate",
        "accounting",
        "missing_budget",
    ],
)
def test_public_plan_constructor_rejects_forged_inputs_before_run(tmp_path, field):
    payload = load_task_suite(manifest(tmp_path, budgets=(2000, 4000))).to_dict()
    item = payload["items"][0]
    if field == "family":
        item["family"] = "made-up"
    elif field == "gold_prompt":
        item["messages"][1]["content"] += "\nGold answer: Ada"
    elif field == "units":
        item["input_units"] = 1
    elif field == "skip":
        item["skip_reason"] = "skip_without_reason"
    elif field == "budget":
        item["budget"] = -1
    elif field == "record":
        item["record"]["answers"] = []
    elif field == "duplicate":
        payload["items"].append(item)
    elif field == "missing_budget":
        payload["items"].pop()
    else:
        payload["tasks"][0]["selected"] += 1
    with pytest.raises(ValueError):
        TaskSuitePlan(payload)


def test_numeric_score_changed_to_boolean_cannot_pass_integrity_check(tmp_path):
    plan = load_task_suite(manifest(tmp_path))
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.run(plan, Offline())
        row = store.connection.execute("SELECT result FROM task_state").fetchone()
        result = json.loads(row[0])
        result["score"]["primary_score"] = True
        store.connection.execute("UPDATE task_state SET result=?", (json.dumps(result),))
        with pytest.raises(ValueError, match="scoring inputs"):
            store.snapshot()


def test_schema_initialization_failure_is_atomic_and_retryable(tmp_path):
    path = tmp_path / "run.sqlite"

    class FailedInit(TaskRunStore):
        def _initialize(self):
            def authorizer(action, table, *_args):
                return (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_CREATE_TABLE and table == "task_state"
                    else sqlite3.SQLITE_OK
                )

            self.connection.set_authorizer(authorizer)
            super()._initialize()

    with pytest.raises(sqlite3.DatabaseError):
        FailedInit(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
    with TaskRunStore(path) as store:
        store.bind(load_task_suite(manifest(tmp_path)), {"provider": "test"})


def test_two_initializers_recheck_schema_under_one_transaction(tmp_path):
    path = tmp_path / "run.sqlite"
    barrier = threading.Barrier(2)

    class ConcurrentInit(TaskRunStore):
        def _pristine(self):
            result = super()._pristine()
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 1:
                barrier.wait(timeout=5)
            return result

    def initialize(_index):
        with ConcurrentInit(path) as store:
            store._validate_database()
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(initialize, range(2))) == [True, True]


def test_existing_database_validation_does_not_need_writer_lock(tmp_path):
    path = tmp_path / "run.sqlite"
    with TaskRunStore(path) as first:
        first.connection.execute("BEGIN IMMEDIATE")
        with TaskRunStore(path) as second:
            second._validate_database()
        first.connection.rollback()


def test_boolean_item_budget_cannot_alias_integer_budget(tmp_path):
    payload = load_task_suite(manifest(tmp_path, budgets=(1,))).to_dict()
    payload["items"][0]["budget"] = True
    with pytest.raises(ValueError, match="budget"):
        TaskSuitePlan(payload)


def test_reports_do_not_expose_mutable_global_capability_registry(tmp_path):
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        report = store.run(load_task_suite(manifest(tmp_path)), Offline())
        report["by_family"]["cite"]["metric_capabilities"]["unsupported"].clear()
        assert store.snapshot()["by_family"]["cite"]["metric_capabilities"]["unsupported"]


@pytest.mark.parametrize(
    "family,field,value",
    [
        ("rag", "documents", None),
        ("rag", "context", None),
        ("rag", "context", ""),
        ("rag", "answers", []),
        ("rag", "relevance", {}),
        ("rag", "examples", []),
        ("rag", "documents", [{"id": "x", "text": "text", "title": None}]),
        ("rag", "documents", [{"id": "x", "text": "text"}, {"id": "x", "text": "text"}]),
        ("rerank", "documents", []),
        ("rerank", "relevance", {"d1": 0, "d2": 0}),
        ("rerank", "relevance", {"d1": -1, "d2": 0}),
        ("rerank", "relevance", {"missing": 1}),
        ("icl", "labels", ["question", "question"]),
        ("icl", "examples", []),
        ("icl", "context", "ignored?"),
        ("icl", "examples", [{"id": "one", "text": "some", "label": "question"}]),
        ("icl", "examples", [{"id": "other", "text": "some", "label": "unsupported"}]),
        (
            "icl",
            "examples",
            [
                {"id": "e", "text": "some", "label": "question"},
                {"id": "e", "text": "more", "label": "question"},
            ],
        ),
    ],
)
def test_invalid_family_records_fail_before_prompt_construction(family, field, value):
    raw = record(family)
    raw[field] = value
    with pytest.raises(ValueError):
        normalize_record(raw, family, "native")


@pytest.mark.parametrize(
    "adapter,family", [("kilt", "recall"), ("msmarco", "rag"), ("processed", "icl")]
)
def test_adapter_cannot_silently_change_task_family(adapter, family):
    with pytest.raises(ValueError):
        normalize_record(record(), family, adapter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"counter_identity": {"unit": "tokens", "implementation": "test"}},
        {"counter": lambda messages: 1},
        {"counter": lambda messages: 1, "counter_identity": {}},
        {
            "counter": lambda messages: True,
            "counter_identity": {"unit": "tokens", "implementation": "test"},
        },
    ],
)
def test_custom_counter_requires_valid_count_and_identity(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_task_suite(manifest(tmp_path), **kwargs)
