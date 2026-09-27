"""Authored IPC/stage tests; not online, native optimizer or scientific results."""

from __future__ import annotations

import io
import json
import os
import subprocess
from dataclasses import asdict

import pytest

from promptwitness.incremental.features import ParentTrace
from reproduce import pipeline_controller as control
from reproduce import process_access as access
from reproduce import role_pipeline as roles
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.strict_scoring import ScoringError

UNFITTED = {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"}


def prompt(identifier, text="authored instruction"):
    return {
        "schema_version": 1,
        "id": identifier,
        "messages": [{"role": "system", "content": text, "id": "instruction"}],
    }


def response(unit, output="authored answer"):
    return {
        "id": unit,
        "output": output,
        "status": "completed",
        "replicate": "sample-0",
        "input_tokens": 12,
        "output_tokens": 3,
        "allocated_seconds": None,
    }


def execution():
    return {
        "model_revision": next(iter(MODEL_REVISIONS.values())),
        "tokenizer_revision": "authored-revision",
        "backend_version": "authored-backend",
        **{
            k: "a" * 64
            for k in (
                "backend_config_digest",
                "template_digest",
                "decoding_digest",
                "scorer_digest",
                "data_digest",
                "tool_environment_digest",
            )
        },
    }


def message(kind="score"):
    result = {
        "format": roles.FORMAT,
        "kind": kind,
        "family": "hotpotqa",
        "model": next(iter(MODEL_REVISIONS)),
        "execution": execution(),
        "candidate": prompt("candidate"),
    }
    if kind == "score":
        result["responses"] = [response("s")]
    else:
        result.update(
            parent=prompt("seed", "old instruction"),
            predictor=UNFITTED,
            structured=True,
            reference={"s": 1},
            parent_observations={},
        )
    return result


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    store = tmp_path / "store"
    for leaf in access.LEAVES:
        (store / leaf).mkdir(parents=True)
    for stage, unit in (("search", "s"), ("selection", "v")):
        write_jsonl(
            store / f"{stage}/inputs/hotpotqa.jsonl",
            [{"id": unit, "messages": [{"role": "user", "content": "authored context"}]}],
        )
        write_jsonl(
            store / f"{stage}/gold/hotpotqa.jsonl", [{"id": unit, "answer": "authored answer"}]
        )
    spec = {
        "run_id": "authored",
        "family": "hotpotqa",
        "model": next(iter(MODEL_REVISIONS)),
        "execution": execution(),
        "configuration": {"evaluator": "authored-full-route"},
        "seed_prompt": prompt("seed", "old instruction"),
        "search_ids": ["s"],
        "selection_ids": ["v"],
    }
    monkeypatch.setattr(control, "text_inventory", lambda path: {})
    monkeypatch.setattr(roles, "load_staged", lambda path: ({}, None, lambda a, b: a == b))
    dispatched = []

    def launch(role, stage, data, scratch, entry, arguments, *, message, timeout):
        # The request is already durably present before the application runs.
        events = [
            json.loads(line)
            for line in (scratch.parents[1] / "events.jsonl").read_text().splitlines()
        ]
        assert events[-1]["kind"] == "request" and events[-1]["message"] == message
        dispatched.append((role, stage, message))
        monkeypatch.setenv("PW_ACCESS_ROLE", role)
        monkeypatch.setenv("PW_ACCESS_STAGE", stage)
        monkeypatch.setenv("PW_LANDLOCK_ABI", "1")
        report = roles.application(data, {"bfcl": tmp_path, "text": tmp_path}, scratch, message)
        report["worker_pid"] = os.getpid() + 1  # authored transport only, not native process proof
        return subprocess.CompletedProcess([], 0, json.dumps(report), "")

    monkeypatch.setattr(control, "launch_role", launch)
    controller = control.PipelineController(
        store, tmp_path / "run", spec, {"bfcl": tmp_path, "text": tmp_path}
    )
    return controller, dispatched


@pytest.mark.parametrize("key,value", [("responses", []), ("current_score", 1), ("gold", {})])
def test_predictor_forbidden_application_fields(key, value):
    request = message("predict")
    request[key] = value
    with pytest.raises(ScoringError, match="forbidden"):
        roles.validate(request, "predictor", "search")


@pytest.mark.parametrize(
    "status,output", [("failed", "text"), ("missing", None), ("completed", None)]
)
def test_failed_or_missing_is_never_score_zero(status, output):
    request = message()
    request["responses"][0].update(status=status, output=output)
    with pytest.raises(ScoringError, match="completed"):
        roles.validate(request, "search_scorer", "search")


def test_duplicate_response_or_changed_model_rejected():
    request = message()
    request["responses"] *= 2
    with pytest.raises(ScoringError, match="duplicate"):
        roles.validate(request, "search_scorer", "search")
    request = message()
    request["execution"]["model_revision"] = "changed"
    with pytest.raises(ScoringError, match="revision"):
        roles.validate(request, "search_scorer", "search")


def test_stage_end_requires_actual_vectors_and_predictor_routes_only_parent(pipeline):
    controller, dispatched = pipeline
    with pytest.raises(ScoringError, match="reference_complete"):
        controller.freeze_candidate(prompt("seed"), prompt("candidate"), UNFITTED, structured=True)
    controller.reference([response("s")])
    frozen = controller.freeze_candidate(
        controller.spec["seed_prompt"], prompt("candidate"), UNFITTED, structured=True
    )
    before = control.copied(frozen)
    prediction = controller.predict("candidate")
    assert prediction["predictor_status"] == "UNFITTED"
    assert prediction["predictions"]["s"]["regression_probability"] is None
    assert "responses" not in dispatched[-1][2]
    assert dispatched[-1][2]["parent_observations"]["s"]["correct"] == 1
    with pytest.raises(ScoringError, match="search_ended"):
        controller.selection("candidate", [response("v")])
    with pytest.raises(ScoringError, match="score vector"):
        controller.end_search(["candidate"])
    controller.score_candidate("candidate", [response("s", "wrong answer")])
    assert frozen == before  # actual current outcomes cannot alter the freeze
    with pytest.raises(ScoringError, match="precede"):
        controller.predict("candidate")
    controller.freeze_candidate(
        prompt("candidate"), prompt("child", "new instruction"), UNFITTED, structured=True
    )
    controller.predict("child")
    assert dispatched[-1][2]["parent_observations"]["s"]["correct"] == 0
    assert dispatched[-1][2]["reference"]["s"] == 1  # reference does not follow ancestry
    controller.end_search(["candidate"])
    with pytest.raises(ScoringError, match="complete vectors"):
        controller.end_selection("candidate")
    controller.selection("candidate", [response("v")])
    receipt = controller.end_selection("candidate")
    assert receipt["final_admission"] == "NOT_GRANTED_BY_THIS_CONTROLLER"
    resumed = control.PipelineController.resume(
        controller.store, controller.directory, controller.runtimes
    )
    assert resumed.events == controller.events
    with pytest.raises(ScoringError, match="already ended"):
        resumed.score_candidate("child", [response("s")])


def test_full_population_not_shrunk_or_predicted(pipeline):
    controller, _ = pipeline
    with pytest.raises(ScoringError, match="complete actual"):
        controller.reference([])
    with pytest.raises(ScoringError, match="complete actual"):
        controller.reference([response("other")])
    assert not any(e["kind"] == "request" for e in controller.events)


def test_original_seed_survives_without_new_candidate_or_search_requery(pipeline):
    controller, dispatched = pipeline
    controller.reference([response("s")])
    controller.end_search(["seed"])
    assert len(dispatched) == 1
    assert not any(e["kind"] == "candidate_frozen" for e in controller.events)
    with pytest.raises(ScoringError, match="complete vectors"):
        controller.end_selection("seed")
    controller.selection("seed", [response("v")])
    receipt = controller.end_selection("seed")
    assert receipt["frozen_prompt"] == controller.spec["seed_prompt"]
    assert len(dispatched) == 2  # one reference, one actual selection vector
    assert dispatched[-1][2]["candidate"] == controller.spec["seed_prompt"]
    resumed = control.PipelineController.resume(
        controller.store, controller.directory, controller.runtimes
    )
    assert resumed._one("selection_ended") == receipt


def test_mixed_survivors_keep_native_seed_choice_and_require_every_actual_vector(pipeline):
    controller, dispatched = pipeline
    controller.reference([response("s")])
    with pytest.raises(ScoringError, match="fully scored"):
        controller.score_candidate("seed", [response("s", "replacement")])
    with pytest.raises(ScoringError, match="precede"):
        controller.predict("seed")
    with pytest.raises(ScoringError, match="distinct new"):
        controller.freeze_candidate(prompt("candidate"), prompt("seed"), UNFITTED, structured=True)
    controller.freeze_candidate(
        controller.spec["seed_prompt"], prompt("candidate"), UNFITTED, structured=True
    )
    controller.score_candidate("candidate", [response("s", "wrong answer")])
    controller.end_search(["seed", "candidate"])
    controller.selection("candidate", [response("v", "wrong answer")])
    with pytest.raises(ScoringError, match="complete vectors"):
        controller.end_selection("seed")
    resumed = control.PipelineController.resume(
        controller.store, controller.directory, controller.runtimes
    )
    resumed.selection("seed", [response("v")])
    receipt = resumed.end_selection("seed")
    assert receipt["frozen_prompt"] == controller.spec["seed_prompt"]
    assert receipt["configuration"] == controller.spec["configuration"]
    assert receipt["execution"] == controller.spec["execution"]
    assert len(dispatched) == 4
    assert resumed._candidate_scores("seed") == [resumed._one("reference_complete")["observations"]]


def test_reference_publication_recovery_uses_original_completed_attempt(pipeline, monkeypatch):
    controller, dispatched = pipeline
    record = controller._record

    def crash(kind, **payload):
        if kind == "reference_complete":
            raise OSError("authored publication/crash boundary")
        return record(kind, **payload)

    monkeypatch.setattr(controller, "_record", crash)
    with pytest.raises(OSError, match="crash"):
        controller.reference([response("s")])
    assert len(dispatched) == 1
    resumed = control.PipelineController.resume(
        controller.store, controller.directory, controller.runtimes
    )
    with pytest.raises(ScoringError, match="provenance"):
        resumed.reference([response("s", "changed")])
    resumed.reference([response("s")])
    assert len(dispatched) == 1
    assert resumed._one("reference_complete")["reference"] == {"s": 1}


def test_returned_freeze_is_detached_and_mutated_run_configuration_rejected(pipeline):
    controller, _ = pipeline
    controller.reference([response("s")])
    freeze = controller.freeze_candidate(
        controller.spec["seed_prompt"], prompt("candidate"), UNFITTED, structured=True
    )
    freeze["candidate"]["messages"][0]["content"] = "changed caller copy"
    assert controller._candidate("candidate")["candidate"] == prompt("candidate")
    controller.spec["configuration"]["evaluator"] = "changed run"
    with pytest.raises(ScoringError, match="configuration changed"):
        controller.score_candidate("candidate", [response("s")])


def test_failed_worker_logs_and_cpu_attempt_remain_not_stage_receipts(pipeline, monkeypatch):
    controller, _ = pipeline
    monkeypatch.setattr(
        control,
        "launch_role",
        lambda *a, **kw: subprocess.CompletedProcess([], 3, "partial", "authored failure"),
    )
    with pytest.raises(ScoringError, match="exit 3"):
        controller.reference([response("s")])
    assert controller.events[-1]["kind"] == "failed"
    assert controller.events[-1]["CPU_wall_seconds"] >= 0
    assert not any(e["kind"] == "reference_complete" for e in controller.events)
    assert (controller.directory / "workers/1/stderr").read_text() == "authored failure"


def test_unresolved_original_request_not_replayed_on_resume(pipeline):
    controller, _ = pipeline
    controller._record("request", attempt=99, message=message())
    with pytest.raises(ScoringError, match="unresolved"):
        control.PipelineController.resume(
            controller.store, controller.directory, controller.runtimes
        )


def test_timeout_retains_both_streams_and_failure(pipeline, monkeypatch):
    controller, _ = pipeline

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("authored", 900, output=b"partial", stderr=b"failure")

    monkeypatch.setattr(control, "launch_role", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        controller.reference([response("s")])
    assert (controller.directory / "workers/1/stdout").read_bytes() == b"partial"
    assert controller.events[-1]["kind"] == "failed"


def test_policy_line_consumed_but_message_unread_until_restriction(tmp_path, monkeypatch):
    for leaf in access.LEAVES:
        (tmp_path / "store" / leaf).mkdir(parents=True)
    code, scratch = tmp_path / "code", tmp_path / "scratch"
    code.mkdir()
    scratch.mkdir()
    entry = code / "entry.py"
    entry.write_text("pass")
    policy = access.build_policy("search_scorer", "search", tmp_path / "store", [code], scratch)
    policy.update(entrypoint=str(entry), arguments=[])
    stream = io.StringIO(json.dumps(policy) + "\n" + json.dumps(message()))
    monkeypatch.setattr(access.sys, "stdin", stream)
    monkeypatch.setattr(access.os, "listdir", lambda p: ["only-thread"])
    monkeypatch.setattr(access.os, "chdir", lambda p: None)
    restricted = []

    def enforce(policy):
        assert stream.getvalue()[stream.tell() :].startswith('{"format"')
        restricted.append(True)
        return 1

    def run(path, **kwargs):
        assert restricted == [True]
        assert json.load(stream) == message()

    monkeypatch.setattr(access, "enforce_policy", enforce)
    monkeypatch.setattr(access.runpy, "run_path", run)
    # Avoid changing the real imported package in this authored worker fixture.
    monkeypatch.setattr(access, "__file__", str(code / "process_access.py"))
    monkeypatch.setitem(access.sys.modules, "reproduce", access.sys.modules["reproduce"])
    access.worker()


def test_launch_serializes_application_on_separate_line(tmp_path, monkeypatch):
    for leaf in access.LEAVES:
        (tmp_path / "store" / leaf).mkdir(parents=True)
    code, scratch = tmp_path / "code", tmp_path / "scratch"
    code.mkdir()
    scratch.mkdir()
    entry = code / "entry.py"
    entry.write_text("pass")
    monkeypatch.setattr(access, "runtime_roots", lambda: (code,))
    seen = []
    monkeypatch.setattr(
        access.subprocess,
        "run",
        lambda args, **kw: seen.append(kw["input"]) or subprocess.CompletedProcess(args, 0, "", ""),
    )
    access.launch_role(
        "search_scorer", "search", tmp_path / "store", scratch, entry, message=message()
    )
    policy, application = seen[0].splitlines()
    assert json.loads(policy)["role"] == "search_scorer"
    assert json.loads(application) == message()
    assert "responses" not in json.loads(policy)


def test_parent_trace_without_current_labels_and_missingness(pipeline):
    controller, _ = pipeline
    request = message("predict")
    result = roles.predict(request, controller.store)
    assert result["predictor_status"] == "UNFITTED"
    assert result["predictions"]["s"]["model_digest"] is None
    request["parent_observations"]["s"] = {**asdict(ParentTrace()), "new_correct": 1}
    with pytest.raises(ScoringError, match="forbidden"):
        roles.validate(request, "predictor", "search")
