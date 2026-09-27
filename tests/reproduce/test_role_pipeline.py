"""Authored IPC/stage tests; not online, native optimizer or scientific results."""

from __future__ import annotations

import io
import json
import os
import subprocess
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from promptwitness.incremental.budget import ResourceLedger, ResourceLimit
from promptwitness.incremental.features import ParentTrace, extract_features
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.predictor import TrainingRow, TransitionPredictor
from promptwitness.parser import parse_prompt
from reproduce import pipeline_controller as control
from reproduce import process_access as access
from reproduce import real_pipeline as real
from reproduce import role_pipeline as roles
from reproduce.incremental_pipeline import IncrementalPipeline
from reproduce.persistent_model import PersistentModel, PhysicalCallFailure
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.prepare_preflight_requests import SEEDS
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


def make_pipeline(tmp_path, monkeypatch, population=1):
    store = tmp_path / "store"
    for leaf in access.LEAVES:
        (store / leaf).mkdir(parents=True)
    search = ["s"] if population == 1 else [f"s{i}" for i in range(population)]
    for stage, units in (("search", search), ("selection", ["v"])):
        write_jsonl(
            store / f"{stage}/inputs/hotpotqa.jsonl",
            [
                {"id": unit, "messages": [{"role": "user", "content": "authored context"}]}
                for unit in units
            ],
        )
        write_jsonl(
            store / f"{stage}/gold/hotpotqa.jsonl",
            [{"id": unit, "answer": "authored answer"} for unit in units],
        )
    spec = {
        "run_id": "authored",
        "family": "hotpotqa",
        "model": next(iter(MODEL_REVISIONS)),
        "execution": execution(),
        "configuration": {"evaluator": "authored-full-route", "max_evaluation_episodes": 2048},
        "seed_prompt": prompt("seed", "old instruction"),
        "search_ids": search,
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


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    return make_pipeline(tmp_path, monkeypatch)


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


def incremental(controller, tmp_path, predictor=UNFITTED):
    # Authored transport costs only; the exact user ceilings are not reduced.
    limit = ResourceLimit(800000, 2000000000, 200000000, 1000)
    ledger = ResourceLedger(
        tmp_path / "resources.sqlite",
        historical_usage={
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "gpu_hours": 0,
        },
        historical_digest="a" * 64,
        global_limit=limit,
        stage_limits={"authored": limit},
        gpu_uuid="GPU-authored-no-allocation",
    )
    controller.freeze_candidate(
        controller.spec["seed_prompt"], prompt("candidate"), predictor, structured=True
    )
    route = IncrementalPipeline(
        controller, ledger, resource_stage="authored", input_cap=32768, output_cap=64
    )
    return route, ledger


def freeze_audit(route, *, scope="FROZEN_TABLE", use_risk=False):
    return route.freeze(
        "candidate",
        replicates={u: "sample-0" for u in route.controller.spec["search_ids"]},
        execution_scope=scope,
        fixture_seed=7,
        use_risk=use_risk,
    )


def test_incremental_eligible_completes_only_unobserved_and_keeps_native_seed(
    tmp_path, monkeypatch
):
    controller, dispatched = make_pipeline(tmp_path, monkeypatch, 64)
    controller.reference([response(u, "wrong answer") for u in controller.spec["search_ids"]])
    route, ledger = incremental(controller, tmp_path)
    plan = freeze_audit(route)
    queried = []

    def execute(request):
        assert controller.events[-1]["kind"] == "generation_request"
        assert ledger.usage()["calls"] == len(queried) + 1  # reserve before actual invocation
        assert "gold" not in request and "reference" not in request
        assert request["candidate"] == prompt("candidate")
        queried.append(request["unit"]["id"])
        return response(queried[-1])

    result = route.evaluate("candidate", execute)
    assert result.status == GateStatus.ELIGIBLE
    assert result.certificate_scope == "MECHANICAL_SEEDED_FIXTURE"
    expected = [
        u for s, n in zip(plan.strata, plan.allocations[0], strict=True) for u in s.permutation[:n]
    ]
    assert queried == expected and len(queried) < 64
    with pytest.raises(ScoringError, match="score vector"):
        controller.end_search(["candidate"])
    with pytest.raises(ScoringError, match="precede"):
        controller.predict("candidate")
    vector = route.complete_survivor("candidate", execute)
    assert vector.unit_ids == tuple(controller.spec["search_ids"]) and vector.scores == (1,) * 64
    assert len(queried) == len(set(queried)) == 64
    assert route.journal.consumed_episodes() == 128  # original reference + all actual queries
    assert ledger.usage()["calls"] == 64 and ledger.usage()["input_tokens"] == 768
    assert ledger.usage()["gpu_hours"] == 0  # no fabricated allocation from scoring CPU walltime
    controller.end_search(["seed", "candidate"])
    controller.selection("seed", [response("v")])
    controller.selection("candidate", [response("v")])
    assert controller.end_selection("seed")["frozen_prompt"] == controller.spec["seed_prompt"]
    assert len(dispatched) == 67  # reference + 64 actual units + two selection vectors
    route.close()
    ledger.close()


def test_incremental_early_rejection_leaves_unknowns_and_partial_parent(tmp_path, monkeypatch):
    controller, _ = make_pipeline(tmp_path, monkeypatch, 64)
    controller.reference([response(u) for u in controller.spec["search_ids"]])
    route, ledger = incremental(controller, tmp_path)
    plan = freeze_audit(route)
    seen = []

    def execute(request):
        seen.append(request["unit"]["id"])
        return response(seen[-1], "wrong answer")

    result = route.evaluate("candidate", execute)
    assert result.status == GateStatus.INELIGIBLE and len(seen) < 64
    assert set(route.journal.observations(plan)) == set(seen)
    assert route.journal.consumed_episodes() == 64 + len(seen)
    with pytest.raises(ScoringError, match="eligible"):
        route.complete_survivor("candidate", execute)
    controller.freeze_candidate(prompt("candidate"), prompt("child"), UNFITTED, structured=True)
    controller.predict("child")
    request = controller.events[-2]["message"]
    assert set(request["parent_observations"]) == set(seen)
    assert len(request["reference"]) == 64  # fixed reference, not queried parent subset
    controller.end_search(["seed"])
    route.close()
    ledger.close()


@pytest.mark.parametrize(
    "failure", ["exception", "unknown_tokens", "wrong_unit", "overrun", "scorer", "physical_known"]
)
def test_incremental_failure_charges_no_zero_no_automatic_replay(
    pipeline, tmp_path, monkeypatch, failure
):
    controller, _ = pipeline
    controller.reference([response("s")])
    route, ledger = incremental(controller, tmp_path)
    plan = freeze_audit(route)
    seen = []
    if failure == "scorer":
        monkeypatch.setattr(
            control,
            "launch_role",
            lambda *a, **kw: subprocess.CompletedProcess([], 2, "", "scorer failed"),
        )

    def execute(request):
        seen.append(request)
        if failure == "exception":
            raise RuntimeError("authored failure")
        if failure == "physical_known":
            raise PhysicalCallFailure(
                "known received cost, bad transport identity", response("other")
            )
        row = response("other" if failure == "wrong_unit" else "s")
        if failure == "unknown_tokens":
            row.update(input_tokens=None, output_tokens=None)
        if failure == "overrun":
            row["output_tokens"] = 65
        return row

    result = route.evaluate("candidate", execute)
    assert result.status == GateStatus.INCONCLUSIVE and len(seen) == 1
    assert route.journal.observations(plan) == {}
    assert route.journal.consumed_episodes() == 2 and ledger.usage()["calls"] == 1
    if failure in ("exception", "unknown_tokens"):
        assert ledger.usage()["calls_with_reserved_or_unknown_token_cost"] == 1
        assert ledger.usage()["input_tokens"] == 32768
    else:
        assert ledger.usage()["calls_with_reserved_or_unknown_token_cost"] == 0
        assert ledger.usage()["output_tokens"] == (65 if failure == "overrun" else 3)
    assert route.evaluate("candidate", execute).status == GateStatus.INCONCLUSIVE
    assert len(seen) == 1  # unchanged failed random unit is not regenerated
    control.PipelineController.resume(controller.store, controller.directory, controller.runtimes)
    route.close()
    ledger.close()


def test_transport_budget_error_swallowed_by_actual_gate_still_stops_child(
    pipeline, tmp_path, monkeypatch
):
    controller, _ = pipeline
    controller.reference([response("s")])
    route, ledger = incremental(controller, tmp_path)
    plan = freeze_audit(route)
    process = PersistentModel.__new__(PersistentModel)
    process.directory = tmp_path / "AUTHORED_MODEL"
    process.directory.mkdir()
    process.ledger, process.allocation_id = ledger, "AUTHORED_ALLOCATION"
    process.profile = {"backend_version": execution()["backend_version"]}
    process.started, process.failed, process.stderr, process.old_sigterm = True, False, None, None
    ledger.begin_gpu(process.allocation_id, "authored", "GPU-authored-no-allocation", start=100)
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)
    child = Mock(stdin=io.StringIO(), stdout=io.StringIO())
    child.poll.return_value = None

    def waited(**_):
        child.poll.return_value = 0
        return 0

    child.wait.side_effect = waited
    process.process = child

    def interrupted():
        raise RuntimeError("AUTHORED GPU budget interruption")

    monkeypatch.setattr(process, "_read", interrupted)

    def execute(request):
        # Literal authored wire, not a real corpus/model qualification.
        return process._execute(
            request["execution"],
            {"id": request["unit"]["id"], "replicate": request["replicate"]},
        )

    result = route.evaluate("candidate", execute)
    assert result.status == GateStatus.INCONCLUSIVE
    assert route.journal.observations(plan) == {}
    assert child.terminate.call_count == child.wait.call_count == 1
    assert process.failed and not process.started
    assert ledger.usage()["calls"] == 1
    assert ledger.usage()["calls_with_reserved_or_unknown_token_cost"] == 1
    process.__exit__(None, None, None)
    assert child.wait.call_count == 1
    assert ledger.connection.execute("SELECT end FROM gpu_allocations").fetchone()[0] is not None
    route.close()
    ledger.close()


def test_audit_freeze_unsupported_unfitted_resume_and_changed_random_units(pipeline, tmp_path):
    controller, _ = pipeline
    controller.reference([response("s")])
    route, ledger = incremental(controller, tmp_path)
    controller.predict("candidate")
    with pytest.raises(ScoringError, match="UNFITTED"):
        freeze_audit(route, use_risk=True)
    plan = freeze_audit(route, scope="UNVERIFIED")
    assert (
        route.evaluate(
            "candidate", lambda request: pytest.fail("unsupported must not execute")
        ).status
        == GateStatus.UNSUPPORTED
    )
    assert ledger.usage()["calls"] == 0
    route.close()
    resumed = control.PipelineController.resume(
        controller.store, controller.directory, controller.runtimes
    )
    again = IncrementalPipeline(
        resumed, ledger, resource_stage="authored", input_cap=32768, output_cap=64
    )
    assert freeze_audit(again, scope="UNVERIFIED") == plan
    with pytest.raises(ScoringError, match="changed"):
        again.freeze(
            "candidate", replicates={"s": "sample-1"}, execution_scope="UNVERIFIED", fixture_seed=7
        )
    again.close()
    ledger.close()


def test_unresolved_generation_or_logical_episode_remains_fail_closed(
    pipeline, tmp_path, monkeypatch
):
    controller, _ = pipeline
    controller.reference([response("s")])
    route, ledger = incremental(controller, tmp_path)
    plan = freeze_audit(route)
    route.journal.reserve(plan, "s")
    result = route.evaluate("candidate", lambda r: pytest.fail("no replay"))
    assert result.status == GateStatus.INCONCLUSIVE and route.journal.consumed_episodes() == 2
    controller._record("generation_request", identifier="candidate", unit="s", call_id="unresolved")
    with pytest.raises(ScoringError, match="unresolved generation"):
        control.PipelineController.resume(
            controller.store, controller.directory, controller.runtimes
        )
    route.close()
    ledger.close()


def test_incremental_does_not_silently_replace_original_budget(pipeline, tmp_path):
    controller, _ = pipeline
    controller.reference([response("s")])
    controller.spec["configuration"]["max_evaluation_episodes"] = 200
    with pytest.raises(ScoringError, match="2048"):
        IncrementalPipeline(
            controller, None, resource_stage="authored", input_cap=32768, output_cap=64
        )


@pytest.mark.parametrize("old_correct", [0, 1])
def test_risk_uses_available_reference_transition_head(pipeline, tmp_path, old_correct):
    pytest.importorskip("sklearn")
    controller, _ = pipeline
    controller.reference([response("s", "authored answer" if old_correct else "wrong answer")])
    model = TransitionPredictor()
    model.fit(
        (
            TrainingRow(
                features=extract_features(
                    parse_prompt(controller.spec["seed_prompt"]),
                    parse_prompt(prompt("candidate")),
                    "authored input",
                ),
                old_correct=old_correct,
                new_correct=1 - old_correct,
                lineage="authored",
                source_group="authored",
                split="fit",
                label_source_digest="a" * 64,
                kind="synthetic",
            ),
        )
    )
    route, ledger = incremental(controller, tmp_path, model.to_dict())
    prediction = controller.predict("candidate")["predictions"]["s"]
    relevant = "regression_probability" if old_correct else "improvement_probability"
    irrelevant = "improvement_probability" if old_correct else "regression_probability"
    assert prediction[relevant] is not None and prediction[irrelevant] is None
    plan = freeze_audit(route, use_risk=True)
    assert plan.population == 1 and plan.strata[0].old_correct == old_correct
    route.close()
    ledger.close()


def actual_vector_fixture(tmp_path, monkeypatch, population=2):
    """Authored executor, actual controller/scorer route; no GPU qualification."""
    controller, dispatched = make_pipeline(tmp_path, monkeypatch, population)
    monkeypatch.setattr(
        real,
        "execution_bindings",
        lambda *args: {
            key: controller.spec["execution"][key]
            for key in ("data_digest", "scorer_digest", "tool_environment_digest")
        },
    )
    original_rows = roles.rows

    def normalized(store, leaf, family):
        return {
            unit: {
                "id": unit,
                "family": family,
                "messages": [
                    {"role": "system", "content": SEEDS[family]},
                    {"role": "user", "content": "all authored context"},
                ],
            }
            for unit in original_rows(store, leaf, family)
        }

    monkeypatch.setattr(real, "rows", normalized)
    model = Mock()
    model.model, model.data_root = controller.spec["model"], controller.store.resolve()
    model.profile = {
        key: value
        for key, value in controller.spec["execution"].items()
        if key not in {"data_digest", "scorer_digest", "tool_environment_digest"}
    }
    queried = []

    def metered(call_id, request, *, purpose):
        assert controller.events[-1]["kind"] == "generation_request"
        assert controller.events[-1]["call_id"] == call_id
        frozen = [
            event
            for event in controller.events
            if event["kind"] == "vector_frozen" and event["purpose"] == purpose
        ]
        assert len(frozen) == 1  # every request was persisted before any response
        assert all("gold" not in r["unit"] for r in frozen[0]["requests"])
        queried.append((purpose, copied(request)))
        return {**response(request["unit"]["id"]), "replicate": request["replicate"]}

    from reproduce.pipeline_controller import copied

    model.metered_task.side_effect = metered
    route = real.RealPipeline(controller, model)
    return route, model, queried, dispatched


def test_real_reference_complete_vector_recovery_and_original_seed_selection(tmp_path, monkeypatch):
    route, model, queried, dispatched = actual_vector_fixture(tmp_path, monkeypatch)
    replicates = {u: "fixed-ref-" + u for u in route.controller.spec["search_ids"]}
    result = route.reference(replicates)
    assert result["reference"] == {"s0": 1, "s1": 1}
    assert len(queried) == 2 and len(dispatched) == 1
    assert route.reference(replicates) == result
    assert len(queried) == 2  # no replay and no extra physical reserve on recovery
    route.controller.end_search(["seed"])
    route.selection("seed", {"v": "fixed-selection-v"})
    receipt = route.controller.end_selection("seed")
    assert receipt["frozen_prompt"] == route.controller.spec["seed_prompt"]
    assert receipt["final_admission"] == "NOT_GRANTED_BY_THIS_CONTROLLER"
    assert queried[-1][0] == "selection" and len(queried) == 3
    assert model.metered_task.call_count == 3
    assert len(dispatched) == 2


@pytest.mark.parametrize("key", ["model", "data_root", "profile"])
def test_real_model_binding_rejects_before_any_call(tmp_path, monkeypatch, key):
    route, model, queried, _ = actual_vector_fixture(tmp_path, monkeypatch)
    setattr(model, key, None)
    with pytest.raises(ScoringError, match="original run freeze"):
        real.RealPipeline(route.controller, model)
    assert not queried


@pytest.mark.parametrize("mapping", [{}, {"s0": "r"}, {"s0": "", "s1": "r"}])
def test_real_complete_random_unit_mapping_required(tmp_path, monkeypatch, mapping):
    route, _, queried, _ = actual_vector_fixture(tmp_path, monkeypatch)
    with pytest.raises(ScoringError, match="random-unit mapping"):
        route.reference(mapping)
    assert not queried
    assert not any(e["kind"] == "vector_frozen" for e in route.controller.events)


def test_real_failed_request_keeps_observed_response_and_cannot_replay(tmp_path, monkeypatch):
    route, model, queried, _ = actual_vector_fixture(tmp_path, monkeypatch)
    successful = model.metered_task.side_effect
    physical = {**response("s1"), "replicate": "r"}

    def fail_second(call_id, request, *, purpose):
        if request["unit"]["id"] == "s1":
            raise PhysicalCallFailure("actual abort witness", physical)
        return successful(call_id, request, purpose=purpose)

    model.metered_task.side_effect = fail_second
    mapping = {u: "r" for u in route.controller.spec["search_ids"]}
    with pytest.raises(PhysicalCallFailure):
        route.reference(mapping)
    failure = route.controller.events[-1]
    assert failure["kind"] == "generation_failed" and failure["response"] == physical
    assert model.metered_task.call_count == 2 and len(queried) == 1
    with pytest.raises(ScoringError, match="do not replay"):
        route.reference(mapping)
    assert model.metered_task.call_count == 2
    assert not any(e["kind"] == "reference_complete" for e in route.controller.events)


def test_real_changed_replicate_or_pending_request_not_restarted(tmp_path, monkeypatch):
    route, model, _, _ = actual_vector_fixture(tmp_path, monkeypatch)
    mapping = {u: "r" for u in route.controller.spec["search_ids"]}
    route.reference(mapping)
    with pytest.raises(ScoringError, match="freeze changed"):
        route.reference({u: "changed" for u in mapping})
    assert model.metered_task.call_count == 2
    route.controller._record(
        "generation_request", call_id="unresolved", identifier="seed", unit="s0", request={}
    )
    with pytest.raises(ScoringError, match="unresolved generation"):
        real.RealPipeline(route.controller, model)
    assert model.metered_task.call_count == 2


def test_real_all_inputs_preflight_before_first_request(tmp_path, monkeypatch):
    route, model, _, _ = actual_vector_fixture(tmp_path, monkeypatch)
    valid = real.rows

    def contaminated(*args):
        inputs = valid(*args)
        inputs["s1"]["gold"] = "forbidden"
        return inputs

    monkeypatch.setattr(real, "rows", contaminated)
    with pytest.raises(ValueError, match="annotation-bearing"):
        route.reference({u: "r" for u in route.controller.spec["search_ids"]})
    assert model.metered_task.call_count == 0


def test_real_selection_requires_search_ended_and_survivor(tmp_path, monkeypatch):
    route, model, _, _ = actual_vector_fixture(tmp_path, monkeypatch)
    with pytest.raises(ScoringError, match="search_ended"):
        route.selection("seed", {"v": "r"})
    route.reference({u: "r" for u in route.controller.spec["search_ids"]})
    route.controller.end_search(["seed"])
    with pytest.raises(ScoringError, match="frozen search survivor"):
        route.selection("unselected", {"v": "r"})
    assert model.metered_task.call_count == 2


def test_real_incremental_reuses_exact_model_ledger_and_single_reservation_callback(
    tmp_path, monkeypatch
):
    route, model, _, _ = actual_vector_fixture(tmp_path, monkeypatch)
    constructor = Mock(return_value="existing-driver")
    monkeypatch.setattr(real, "IncrementalPipeline", constructor)
    driver, execute = route.incremental()
    assert driver == "existing-driver" and execute == model.task
    constructor.assert_called_once_with(
        route.controller,
        model.ledger,
        resource_stage=real.STAGE,
        input_cap=32768,
        output_cap=64,
    )
    model.metered_task.assert_not_called()


@pytest.mark.parametrize("family", ["bfcl", "hotpotqa", "instruction_following"])
def test_real_identity_uses_training_and_actual_scorer_bytes_never_final(
    tmp_path, monkeypatch, family
):
    store, runtime = tmp_path / "store", tmp_path / "runtime"
    for pool in ("fit", "search", "selection"):
        for leaf in ("inputs", "gold"):
            target = store / pool / leaf / (family + ".jsonl")
            target.parent.mkdir(parents=True)
            target.write_bytes(b"authored training bytes")
    runtime.mkdir()
    (runtime / "code.py").write_bytes(b"authored scorer code")
    (runtime / "resource").write_bytes(b"authored resource")
    monkeypatch.setattr(real, "bfcl_inventory", lambda p: {"files": ["code.py"]})
    monkeypatch.setattr(
        real,
        "text_inventory",
        lambda p: {
            "code_files": ["code.py"],
            "resource_files": ["resource"],
        },
    )
    original = real.execution_bindings(store, family, {"bfcl": runtime, "text": runtime})
    assert all(len(value) == 64 for value in original.values())
    (store / "search/gold" / (family + ".jsonl")).write_bytes(b"changed training bytes")
    changed = real.execution_bindings(store, family, {"bfcl": runtime, "text": runtime})
    assert changed["data_digest"] != original["data_digest"]
    assert changed["scorer_digest"] == original["scorer_digest"]
    (runtime / "code.py").write_bytes(b"changed scorer code")
    assert (
        real.execution_bindings(store, family, {"bfcl": runtime, "text": runtime})["scorer_digest"]
        != changed["scorer_digest"]
    )
    assert not (store / "final").exists()


def test_original_reference_cpu_recovery_keeps_failed_worker_without_generation(
    tmp_path, monkeypatch
):
    from reproduce.recover_reference import original_responses

    route, model, _, dispatched = actual_vector_fixture(tmp_path, monkeypatch)
    launcher = control.launch_role
    monkeypatch.setattr(
        control,
        "launch_role",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "denied runtime"),
    )
    with pytest.raises(ScoringError, match="worker exit"):
        route.reference({u: "fixed" for u in route.controller.spec["search_ids"]})
    assert model.metered_task.call_count == 2
    monkeypatch.setattr(control, "launch_role", launcher)
    result = route.controller.reference(original_responses(route.controller))
    assert result["reference"] == {"s0": 1, "s1": 1}
    assert model.metered_task.call_count == 2 and len(dispatched) == 1
    assert len([e for e in route.controller.events if e["kind"] == "failed"]) == 1


@pytest.mark.parametrize("defect", ["generation_failed", "changed_random_unit", "partial_vector"])
def test_original_reference_cpu_recovery_never_replays_or_fills(tmp_path, monkeypatch, defect):
    from reproduce.recover_reference import original_responses

    route, model, _, dispatched = actual_vector_fixture(tmp_path, monkeypatch)
    route.reference({u: "fixed" for u in route.controller.spec["search_ids"]})
    if defect == "generation_failed":
        event = next(e for e in route.controller.events if e["kind"] == "generation_result")
        event["kind"] = "generation_failed"
    elif defect == "changed_random_unit":
        event = next(e for e in route.controller.events if e["kind"] == "generation_result")
        event["response"]["replicate"] = "changed"
    else:
        next(e for e in route.controller.events if e["kind"] == "vector_frozen")["requests"].pop()
    with pytest.raises(ScoringError):
        original_responses(route.controller)
    assert model.metered_task.call_count == 2 and len(dispatched) == 1


@pytest.mark.parametrize("case", ["eligible", "rejected", "failed", "unsupported", "wrong_prompt"])
def test_mipro_gate_connects_original_driver_without_zero_or_replay(tmp_path, monkeypatch, case):
    import sys
    from types import ModuleType, SimpleNamespace

    from reproduce.mipro_search import MIPROGateEvaluator

    class Result:
        def __init__(self, score, results):
            self.score, self.results = score, results

    class Pruned(Exception):
        pass

    dspy, evaluate_module, optuna = (
        ModuleType("dspy"),
        ModuleType("dspy.evaluate.evaluate"),
        ModuleType("optuna"),
    )
    dspy.Prediction = lambda **kwargs: SimpleNamespace(**kwargs)
    evaluate_module.EvaluationResult, optuna.TrialPruned = Result, Pruned
    for module in (dspy, evaluate_module, optuna):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    controller, _ = make_pipeline(tmp_path, monkeypatch, 64)
    controller.reference([response(u) for u in controller.spec["search_ids"]])
    route, ledger = incremental(controller, tmp_path)
    freeze_audit(route, scope="UNVERIFIED" if case == "unsupported" else "FROZEN_TABLE")
    queried = []

    def execute(request):
        queried.append(request["unit"]["id"])
        if case == "failed":
            raise RuntimeError("authored execution failed")
        return response(queried[-1], "wrong" if case == "rejected" else "authored answer")

    predictor = SimpleNamespace(
        signature=SimpleNamespace(
            instructions="wrong" if case == "wrong_prompt" else "authored instruction",
            input_fields={"task_input": None},
            output_fields={"task_output": None},
        ),
        demos=[],
    )
    program = SimpleNamespace(predictors=lambda: [predictor])
    native = MIPROGateEvaluator(route, execute, lambda p: "candidate")
    batch = [{"unit_id": controller.spec["search_ids"][0]}]
    try:
        if case == "eligible":
            result = native(
                program, devset=batch, callback_metadata={"metric_key": "eval_minibatch"}
            )
            assert result.score == 100 and len(result.results) == 1
            assert result.results[0][1].task_output == "authored answer"
            assert len(queried) == ledger.usage()["calls"] == 64
            assert set(controller._candidate_scores("candidate")[0]) == set(
                controller.spec["search_ids"]
            )
            native(program, devset=batch, callback_metadata={"metric_key": "eval_full"})
            assert len(queried) == 64  # complete original census reused, not free invented scores
        else:
            with pytest.raises(Pruned if case == "rejected" else ScoringError):
                native(program, devset=batch, callback_metadata={"metric_key": "eval_full"})
            assert len(queried) == {"rejected": 4, "failed": 1}.get(case, 0)
            assert not any(
                e["kind"] == "native_mipro_evaluation_complete" for e in controller.events
            )
            if case == "rejected":
                assert any(e["kind"] == "native_mipro_pruned" for e in controller.events)
            if case == "failed":
                assert ledger.usage()["calls"] == 1
                assert any(e["kind"] == "generation_failed" for e in controller.events)
    finally:
        route.close()
        ledger.close()
