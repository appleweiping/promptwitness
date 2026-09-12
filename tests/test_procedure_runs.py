"""Versioned procedure execution without providers, paid calls or code execution."""

import json
import sqlite3
from typing import ClassVar

import pytest
from test_procedure_plan import ANSWER, authored_case, small_plan

from promptwitness.procedure_plan import build_procedure_plan
from promptwitness.procedure_scores import ProcedureScoreLimitError, ProcedureScoreLimits
from promptwitness.task_data import digest
from promptwitness.task_runs import TaskRunConflict, TaskRunStore, _validated_plan


class OfflineProcedure:
    identity: ClassVar = {"provider": "offline-author-oracle", "model": "fixed-v1"}

    def __init__(self, response=ANSWER):
        self.response = response
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.response


def test_procedure_report_denominators_generation_and_resume(tmp_path):
    plan = build_procedure_plan(
        [
            {"id": "math", "cases": [authored_case()], "generation": {"max_tokens": 91}},
            {
                "id": "code",
                "cases": [authored_case(code=True, bucket="0.5k")],
                "generation": {"max_tokens": 1024},
            },
        ],
        suite_id="two-task",
        budgets=[1, 4096],
    )
    provider = OfflineProcedure()
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as store:
        partial = store.run(plan, provider, max_cases=1)
        assert partial["coverage"]["succeeded"] == 1
    with TaskRunStore(database) as store:
        report = store.run(plan, provider)
        assert store.run(plan, provider) == report
        assert len(provider.requests) == 2
    assert report["format"] == "promptwitness.procedure-report/v1"
    assert "seven_family_execution_complete" not in report
    coverage = report["coverage"]
    assert coverage["planned"] == 4 and coverage["skipped"] == 2
    assert coverage["succeeded"] == 2 and coverage["scored"] == 1
    assert coverage["score_coverage"] == 0.25
    assert coverage["unsupported_primary"] == 1
    assert "mean_primary_score" not in coverage
    assert report["by_family"]["countdown"]["mean_primary_score"] == 1
    assert not report["by_family"]["path_traversal"]["configured"]
    assert report["by_budget_family"]["1"]["countdown"]["skipped"] == 1
    assert report["by_output_bucket_family"]["0.5k"]["pseudo_to_code"]["planned"] == 2
    assert not report["six_family_execution_complete"]
    assert not report["six_family_scoring_complete"]
    assert [request.generation["max_tokens"] for request in provider.requests] == [91, 1024]
    assert not any("GOLD_ONLY" in json.dumps(request.to_dict()) for request in provider.requests)


def test_interrupted_reservation_cas_explicit_retry_and_completed_replay(tmp_path):
    plan, provider = small_plan(), OfflineProcedure()
    database = tmp_path / "run.sqlite"
    with TaskRunStore(database) as first, TaskRunStore(database) as second:
        first.bind(plan, provider.identity)
        item = first.snapshot()["results"][0]
        request, revision = first.reserve(item["item_id"], expected_revision=0)
        assert second.snapshot()["results"][0]["status"] == "running"
        with pytest.raises(TaskRunConflict):
            second.reserve(item["item_id"], expected_revision=0)
    with TaskRunStore(database) as store:
        assert store.run(plan, provider)["coverage"]["running"] == 1
        assert not provider.requests
        store.retry(item["item_id"], expected_revision=revision)
        with pytest.raises(TaskRunConflict):
            store.complete(item["item_id"], ANSWER, expected_revision=revision)
        report = store.run(plan, provider)
        assert len(provider.requests) == 1
        saved = report["results"][0]
        assert saved["status"] == "succeeded" and saved["attempts"] == 2
        assert saved["result"]["request_digest"] == request.digest
        assert store.run(plan, provider) == report


def test_incomplete_response_and_scoring_budget_are_failures_not_zero_scores(tmp_path):
    plan = small_plan(score_limits=ProcedureScoreLimits(max_prediction_bytes=10))
    provider = OfflineProcedure(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "SENSITIVE_PARTIAL"},
                }
            ]
        }
    )
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        row = store.run(plan, provider)["results"][0]
        assert row["error"] == "ValueError" and row["result"] is None
        provider.response = ANSWER
        assert store.run(plan, provider)["coverage"]["failed"] == 1
        store.retry(row["item_id"], expected_revision=row["revision"])
        row = store.run(plan, provider)["results"][0]
        assert row["error"] == "ProcedureScoreLimitError" and row["result"] is None
        assert store.snapshot()["coverage"]["scored"] == 0
    assert b"SENSITIVE_PARTIAL" not in (tmp_path / "run.sqlite").read_bytes()


def test_prediction_limit_before_writer_and_no_mutation_on_score_error(tmp_path):
    plan = small_plan(score_limits=ProcedureScoreLimits(max_prediction_bytes=10))
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.bind(plan, OfflineProcedure.identity)
        identifier = plan.payload["items"][0]["id"]
        _, revision = store.reserve(identifier, expected_revision=0)
        before = store.snapshot()
        for response in (ANSWER, "x" * (4 * 1024 * 1024 + 1)):
            with pytest.raises(ProcedureScoreLimitError):
                store.complete(identifier, response, expected_revision=revision)
            assert store.snapshot() == before
        with pytest.raises(ValueError, match="Unicode"):
            store.complete(identifier, "\ud800", expected_revision=revision)
        assert store.snapshot() == before


def test_opt_in_larger_prediction_uses_pinned_score_policy(tmp_path):
    limits = ProcedureScoreLimits(max_prediction_bytes=2 * 1024 * 1024, max_line_bytes=256 * 1024)
    plan = small_plan(cases=[authored_case(code=True)], score_limits=limits)
    prediction = "```cpp\n" + ("x" * 200_000 + "\n") * 6 + "```"
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        report = store.run(plan, OfflineProcedure(prediction))
        assert report["coverage"]["succeeded"] == 1
        assert report["coverage"]["scored"] == 0
        score = report["results"][0]["result"]["score"]
        assert score["score_limits"] == limits.to_dict()
        assert not score["diagnostics"]["execution_attempted"]


def test_complete_transaction_rolls_back_when_event_write_fails(tmp_path):
    plan = small_plan()
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.bind(plan, OfflineProcedure.identity)
        identifier = plan.payload["items"][0]["id"]
        _, revision = store.reserve(identifier, expected_revision=0)
        before = store.snapshot()
        store.connection.set_authorizer(
            lambda action, table, *_: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_INSERT and table == "task_event"
                else sqlite3.SQLITE_OK
            )
        )
        with pytest.raises(sqlite3.DatabaseError):
            store.complete(identifier, ANSWER, expected_revision=revision)
        store.connection.set_authorizer(lambda *_: sqlite3.SQLITE_OK)
        assert store.snapshot() == before
        store.complete(identifier, ANSWER, expected_revision=revision)
        assert store.snapshot()["coverage"]["scored"] == 1


def test_saved_prediction_is_rescored_and_unbound_versions_rejected(tmp_path):
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        report = store.run(small_plan(), OfflineProcedure())
        row = report["results"][0]
        result = row["result"]
        result["prediction"] = "incorrect"
        store.connection.execute(
            "UPDATE task_state SET result=? WHERE item_id=?", (json.dumps(result), row["item_id"])
        )
        with pytest.raises(ValueError, match="pinned scoring"):
            store.snapshot()
    for raw in ({"format": "future"}, [], {}):
        with pytest.raises(ValueError):
            _validated_plan(raw)


def test_plan_digest_rehash_does_not_authorize_gold_injection(tmp_path):
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        plan = small_plan()
        store.bind(plan, OfflineProcedure.identity)
        changed = plan.to_dict()
        changed["items"][0]["messages"][1]["content"] += "GOLD_ONLY"
        store.connection.execute(
            "UPDATE task_run SET plan=?,plan_digest=?", (json.dumps(changed), digest(changed))
        )
        with pytest.raises(ValueError, match="gold-free template"):
            store.snapshot()


def test_changed_settings_gold_or_model_refuse_rebinding(tmp_path):
    plan = small_plan()
    with TaskRunStore(tmp_path / "run.sqlite") as store:
        store.bind(plan, OfflineProcedure.identity)
        before = store.snapshot()
        for changed in (
            small_plan(generation={"max_tokens": 55}),
            small_plan(score_limits=ProcedureScoreLimits(max_prediction_bytes=77)),
            small_plan(seed="new"),
        ):
            with pytest.raises(TaskRunConflict):
                store.bind(changed, OfflineProcedure.identity)
        with pytest.raises(TaskRunConflict):
            store.bind(plan, {"provider": "different"})
        with pytest.raises(ValueError, match="typed suite"):
            store.bind({}, OfflineProcedure.identity)
        assert store.snapshot() == before
