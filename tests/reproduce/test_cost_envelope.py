"""Fabricated profiles check that consumed cost cannot be omitted or replayed."""

import json
import sqlite3
from pathlib import Path

import pytest

from reproduce.summarize_cost_envelope import summarize


@pytest.fixture
def evidence(tmp_path):
    root = Path(__file__).parents[2]
    plan = json.loads(
        (root / "configs/research/preflight-cost-envelope-v1.json").read_text(encoding="utf-8")
    )
    plan.update(
        prior_charged_requests=2,
        prior_allocated_gpu_hours=25 / 3600,
        aggregate_g0_attempt_ceiling=302,
    )
    prior = {
        "charged_attempts": 2,
        "allocated_gpu_hours": 25 / 3600,
        "input_tokens": 4,
        "output_tokens": 2,
        "models": [],
        "startup_failures": [{"ledger_model": "prior:failed", "elapsed_seconds": 5}],
    }
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "CREATE TABLE runs(model TEXT PRIMARY KEY,revision TEXT,requests_sha TEXT,"
            "elapsed REAL,cold_start REAL)"
        )
        connection.execute(
            "CREATE TABLE attempts(model TEXT,id TEXT,role TEXT,status TEXT,PRIMARY KEY(model,id))"
        )
        for model in ("prior1", "prior2"):
            connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (model, "r", "s", 10, 1))
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?)", (model, "unit", "task", "completed")
            )
            prior["models"].append(
                {
                    "ledger_model": model,
                    "revision": "r",
                    "request_file_sha256": "s",
                    "elapsed_seconds": 10,
                    "cold_start_seconds": 1,
                    "requests": 1,
                }
            )
        connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?)", ("prior:failed", "r", "s", 5, None)
        )
        records = []
        for model in plan["models"]:
            ledger_model = model["id"] + "@" + plan["probe_id"]
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (ledger_model, model["revision"], plan["request_file_sha256"], 112, 10),
            )
            for family in ("bfcl", "hotpotqa", "instruction_following"):
                for role, profile, count in (
                    ("task", "base", 16),
                    ("task", "eight_demo_stress_proxy", 16),
                    ("proposer", "serial_training_context_proxy", 2 if family == "hotpotqa" else 1),
                ):
                    for index in range(count):
                        item_id = f"{family}:{profile}:{index}"
                        connection.execute(
                            "INSERT INTO attempts VALUES (?,?,?,?)",
                            (ledger_model, item_id, role, "completed"),
                        )
                        records.append(
                            {
                                "model": model["id"],
                                "ledger_model": ledger_model,
                                "revision": model["revision"],
                                "id": item_id,
                                "family": family,
                                "role": role,
                                "profile": profile,
                                "status": "completed",
                                "requests_sha256": plan["request_file_sha256"],
                                "script_sha256": plan["script_sha256"],
                                "scoring_executed": False,
                                "torch_version": plan["runtime"]["torch"],
                                "cpu_threads": 4,
                                "max_input_tokens": 32768,
                                "disable_compile": True,
                                "proposers_batch_one": True,
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "max_new_tokens": plan["max_new_tokens"][
                                    "proposer" if role == "proposer" else family
                                ],
                                "batch_size": 1,
                                "batch_elapsed_seconds": 1,
                                "allocated_seconds_share": 1,
                            }
                        )
        failure_run = plan["models"][0]["id"] + "@" + plan["probe_id"]
        archive = failure_run + ":startup-failed1"
        connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?)",
            (archive, plan["models"][0]["revision"], plan["request_file_sha256"], 7, None),
        )
    values = {
        "plan.json": plan,
        "prior.json": prior,
        "failure.json": {
            "ledger_run": failure_run,
            "archive_run": archive,
            "model_requests_attempted": 0,
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    return (
        ledger,
        [records_path],
        tmp_path / "plan.json",
        tmp_path / "prior.json",
        [tmp_path / "failure.json"],
    )


def test_prior_cold_failed_and_warm_costs_retained(evidence):
    result = summarize(*evidence)
    assert result["completed_requests_all_g0_phases"] == 302
    assert result["input_tokens_all_g0_phases"] == 3004
    assert result["output_tokens_all_g0_phases"] == 602
    assert result["allocated_gpu_hours_all_g0_phases"] == pytest.approx((25 + 7 + 336) / 3600)
    assert not result["scores_or_effectiveness_measured"]
    assert len(result["startup_failures_all_g0_phases"]) == 2


def test_duplicate_and_missing_raw_records_fail_closed(evidence):
    ledger, records, plan, prior, failures = evidence
    with pytest.raises(ValueError, match="duplicate"):
        summarize(ledger, records * 2, plan, prior, failures)
    rows = records[0].read_text(encoding="utf-8").splitlines()
    records[0].write_text("\n".join(rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        summarize(*evidence)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("scoring_executed", True, "label access"),
        ("script_sha256", "changed", "provenance"),
        ("torch_version", "changed", "runtime"),
        ("output_tokens", 9000, "bounds"),
        ("batch_size", True, "count field"),
        ("allocated_seconds_share", float("nan"), "cost field"),
        ("allocated_seconds_share", 0.5, "does not sum"),
    ],
)
def test_invalid_profile_or_provenance_rejected(evidence, field, value, message):
    path = evidence[1][0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        summarize(*evidence)


def test_past_consumption_cannot_be_reset(evidence):
    with sqlite3.connect(evidence[0]) as connection:
        connection.execute("UPDATE runs SET elapsed=0 WHERE model='prior1'")
    with pytest.raises(ValueError, match="past run changed"):
        summarize(*evidence)


def test_unfinished_attempt_not_treated_as_completed(evidence):
    with sqlite3.connect(evidence[0]) as connection:
        connection.execute("UPDATE attempts SET status='failed' WHERE rowid=3")
    with pytest.raises(ValueError, match="no dropped failures"):
        summarize(*evidence)


def test_generation_cannot_be_disguised_as_startup(evidence):
    failure = json.loads(evidence[4][0].read_text(encoding="utf-8"))
    with sqlite3.connect(evidence[0]) as connection:
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?)",
            (failure["archive_run"], "attempt", "task", "failed"),
        )
    with pytest.raises(ValueError, match="startup failure contains inference"):
        summarize(*evidence)
