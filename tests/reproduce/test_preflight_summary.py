"""Mechanical cost-evidence checks; these fabricated fixtures are not model results."""

import hashlib
import json
import sqlite3

import pytest

from reproduce.summarize_preflight import summarize


@pytest.fixture
def evidence(tmp_path):
    ledger_path = tmp_path / "cost.sqlite"
    plan_path = tmp_path / "plan.json"
    records_path = tmp_path / "records.jsonl"
    connection = sqlite3.connect(ledger_path)
    connection.execute(
        "CREATE TABLE runs(model TEXT PRIMARY KEY,revision TEXT,requests_sha TEXT,"
        "elapsed REAL,cold_start REAL)"
    )
    connection.execute(
        "CREATE TABLE attempts(model TEXT,id TEXT,role TEXT,status TEXT,PRIMARY KEY(model,id))"
    )
    records = []
    for model in ("test-model-a", "test-model-b"):
        connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (model, "r1", "s1", 110, 10))
        for family, role, count in (
            ("bfcl", "task", 32),
            ("hotpotqa", "task", 32),
            ("instruction_following", "task", 32),
            ("bfcl", "proposer", 2),
            ("hotpotqa", "proposer", 1),
            ("instruction_following", "proposer", 1),
        ):
            for index in range(count):
                item_id = f"{family}:{role}:{index}"
                connection.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?)", (model, item_id, role, "completed")
                )
                records.append(
                    {
                        "model": model,
                        "id": item_id,
                        "role": role,
                        "family": family,
                        "status": "completed",
                        "revision": "r1",
                        "requests_sha256": "s1",
                        "scoring_executed": False,
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "max_new_tokens": 4,
                        "batch_size": 4,
                        "allocated_seconds_share": 1,
                    }
                )
    connection.commit()
    connection.close()
    records_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    plan_path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "test-model-a", "revision": "r1"},
                    {"id": "test-model-b", "revision": "r1"},
                ],
                "request_file_sha256": "s1",
            }
        ),
        encoding="utf-8",
    )
    return ledger_path, records_path, plan_path, records


def test_complete_evidence_is_cost_only(evidence):
    ledger, records, plan, _ = evidence
    result = summarize(ledger, [records], plan)
    assert result["charged_attempts"] == 200
    assert result["input_tokens"] == 600
    assert result["output_tokens"] == 400
    assert result["allocated_gpu_hours"] == pytest.approx(220 / 3600)
    assert result["peak_gpu_memory_mib"] is None
    assert all(row["other_warm_allocated_seconds"] == 0 for row in result["models"])


def test_duplicate_records_are_not_free(evidence):
    ledger, records, plan, _ = evidence
    with pytest.raises(ValueError, match="duplicate"):
        summarize(ledger, [records, records], plan)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_tokens", True, "invalid token"),
        ("input_tokens", -1, "invalid token"),
        ("output_tokens", 5, "output bounds"),
        ("batch_size", 0, "batch/output bounds"),
        ("allocated_seconds_share", float("nan"), "elapsed-time"),
        ("allocated_seconds_share", -1, "elapsed-time"),
        ("scoring_executed", True, "provenance"),
        ("revision", "changed", "provenance"),
        ("requests_sha256", "changed", "provenance"),
        ("family", "unknown", "family counts"),
    ],
)
def test_bad_evidence_fails_closed(evidence, field, value, message):
    ledger, records_path, plan, rows = evidence
    rows[0][field] = value
    records_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        summarize(ledger, [records_path], plan)


@pytest.mark.parametrize("status", ["reserved", "failed"])
def test_crash_or_failure_requires_explicit_accounting(evidence, status):
    ledger, records, plan, _ = evidence
    with sqlite3.connect(ledger) as connection:
        connection.execute("UPDATE attempts SET status=? WHERE rowid=1", (status,))
    with pytest.raises(ValueError, match="failure cost"):
        summarize(ledger, [records], plan)


@pytest.mark.parametrize("elapsed,cold", [(None, 10), (10, None), (2, 3), (50, 10)])
def test_unfinished_or_inconsistent_run_time_is_rejected(evidence, elapsed, cold):
    ledger, records, plan, _ = evidence
    with sqlite3.connect(ledger) as connection:
        connection.execute("UPDATE runs SET elapsed=?,cold_start=? WHERE rowid=1", (elapsed, cold))
    with pytest.raises(ValueError, match="timing"):
        summarize(ledger, [records], plan)


def test_input_plan_cannot_silently_change(evidence):
    ledger, records, plan, _ = evidence
    config = json.loads(plan.read_text(encoding="utf-8"))
    config["request_file_sha256"] = "changed"
    plan.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        summarize(ledger, [records], plan)


def test_missing_record_is_not_zero_cost(evidence):
    ledger, records_path, plan, rows = evidence
    records_path.write_text("\n".join(json.dumps(row) for row in rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        summarize(ledger, [records_path], plan)


@pytest.fixture
def repair_evidence(evidence, tmp_path):
    ledger, records_path, prior_path, original_rows = evidence
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    plan = {
        **prior,
        "probe_id": "infra1",
        "script_sha256": "script1",
        "source_plan_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "runtime": {"torch": "2.7.1+cu118"},
        "cpu_threads": 4,
        "batch_size": 8,
    }
    plan_path = tmp_path / "repair-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    rows = []
    with sqlite3.connect(ledger) as connection:
        for model in ("test-model-a", "test-model-b"):
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)", (model + "@infra1", "r1", "s1", 210, 10)
            )
        for row in original_rows:
            updated = {
                **row,
                "ledger_model": row["model"] + "@infra1",
                "probe_id": "infra1",
                "script_sha256": "script1",
                "torch_version": "2.7.1+cu118",
                "cpu_threads": 4,
                "batch_size": 8,
                "allocated_seconds_share": 2,
            }
            rows.append(updated)
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?)",
                (updated["ledger_model"], updated["id"], updated["role"], "completed"),
            )
    repair_path = tmp_path / "repair-records.jsonl"
    repair_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return ledger, [records_path, repair_path], plan_path, prior_path


def test_repair_preserves_all_prior_costs(repair_evidence):
    ledger, records, plan, prior = repair_evidence
    result = summarize(ledger, records, plan, prior_plan_path=prior)
    assert result["charged_attempts"] == 400
    assert result["allocated_gpu_hours"] == pytest.approx(640 / 3600)
    assert result["input_tokens"] == 1200
    assert len(result["models"]) == 4


def test_repair_cannot_omit_prior_consumption(repair_evidence):
    ledger, records, plan, _ = repair_evidence
    with pytest.raises(ValueError, match="original plan"):
        summarize(ledger, records, plan)


def test_repair_plan_hash_must_match(repair_evidence):
    ledger, records, plan, prior = repair_evidence
    prior.write_text(prior.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source plan hash"):
        summarize(ledger, records, plan, prior_plan_path=prior)


def test_repair_runtime_must_match(repair_evidence):
    ledger, records, plan, prior = repair_evidence
    rows = [json.loads(line) for line in records[1].read_text(encoding="utf-8").splitlines()]
    rows[0]["torch_version"] = "unknown"
    records[1].write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen runtime"):
        summarize(ledger, records, plan, prior_plan_path=prior)


def test_explicit_zero_request_startup_failure_still_costs(repair_evidence, tmp_path):
    ledger, records, plan, prior = repair_evidence
    run = "test-model-a@infra1"
    archived = run + ":startup-failed1"
    with sqlite3.connect(ledger) as connection:
        connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (archived, "r1", "s1", 5, None))
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps({"ledger_run": run, "archive_run": archived, "model_requests_attempted": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="planned model runs"):
        summarize(ledger, records, plan, prior_plan_path=prior)
    result = summarize(
        ledger, records, plan, prior_plan_path=prior, startup_failure_paths=[failure]
    )
    assert result["allocated_gpu_hours"] == pytest.approx(645 / 3600)
    assert result["startup_failures"][0]["elapsed_seconds"] == 5


def test_failed_generation_cannot_be_reclassified_startup(repair_evidence, tmp_path):
    ledger, records, plan, prior = repair_evidence
    run = "test-model-a@infra1"
    archived = run + ":startup-failed1"
    with sqlite3.connect(ledger) as connection:
        connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (archived, "r1", "s1", 5, None))
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?)", (archived, "x", "task", "failed")
        )
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps({"ledger_run": run, "archive_run": archived, "model_requests_attempted": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero-request"):
        summarize(ledger, records, plan, prior_plan_path=prior, startup_failure_paths=[failure])
