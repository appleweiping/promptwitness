"""Synthetic fixtures enforce cost-context boundaries, not research performance."""

import copy
import hashlib
import json
import shutil
import sqlite3

import pytest

from reproduce.prepare_native_context import task_messages, validate_context_audit
from reproduce.run_native_context_probe import validate_requests
from reproduce.summarize_native_context import summarize


@pytest.fixture
def audit():
    return {
        "format": "promptwitness.delta.cost-demonstration-audit/v1",
        "source_commit": "99b1ee970490a2d0d5664663eb0b1acc410b944c",
        "scoped_to_all_prior_fit_responses": True,
        "final_test_accessed": False,
        "new_model_calls": 0,
        "language_detector_seed": 11,
        "fit_only_units": [
            {"unit": str(i), "strict_all_constraints_pass": i != 0} for i in range(5)
        ],
        "accepted_count": 4,
        "first_four_accepted_for_cost_context": ["1", "2", "3", "4"],
    }


def test_cost_demos_are_distinct_passing_fit_only(audit):
    assert validate_context_audit(audit, set("01234")) == ["1", "2", "3", "4"]
    with pytest.raises(ValueError, match="fit-only"):
        validate_context_audit(audit, set("0123"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("first_four_accepted_for_cost_context", ["0", "1", "2", "3"]),
        ("first_four_accepted_for_cost_context", ["1", "1", "2", "3"]),
        ("accepted_count", 5),
    ],
)
def test_failed_or_duplicated_demos_cannot_be_accepted(audit, field, value):
    audit[field] = value
    with pytest.raises(ValueError, match="disagree"):
        validate_context_audit(audit, set("01234"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("scoped_to_all_prior_fit_responses", False),
        ("new_model_calls", 1),
        ("final_test_accessed", True),
        ("language_detector_seed", None),
    ],
)
def test_unsupported_audit_provenance_fails_closed(audit, field, value):
    audit[field] = value
    with pytest.raises(ValueError, match="provenance"):
        validate_context_audit(audit, set("01234"))


def test_original_contexts_and_messages_are_preserved():
    row = {
        "question": "Q?",
        "context": {
            "title": ["A", "B"],
            "sentences": [["one.", "two."], ["three."]],
        },
    }
    original = copy.deepcopy(row)
    assert task_messages("hotpotqa", row)[0]["content"] == (
        "Context:\nA: one.two.\n\nB: three.\n\nQuestion: Q?"
    )
    assert row == original
    instruction = {"messages": [{"role": "user", "content": "all constraints"}]}
    messages = task_messages("instruction_following", instruction)
    messages[0]["content"] = "changed"
    assert instruction["messages"][0]["content"] == "all constraints"
    with pytest.raises(ValueError, match="unsupported"):
        task_messages("unknown", instruction)


@pytest.fixture
def requests():
    rows = []
    for family, role, profile, count, cap in (
        ("bfcl", "task", "base", 16, 1024),
        ("hotpotqa", "task", "base", 16, 64),
        ("instruction_following", "task", "base", 16, 4096),
        ("instruction_following", "task", "four_valid_demo_cost_context", 16, 4096),
        ("hotpotqa", "proposer", "official_json_proposer_cost_fixture", 1, 2048),
        ("instruction_following", "proposer", "official_json_proposer_cost_fixture", 1, 2048),
    ):
        for index in range(count):
            messages = [{"role": "system", "content": "fixed interface"}]
            if profile == "four_valid_demo_cost_context":
                for demo_index in range(4):
                    messages.extend(
                        [
                            {"role": "user", "content": f"fit-only {demo_index}"},
                            {"role": "assistant", "content": f"verified response {demo_index}"},
                        ]
                    )
            messages.append({"role": "user", "content": "original input"})
            rows.append(
                {
                    "id": f"{family}:{profile}:{index}",
                    "family": family,
                    "role": role,
                    "profile": profile,
                    "max_new_tokens": cap,
                    "messages": messages,
                }
            )
    for family, profile in (
        ("bfcl", "base"),
        ("hotpotqa", "base"),
        ("instruction_following", "base"),
        ("instruction_following", "four_valid_demo_cost_context"),
    ):
        row = copy.deepcopy(
            next(r for r in rows if r["family"] == family and r["profile"] == profile)
        )
        row["repeat_of"] = row["id"]
        row["id"] = "repeat:" + row["id"]
        rows.append(row)
    return rows


def test_all_seventy_cost_requests_and_identical_repeats(requests):
    validate_requests(requests)
    requests[-1]["messages"][-1]["content"] = "not an identical request"
    with pytest.raises(ValueError, match="body changed"):
        validate_requests(requests)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("max_new_tokens", 1, "caps"),
        ("profile", "changed", "counts"),
        ("messages", [{"role": "user", "content": "lost interface"}], "demonstration"),
    ],
)
def test_changed_task_or_cap_rejected(requests, field, value, match):
    requests[0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_requests(requests)


def test_dropped_or_replayed_calibration_unit_rejected(requests):
    with pytest.raises(ValueError, match="70 charged"):
        validate_requests(requests[:-1])
    requests[-1]["id"] = requests[0]["id"]
    with pytest.raises(ValueError, match="70 charged"):
        validate_requests(requests)


@pytest.fixture
def ledger_evidence(tmp_path, requests):
    archived = tmp_path / "prior.sqlite"
    current = tmp_path / "current.sqlite"
    with sqlite3.connect(archived) as connection:
        connection.execute(
            "CREATE TABLE runs(model TEXT PRIMARY KEY,revision TEXT,requests_sha TEXT,"
            "elapsed REAL,cold_start REAL)"
        )
        connection.execute(
            "CREATE TABLE attempts(model TEXT,id TEXT,role TEXT,status TEXT,PRIMARY KEY(model,id))"
        )
        connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("old", "r", "s", 12, 2))
        connection.execute(
            "INSERT INTO attempts VALUES (?,?,?,?)", ("old", "unit", "task", "completed")
        )
        connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?)", ("old:startup-failed", "r", "s", 3, None)
        )
    prior_sha = hashlib.sha256(archived.read_bytes()).hexdigest()
    prior = {
        "ledger_sha256": prior_sha,
        "completed_requests_all_g0_phases": 1,
        "input_tokens_all_g0_phases": 4,
        "output_tokens_all_g0_phases": 2,
        "allocated_gpu_hours_all_g0_phases": 15 / 3600,
        "startup_failures_all_g0_phases": [
            {"ledger_model": "old:startup-failed", "elapsed_seconds": 3}
        ],
    }
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text("\n".join(json.dumps(r) for r in requests), encoding="utf-8")
    requests_sha = hashlib.sha256(requests_path.read_bytes()).hexdigest()
    plan = {
        "request_file_sha256": requests_sha,
        "prior_ledger_sha256": prior_sha,
        "prior_summary_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "models": [{"id": name, "revision": "r"} for name in ("q", "o", "m")],
        "script_sha256": "worker",
        "probe_id": "native-context-v1",
        "runtime": {"torch": "2.7.1+cu118"},
        "aggregate_g0_attempt_ceiling": 211,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    shutil.copy2(archived, current)
    records = []
    with sqlite3.connect(current) as connection:
        for name in ("q", "o", "m"):
            model = name + "@native-context-v1"
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)", (model, "r", requests_sha, 75, 5)
            )
            for row in requests:
                connection.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?)",
                    (model, row["id"], row["role"], "completed"),
                )
                records.append(
                    {
                        **{key: value for key, value in row.items() if key != "messages"},
                        "repeat_of": row.get("repeat_of"),
                        "model": name,
                        "ledger_model": model,
                        "revision": "r",
                        "requests_sha256": requests_sha,
                        "script_sha256": "worker",
                        "status": "completed",
                        "scoring_executed": False,
                        "batch_size": 1,
                        "cpu_threads": 4,
                        "torch_version": "2.7.1+cu118",
                        "max_input_tokens": 32768,
                        "disable_compile": True,
                        "deterministic_algorithms": True,
                        "cublas_workspace_config": ":4096:8",
                        "decoding": {"do_sample": False, "enable_thinking": False},
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "output": "fixture",
                        "allocated_seconds_share": 1,
                        "batch_elapsed_seconds": 1,
                    }
                )
    raw_path = tmp_path / "records.jsonl"
    raw_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return current, archived, [raw_path], plan_path, requests_path, prior_path


def test_all_old_and_repeat_consumption_preserved(ledger_evidence):
    result = summarize(*ledger_evidence)
    assert result["consumed_all_g0_phases"] == {
        "requests": 211,
        "input_tokens": 2104,
        "output_tokens": 422,
        "allocated_gpu_hours": pytest.approx(240 / 3600),
    }
    assert len(result["prior_startup_failures_retained"]) == 1
    assert all(
        model["sampled_repeat_consistency"] for model in result["models_native_context_phase"]
    )
    assert all(
        sum(p["requests"] for p in model["profiles"]) == 66
        for model in result["models_native_context_phase"]
    )


def test_old_cost_cannot_be_reset(ledger_evidence):
    with sqlite3.connect(ledger_evidence[0]) as connection:
        connection.execute("UPDATE runs SET elapsed=0 WHERE model='old'")
    with pytest.raises(ValueError, match="past ledger"):
        summarize(*ledger_evidence)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("batch_size", True, "execution"),
        ("deterministic_algorithms", False, "execution"),
        ("scoring_executed", True, "label access"),
        ("script_sha256", "wrong", "provenance"),
        ("input_tokens", float("nan"), "token"),
        ("output_tokens", 9999, "token"),
        ("allocated_seconds_share", float("inf"), "cost field"),
        ("allocated_seconds_share", 0.5, "time accounting"),
    ],
)
def test_changed_raw_metadata_fails_closed(ledger_evidence, field, value, match):
    path = ledger_evidence[2][0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        summarize(*ledger_evidence)


def test_nonidentical_repeat_is_reported_not_hidden(ledger_evidence):
    path = ledger_evidence[2][0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["repeat_of"] is not None)["output"] = "different"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = summarize(*ledger_evidence)
    assert result["models_native_context_phase"][0]["sampled_repeat_consistency"] is False
    assert result["consumed_all_g0_phases"]["requests"] == 211


def test_uncharged_and_missing_records_rejected(ledger_evidence):
    with pytest.raises(ValueError, match="duplicate"):
        summarize(
            ledger_evidence[0], ledger_evidence[1], ledger_evidence[2] * 2, *ledger_evidence[3:]
        )
    with sqlite3.connect(ledger_evidence[0]) as connection:
        connection.execute(
            "UPDATE attempts SET status='failed' WHERE model='q@native-context-v1' AND rowid=2"
        )
    with pytest.raises(ValueError, match="provenance"):
        summarize(*ledger_evidence)


def test_completed_model_boundary_reports_unmeasured_costs_explicitly(ledger_evidence):
    current, _, records, *_ = ledger_evidence
    with sqlite3.connect(current) as connection:
        connection.execute(
            "DELETE FROM attempts WHERE model IN (?,?)",
            ("o@native-context-v1", "m@native-context-v1"),
        )
        connection.execute(
            "DELETE FROM runs WHERE model IN (?,?)", ("o@native-context-v1", "m@native-context-v1")
        )
    rows = [json.loads(line) for line in records[0].read_text(encoding="utf-8").splitlines()]
    records[0].write_text(
        "\n".join(json.dumps(r) for r in rows if r["model"] == "q"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing model run"):
        summarize(*ledger_evidence)
    result = summarize(*ledger_evidence, completed_model_boundary_only=True)
    assert not result["calibration_complete"]
    assert result["consumed_all_g0_phases"]["requests"] == 71
    assert result["remaining_model_costs_not_measured"] == ["o", "m"]
    assert result["remaining_planned_calls_not_executed"] == 140
