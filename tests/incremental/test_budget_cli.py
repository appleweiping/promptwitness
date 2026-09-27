import json

import pytest

pytest.importorskip("scipy")

from promptwitness.cli import main
from promptwitness.incremental.budget import ResourceLedger, ResourceLimit, allocated_gpu_seconds
from promptwitness.incremental.journal import BudgetExhausted
from promptwitness.incremental.sampling import digest


def ledger(tmp_path, **changes):
    params = {
        "historical_usage": {
            "calls": 770,
            "input_tokens": 1448879,
            "output_tokens": 178514,
            "gpu_hours": 3.2538161689659404,
        },
        "historical_digest": digest("validated historical ledger"),
        "global_limit": ResourceLimit(800000, 2000000000, 200000000, 1000),
        "stage_limits": {"unblocking": ResourceLimit(4, 100, 100, 8)},
        "gpu_uuid": "GPU-assigned",
    }
    return ResourceLedger(tmp_path / "resources.sqlite", **(params | changes))


def test_union_overlap_not_sum_request_latency_or_gpu_busy_time():
    assert (
        allocated_gpu_seconds(
            [("GPU-a", 0, 10), ("GPU-a", 5, 15), ("GPU-b", 5, 8), ("GPU-a", 16, 20)]
        )
        == 22
    )
    assert allocated_gpu_seconds([]) == 0
    with pytest.raises(ValueError):
        allocated_gpu_seconds([("GPU-a", 10, 9)])


def test_actual_prior_usage_caps_unknown_failure_and_no_reset(tmp_path):
    store = ledger(tmp_path)
    assert store.usage()["calls"] == 770
    store.reserve_call("attempt1", "unblocking", digest("request"), input_cap=20, output_cap=30)
    assert store.usage()["calls"] == 771
    store.settle_call("attempt1", succeeded=True, input_tokens=5, output_tokens=9)
    assert store.usage("unblocking")["input_tokens"] == 5
    store.reserve_call("attempt2", "unblocking", digest("request"), input_cap=40, output_cap=60)
    store.settle_call("attempt2", succeeded=False, input_tokens=None, output_tokens=None)
    assert store.usage("unblocking")["output_tokens"] == 69
    assert store.usage("unblocking")["calls_with_reserved_or_unknown_token_cost"] == 1
    with pytest.raises(BudgetExhausted):
        store.reserve_call("attempt3", "unblocking", digest("new"), input_cap=1, output_cap=40)
    store.close()
    recovered = ledger(tmp_path)
    assert recovered.usage()["calls"] == 772
    recovered.close()
    with pytest.raises(ValueError):
        ledger(tmp_path, historical_digest=digest("reset forbidden"))


def test_incurring_overrun_retained_and_duplicate_resolution_rejected(tmp_path):
    store = ledger(tmp_path)
    store.reserve_call("attempt", "unblocking", digest("r"), input_cap=5, output_cap=5)
    with pytest.raises(BudgetExhausted):
        store.settle_call("attempt", succeeded=True, input_tokens=8, output_tokens=3)
    assert store.usage("unblocking")["input_tokens"] == 8
    with pytest.raises(ValueError):
        store.settle_call("attempt", succeeded=True, input_tokens=0, output_tokens=0)
    store.close()


def test_gpu_uuid_idle_startup_overlap_and_stage_deadline(tmp_path):
    store = ledger(tmp_path)
    store.begin_gpu("load1", "unblocking", "GPU-assigned", start=100)
    store.begin_gpu("overlap", "unblocking", "GPU-assigned", start=110)
    store.end_gpu("load1", end=200)
    store.end_gpu("overlap", end=250)
    assert store.usage("unblocking")["gpu_hours"] == 150 / 3600
    assert store.usage()["gpu_hours"] == 3.2538161689659404 + 150 / 3600
    with pytest.raises(ValueError):
        store.begin_gpu("other", "unblocking", "GPU-other", start=300)
    with pytest.raises(ValueError):
        store.end_gpu("load1", end=300)
    store.begin_gpu("idle", "unblocking", "GPU-assigned", start=300)
    with pytest.raises(BudgetExhausted):
        store.reserve_call(
            "too_late", "unblocking", digest("r"), input_cap=1, output_cap=1, now=300 + 8 * 3600
        )
    store.end_gpu("idle", end=301)
    store.close()


def test_cli_complete_recorded_table_plan_run_report_and_no_overwrite(tmp_path, capsys):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({str(i): 1 for i in range(20)}), encoding="utf-8")
    plan = tmp_path / "plan.json"
    args = [
        "delta",
        "plan",
        str(reference),
        "--candidate-digest",
        digest("candidate"),
        "--execution-digest",
        digest("execution"),
        "--fixture-seed",
        "11",
        "--output",
        str(plan),
    ]
    assert main(args) == 0
    assert main(args) == 1
    store = tmp_path / "audit.sqlite"
    result = main(
        [
            "delta",
            "run",
            "--plan",
            str(plan),
            "--outcomes",
            str(reference),
            "--ledger",
            str(store),
            "--run-id",
            "run",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    outcome = json.loads(captured.out)
    assert outcome["new_real_model_calls"] == 0
    assert outcome["mode"] == "RECORDED_TABLE_MECHANISM_ONLY"
    assert main(["delta", "report", "--ledger", str(store), "--run-id", "run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["charged_candidate_query_attempts"] == 20
    assert main(["delta", "report", "--ledger", str(store), "--run-id", "missing"]) == 1


def test_cli_duplicate_json_missing_outcomes_and_readonly_report(tmp_path):
    reference = tmp_path / "duplicate.json"
    reference.write_text('{"x":0,"x":1}', encoding="utf-8")
    assert (
        main(
            [
                "delta",
                "plan",
                str(reference),
                "--candidate-digest",
                digest("c"),
                "--execution-digest",
                digest("e"),
            ]
        )
        == 1
    )
    assert (
        main(["delta", "report", "--ledger", str(tmp_path / "absent.sqlite"), "--run-id", "r"]) == 1
    )
    assert not (tmp_path / "absent.sqlite").exists()
