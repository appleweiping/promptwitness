"""Independent adversarial plan probes; no runtime changes and no external data."""

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_plan import (
    ProcedurePlanLimits,
    ProcedureSuitePlan,
    build_procedure_plan,
    load_procedure_suite,
    procedure_messages,
)


def review_case(identifier="one", *, demonstration="GOLD_SENTINEL"):
    return ProcedureCase(
        "countdown",
        identifier,
        "2k",
        {"numbers": [1, 2, 3, 4], "target": 10, "min_intermediate": 1, "max_intermediate": 2000},
        {
            "solution": ["1 + 2 = 3", "3 + 3 = 6", "6 + 4 = 10"],
            "solution_text": "1 + 2 = 3\n3 + 3 = 6\n6 + 4 = 10",
            "demonstration": demonstration,
            "search_steps": 0,
            "num_search_tokens": 0,
        },
        {"kind": "authored", "label": "SOURCE_SENTINEL"},
    )


def manifest(tmp_path, cases, *, data_limits=None, tasks=1):
    content = json.dumps([item.to_dict() for item in cases]).encode("utf-8")
    (tmp_path / "cases.json").write_bytes(content)
    value = {
        "format": "promptwitness.procedure-suite/v1",
        "suite_id": "adversarial",
        "budgets": [4096],
        "tasks": [
            {"id": f"task-{index}", "adapter": "cases", "path": "cases.json"}
            for index in range(tasks)
        ],
    }
    if data_limits is not None:
        value["data_limits"] = data_limits
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, len(content)


def test_mixed_budget_types_raise_controlled_value_error():
    with pytest.raises(ValueError):
        build_procedure_plan(
            [{"id": "t", "cases": [review_case()]}], suite_id="s", budgets=[1, "2"]
        )


@pytest.mark.parametrize(
    "field",
    ["max_case_bytes", "max_total_case_bytes", "max_total_read_bytes", "max_records"],
)
def test_typed_case_file_honors_all_declared_data_admission_limits(tmp_path, field):
    path, _ = manifest(tmp_path, [review_case("one"), review_case("two")], data_limits={field: 1})
    with pytest.raises(ValueError):
        load_procedure_suite(path)


def test_suite_aggregate_read_budget_does_not_reset_for_each_task(tmp_path):
    cases = [review_case()]
    _, file_bytes = manifest(tmp_path, cases)
    path, _ = manifest(
        tmp_path, cases, data_limits={"max_total_read_bytes": file_bytes + 10}, tasks=2
    )
    with pytest.raises(ValueError):
        load_procedure_suite(path)


def test_plan_rejects_oversized_native_record_before_to_dict_expansion(monkeypatch):
    item = review_case(demonstration="x" * 8192)

    def forbidden(self):
        raise AssertionError("over-budget native case was expanded before plan admission")

    monkeypatch.setattr(ProcedureCase, "to_dict", forbidden)
    with pytest.raises(ValueError):
        build_procedure_plan(
            [{"id": "t", "cases": [item]}],
            suite_id="s",
            budgets=[4096],
            limits=ProcedurePlanLimits(max_plan_bytes=2048),
        )


def test_gold_and_provenance_changes_never_change_provider_messages():
    first = review_case()
    raw = first.to_dict()
    raw["reference"]["demonstration"] = "OTHER_GOLD_SENTINEL"
    raw["source"]["label"] = "OTHER_SOURCE_SENTINEL"
    second = ProcedureCase.from_dict(raw)
    assert first.digest != second.digest
    assert procedure_messages(first) == procedure_messages(second)
    text = json.dumps(procedure_messages(first))
    assert "SENTINEL" not in text
    assert "reference" not in json.loads(procedure_messages(first)[1]["content"])


def test_materialized_message_tamper_is_rejected_and_sampling_is_order_independent():
    items = [review_case("first"), review_case("second")]
    first = build_procedure_plan(
        [{"id": "t", "cases": items}], suite_id="s", budgets=[4096], seed="fixed"
    )
    second = build_procedure_plan(
        [{"id": "t", "cases": list(reversed(items))}], suite_id="s", budgets=[4096], seed="fixed"
    )
    assert first.digest == second.digest
    raw = first.to_dict()
    raw["items"][0]["messages"][1]["content"] += "GOLD_LEAK"
    with pytest.raises(ValueError):
        ProcedureSuitePlan(raw)


def test_final_frame_task_and_item_admission_has_exact_byte_boundary():
    specs = [{"id": "t", "cases": [review_case()]}]
    limits = ProcedurePlanLimits(max_plan_bytes=10_000)
    # The budget is itself in the serialized envelope. Iterate to its small
    # fixed point; the expected size is independently standard-json encoded.
    for _ in range(4):
        plan = build_procedure_plan(specs, suite_id="s", budgets=[4096], limits=limits)
        size = len(
            json.dumps(
                plan.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        )
        if size == limits.max_plan_bytes:
            break
        limits = replace(limits, max_plan_bytes=size)
    assert size == limits.max_plan_bytes
    with pytest.raises(ValueError):
        build_procedure_plan(
            specs, suite_id="s", budgets=[4096], limits=replace(limits, max_plan_bytes=size - 1)
        )


def test_aggregate_node_admission_precedes_final_plan_constructor(monkeypatch):
    from promptwitness import procedure_plan

    def count(value):
        if isinstance(value, dict):
            return 1 + sum(1 + count(child) for child in value.values())
        if isinstance(value, list):
            return 1 + sum(map(count, value))
        return 1

    specs = [{"id": "t", "cases": [review_case("one"), review_case("two")]}]
    plan = build_procedure_plan(specs, suite_id="s", budgets=[4096])
    node_count = count(plan.to_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate node excess reached immutable plan copying")

    monkeypatch.setattr(procedure_plan, "ProcedureSuitePlan", forbidden)
    with pytest.raises(ValueError):
        build_procedure_plan(
            specs,
            suite_id="s",
            budgets=[4096],
            limits=ProcedurePlanLimits(max_plan_nodes=node_count - 1),
        )


def test_sampling_reuses_prefix_without_changing_independent_sha256_order():
    items = [review_case(str(index)) for index in range(12)]
    seed, task = "é" * 1000, "sample"

    def rank(case):
        raw = json.dumps(
            [seed, task, case.case_id], ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    expected = [item.case_id for item in sorted(items, key=rank)[:3]]
    plan = build_procedure_plan(
        [{"id": task, "cases": items, "limit": 3}], suite_id="s", budgets=[4096], seed=seed
    )
    assert [item["record_id"] for item in plan.payload["items"]] == expected


def test_suite_primary_record_and_canonical_case_budgets_are_shared(tmp_path):
    item = review_case()
    canonical_size = len(
        json.dumps(
            item.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    for limits in ({"max_records": 1}, {"max_total_case_bytes": canonical_size + 10}):
        path, _ = manifest(tmp_path, [item], data_limits=limits, tasks=2)
        with pytest.raises(ValueError):
            load_procedure_suite(path)


def test_unknown_adapter_and_invalid_budget_do_not_read_case_file(tmp_path, monkeypatch):
    from promptwitness import procedure_plan

    path, _ = manifest(tmp_path, [review_case()])
    value = json.loads(path.read_text())
    value["budgets"] = [1, "2"]
    path.write_text(json.dumps(value))
    original = procedure_plan._read
    calls = []

    def guarded(path, limit):
        calls.append(path.name)
        assert path.name == "suite.json"
        return original(path, limit)

    monkeypatch.setattr(procedure_plan, "_read", guarded)
    with pytest.raises(ValueError):
        load_procedure_suite(path)
    assert calls == ["suite.json"]


def test_longproc_loaders_receive_remaining_suite_budgets(tmp_path, monkeypatch):
    from promptwitness import procedure_plan

    item = review_case()
    value = {
        "format": "promptwitness.procedure-suite/v1",
        "suite_id": "adversarial",
        "budgets": [4096],
        "data_limits": {"max_total_read_bytes": 30, "max_total_case_bytes": 2000, "max_records": 3},
        "tasks": [
            {"id": name, "adapter": "longproc", "path": "unused-root", "dataset": "countdown_2k"}
            for name in ("one", "two")
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value))
    observed = []

    def load(path, dataset, *, limits):
        observed.append(
            (limits.max_total_read_bytes, limits.max_total_case_bytes, limits.max_records)
        )
        return SimpleNamespace(
            cases=(item,),
            inventory={
                "available": 1,
                "excluded": 0,
                "read_bytes": 10,
                "normalized_case_bytes": 700,
            },
        )

    monkeypatch.setattr(procedure_plan, "load_longproc_dataset", load)
    assert len(load_procedure_suite(path).payload["items"]) == 2
    assert observed == [(30, 2000, 3), (20, 1300, 2)]


def test_nonregular_source_is_rejected_before_potentially_blocking_open(tmp_path, monkeypatch):
    from promptwitness.procedure_plan import _read

    def forbidden(*args, **kwargs):
        raise AssertionError("nonregular source reached a potentially blocking open")

    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_mode=stat.S_IFIFO))
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="regular"):
        _read(tmp_path / "named-pipe", 1024)
