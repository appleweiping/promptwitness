"""Plan invariants and immutable gold isolation, using small authored problems."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_json_data import canonical_bytes
from promptwitness.procedure_plan import (
    ProcedurePlanLimits,
    ProcedureSuitePlan,
    build_procedure_plan,
    load_procedure_suite,
    procedure_messages,
)
from promptwitness.procedure_scores import ProcedureScoreLimits

ANSWER = "<Solution>\n2 + 2 = 4\n4 + 4 = 8\n8 * 3 = 24\n</Solution>"


def authored_case(identifier="private-id", bucket="2k", *, code=False):
    if code:
        return ProcedureCase(
            "pseudo_to_code",
            identifier,
            bucket,
            {"pseudocode_lines": ["Print 42."]},
            {"code_lines": ["GOLD_ONLY"], "testcases": [[[], ["42"]]]},
            {"kind": "authored", "label": "SOURCE_ONLY"},
        )
    return ProcedureCase(
        "countdown",
        identifier,
        bucket,
        {"numbers": [2, 2, 3, 4], "target": 24, "min_intermediate": 1, "max_intermediate": 2000},
        {
            "solution": ["2 + 2 = 4", "4 + 4 = 8", "8 * 3 = 24"],
            "solution_text": "GOLD_ONLY",
            "demonstration": "DEMONSTRATION_ONLY",
            "search_steps": 0,
            "num_search_tokens": 0,
        },
        {"kind": "authored", "label": "SOURCE_ONLY"},
    )


def small_plan(*, cases=None, **kwargs):
    return build_procedure_plan(
        [{"id": "arithmetic", "cases": cases if cases is not None else [authored_case()]}],
        suite_id="authored-suite",
        budgets=kwargs.pop("budgets", [4096]),
        **kwargs,
    )


def test_roundtrip_immutable_messages_and_full_utf8_byte_accounting():
    case = authored_case("私有标识")
    before = case.to_dict()
    plan = small_plan(cases=[case], budgets=[4096, 1])
    assert plan.to_dict()["budgets"] == [1, 4096]
    assert ProcedureSuitePlan(plan.to_dict()).digest == plan.digest
    item = plan.payload["items"][0]
    wire = canonical_bytes(item["messages"]).decode("utf-8")
    assert not any(
        marker in wire
        for marker in (
            "私有标识",
            "GOLD_ONLY",
            "DEMONSTRATION_ONLY",
            "SOURCE_ONLY",
            "solution_text",
        )
    )
    assert item["input_units"] == len(canonical_bytes(procedure_messages(case)))
    assert item["skip_reason"] == "input_budget_exceeded"
    assert plan.payload["items"][1]["skip_reason"] is None
    assert case.to_dict() == before
    with pytest.raises(TypeError):
        item["generation"]["max_tokens"] = 7
    with pytest.raises(FrozenInstanceError):
        plan.payload = {}
    detached = plan.to_dict()
    detached["items"][0]["record"]["input"]["numbers"][0] = 9
    assert plan.to_dict()["items"][0]["record"]["input"]["numbers"][0] == 2


def test_sampling_accounting_is_explicit_stable_and_per_task_generation():
    cases = [authored_case(str(index)) for index in range(8)]
    specs = [
        {
            "id": "a",
            "cases": cases,
            "limit": 3,
            "available": 10,
            "excluded": 2,
            "generation": {"max_tokens": 77, "temperature": 0.5, "seed": 12},
        }
    ]
    first = build_procedure_plan(specs, suite_id="s", budgets=[4096, 8192], seed="stable")
    specs[0]["cases"] = list(reversed(cases))
    second = build_procedure_plan(specs, suite_id="s", budgets=[8192, 4096], seed="stable")
    assert first.digest == second.digest
    task = first.to_dict()["tasks"][0]
    assert [
        task[key] for key in ("available", "eligible", "excluded", "selected", "sampled_out")
    ] == [10, 8, 2, 3, 5]
    assert len(first.payload["items"]) == 6
    assert all(item["generation"]["max_tokens"] == 77 for item in first.payload["items"])
    changed = build_procedure_plan(specs, suite_id="s", budgets=[4096, 8192], seed="changed")
    assert first.digest != changed.digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "promptwitness.procedure-plan/v2"),
        ("scorer_version", "other"),
        ("template_version", "other"),
        ("manifest_sha256", "x"),
        ("suite_id", ""),
        ("seed", True),
        ("counter", {}),
        ("budgets", [True]),
        ("budgets", [4096, 4096]),
        ("generation", {"max_tokens": 1, "messages": []}),
        ("limits", {}),
        ("score_limits", {}),
        ("tasks", []),
        ("items", []),
    ],
)
def test_tampered_top_level_rejected(field, value):
    payload = small_plan().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        ProcedureSuitePlan(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "wrong"),
        ("family", "path_traversal"),
        ("output_bucket", "8k"),
        ("task_id", "missing"),
        ("record_id", "wrong"),
        ("budget", 2),
        ("input_units", True),
        ("skip_reason", "anything"),
        ("generation", {"max_tokens": 77}),
        ("messages", []),
    ],
)
def test_tampered_item_rejected(field, value):
    payload = small_plan().to_dict()
    payload["items"][0][field] = value
    with pytest.raises(ValueError):
        ProcedureSuitePlan(payload)


def test_reject_missing_budget_duplicate_item_and_inconsistent_record():
    base = small_plan(budgets=[4096, 8192]).to_dict()
    variants = []
    missing = copy.deepcopy(base)
    missing["items"].pop()
    variants.append(missing)
    duplicate = copy.deepcopy(base)
    duplicate["items"].append(duplicate["items"][0])
    variants.append(duplicate)
    altered = copy.deepcopy(base)
    altered["items"][1]["record"]["reference"]["solution_text"] = "Changed off-wire gold"
    variants.append(altered)
    for payload in variants:
        with pytest.raises(ValueError):
            ProcedureSuitePlan(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("available", 2),
        ("eligible", 2),
        ("excluded", 1),
        ("selected", 0),
        ("sampled_out", 1),
        ("source", []),
        ("generation", {"max_tokens": 0}),
        ("output_bucket", "bad"),
        ("family", "bad"),
    ],
)
def test_task_accounting_cannot_be_forged(field, value):
    payload = small_plan().to_dict()
    payload["tasks"][0][field] = value
    with pytest.raises(ValueError):
        ProcedureSuitePlan(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budgets": []},
        {"budgets": [0]},
        {"budgets": [True]},
        {"budgets": [4096, 4096]},
        {"limits": False},
        {"score_limits": False},
        {"generation": {"max_tokens": 1_000_001}},
        {"manifest_sha256": ""},
        {"limits": ProcedurePlanLimits(max_items=1), "budgets": [4096, 8192]},
        {"limits": ProcedurePlanLimits(max_plan_nodes=5)},
    ],
)
def test_invalid_builder_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        small_plan(**kwargs)


def test_limits_and_messages_require_typed_values():
    for field in asdict(ProcedurePlanLimits()):
        with pytest.raises(ValueError):
            ProcedurePlanLimits(**{field: True})
    with pytest.raises(ValueError):
        procedure_messages({})
    plan = small_plan(score_limits=ProcedureScoreLimits(max_prediction_bytes=123))
    assert plan.payload["score_limits"]["max_prediction_bytes"] == 123


def test_local_manifest_roundtrip_and_raw_file_hash(tmp_path):
    cases = tmp_path / "cases.json"
    content = canonical_bytes([authored_case().to_dict()])
    cases.write_bytes(content)
    path = tmp_path / "suite.json"
    value = {
        "format": "promptwitness.procedure-suite/v1",
        "suite_id": "local",
        "budgets": [4096],
        "tasks": [{"id": "a", "adapter": "cases", "path": cases.name}],
        "limits": {"max_items": 2},
        "score_limits": {"max_prediction_bytes": 200},
    }
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    plan = load_procedure_suite(path)
    assert plan.payload["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert plan.payload["tasks"][0]["source"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert plan.payload["limits"]["max_items"] == 2
    assert plan.payload["score_limits"]["max_prediction_bytes"] == 200
    assert cases.read_bytes() == content
    for field, invalid in (("format", "bad"), ("limits", {"unknown": 1}), ("data_limits", [])):
        changed = {**value, field: invalid}
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_procedure_suite(path)
