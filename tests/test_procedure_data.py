"""Authored procedure fixtures, independent of the upstream dataset or evaluators."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from promptwitness.procedure_data import (
    DATASET_NAMES,
    FAMILIES,
    ProcedureCase,
    ProcedureDataError,
    ProcedureDataLimits,
    ProcedureDataset,
    audit_longproc_data,
    load_longproc_dataset,
    normalize_longproc_record,
)
from promptwitness.procedure_json_data import canonical_bytes, load_json


def native_record(family: str) -> dict:
    edge = {"src": "Alpha", "dst": "Beta", "transit": "ferry"}
    records = {
        "countdown": {
            "nums": [1, 2, 3, 4],
            "target": 10,
            "solution": ["1+2=3", "3+3=6", "6+4=10"],
            "solution_text": "1 + 2 = 3\n3 + 3 = 6\n6 + 4 = 10",
            "demonstration": "GOLD-CANARY countdown trace",
            "search_steps": 0.5,
            "num_search_tokens": 50,
        },
        "path_traversal": {
            "problem_description": "A hypothetical network.",
            "context_repr": [edge],
            "question_repr": ["Alpha", "Beta"],
            "answer_repr": [edge],
            "context_nl": "Alpha leads to Beta by ferry.",
            "question_nl": "Alpha to Beta?",
            "answer_nl": "GOLD-CANARY declared route",
        },
        "html_to_tsv": {
            "output_length_group": "0.5k",
            "task_id": "authored-table",
            "website_id": "table",
            "html_path": "webpage/table.html",
            "task_topic": "Names",
            "task_description": "Name",
            "gt": "name\nGOLD-CANARY\n",
            "tsv_header": "name",
            "filtering_instruction": "",
        },
        "tom_tracking": {
            "story_components": "One person, one object.",
            "story": "Step 0: A sees a key.",
            "question": "What does A believe?",
            "solution": "Heading\n- Step 0: GOLD-CANARY\nDone",
            "answer": ["GOLD-CANARY"],
        },
        "travel_planning": {
            "ground_truth_cities": "Alpha**Beta",
            "ground_truth_durations": "2**3",
            "num_cities": 2,
            "total_days": 4,
            "constraints": [
                {"type": "duration", "city": "Beta", "num_days": 3},
                {"type": "fixed", "city": "Beta", "start_day": 2, "end_day": 4},
                {"type": "duration", "city": "Alpha", "num_days": 2},
            ],
            "connected_cities": [["Alpha", "Beta"]],
            "original_question_text": "Two cities, four days.",
            "disambig_question_text": "Alpha to Beta, four days.",
            "ground_truth_plan": "GOLD-CANARY travel plan",
            "id": "authored-trip",
            "estimated_output_tokens": 1024,
            "solving_procedure": "GOLD-CANARY travel search",
        },
        "pseudo_to_code": {
            "problem_id": "authored-code",
            "pseudocode_lines": ["print a string"],
            "code_lines": ['print("GOLD-CANARY")'],
            "testcases": [[["input"], ["GOLD-CANARY"]]],
        },
    }
    return copy.deepcopy(records[family])


def make_case(family: str = "countdown") -> ProcedureCase:
    return normalize_longproc_record(
        family,
        native_record(family),
        case_id="authored-1",
        output_bucket="2k" if family == "travel_planning" else "0.5k",
        source={"kind": "authored", "label": "independent fixture"},
        html="<table><tr><td>inert</td></tr></table>" if family == "html_to_tsv" else None,
    )


def write_dataset(root: Path, name: str, rows: list | None = None) -> Path:
    family, bucket = name.rsplit("_", 1)
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    row = native_record(family)
    if family == "html_to_tsv":
        row["output_length_group"] = bucket
        webpage = directory / "webpage"
        webpage.mkdir(exist_ok=True)
        (webpage / "table.html").write_text("<script>neverExecute()</script>é", encoding="utf-8")
    filename = "travel_planning_all.json" if family == "travel_planning" else f"{name}.json"
    path = directory / filename
    path.write_text(
        json.dumps(rows if rows is not None else [row], ensure_ascii=False), encoding="utf-8"
    )
    (directory / "prompts.yaml").write_text(
        "!!python/object/apply:neverExecute []", encoding="utf-8"
    )
    if family == "travel_planning":
        demo = native_record(family)
        del demo["estimated_output_tokens"], demo["solving_procedure"]
        (directory / "travel_planning_icl_examples.json").write_text(
            json.dumps([demo]), encoding="utf-8"
        )
    return path


@pytest.mark.parametrize("family", FAMILIES)
def test_six_family_roundtrip_private_reference_and_deep_immutability(family):
    original = make_case(family)
    assert "GOLD-CANARY" not in json.dumps(original.to_dict()["input"])
    assert "GOLD-CANARY" in json.dumps(original.to_dict()["reference"])
    assert ProcedureCase.from_dict(original.to_dict()).digest == original.digest
    with pytest.raises(FrozenInstanceError):
        original.case_id = "changed"
    with pytest.raises(TypeError):
        original.input["changed"] = 1
    serialized = original.to_dict()
    serialized["reference"].clear()
    assert original.reference
    assert len(original.digest) == 64


@pytest.mark.parametrize("family", FAMILIES)
def test_native_unknown_missing_fields_rejected_without_source_text(family):
    row = native_record(family)
    row["PRIVATE_UNKNOWN"] = "private"
    with pytest.raises(ProcedureDataError) as caught:
        normalize_longproc_record(
            family,
            row,
            case_id="one",
            output_bucket="0.5k",
            source={"kind": "authored", "label": "test"},
        )
    assert "PRIVATE" not in str(caught.value)
    del row["PRIVATE_UNKNOWN"]
    row.pop(next(iter(row)))
    with pytest.raises(ProcedureDataError):
        normalize_longproc_record(
            family,
            row,
            case_id="one",
            output_bucket="0.5k",
            source={"kind": "authored", "label": "test"},
        )


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_official_definitions_load_authored_files_without_executing_templates(tmp_path, name):
    write_dataset(tmp_path, name)
    dataset = load_longproc_dataset(tmp_path, name)
    expected = 0 if name == "travel_planning_8k" else 1
    assert len(dataset.cases) == expected
    assert dataset.inventory["available"] == 1
    assert dataset.inventory["selected"] == expected
    assert dataset.inventory["excluded"] == 1 - expected
    assert dataset.to_dict()["format"] == "promptwitness.procedure-dataset/v1"
    if expected:
        case = dataset.cases[0]
        assert case.output_bucket == name.rsplit("_", 1)[1]
        assert case.source["row_index"] == 0
        assert (
            case.source["file_sha256"]
            == hashlib.sha256((tmp_path / case.source["file"]).read_bytes()).hexdigest()
        )
        assert (
            case.source["prompt_sha256"]
            == hashlib.sha256((tmp_path / case.family / "prompts.yaml").read_bytes()).hexdigest()
        )
    assert load_longproc_dataset(tmp_path, name).to_dict() == dataset.to_dict()


def test_travel_exact_half_open_output_boundaries_and_excluded_rows_validated(tmp_path):
    rows = []
    for estimate in [0, 2047, 2048, 4095, 4096, 8191, 8192]:
        row = native_record("travel_planning")
        row["estimated_output_tokens"] = estimate
        rows.append(row)
    path = write_dataset(tmp_path, "travel_planning_2k", rows)
    low = load_longproc_dataset(tmp_path, "travel_planning_2k")
    high = load_longproc_dataset(tmp_path, "travel_planning_8k")
    assert [case.source["row_index"] for case in low.cases] == [0, 1]
    assert [case.source["row_index"] for case in high.cases] == [4, 5]
    assert low.inventory["excluded"] == high.inventory["excluded"] == 5
    assert low.cases[0].input["cities"][0] == {
        "city": "Beta",
        "duration": 3,
        "fixed_start": 2,
        "fixed_end": 4,
    }
    assert low.cases[0].reference["stays"][1]["start_day"] == 2
    rows[-1]["constraints"][0]["num_days"] = True
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ProcedureDataError):
        load_longproc_dataset(tmp_path, "travel_planning_2k")


def test_full_audit_keeps_all_six_families_and_hash_only_inventory(tmp_path):
    for name in DATASET_NAMES:
        write_dataset(tmp_path, name)
    report = audit_longproc_data(tmp_path)
    assert report["families"] == list(FAMILIES)
    assert len(report["datasets"]) == 16
    assert len(report["files"]) == 23  # 15 record files + 6 prompts + demo + one HTML
    assert report["schema_valid"] is True
    assert report["reference_semantics_verified"] is report["model_quality_measured"] is False
    assert "GOLD-CANARY" not in json.dumps(report)
    assert "neverExecute" not in json.dumps(report)


def test_html_asset_binding_duplicates_and_stale_export(tmp_path):
    row = native_record("html_to_tsv")
    write_dataset(tmp_path, "html_to_tsv_0.5k", [row, copy.deepcopy(row)])
    dataset = load_longproc_dataset(tmp_path, "html_to_tsv_0.5k")
    assert dataset.inventory["html_references"] == 2
    assert dataset.inventory["html_unique_files"] == 1
    assert dataset.cases[0].case_id != dataset.cases[1].case_id
    data = dataset.cases[0].to_dict()
    assert "<script>" in data["input"]["html"]
    data["input"]["html"] += "changed"
    with pytest.raises(ProcedureDataError, match="source asset"):
        ProcedureCase.from_dict(data)


@pytest.mark.parametrize(
    "path",
    [
        "../secret.html",
        "/secret.html",
        "C:/secret.html",
        "webpage/../table.html",
        "webpage\\table.html",
        "https://x",
    ],
)
def test_external_html_paths_rejected_before_access(tmp_path, path):
    row = native_record("html_to_tsv")
    row["html_path"] = path
    write_dataset(tmp_path, "html_to_tsv_0.5k", [row])
    with pytest.raises(ProcedureDataError, match="confined"):
        load_longproc_dataset(tmp_path, "html_to_tsv_0.5k")


def test_symlinked_asset_is_rejected(tmp_path):
    write_dataset(tmp_path, "html_to_tsv_0.5k")
    path = tmp_path / "html_to_tsv/webpage/table.html"
    original = path.read_bytes()
    path.unlink()
    elsewhere = tmp_path / "elsewhere.html"
    elsewhere.write_bytes(original)
    try:
        path.symlink_to(elsewhere)
    except OSError:
        pytest.skip("OS does not grant symlink creation")
    with pytest.raises(ProcedureDataError, match="symlinks"):
        load_longproc_dataset(tmp_path, "html_to_tsv_0.5k")


@pytest.mark.parametrize(
    "raw",
    [
        b'[{"a":1,"a":2}]',
        b"[NaN]",
        b"[1e999]",
        b"\xff",
        b'[{"private":',
        b'["\\ud800"]',
        b"[" * 40 + b"]" * 40,
    ],
)
def test_strict_json_faults_are_redacted(tmp_path, raw):
    path = write_dataset(tmp_path, "countdown_0.5k")
    path.write_bytes(raw)
    with pytest.raises(ProcedureDataError) as caught:
        load_longproc_dataset(tmp_path, "countdown_0.5k")
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_file_bytes", 1),
        ("max_case_bytes", 20),
        ("max_total_read_bytes", 1),
        ("max_total_case_bytes", 20),
    ],
)
def test_loader_explicit_budgets_do_not_truncate(tmp_path, field, value):
    write_dataset(tmp_path, "countdown_0.5k")
    with pytest.raises(ProcedureDataError, match="budget"):
        load_longproc_dataset(
            tmp_path, "countdown_0.5k", limits=replace(ProcedureDataLimits(), **{field: value})
        )


def test_html_and_record_count_limits(tmp_path):
    write_dataset(tmp_path, "html_to_tsv_0.5k")
    with pytest.raises(ProcedureDataError, match="budget"):
        load_longproc_dataset(
            tmp_path, "html_to_tsv_0.5k", limits=ProcedureDataLimits(max_html_bytes=1)
        )
    row = native_record("countdown")
    write_dataset(tmp_path, "countdown_0.5k", [row, row])
    with pytest.raises(ProcedureDataError, match="length"):
        load_longproc_dataset(tmp_path, "countdown_0.5k", limits=ProcedureDataLimits(max_records=1))


def test_bad_reference_semantics_preserved_not_repaired():
    row = native_record("countdown")
    row["solution"] = ["1 + 2 = 999"]
    result = normalize_longproc_record(
        "countdown",
        row,
        case_id="bad-gold",
        output_bucket="0.5k",
        source={"kind": "authored", "label": "deliberately wrong"},
    )
    assert result.reference["solution"] == ("1 + 2 = 999",)


def test_derived_fields_cannot_be_mutated_or_bool_aliased():
    for family, field, replacement in [
        ("tom_tracking", "trace", ["changed"]),
        ("travel_planning", "stays", []),
    ]:
        data = make_case(family).to_dict()
        data["reference"][field] = replacement
        with pytest.raises(ProcedureDataError):
            ProcedureCase.from_dict(data)
    data = make_case("travel_planning").to_dict()
    data["input"]["cities"][0]["fixed_start"] = True
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)
    data = make_case().to_dict()
    data["input"]["min_intermediate"] = True
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)


def test_case_envelope_dataset_types_and_duplicate_ids(tmp_path):
    case = make_case()
    data = case.to_dict()
    data["extra"] = 1
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)
    del data["extra"]
    data["format"] = "future"
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)
    with pytest.raises(ProcedureDataError):
        replace(case, family="other")
    with pytest.raises(ProcedureDataError):
        replace(case, output_bucket="input-8192")
    with pytest.raises(ProcedureDataError):
        ProcedureDataset("countdown_0.5k", "countdown", "0.5k", (case, case), {})
    for name in ("travel_planning_0.5k", "pseudo_to_code_8k", "../bad", None):
        with pytest.raises(ProcedureDataError):
            load_longproc_dataset(tmp_path, name)
    with pytest.raises(ProcedureDataError):
        load_longproc_dataset(tmp_path, "countdown_0.5k", limits={})
    with pytest.raises(ProcedureDataError):
        audit_longproc_data(tmp_path, limits={})


@pytest.mark.parametrize("value", [True, 0, -1, 2**53, float("nan"), "10"])
def test_limits_reject_nonportable_values(value):
    with pytest.raises(ProcedureDataError):
        ProcedureDataLimits(max_records=value)


def test_json_encoding_matches_canonical_json_and_enforces_escaped_bytes():
    value = {"unicode": "é😀" * 5000, "control": '\x00\\\n"', "nested": [True, None, 0.25]}
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_bytes(value, limit=len(expected)) == expected
    assert load_json(expected, limit=len(expected)) == value
    with pytest.raises(ProcedureDataError, match="byte budget"):
        canonical_bytes(value, limit=len(expected) - 1)
    for value in ({1: "bad"}, {"bad": object()}, [2**53], [float("inf")]):
        with pytest.raises(ProcedureDataError):
            canonical_bytes(value)


def test_pending_siblings_counted_before_iterating_nested_sequence():
    class NoIteration(list):
        def __iter__(self):
            pytest.fail("over-budget nested sequence was iterated")

    # Outer siblings already consume the remaining node allowance.
    with pytest.raises(ProcedureDataError, match="structural budget"):
        canonical_bytes([None] * 8 + [NoIteration([None] * 8)], node_limit=12)


@pytest.mark.parametrize(
    "family,section,key,value",
    [
        ("countdown", "input", "max_intermediate", 1999),
        ("countdown", "reference", "search_steps", -0.1),
        ("countdown", "input", "numbers", [1, 2, 3, True]),
        ("path_traversal", "input", "edges", [{"source": "A", "target": "B", "method": "walk"}]),
        (
            "path_traversal",
            "input",
            "edges",
            [
                {"source": "A", "target": "B", "method": "bus"},
                {"source": "A", "target": "C", "method": "bus"},
            ],
        ),
        ("html_to_tsv", "input", "header", ["two\tfields"]),
        ("pseudo_to_code", "reference", "testcases", [["not lines", []]]),
        ("travel_planning", "input", "num_cities", 3),
        ("travel_planning", "input", "flights", [["Alpha"]]),
        (
            "travel_planning",
            "input",
            "constraints",
            [
                {"type": "fixed", "city": "Beta", "start_day": 2, "end_day": 4},
            ],
        ),
        (
            "travel_planning",
            "input",
            "constraints",
            [
                {"type": "duration", "city": "Beta", "num_days": 3},
                {"type": "duration", "city": "Beta", "num_days": 3},
            ],
        ),
        (
            "travel_planning",
            "input",
            "constraints",
            [
                {"type": "fixed", "city": "Beta", "start_day": 2, "end_day": 4},
                {"type": "fixed", "city": "Beta", "start_day": 2, "end_day": 4},
            ],
        ),
        ("travel_planning", "input", "constraints", [{"type": "arbitrary-code"}]),
        ("travel_planning", "input", "constraints", ["not object"]),
        ("travel_planning", "reference", "ground_truth_durations", "2"),
        ("travel_planning", "reference", "ground_truth_durations", "2**3.0"),
    ],
)
def test_typed_procedure_constraints_are_closed(family, section, key, value):
    data = make_case(family).to_dict()
    data[section][key] = value
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)


def test_inventory_is_immutable_closed_and_rejects_misbound_digests(tmp_path):
    write_dataset(tmp_path, "countdown_0.5k")
    dataset = load_longproc_dataset(tmp_path, "countdown_0.5k")
    for field, value in (
        ("selected", True),
        ("excluded", 4),
        ("case_digest_sha256", "f" * 64),
        ("all_record_cases_sha256", "bad"),
    ):
        inventory = dataset.to_dict()["inventory"]
        inventory[field] = value
        with pytest.raises(ProcedureDataError):
            replace(dataset, inventory=inventory)
    for change in ("unknown_kind", "duplicate", "reverse"):
        inventory = dataset.to_dict()["inventory"]
        if change == "unknown_kind":
            inventory["files"][0]["kind"] = "plugin"
        elif change == "duplicate":
            inventory["files"].append(inventory["files"][0])
        else:
            inventory["files"].reverse()
        with pytest.raises(ProcedureDataError):
            replace(dataset, inventory=inventory)


def test_longproc_source_identity_and_asset_inventory_are_checked(tmp_path):
    write_dataset(tmp_path, "countdown_0.5k")
    case = load_longproc_dataset(tmp_path, "countdown_0.5k").cases[0]
    for field, value in (
        ("dataset", "countdown_2k"),
        ("kind", "untrusted"),
        ("file_sha256", "bad"),
        ("row_index", True),
    ):
        data = case.to_dict()
        data["source"][field] = value
        with pytest.raises(ProcedureDataError):
            ProcedureCase.from_dict(data)
    data = case.to_dict()
    data["source"]["assets"] = [{"path": "unused", "sha256": "0" * 64, "size": 0}]
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)
    write_dataset(tmp_path, "html_to_tsv_0.5k")
    data = load_longproc_dataset(tmp_path, "html_to_tsv_0.5k").cases[0].to_dict()
    data["source"]["assets"] = []
    with pytest.raises(ProcedureDataError):
        ProcedureCase.from_dict(data)


def test_missing_and_nonregular_files_and_invalid_root_are_controlled(tmp_path):
    with pytest.raises(ProcedureDataError):
        load_longproc_dataset(tmp_path, "countdown_0.5k")
    with pytest.raises(ProcedureDataError):
        load_longproc_dataset(None, "countdown_0.5k")
    path = write_dataset(tmp_path, "countdown_0.5k")
    with pytest.raises(ProcedureDataError):
        load_longproc_dataset(path, "countdown_0.5k")
    path.unlink()
    path.mkdir()
    with pytest.raises(ProcedureDataError, match="regular"):
        load_longproc_dataset(tmp_path, "countdown_0.5k")


def test_small_corruption_after_first_html_reference_cannot_mix_source_versions(
    tmp_path, monkeypatch
):
    from promptwitness.procedure_data import _Reader

    row = native_record("html_to_tsv")
    write_dataset(tmp_path, "html_to_tsv_0.5k", [row, row])
    original_read = _Reader.read
    seen = 0

    def changing_read(self, name, kind):
        nonlocal seen
        if kind == "html":
            seen += 1
            if seen == 2:
                (tmp_path / name).write_text("different source", encoding="utf-8")
        return original_read(self, name, kind)

    monkeypatch.setattr(_Reader, "read", changing_read)
    with pytest.raises(ProcedureDataError, match="between references"):
        load_longproc_dataset(tmp_path, "html_to_tsv_0.5k")


def test_low_record_budget_precedes_normalization(tmp_path, monkeypatch):
    import promptwitness.procedure_data as module

    write_dataset(tmp_path, "countdown_0.5k")
    monkeypatch.setattr(
        module,
        "normalize_longproc_record",
        lambda *args, **kwargs: pytest.fail("over-budget record normalized"),
    )
    with pytest.raises(ProcedureDataError, match="budget"):
        load_longproc_dataset(
            tmp_path, "countdown_0.5k", limits=ProcedureDataLimits(max_case_bytes=16)
        )
