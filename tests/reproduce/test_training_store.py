"""Authored complete training-store checks; these are not benchmark results."""

from __future__ import annotations

import copy
import json

import pytest

from reproduce import prepare_training_store as prep
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    source, bfcl_fit, text_fit = [tmp_path / name for name in ("source", "bfcl-fit", "text-fit")]
    source.mkdir()
    proposal = {"status": "METADATA_PROPOSAL_NOT_FORMAL_FREEZE"}
    original, inputs, gold = {}, {}, {}
    for family in prep.FAMILIES:
        pools, original[family], inputs[family], gold[family] = {}, {}, {}, {}
        for pool in prep.POOLS:
            unit, group = f"{family}-{pool}", f"{family}-group-{pool}"
            member = {"id": unit, "groups": [group]}
            if family == "bfcl":
                category = "irrelevance" if pool == "selection" else "simple_python"
                member["category"] = category
                row = {
                    "id": unit,
                    "function": [{"name": group}],
                    "question": [[{"role": "user", "content": "authored"}]],
                }
                annotation = {
                    "id": unit,
                    "ground_truth": None if category == "irrelevance" else [{group: {"x": [1]}}],
                }
                converted = {**row, "category": category}
            elif family == "hotpotqa":
                row = {
                    "id": unit,
                    "question": "authored question",
                    "answer": "authored answer",
                    "context": {"title": [group], "sentences": [["authored context"]]},
                }
                converted, annotation = prep.text.convert(family, row, member)
            else:
                row = {
                    "key": unit,
                    "messages": [{"role": "user", "content": "authored instruction"}],
                    "ground_truth": repr([{"instruction_id": [group], "kwargs": [{}]}]),
                }
                converted, annotation = prep.text.convert(family, row, member)
            pools[pool] = [member]
            original[family][unit] = row
            inputs[family][unit], gold[family][unit] = converted, annotation
        if family == "bfcl":
            pools["final_derived"] = [{"id": "never-decode-final", "groups": ["final-group"]}]
        proposal[family] = {"pools": pools}
        fit_root = bfcl_fit if family == "bfcl" else text_fit
        unit = pools["fit"][0]["id"]
        for leaf in ("fit/inputs", "fit/gold", "fit/records"):
            (fit_root / leaf).mkdir(parents=True, exist_ok=True)
        write_jsonl(fit_root / f"fit/inputs/{family}.jsonl", [inputs[family][unit]])
        write_jsonl(fit_root / f"fit/gold/{family}.jsonl", [gold[family][unit]])
        for index, (model, revision) in enumerate(MODEL_REVISIONS.items()):
            write_jsonl(
                fit_root / f"fit/records/{family}-{index}.jsonl",
                [
                    {
                        "id": unit,
                        "model": model,
                        "revision": revision,
                        "status": "completed",
                        "output": "authored",
                        "original_scoring_executed": False,
                    }
                ],
            )
    pools_path = tmp_path / "pools.json"
    pools_path.write_text(json.dumps(proposal), encoding="utf-8")
    calls = []

    def git_rows(src, relative, selected):
        calls.append((relative, set(selected)))
        assert "never-decode-final" not in selected
        is_gold = "possible_answer" in relative
        return [
            copy.deepcopy((gold if is_gold else original)["bfcl"][unit])
            for unit in sorted(selected)
        ]

    def selected_parquet(src, family, selected):
        calls.append((family, set(selected)))
        return [copy.deepcopy(original[family][unit]) for unit in sorted(selected)]

    monkeypatch.setattr(prep, "verify_bfcl_source", lambda _: None)
    monkeypatch.setattr(prep, "git_rows", git_rows)
    monkeypatch.setattr(prep.text, "selected_parquet", selected_parquet)
    return source, pools_path, bfcl_fit, text_fit, proposal, original, gold, calls


def prepare(fixture, tmp_path):
    source, pools, bfcl_fit, text_fit, *_ = fixture
    return prep.prepare(source, source, pools, bfcl_fit, text_fit, tmp_path / "store")


def test_all_three_pools_and_all_families_preserved(fixture, tmp_path):
    before = fixture[1].read_bytes()
    report = prepare(fixture, tmp_path)
    assert fixture[1].read_bytes() == before
    assert report["counts"] == {
        family: {pool: 1 for pool in prep.POOLS} for family in prep.FAMILIES
    }
    assert report["previous_fit_inputs_and_gold_equal"] is True
    assert report["reference_scores"] == "NOT_GENERATED"
    assert report["final_content_read"] is False
    assert report["status"] == "PREPARED_NOT_SCIENTIFIC_FREEZE"
    store = tmp_path / "store"
    assert len(list(store.rglob("*.jsonl"))) == 24
    for family in prep.FAMILIES:
        for pool in prep.POOLS:
            row = prep.read_rows(store / f"{pool}/inputs/{family}.jsonl")
            assert set(row) == {f"{family}-{pool}"}
            assert "ground_truth" not in next(iter(row.values()))
            assert "answer" not in next(iter(row.values()))
    for leaf in report["empty_leaves"]:
        assert not list((store / leaf).iterdir())
    assert (
        prep.read_rows(store / "selection/gold/bfcl.jsonl")["bfcl-selection"]["ground_truth"]
        is None
    )


@pytest.mark.parametrize(
    "defect", ["duplicate", "overlap", "empty", "wrong_pool", "category", "state"]
)
def test_membership_failure_before_benchmark_read(fixture, tmp_path, monkeypatch, defect):
    _, path, _, _, proposal, *_ = fixture
    pools = proposal["bfcl"]["pools"]
    if defect == "duplicate":
        pools["selection"][0]["id"] = pools["fit"][0]["id"]
    elif defect == "overlap":
        pools["search"][0]["groups"] = pools["final_derived"][0]["groups"]
    elif defect == "empty":
        pools["selection"] = []
    elif defect == "wrong_pool":
        pools["typo"] = []
    elif defect == "category":
        pools["search"][0]["category"] = "online"
    else:
        proposal["status"] = "FORMAL_FREEZE"
    path.write_text(json.dumps(proposal), encoding="utf-8")
    monkeypatch.setattr(
        prep, "verify_bfcl_source", lambda _: pytest.fail("data source before validation")
    )
    with pytest.raises(ScoringError):
        prepare(fixture, tmp_path)
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "defect", ["group", "missing_annotation", "fit_changed", "response_model", "response_id"]
)
def test_actual_source_and_previous_fit_semantics(fixture, tmp_path, defect):
    _, _, bfcl_fit, text_fit, _, original, gold, _ = fixture
    if defect == "group":
        original["hotpotqa"]["hotpotqa-search"]["context"]["title"] = ["changed"]
    elif defect == "missing_annotation":
        gold["bfcl"]["bfcl-search"]["ground_truth"] = []
    elif defect == "fit_changed":
        original["instruction_following"]["instruction_following-fit"]["messages"][0]["content"] = (
            "changed"
        )
    else:
        path = (bfcl_fit if defect == "response_id" else text_fit) / (
            "fit/records/bfcl-0.jsonl"
            if defect == "response_id"
            else "fit/records/hotpotqa-0.jsonl"
        )
        row = next(iter(prep.read_rows(path).values()))
        row["id" if defect == "response_id" else "model"] = "unapproved"
        path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ScoringError):
        prepare(fixture, tmp_path)
    assert not (tmp_path / "store").exists()


def test_existing_and_partial_outputs_not_overwritten(fixture, tmp_path, monkeypatch):
    real_write = prep.write_jsonl
    count = 0

    def failing_write(path, rows):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("authored I/O failure")
        return real_write(path, rows)

    monkeypatch.setattr(prep, "write_jsonl", failing_write)
    with pytest.raises(OSError, match="I/O failure"):
        prepare(fixture, tmp_path)
    assert not (tmp_path / "store/preparation.json").exists()
    before = (tmp_path / "store/fit/inputs/bfcl.jsonl").read_bytes()
    monkeypatch.setattr(prep, "write_jsonl", real_write)
    with pytest.raises(FileExistsError):
        prepare(fixture, tmp_path)
    assert (tmp_path / "store/fit/inputs/bfcl.jsonl").read_bytes() == before
