"""Authored fit-selection and failure semantics, not research performance."""

from __future__ import annotations

import json
import subprocess

import pytest

from reproduce import check_text_fit as audit
from reproduce import prepare_text_fit as prep
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    hotpot = {
        "id": "h",
        "question": "authored question",
        "answer": "authored answer",
        "context": {"title": ["title", "distractor"], "sentences": [["a"], ["b"]]},
    }
    annotation = [{"instruction_id": ["punctuation:no_comma"], "kwargs": [{}]}]
    instruction = {
        "key": "i",
        "messages": [{"role": "user", "content": "authored prompt"}],
        "ground_truth": repr(annotation),
    }
    raw = {"hotpotqa": hotpot, "instruction_following": instruction}
    fit = {
        "hotpotqa": {"id": "h", "groups": ["distractor", "title"]},
        "instruction_following": {"id": "i", "groups": ["punctuation:no_comma"]},
    }
    proposal = {
        "status": "METADATA_PROPOSAL_NOT_FORMAL_FREEZE",
        **{
            family: {"pools": {"fit": [member], "search": [], "selection": []}}
            for family, member in fit.items()
        },
    }
    pools = tmp_path / "pools.json"
    pools.write_text(json.dumps(proposal))
    requests = []
    for family, member in fit.items():
        converted, _ = prep.convert(family, raw[family], member)
        requests.append({"id": family + ":" + member["id"], "messages": converted["messages"]})
    write_jsonl(source / "preflight-context-requests.jsonl", requests)
    # A nonfit malformed body proves selection happens before whole-line decoding.
    with (source / "preflight-context-requests.jsonl").open("a") as stream:
        stream.write('{"id":"hotpotqa:nonfit","answer": MALFORMED}\n')
    monkeypatch.setattr(prep, "selected_parquet", lambda src, family, ids: [raw[family]])
    archives = []
    for index, (model, revision) in enumerate(MODEL_REVISIONS.items()):
        path = tmp_path / f"archive-{index}.jsonl"
        write_jsonl(
            path,
            [
                {
                    "id": family + ":" + member["id"],
                    "family": family,
                    "role": "task",
                    "model": model,
                    "revision": revision,
                    "status": "completed",
                    "output": "authored answer",
                    "scoring_executed": False,
                }
                for family, member in fit.items()
            ],
        )
        archives.append(path)
    return source, pools, archives, raw


def prepare_store(prepared, tmp_path):
    source, pools, archives, _ = prepared
    store = tmp_path / "store"
    report = prep.prepare(source, pools, archives, store)
    return store, report


def test_fixed_metadata_and_both_models_preserved(prepared, tmp_path):
    _, pools, _, _ = prepared
    before = pools.read_bytes()
    store, report = prepare_store(prepared, tmp_path)
    assert report["fit_inputs"] == {family: 1 for family in prep.FAMILIES}
    assert all(list(models.values()) == [1, 1] for models in report["prior_fit_responses"].values())
    assert pools.read_bytes() == before
    assert all(
        not list((store / leaf).iterdir()) for leaf in prep.LEAVES if not leaf.startswith("fit/")
    )
    assert (
        "distractor: b"
        in json.loads((store / "fit/inputs/hotpotqa.jsonl").read_text())["messages"][1]["content"]
    )
    assert "answer" not in json.loads((store / "fit/inputs/hotpotqa.jsonl").read_text())


@pytest.mark.parametrize("defect", ["duplicate", "group_overlap", "formal_freeze", "wrong_pools"])
def test_membership_errors_prevent_data_read(prepared, tmp_path, monkeypatch, defect):
    source, pools, archives, _ = prepared
    proposal = json.loads(pools.read_text())
    if defect == "formal_freeze":
        proposal["status"] = "FORMAL_FREEZE"
    elif defect == "wrong_pools":
        proposal["hotpotqa"]["pools"]["final_derived"] = []
    else:
        member = dict(proposal["hotpotqa"]["pools"]["fit"][0])
        if defect == "group_overlap":
            member["id"] = "other"
        proposal["hotpotqa"]["pools"]["search"] = [member]
    pools.write_text(json.dumps(proposal))
    monkeypatch.setattr(prep, "selected_parquet", lambda *a: pytest.fail("read before validation"))
    with pytest.raises(ScoringError):
        prep.prepare(source, pools, archives, tmp_path / "store")


@pytest.mark.parametrize(
    "defect", ["completion", "changed_group", "changed_prompt", "missing_model"]
)
def test_input_archive_errors_fail_before_store_creation(prepared, tmp_path, defect):
    source, pools, archives, raw = prepared
    if defect == "completion":
        raw["instruction_following"]["messages"][0]["role"] = "assistant"
    elif defect == "changed_group":
        raw["hotpotqa"]["context"]["title"][0] = "changed"
    elif defect == "changed_prompt":
        raw["hotpotqa"]["question"] = "different request"
    else:
        archives = archives[:1]
    store = tmp_path / "store"
    with pytest.raises(ScoringError):
        prep.prepare(source, pools, archives, store)
    assert not store.exists()


@pytest.fixture
def scored(monkeypatch, prepared, tmp_path):
    store, _ = prepare_store(prepared, tmp_path)
    work = tmp_path / "scratch"
    work.mkdir()
    monkeypatch.setenv("PW_ACCESS_ROLE", "fit_learner")
    monkeypatch.setenv("PW_ACCESS_STAGE", "fit")
    monkeypatch.setenv("PW_LANDLOCK_ABI", "1")

    class Checker:
        def __init__(self, name):
            pass

        def build_description(self, **kwargs):
            pass

        def check_following(self, text):
            return "," not in text

    monkeypatch.setattr(
        audit,
        "load_staged",
        lambda _: ({"punctuation:no_comma": Checker}, None, lambda a, b: a == b),
    )
    monkeypatch.setattr(audit, "mechanical_checks", lambda *a: ["authored"])
    return store, work


def test_only_aggregates_public_and_private_scores_exclusive(scored, tmp_path):
    store, work = scored
    report = audit.score_fit(store, tmp_path / "runtime", work)
    for value in report["families"].values():
        assert [row["correct"] for row in value["models"]] == [1, 1]
    assert '"checks"' not in json.dumps(report)
    private = json.loads((work / "text-fit-scores.json").read_text())
    assert private["hotpotqa"][0]["checks"][0]["unit"] == "h"
    with pytest.raises(FileExistsError):
        audit.score_fit(store, tmp_path / "runtime", work)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("output", None),
        ("revision", "other"),
        ("original_scoring_executed", True),
    ],
)
def test_failed_missing_or_changed_response_not_zero(scored, tmp_path, field, value):
    store, work = scored
    path = store / "fit/records/hotpotqa-0.jsonl"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ScoringError):
        audit.score_fit(store, tmp_path / "runtime", work)
    assert not (work / "text-fit-scores.json").exists()


def test_reject_unrestricted_role_before_native_import(monkeypatch, tmp_path):
    monkeypatch.delenv("PW_ACCESS_ROLE", raising=False)
    monkeypatch.setattr(audit, "load_staged", lambda _: pytest.fail("unrestricted import"))
    with pytest.raises(ScoringError, match="restricted"):
        audit.score_fit(tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_failure_preserves_logs_not_result(monkeypatch, tmp_path, timeout):
    monkeypatch.setattr(audit, "staged_inventory", lambda _: {})

    def launch(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired("authored", 1, output=b"partial", stderr=b"error")
        return subprocess.CompletedProcess([], 1, "partial", "error")

    monkeypatch.setattr(audit, "launch_role", launch)
    work = tmp_path / "work"
    with pytest.raises(subprocess.TimeoutExpired if timeout else ScoringError):
        audit.check(tmp_path, tmp_path, work, timeout=1)
    assert (work / "worker.stdout").read_bytes() == b"partial"
    assert (work / "worker.stderr").read_bytes() == b"error"
