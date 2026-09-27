"""Authored fit preparation/scoring mechanics, not dataset performance."""

from __future__ import annotations

import json
import subprocess

import pytest

from reproduce import check_bfcl_fit as audit
from reproduce import prepare_bfcl_fit as prep
from reproduce.strict_scoring import ScoringError


def test_nonfit_bodies_are_not_decoded():
    rows = ['{"id":"other","ground_truth": MALFORMED}', '{"id":"fit","ground_truth":[]}']
    assert list(prep.selected_lines(rows, {"fit"})) == [{"id": "fit", "ground_truth": []}]


def test_sorted_archive_only_selected_body_decoded():
    rows = [
        '{"family":"bfcl","id":"bfcl:other","output": MALFORMED}',
        '{"family":"bfcl","id":"bfcl:fit","output":"\\"id\\": 0"}',
    ]
    assert len(list(prep.selected_lines(rows, {"bfcl:fit"}, archive=True))) == 1


@pytest.mark.parametrize("line", ['{"key":"a"}', '{"id":"a","nested":{"id":"b"}}'])
def test_ambiguous_archive_ids_fail_before_scoring(line):
    with pytest.raises(ScoringError, match="identifier"):
        list(prep.selected_lines([line], {"a"}, archive=True))


def test_final_archive_id_rejected_before_body_decode():
    with pytest.raises(ScoringError, match="final-derived"):
        list(
            prep.selected_lines(
                ['{"id":"final","output": MALFORMED}'], {"fit"}, archive=True, forbidden={"final"}
            )
        )


def test_duplicate_selected_records_are_errors():
    with pytest.raises(ScoringError, match="duplicate"):
        prep.unique_rows([{"id": "a"}, {"id": "a"}])


@pytest.fixture
def fit_store(monkeypatch, tmp_path):
    root = tmp_path / "store"
    for leaf in prep.LEAVES:
        (root / leaf).mkdir(parents=True)
    functions = [{"name": "authored", "parameters": {"type": "object", "properties": {}}}]
    prep.write_jsonl(
        root / "fit/inputs/bfcl.jsonl",
        [{"id": "a", "category": "simple_python", "function": functions, "question": []}],
    )
    prep.write_jsonl(
        root / "fit/gold/bfcl.jsonl", [{"id": "a", "ground_truth": [{"authored": {}}]}]
    )
    for index, (model, revision) in enumerate(prep.MODEL_REVISIONS.items()):
        prep.write_jsonl(
            root / f"fit/records/bfcl-{index}.jsonl",
            [
                {
                    "id": "a",
                    "model": model,
                    "revision": revision,
                    "status": "completed",
                    "output": '[{"name":"authored","arguments":{}}]',
                    "original_scoring_executed": False,
                }
            ],
        )
    monkeypatch.setenv("PW_ACCESS_ROLE", "fit_learner")
    monkeypatch.setenv("PW_ACCESS_STAGE", "fit")
    monkeypatch.setenv("PW_LANDLOCK_ABI", "1")
    calls = []

    def load(runtime, work, model):
        def check(functions, decoded, answers, category):
            calls.append((model, answers, category))
            return {"valid": True}

        return check, {"revision": "authored", "native_module": "authored.py"}

    monkeypatch.setattr(audit, "load_staged_bfcl", load)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return root, scratch, calls


def test_two_models_receive_gold_and_private_scores(fit_store, tmp_path):
    store, scratch, calls = fit_store
    result = audit.score_fit(store, tmp_path / "runtime", scratch)
    assert [row["correct"] for row in result["models"]] == [1, 1]
    assert len(calls) == 2
    assert all(row[1] == [{"authored": {}}] for row in calls)
    private = json.loads((scratch / "fit-scores.json").read_text())
    assert len(private["models"]) == 2
    assert "checks" not in result
    with pytest.raises(FileExistsError):
        audit.score_fit(store, tmp_path / "runtime", scratch)


def test_unrestricted_or_wrong_role_cannot_score(monkeypatch, tmp_path):
    monkeypatch.delenv("PW_ACCESS_ROLE", raising=False)
    with pytest.raises(ScoringError, match="restricted"):
        audit.score_fit(tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("output", None),
        ("revision", "other"),
        ("original_scoring_executed", True),
    ],
)
def test_invalid_record_never_becomes_zero(fit_store, tmp_path, field, value):
    store, scratch, _ = fit_store
    path = store / "fit/records/bfcl-0.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ScoringError):
        audit.score_fit(store, tmp_path / "runtime", scratch)
    assert not (scratch / "fit-scores.json").exists()


def test_failed_worker_preserves_logs_not_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "staged_inventory", lambda _: {})
    monkeypatch.setattr(
        audit,
        "launch_role",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "partial", "original error"),
    )
    work = tmp_path / "scratch"
    with pytest.raises(ScoringError, match="worker failed"):
        audit.check(tmp_path, tmp_path, work, timeout=30)
    assert (work / "worker.stdout").read_text() == "partial"
    assert (work / "worker.stderr").read_text() == "original error"


def test_timed_out_worker_preserves_exact_partial_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "staged_inventory", lambda _: {})

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("authored", 1, output=b"partial\xff", stderr=b"error")

    monkeypatch.setattr(audit, "launch_role", timed_out)
    work = tmp_path / "scratch"
    with pytest.raises(subprocess.TimeoutExpired):
        audit.check(tmp_path, tmp_path, work, timeout=1)
    assert (work / "worker.stdout").read_bytes() == b"partial\xff"
    assert (work / "worker.stderr").read_bytes() == b"error"


@pytest.fixture
def prepare_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(prep, "verify_bfcl_source", lambda _: None)
    member = {"id": "simple_python_0", "category": "simple_python", "groups": ["authored"]}
    pools = tmp_path / "pools.json"
    pools.write_text(
        json.dumps(
            {
                "status": "METADATA_PROPOSAL_NOT_FORMAL_FREEZE",
                "bfcl": {"pools": {"fit": [member], "final_derived": []}},
            }
        )
    )
    raw = {"id": member["id"], "function": [{"name": "authored"}], "question": ["input"]}
    monkeypatch.setattr(
        prep,
        "git_rows",
        lambda source, relative, selected: (
            [{"id": member["id"], "ground_truth": [{"authored": {}}]}]
            if "possible_answer" in relative
            else [raw]
        ),
    )
    training = tmp_path / "training"
    training.mkdir()
    # Original BFCL JSONL has id first, unlike this helper's sorted-key outputs.
    (training / "bfcl-simple-python.jsonl").write_text(json.dumps(raw), encoding="utf-8")
    archives = []
    for index, (model, revision) in enumerate(prep.MODEL_REVISIONS.items()):
        path = tmp_path / f"archive-{index}.jsonl"
        prep.write_jsonl(
            path,
            [
                {
                    "id": "bfcl:" + member["id"],
                    "family": "bfcl",
                    "role": "task",
                    "model": model,
                    "revision": revision,
                    "status": "completed",
                    "output": "[]",
                    "scoring_executed": False,
                }
            ],
        )
        archives.append(path)
    return pools, training, archives


def test_prepare_preserves_membership_and_selects_all_models(prepare_inputs, tmp_path):
    pools, training, archives = prepare_inputs
    before = pools.read_bytes()
    store = tmp_path / "store"
    result = prep.prepare(tmp_path, pools, training, archives, store)
    assert list(result["prior_fit_responses"].values()) == [1, 1]
    assert result["fit_inputs"] == 1
    assert result["original_fit_input_equality"] is True
    assert pools.read_bytes() == before
    assert all(
        not list((store / leaf).iterdir()) for leaf in prep.LEAVES if not leaf.startswith("fit/")
    )


def test_changed_original_input_fails_before_store_creation(prepare_inputs, tmp_path):
    pools, training, archives = prepare_inputs
    path = training / "bfcl-simple-python.jsonl"
    row = json.loads(path.read_text())
    row["question"] = ["changed"]
    path.write_text(json.dumps(row))
    store = tmp_path / "store"
    with pytest.raises(ScoringError, match="differs"):
        prep.prepare(tmp_path, pools, training, archives, store)
    assert not store.exists()


def test_missing_model_is_not_silently_removed(prepare_inputs, tmp_path):
    pools, training, archives = prepare_inputs
    with pytest.raises(ScoringError, match="missing prior"):
        prep.prepare(tmp_path, pools, training, archives[:1], tmp_path / "store")
