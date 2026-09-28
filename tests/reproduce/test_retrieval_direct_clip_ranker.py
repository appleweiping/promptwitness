"""Input-only full-pool CLIP wiring; fake features are not model evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_direct_clip_ranker as ranker_module
from reproduce.process_access import LEAVES
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_direct_clip_ranker import (
    DirectClipRanker,
    load_input_only,
    require_complete_ranking,
)
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger


def _inputs(directory: Path, dataset: str, rows: list[dict], pools: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{dataset}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / f"{dataset}-gallery.json").write_text(json.dumps(pools), encoding="utf-8")


def _row(identifier: str, reference: str = "reference", category: str = "") -> dict:
    return {
        "id": identifier,
        "reference_id": reference,
        "modification": "make it blue",
        "category": category,
    }


def _fake_model(monkeypatch, *, fail_open: bool = False) -> None:
    class ImageContext:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self.path

        def __exit__(self, *_args):
            return False

    class FakeEncoder:
        def __init__(self, _checkpoint):
            self.checkpoint_sha256 = "authored-fake"
            self.costs = {"image_forward_calls": 0, "text_forward_calls": 0}

        def encode_image(self, image):
            self.costs["image_forward_calls"] += 1
            return image.name

        def encode_text(self, value):
            self.costs["text_forward_calls"] += 1
            return value

    torch = ModuleType("torch")
    torch.stack = tuple
    pil = ModuleType("PIL")
    image = ModuleType("Image")

    def open_image(path):
        if fail_open:
            raise OSError("image decode failed")
        return ImageContext(path)

    image.open = open_image
    pil.Image = image
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(ranker_module, "ClipCPUEncoder", FakeEncoder)

    def fake_rank(gallery, ids, reference, text, weight):
        assert gallery == ids
        assert reference == "reference" and text == "make it blue" and weight == 0.5
        return ("target", *(item for item in ids if item != "target"))

    monkeypatch.setattr(ranker_module, "rank_direct_composed", fake_rank)


def test_input_only_schema_and_category_membership(tmp_path):
    inputs = tmp_path / "inputs"
    _inputs(inputs, "cirr", [_row("q1")], {"cirr": ["reference", "target"]})
    assert load_input_only(inputs, "cirr") == (
        {"q1": ("reference", "make it blue", "cirr")},
        {"cirr": ("reference", "target")},
    )
    _inputs(
        inputs,
        "fashioniq",
        [_row("q1", "reference", "dress")],
        {"dress": ["reference", "target"], "shirt": ["s"], "toptee": ["t"]},
    )
    assert load_input_only(inputs, "fashioniq")[0]["q1"][2] == "dress"
    with pytest.raises(ValueError, match="reference"):
        _inputs(
            inputs,
            "fashioniq",
            [_row("q1", "shirt-reference", "dress")],
            {"dress": ["reference", "target"], "shirt": ["shirt-reference"], "toptee": ["t"]},
        )
        load_input_only(inputs, "fashioniq")


def test_gold_fields_duplicate_ids_and_incomplete_gallery_rejected(tmp_path):
    inputs = tmp_path / "inputs"
    row = _row("q1")
    _inputs(inputs, "cirr", [{**row, "target_id": "target"}], {"cirr": ["reference", "target"]})
    with pytest.raises(ValueError, match="input-only"):
        load_input_only(inputs, "cirr")
    _inputs(inputs, "cirr", [row, row], {"cirr": ["reference", "target"]})
    with pytest.raises(ValueError, match="unique query"):
        load_input_only(inputs, "cirr")
    _inputs(inputs, "cirr", [row], {"cirr": ["reference", "reference"]})
    with pytest.raises(ValueError, match="ordered unique"):
        load_input_only(inputs, "cirr")


def test_full_gallery_index_query_and_replay_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    inputs = tmp_path / "inputs"
    _inputs(inputs, "cirr", [_row("q1")], {"cirr": ["reference", "target"]})
    paths = {name: tmp_path / name for name in ("reference", "target")}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    try:
        ranker = DirectClipRanker(
            input_dir=inputs,
            dataset="cirr",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix="authored",
        )
        assert ranker.rank_one("q1") == ("target", "reference")
        with pytest.raises(ValueError, match="outside"):
            ranker.rank_one("not-in-input")
        with pytest.raises(ValueError, match="replayed"):
            ranker.rank_one("q1")
        summary = ledger.summary()
        assert (summary["attempts"], summary["known_forward_calls"]) == (5, 3)
        assert (summary["failed"], summary["unresolved"]) == (0, 0)
    finally:
        ledger.close()


def test_supplied_description_uses_same_reference_and_distinct_physical_receipts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    seen = []

    def fake_rank(gallery, ids, reference, text, weight):
        seen.append((gallery, ids, reference, text, weight))
        return ("target", "reference")

    monkeypatch.setattr(ranker_module, "rank_direct_composed", fake_rank)
    inputs = tmp_path / "inputs"
    _inputs(inputs, "cirr", [_row("q1")], {"cirr": ["reference", "target"]})
    paths = {name: tmp_path / name for name in ("reference", "target")}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    try:
        ranker = DirectClipRanker(
            input_dir=inputs,
            dataset="cirr",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix="authored",
        )
        assert ranker.rank_description("q1", "child-a:q1", "a blue square") == (
            "target",
            "reference",
        )
        assert seen == [
            (("reference", "target"), ("reference", "target"), "reference", "a blue square", 0.5)
        ]
        with pytest.raises(ValueError, match="replayed"):
            ranker.rank_description("q1", "child-a:q1", "a red square")
        assert len(seen) == 1
        with pytest.raises(ValueError, match="nonempty target description"):
            ranker.rank_description("q1", "child-b:q1", "   ")
        summary = ledger.summary()
        assert (summary["attempts"], summary["known_forward_calls"]) == (5, 3)
        assert summary["failed"] == summary["unresolved"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("stage", ["fit", "selection"])
def test_fit_or_selection_ranker_charges_its_own_input_only_stage(tmp_path, monkeypatch, stage):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    inputs = tmp_path / stage / "inputs"
    _inputs(inputs, "cirr", [_row("stage-q")], {"cirr": ["reference", "target"]})
    paths = {name: tmp_path / name for name in ("reference", "target")}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / f"{stage}-work.sqlite")
    try:
        ranker = DirectClipRanker(
            input_dir=inputs,
            dataset="cirr",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix=f"authored-{stage}",
            stage=stage,
        )
        assert ranker.rank_one("stage-q") == ("target", "reference")
        assert ranker.rank_description("stage-q", "child:q", "make it blue") == (
            "target",
            "reference",
        )
        stages = ledger.connection.execute(
            "SELECT DISTINCT stage FROM retrieval_work ORDER BY stage"
        ).fetchall()
        assert stages == [(stage,), ("shared",)]
    finally:
        ledger.close()


def test_final_ranker_is_not_admitted(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="only fit, search or selection"):
        DirectClipRanker(
            input_dir=tmp_path,
            dataset="cirr",
            image_paths={},
            checkpoint=tmp_path / "missing.pt",
            image_weight=0.5,
            ledger=None,
            attempt_prefix="authored-final",
            stage="final",
        )


@pytest.mark.parametrize(
    "ranking",
    [
        ("target",),
        ("reference", "target", "extra"),
        ("target", "target"),
        ("reference", "wrong-category"),
        ["target", "reference"],
    ],
)
def test_incomplete_or_wrong_category_ranking_is_rejected(ranking):
    with pytest.raises(ValueError, match="complete category gallery"):
        require_complete_ranking(ranking, ("reference", "target"))
    assert require_complete_ranking(("target", "reference"), ("reference", "target")) == (
        "target",
        "reference",
    )


def test_invalid_selection_ranking_is_charged_as_failed_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    monkeypatch.setattr(ranker_module, "rank_direct_composed", lambda *_args: ("target",))
    inputs = tmp_path / "selection/inputs"
    _inputs(inputs, "cirr", [_row("selection-q")], {"cirr": ["reference", "target"]})
    paths = {name: tmp_path / name for name in ("reference", "target")}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "selection-invalid-work.sqlite")
    try:
        ranker = DirectClipRanker(
            input_dir=inputs,
            dataset="cirr",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix="authored-selection-invalid",
            stage="selection",
        )
        with pytest.raises(ValueError, match="complete category gallery"):
            ranker.rank_one("selection-q")
        assert ledger.summary()["failed"] == 1
        assert ledger.connection.execute(
            "SELECT stage,status FROM retrieval_work WHERE kind='rank_callback'"
        ).fetchall() == [("selection", "failed")]
    finally:
        ledger.close()


def test_fashioniq_uses_only_each_category_full_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    inputs = tmp_path / "inputs"
    pools = {
        "dress": ["reference", "target"],
        "shirt": ["reference", "target", "shirt-other"],
        "toptee": ["reference", "target", "toptee-other-a", "toptee-other-b"],
    }
    _inputs(
        inputs,
        "fashioniq",
        [
            _row("dress-q", category="dress"),
            _row("shirt-q", category="shirt"),
            _row("toptee-q", category="toptee"),
        ],
        pools,
    )
    paths = {
        image_id: tmp_path / image_id
        for image_id in dict.fromkeys(item for pool in pools.values() for item in pool)
    }
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    try:
        ranker = DirectClipRanker(
            input_dir=inputs,
            dataset="fashioniq",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix="authored-fashion",
        )
        for category in ("dress", "shirt", "toptee"):
            expected = ("target", *(item for item in pools[category] if item != "target"))
            assert ranker.rank_one(f"{category}-q") == expected
        assert ledger.summary()["known_forward_calls"] == len(paths) + 3
    finally:
        ledger.close()


def test_image_open_failure_is_charged_before_encoder_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch, fail_open=True)
    inputs = tmp_path / "inputs"
    _inputs(inputs, "cirr", [_row("q1")], {"cirr": ["reference", "target"]})
    paths = {name: tmp_path / name for name in ("reference", "target")}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    try:
        with pytest.raises(OSError, match="decode failed"):
            DirectClipRanker(
                input_dir=inputs,
                dataset="cirr",
                image_paths=paths,
                checkpoint=tmp_path / "fake.pt",
                image_weight=0.5,
                ledger=ledger,
                attempt_prefix="authored",
            )
        assert ledger.summary()["attempts"] == 2
        assert ledger.summary()["failed"] == 1
        assert ledger.summary()["known_forward_calls"] == 0
    finally:
        ledger.close()


def test_authored_input_to_full_rank_to_restricted_gate(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux Landlock worker required")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_model(monkeypatch)
    store = tmp_path / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    pool = ("reference", "target", *(f"other-{i}" for i in range(9)))
    rows = [_row(f"q{i}") for i in range(64)]
    input_dir = store / "search/inputs"
    _inputs(input_dir, "cirr", rows, {"cirr": pool})
    (store / "search/gold/cirr.jsonl").write_text(
        "".join(
            json.dumps({"id": row["id"], "target_id": "target", "subset": ["reference", "target"]})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    old_rank = ("reference", *pool[2:], "target")
    old = score_rankings_restricted(
        store, scratch, "cirr", "search", {row["id"]: old_rank for row in rows}
    )
    reference = {
        query_id: observation["primary_hit"]
        for query_id, observation in old["observations"].items()
    }
    assert set(reference.values()) == {0}
    paths = {image_id: tmp_path / image_id for image_id in pool}
    for path in paths.values():
        path.touch()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    plan = make_plan(
        reference,
        candidate_digest=digest("authored-direct-child"),
        execution_digest=digest("authored-direct-clip"),
        fixture_seed=11,
    )
    journal = AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="authored-direct-cir",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
    )
    try:
        ranker = DirectClipRanker(
            input_dir=input_dir,
            dataset="cirr",
            image_paths=paths,
            checkpoint=tmp_path / "fake.pt",
            image_weight=0.5,
            ledger=ledger,
            attempt_prefix="authored-direct",
        )
        bridge = RetrievalAuditBridge(
            store=store,
            scratch_root=scratch,
            dataset="cirr",
            plan=plan,
            journal=journal,
            work_ledger=ledger,
            unit_ids=tuple(reference),
            rank_one=ranker.rank_one,
        )
        verdict = bridge.evaluate(
            contract=ContractCheck(ContractStatus.VALID, ()), execution_scope="FROZEN_TABLE"
        )
        assert verdict.status == GateStatus.ELIGIBLE
        assert bridge.complete_survivor().scores == (1,) * 64
        assert ledger.summary()["known_forward_calls"] == len(pool) + 64
        assert ledger.summary()["unresolved"] == 0
    finally:
        journal.close()
        ledger.close()
