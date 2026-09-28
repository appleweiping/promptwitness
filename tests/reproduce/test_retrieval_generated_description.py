"""Authored transport checks; these are not real caption/model experiments."""

import copy
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from promptwitness.incremental.budget import ResourceLedger
from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import check_retrieval_rank_session, retrieval_rank_role
from reproduce.online_resources import CEILING, GPU, STAGE
from reproduce.persistent_model import PersistentModel
from reproduce.process_access import LEAVES
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_generated_description import GeneratedDescriptionRanker, description_wire
from reproduce.retrieval_rank_role import RestrictedRankerSession
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.torch_runtime import RETRIEVAL_DESCRIPTION_CAP


def candidate(content="Describe {{ reference_caption }} after {{ modification }}"):
    return {
        "schema_version": 1,
        "id": "authored",
        "messages": [{"role": "user", "id": "request", "content": content}],
    }


def input_dir(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "cirr.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "reference_id": "r1",
                "modification": "make it blue",
                "category": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "cirr-gallery.json").write_text(
        json.dumps({"cirr": ["r1", "target"]}), encoding="utf-8"
    )
    return directory


def authored_execution():
    return {
        "model_revision": "AUTHORED",
        "tokenizer_revision": "AUTHORED",
        "backend_version": "AUTHORED_NO_INFERENCE",
        **{
            key: "a" * 64
            for key in (
                "backend_config_digest",
                "template_digest",
                "decoding_digest",
                "scorer_digest",
                "data_digest",
                "tool_environment_digest",
            )
        },
    }


def test_description_wire_binds_exact_input_and_does_not_mutate_candidate():
    prompt = candidate()
    original = copy.deepcopy(prompt)
    wire = description_wire(
        prompt,
        query_id="q1",
        reference_caption="a red coat",
        modification="make it blue",
        replicate="seed-11",
    )
    assert prompt == original
    assert wire == {
        "id": "q1",
        "replicate": "seed-11",
        "role": "task",
        "family": "cir_description",
        "messages": [{"role": "user", "content": "Describe a red coat after make it blue"}],
        "max_new_tokens": RETRIEVAL_DESCRIPTION_CAP,
    }
    prefix = description_wire(
        candidate("Return one concise target description."),
        query_id="q1",
        reference_caption="a red coat",
        modification="make it blue",
        replicate="seed-11",
    )
    assert prefix["messages"][-1]["content"].startswith("Reference image caption: a red coat")
    assert "Requested modification: make it blue" in prefix["messages"][-1]["content"]


@pytest.mark.parametrize(
    "prompt",
    [
        candidate("{{ reference_caption }}"),
        candidate("{{ target_id }}"),
        candidate("{{ malformed"),
        {**candidate(), "tools": [{"name": "lookup", "parameters": {"type": "object"}}]},
        {**candidate(), "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
    ],
)
def test_description_wire_rejects_partial_annotation_or_unsupported_prompt(prompt):
    with pytest.raises((ValueError, KeyError)):
        description_wire(
            prompt,
            query_id="q1",
            reference_caption="a red coat",
            modification="make it blue",
            replicate="seed-11",
        )


class FakeModel:
    def __init__(self, output="a blue coat"):
        self.output = output
        self.calls = []

    def metered(self, call_id, execution, wire, *, purpose):
        self.calls.append((call_id, execution, wire, purpose))
        return {
            "id": wire["id"],
            "replicate": wire["replicate"],
            "status": "completed",
            "output": self.output,
        }


class FakeRanker:
    def __init__(self):
        self.calls = []

    def rank_description(self, query_id, request_id, description):
        self.calls.append((query_id, request_id, description))
        return ("target", "r1")


def test_generated_description_reaches_ranker_only_after_metered_response(tmp_path):
    prompt = candidate()
    model, ranker = FakeModel(), FakeRanker()
    adapter = GeneratedDescriptionRanker(
        input_dir=input_dir(tmp_path),
        dataset="cirr",
        captions={"r1": "a red coat"},
        candidate=prompt,
        model=model,
        ranker=ranker,
        execution={"frozen": "authored"},
        replicate="seed-11",
        attempt_prefix="child-a",
    )
    prompt["messages"][0]["content"] = "changed after freeze"
    assert adapter.rank_one("q1") == ("target", "r1")
    assert model.calls[0][0] == "child-a:generate:q1"
    assert model.calls[0][2]["messages"] == [
        {"role": "user", "content": "Describe a red coat after make it blue"}
    ]
    assert model.calls[0][3] == "search"
    assert ranker.calls == [("q1", "child-a:rank:q1", "a blue coat")]
    with pytest.raises(ValueError, match="outside"):
        adapter.rank_one("not-a-query")
    assert len(model.calls) == 1


def test_empty_completed_model_output_is_not_ranked_or_retried(tmp_path):
    model, ranker = FakeModel(" \n "), FakeRanker()
    adapter = GeneratedDescriptionRanker(
        input_dir=input_dir(tmp_path),
        dataset="cirr",
        captions={"r1": "a red coat"},
        candidate=candidate(),
        model=model,
        ranker=ranker,
        execution={},
        replicate="seed-11",
        attempt_prefix="child-b",
    )
    with pytest.raises(ValueError, match="empty"):
        adapter.rank_one("q1")
    assert len(model.calls) == 1
    assert ranker.calls == []


def test_missing_or_extra_caption_is_rejected_before_model_call(tmp_path):
    directory = input_dir(tmp_path)
    for captions in ({}, {"r1": "red coat", "target": "gold-like extra"}):
        model = FakeModel()
        with pytest.raises(ValueError, match="one nonempty fixed caption"):
            GeneratedDescriptionRanker(
                input_dir=directory,
                dataset="cirr",
                captions=captions,
                candidate=candidate(),
                model=model,
                ranker=FakeRanker(),
                execution={},
                replicate="seed-11",
                attempt_prefix="child-c",
            )
        assert model.calls == []


def test_adapter_charges_original_physical_ledger_before_rank_and_replay(tmp_path, monkeypatch):
    directory = input_dir(tmp_path)
    ledger = ResourceLedger(
        tmp_path / "AUTHORED.sqlite",
        historical_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        historical_digest="a" * 64,
        global_limit=CEILING,
        stage_limits={STAGE: CEILING},
        gpu_uuid=GPU,
    )
    execution = authored_execution()
    model = PersistentModel.__new__(PersistentModel)
    model.directory = tmp_path / "AUTHORED_process"
    model.directory.mkdir()
    model.ledger, model.allocation_id = ledger, "AUTHORED"
    model.profile = {"backend_version": execution["backend_version"]}
    model.failed = False

    def generate(_execution, wire):
        assert wire["family"] == "cir_description"
        assert ledger.usage()["calls"] == 1  # reservation precedes inference
        return {
            "id": wire["id"],
            "replicate": wire["replicate"],
            "status": "completed",
            "input_tokens": 17,
            "output_tokens": 4,
            "allocated_seconds": 0.25,
            "output": "a blue coat",
        }

    monkeypatch.setattr(model, "_execute", generate)
    ranker = FakeRanker()
    adapter = GeneratedDescriptionRanker(
        input_dir=directory,
        dataset="cirr",
        captions={"r1": "a red coat"},
        candidate=candidate(),
        model=model,
        ranker=ranker,
        execution=execution,
        replicate="seed-11",
        attempt_prefix="physical-child",
    )
    try:
        assert adapter.rank_one("q1") == ("target", "r1")
        assert ledger.usage()["calls"] == 1
        assert ledger.usage()["input_tokens"] == 17
        assert ledger.usage()["output_tokens"] == 4
        assert ranker.calls == [("q1", "physical-child:rank:q1", "a blue coat")]
        with pytest.raises(sqlite3.IntegrityError):
            adapter.rank_one("q1")
        assert ledger.usage()["calls"] == 1
        assert len(ranker.calls) == 1
    finally:
        ledger.close()


def test_authored_model_accounting_to_landlock_ranker_and_restricted_gate(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux Landlock ranker and scorer workers required")
    monkeypatch.setattr(
        retrieval_rank_role, "ENTRYPOINT", Path(check_retrieval_rank_session.__file__)
    )
    store = tmp_path / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    for leaf in ("fit/gold", "search/gold", "selection/gold", "final/gold"):
        (store / leaf / "sentinel.txt").write_text("AUTHORED-GOLD", encoding="utf-8")
    pool = ("reference", "target", *(f"other-{index}" for index in range(9)))
    rows = [
        {
            "id": f"q{index}",
            "reference_id": "reference",
            "modification": "make it blue",
            "category": "",
        }
        for index in range(64)
    ]
    inputs = store / "search/inputs"
    (inputs / "cirr.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (inputs / "cirr-gallery.json").write_text(json.dumps({"cirr": pool}), encoding="utf-8")
    (store / "search/gold/cirr.jsonl").write_text(
        "".join(
            json.dumps({"id": row["id"], "target_id": "target", "subset": ["reference", "target"]})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    (inputs / "model.pt").write_bytes(b"authored-not-a-model")
    image_paths = {}
    for image_id in pool:
        relative = f"images/{image_id}.png"
        path = inputs / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"authored-not-an-image")
        image_paths[image_id] = relative
    score_scratch = tmp_path / "score-scratch"
    rank_scratch = tmp_path / "rank-scratch"
    score_scratch.mkdir()
    rank_scratch.mkdir()
    old_rank = ("reference", *pool[2:], "target")
    old = score_rankings_restricted(
        store, score_scratch, "cirr", "search", {row["id"]: old_rank for row in rows}
    )
    reference = {
        query_id: observation["primary_hit"]
        for query_id, observation in old["observations"].items()
    }
    assert set(reference.values()) == {0}
    plan = make_plan(
        reference,
        candidate_digest=digest("authored-generated-child"),
        execution_digest=digest("authored-generated-ranker"),
        fixture_seed=11,
    )
    journal = AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="authored-generated-cir",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
    )
    model_ledger = ResourceLedger(
        tmp_path / "AUTHORED-model.sqlite",
        historical_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        historical_digest="a" * 64,
        global_limit=CEILING,
        stage_limits={STAGE: CEILING},
        gpu_uuid=GPU,
    )
    model = PersistentModel.__new__(PersistentModel)
    model.directory = tmp_path / "AUTHORED-model-process"
    model.directory.mkdir()
    model.ledger, model.allocation_id = model_ledger, "AUTHORED"
    model.failed = False

    def generate(_execution, wire):
        assert wire["messages"] == [
            {"role": "user", "content": "Describe a red coat after make it blue"}
        ]
        return {
            "id": wire["id"],
            "replicate": wire["replicate"],
            "status": "completed",
            "input_tokens": 17,
            "output_tokens": 4,
            "allocated_seconds": 0.25,
            "output": "authored target first",
        }

    monkeypatch.setattr(model, "_execute", generate)
    outer = RetrievalWorkLedger(tmp_path / "outer-work.sqlite")
    try:
        with RestrictedRankerSession(
            store=store,
            scratch=rank_scratch,
            dataset="cirr",
            image_paths=image_paths,
            checkpoint="model.pt",
            image_weight=0.5,
            attempt_prefix="authored-generated",
        ) as session:
            assert session.ready["pid"] != os.getpid()
            generated = GeneratedDescriptionRanker(
                input_dir=inputs,
                dataset="cirr",
                captions={"reference": "a red coat"},
                candidate=candidate(),
                model=model,
                ranker=session,
                execution=authored_execution(),
                replicate="seed-11",
                attempt_prefix="authored-generated-child",
            )
            bridge = RetrievalAuditBridge(
                store=store,
                scratch_root=score_scratch,
                dataset="cirr",
                plan=plan,
                journal=journal,
                work_ledger=outer,
                unit_ids=tuple(reference),
                rank_one=generated.rank_one,
            )
            verdict = bridge.evaluate(
                contract=ContractCheck(ContractStatus.VALID, ()),
                execution_scope="FROZEN_TABLE",
            )
            assert verdict.status == GateStatus.ELIGIBLE
            assert bridge.complete_survivor().scores == (1,) * 64
        assert model_ledger.usage()["calls"] == 64
        assert model_ledger.usage()["input_tokens"] == 64 * 17
        assert model_ledger.usage()["output_tokens"] == 64 * 4
        assert outer.summary()["attempts"] == 128
        inner = RetrievalWorkLedger(rank_scratch / "ranker-work.sqlite")
        try:
            assert inner.summary()["attempts"] == 64
            assert inner.summary()["unresolved"] == 0
        finally:
            inner.close()
    finally:
        outer.close()
        model_ledger.close()
        journal.close()
