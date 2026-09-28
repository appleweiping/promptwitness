"""Authored retrieval rankings through the existing logical audit gate."""

from __future__ import annotations

import json
import sys

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.process_access import LEAVES
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger


@pytest.fixture
def work_ledger(tmp_path):
    ledger = RetrievalWorkLedger(tmp_path / "retrieval-work.sqlite")
    try:
        yield ledger
    finally:
        ledger.close()


def _plan(reference):
    return make_plan(
        reference,
        candidate_digest=digest("authored-child"),
        execution_digest=digest("authored-ranker"),
        fixture_seed=11,
    )


def _journal(tmp_path, plan):
    return AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="authored-cir",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
    )


def _valid_contract():
    return ContractCheck(ContractStatus.VALID, ())


def test_fixed_gate_prefix_and_actual_survivor_vector(tmp_path, monkeypatch, work_ledger):
    reference = {f"q{i}": 0 for i in range(64)}
    plan = _plan(reference)
    journal = _journal(tmp_path, plan)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ranked = []
    scored = []

    def rank_one(unit):
        ranked.append(unit)
        return ("target", "reference", "other")

    def scorer(store, work, dataset, stage, rankings):
        assert (dataset, stage) == ("cirr", "search")
        assert set(rankings) == {ranked[-1]}
        scored.append(next(iter(rankings)))
        return {"observations": {scored[-1]: {"primary_hit": int(scored[-1] != "q0")}}}

    monkeypatch.setattr(bridge_module, "score_rankings_restricted", scorer)
    bridge = RetrievalAuditBridge(
        store=tmp_path / "not-read-by-fake",
        scratch_root=scratch,
        dataset="cirr",
        plan=plan,
        journal=journal,
        work_ledger=work_ledger,
        unit_ids=tuple(reference),
        rank_one=rank_one,
    )
    try:
        with pytest.raises(ValueError, match="eligible"):
            bridge.complete_survivor()
        verdict = bridge.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE")
        assert verdict.status == GateStatus.ELIGIBLE
        assert 0 < len(ranked) < 64
        assert ranked == list(plan.strata[0].permutation[: len(ranked)])
        prefix = tuple(ranked)
        vector = bridge.complete_survivor()
        assert vector.unit_ids == tuple(reference)
        assert vector.scores == tuple(int(unit != "q0") for unit in reference)
        assert len(ranked) == len(scored) == 64 == len(set(ranked))
        assert tuple(ranked[: len(prefix)]) == prefix
        assert journal.consumed_episodes() == 128
        assert work_ledger.summary()["attempts"] == 128
        assert work_ledger.summary()["unresolved"] == 0
        resumed = RetrievalAuditBridge(
            store=bridge.store,
            scratch_root=scratch,
            dataset="cirr",
            plan=plan,
            journal=journal,
            work_ledger=work_ledger,
            unit_ids=tuple(reference),
            rank_one=rank_one,
        )
        assert (
            resumed.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE").status
            == GateStatus.ELIGIBLE
        )
        assert len(ranked) == 64
    finally:
        journal.close()


def test_survivor_failure_is_charged_without_replay_or_zero(tmp_path, monkeypatch, work_ledger):
    plan = _plan({f"q{i}": 0 for i in range(64)})
    journal = _journal(tmp_path, plan)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fail_completion = False
    calls = []

    def rank_one(unit):
        calls.append(unit)
        return ("bad",) if fail_completion else ("target", "reference", "other")

    def scorer(store, work, dataset, stage, rankings):
        if next(iter(rankings.values())) == ("bad",):
            raise ValueError("failed full-pool ranking")
        unit = next(iter(rankings))
        return {"observations": {unit: {"primary_hit": 1}}}

    monkeypatch.setattr(bridge_module, "score_rankings_restricted", scorer)
    bridge = RetrievalAuditBridge(
        store=tmp_path,
        scratch_root=scratch,
        dataset="cirr",
        plan=plan,
        journal=journal,
        work_ledger=work_ledger,
        unit_ids=tuple(f"q{i}" for i in range(64)),
        rank_one=rank_one,
    )
    try:
        assert (
            bridge.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE").status
            == GateStatus.ELIGIBLE
        )
        observed = journal.observations(plan)
        fail_completion = True
        with pytest.raises(ValueError, match="failed full-pool"):
            bridge.complete_survivor()
        assert journal.observations(plan) == observed
        assert journal.consumed_episodes() == 64 + len(observed) + 1
        before_retry = len(calls)
        with pytest.raises(ValueError, match="attempt limit"):
            bridge.complete_survivor()
        assert len(calls) == before_retry
        assert work_ledger.summary()["failed"] == 1
        assert work_ledger.summary()["unresolved"] == 0
    finally:
        journal.close()


def test_failed_rank_is_charged_and_never_imputed_or_replayed(tmp_path, monkeypatch, work_ledger):
    plan = _plan({f"q{i}": 0 for i in range(64)})
    journal = _journal(tmp_path, plan)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    calls = []

    def rank_one(unit):
        calls.append(unit)
        return ("incomplete",)

    def scorer(*args, **kwargs):
        raise ValueError("incomplete full-pool ranking")

    monkeypatch.setattr(bridge_module, "score_rankings_restricted", scorer)
    bridge = RetrievalAuditBridge(
        store=tmp_path,
        scratch_root=scratch,
        dataset="fashioniq",
        plan=plan,
        journal=journal,
        work_ledger=work_ledger,
        unit_ids=tuple(f"q{i}" for i in range(64)),
        rank_one=rank_one,
    )
    try:
        verdict = bridge.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE")
        assert verdict.status == GateStatus.INCONCLUSIVE
        assert journal.observations(plan) == {}
        assert journal.consumed_episodes() == 65
        assert (
            bridge.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE").status
            == GateStatus.INCONCLUSIVE
        )
        assert len(calls) == 1
        with pytest.raises(ValueError, match="eligible"):
            bridge.complete_survivor()
        assert work_ledger.summary()["attempts"] == 2
        assert work_ledger.summary()["failed"] == 1
    finally:
        journal.close()


def test_unverified_execution_never_calls_ranker(tmp_path, work_ledger):
    plan = _plan({f"q{i}": 0 for i in range(64)})
    journal = _journal(tmp_path, plan)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def rank_one(unit):
        raise AssertionError("ranker must not run")

    bridge = RetrievalAuditBridge(
        store=tmp_path,
        scratch_root=scratch,
        dataset="cirr",
        plan=plan,
        journal=journal,
        work_ledger=work_ledger,
        unit_ids=tuple(f"q{i}" for i in range(64)),
        rank_one=rank_one,
    )
    try:
        verdict = bridge.evaluate(contract=_valid_contract(), execution_scope="UNVERIFIED")
        assert verdict.status == GateStatus.UNSUPPORTED
        assert journal.consumed_episodes() == 64
        assert work_ledger.summary()["attempts"] == 0
    finally:
        journal.close()


def test_selector_population_must_match_frozen_plan(tmp_path, work_ledger):
    plan = _plan({f"q{i}": 0 for i in range(64)})
    journal = _journal(tmp_path, plan)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    try:
        with pytest.raises(ValueError, match="selector population"):
            RetrievalAuditBridge(
                store=tmp_path,
                scratch_root=scratch,
                dataset="cirr",
                plan=plan,
                journal=journal,
                work_ledger=work_ledger,
                unit_ids=("q0",),
                rank_one=lambda unit: ("target",),
            )
    finally:
        journal.close()


def test_real_linux_scorer_and_gate_on_authored_rankings(tmp_path, work_ledger):
    if sys.platform != "linux":
        pytest.skip("Linux Landlock worker required")
    store = tmp_path / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    pool = ("reference", "target", *(f"other-{i}" for i in range(9)))
    inputs = [
        {"id": f"q{i}", "reference_id": "reference", "modification": "make it blue", "category": ""}
        for i in range(64)
    ]
    labels = [
        {"id": f"q{i}", "target_id": "target", "subset": ["reference", "target"]} for i in range(64)
    ]
    (store / "search/inputs/cirr.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inputs), encoding="utf-8"
    )
    (store / "search/gold/cirr.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    (store / "search/inputs/cirr-gallery.json").write_text(
        json.dumps({"cirr": pool}), encoding="utf-8"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    old_ranking = ("reference", *pool[2:], "target")
    old = score_rankings_restricted(
        store, scratch, "cirr", "search", {row["id"]: old_ranking for row in inputs}
    )
    reference = {
        unit: observation["primary_hit"] for unit, observation in old["observations"].items()
    }
    assert set(reference.values()) == {0}
    plan = _plan(reference)
    journal = _journal(tmp_path, plan)
    bridge = RetrievalAuditBridge(
        store=store,
        scratch_root=scratch,
        dataset="cirr",
        plan=plan,
        journal=journal,
        work_ledger=work_ledger,
        unit_ids=tuple(reference),
        rank_one=lambda unit: ("target", "reference", *pool[2:]),
    )
    try:
        verdict = bridge.evaluate(contract=_valid_contract(), execution_scope="FROZEN_TABLE")
        assert verdict.status == GateStatus.ELIGIBLE
        assert bridge.complete_survivor().scores == (1,) * 64
        assert journal.consumed_episodes() == 128
        assert work_ledger.summary()["attempts"] == 128
    finally:
        journal.close()
