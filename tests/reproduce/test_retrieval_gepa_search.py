"""Authored GEPA gate tests; no official data, model, or optimizer claim."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_gepa_search import RetrievalGEPAGateController, gepa_prompt
from reproduce.retrieval_scoring import RetrievalGold, score_ranking
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.strict_scoring import ScoringError

UNITS = tuple(f"q{index}" for index in range(64))
POOL = ("reference", "target", *(f"other-{index}" for index in range(10)))


def _fixture(tmp_path, monkeypatch, *, seed_scores, child_hits, bad_document=False):
    events = []
    seed = {"instruction": "authored seed"}
    child = {"instruction": "authored child"}
    reference = ActualScoreVector(UNITS, tuple(seed_scores))
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    journals = []

    def scorer(_store, _scratch, dataset, stage, rankings):
        assert (dataset, stage) == ("cirr", "search")
        observations = {}
        for unit, ranking in rankings.items():
            gold = RetrievalGold(unit, "cirr", "reference", "target", category="", subset=POOL)
            observations[unit] = {
                "primary_hit": score_ranking(gold, candidate_ids=POOL, ranking=ranking).primary_hit
            }
        return {"observations": observations}

    monkeypatch.setattr(bridge_module, "score_rankings_restricted", scorer)

    def freeze(candidate, parent_document, parent_scores):
        events.append("freeze")
        assert parent_document == gepa_prompt(seed, "seed")
        assert parent_scores is reference
        document = gepa_prompt(candidate, "child")
        plan = make_plan(
            dict(zip(UNITS, seed_scores, strict=True)),
            candidate_digest=digest(document),
            execution_digest=digest("authored-gepa-execution"),
            fixture_seed=11,
        )
        journal = AuditJournal(
            tmp_path / "audit.sqlite",
            run_id="authored-gepa",
            reference_digest=plan.reference_digest,
            execution_digest=plan.execution_digest,
            reference_episodes=plan.population,
        )
        journal.freeze(plan)
        journals.append(journal)

        def rank_one(unit):
            events.append("rank")
            if child_hits[UNITS.index(unit)]:
                return ("target", "reference", *POOL[2:])
            return ("reference", *POOL[2:], "target")

        bridge = RetrievalAuditBridge(
            store=tmp_path,
            scratch_root=scratch,
            dataset="cirr",
            plan=plan,
            journal=journal,
            work_ledger=ledger,
            unit_ids=UNITS,
            rank_one=rank_one,
        )
        if bad_document:
            document["messages"][0]["content"] = "mismatched"
        return "child", document, bridge

    controller = RetrievalGEPAGateController(
        seed_candidate=seed,
        seed_identifier="seed",
        seed_scores=reference,
        propose=lambda _parent, _feedback, _components: {"instruction": "authored child"},
        freeze=freeze,
        contract=ContractCheck(ContractStatus.VALID, ()),
        execution_scope="FROZEN_TABLE",
    )
    return controller, seed, child, events, ledger, journals


def _proposal(child, *, before=(0.0,), after=(1.0,)):
    before = list(before)
    after = list(after)
    return SimpleNamespace(
        candidate=child,
        parent_program_ids=[0],
        subsample_indices=list(range(len(before))),
        subsample_scores_before=before,
        subsample_scores_after=after,
        eval_before=SimpleNamespace(
            scores=before, outputs=[("actual parent output",)] * len(before)
        ),
        eval_after=SimpleNamespace(scores=after, outputs=[("actual child output",)] * len(after)),
    )


def _close(ledger, journals):
    ledger.close()
    for journal in journals:
        journal.close()


def test_gepa_document_preserves_all_named_native_components():
    assert gepa_prompt({"z": "last", "a": "first"}, "id")["messages"] == [
        {"id": "component:a", "role": "system", "content": "first"},
        {"id": "component:z", "role": "system", "content": "last"},
    ]
    with pytest.raises(ScoringError, match="named original"):
        gepa_prompt({"a": 3}, "id")


def test_preoutcome_freeze_and_actual_complete_survivor(tmp_path, monkeypatch):
    controller, seed, child, events, ledger, journals = _fixture(
        tmp_path, monkeypatch, seed_scores=(0,) * 64, child_hits=(1,) * 64
    )
    try:
        assert controller.propose_new_texts(seed, {}, ["instruction"]) == child
        assert events == ["freeze"]
        assert controller.should_accept(
            _proposal(child), SimpleNamespace(program_candidates=[seed])
        )
        assert events[0] == "freeze" and events.count("rank") == 64
        assert controller.search_scores(child) == ActualScoreVector(UNITS, (1,) * 64)
        assert journals[0].consumed_episodes() == 128
        assert ledger.summary()["unresolved"] == 0
    finally:
        _close(ledger, journals)


def test_native_nonimprovement_does_not_read_audit_outcomes(tmp_path, monkeypatch):
    controller, seed, child, events, ledger, journals = _fixture(
        tmp_path, monkeypatch, seed_scores=(0,) * 64, child_hits=(1,) * 64
    )
    try:
        controller.propose_new_texts(seed, {}, ["instruction"])
        proposal = _proposal(child, before=(1.0,), after=(0.0,))
        assert not controller.should_accept(proposal, SimpleNamespace(program_candidates=[seed]))
        assert controller.reject_reason(proposal, None).startswith("native GEPA minibatch")
        assert events == ["freeze"]
        with pytest.raises(KeyError):
            controller.search_scores(child)
    finally:
        _close(ledger, journals)


def test_actual_regression_rejects_without_selector_vector(tmp_path, monkeypatch):
    seed_scores = (0, *(1 for _ in range(63)))
    child_hits = (1, *(0 for _ in range(63)))
    controller, seed, child, events, ledger, journals = _fixture(
        tmp_path, monkeypatch, seed_scores=seed_scores, child_hits=child_hits
    )
    try:
        controller.propose_new_texts(seed, {}, ["instruction"])
        proposal = _proposal(child)  # q0 really improves; most other queries regress.
        assert not controller.should_accept(proposal, SimpleNamespace(program_candidates=[seed]))
        assert controller.reject_reason(proposal, None) == "retrieval audit INELIGIBLE"
        assert 0 < events.count("rank") < 64
        with pytest.raises(KeyError):
            controller.search_scores(child)
    finally:
        _close(ledger, journals)


@pytest.mark.parametrize("case", ["truncated_scores", "missing_outputs"])
def test_native_minibatch_requires_each_actual_output(tmp_path, monkeypatch, case):
    controller, seed, child, events, ledger, journals = _fixture(
        tmp_path, monkeypatch, seed_scores=(0,) * 64, child_hits=(1,) * 64
    )
    try:
        controller.propose_new_texts(seed, {}, ["instruction"])
        proposal = _proposal(child, before=(0.0,) * 8, after=(1.0,) * 8)
        if case == "truncated_scores":
            proposal.subsample_scores_before = [0.0]
            proposal.subsample_scores_after = [1.0]
            proposal.eval_before.scores = [0.0]
            proposal.eval_after.scores = [1.0]
        else:
            proposal.eval_after.outputs = [None] * 8
        with pytest.raises(ScoringError, match="complete actual binary"):
            controller.should_accept(proposal, SimpleNamespace(program_candidates=[seed]))
        assert events == ["freeze"]
        with pytest.raises(KeyError):
            controller.search_scores(child)
    finally:
        _close(ledger, journals)


@pytest.mark.parametrize("case", ["unfrozen", "merge", "missing", "mismatch", "bad_scores"])
def test_invalid_native_boundary_fails_closed(tmp_path, monkeypatch, case):
    controller, seed, child, events, ledger, journals = _fixture(
        tmp_path,
        monkeypatch,
        seed_scores=(0,) * 64,
        child_hits=(1,) * 64,
        bad_document=case == "mismatch",
    )
    try:
        if case not in ("unfrozen", "mismatch"):
            controller.propose_new_texts(seed, {}, ["instruction"])
        proposal = _proposal(child)
        if case == "merge":
            proposal.parent_program_ids = [0, 1]
        if case == "missing":
            proposal.parent_program_ids = [1]
        if case == "bad_scores":
            proposal.subsample_scores_after = [float("nan")]
        with pytest.raises(ScoringError):
            if case == "mismatch":
                controller.propose_new_texts(seed, {}, ["instruction"])
            else:
                controller.should_accept(proposal, SimpleNamespace(program_candidates=[seed]))
        assert events.count("rank") == 0
    finally:
        _close(ledger, journals)


def test_incomplete_seed_or_unknown_proposal_is_rejected(tmp_path):
    with pytest.raises(ScoringError, match="complete binary"):
        RetrievalGEPAGateController(
            seed_candidate={"instruction": "seed"},
            seed_identifier="seed",
            seed_scores=ActualScoreVector(("q0", "q0"), (0, 1)),
            propose=lambda *_: {},
            freeze=lambda *_: None,
            contract=ContractCheck(ContractStatus.VALID, ()),
            execution_scope="FROZEN_TABLE",
        )
