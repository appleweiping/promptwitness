"""Authored full-ranking checks; no benchmark, encoder, model or real costs."""

from __future__ import annotations

import math

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.gate import GateStatus, evaluate_candidate
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import complete_survivor_scores
from promptwitness.incremental.sampling import digest, make_plan
from reproduce.retrieval_scoring import (
    RetrievalGold,
    compose_vectors,
    rank_cosine,
    score_ranking,
    summarize_rankings,
)


def cirr(query="q", *, subset=("reference", "target")):
    return RetrievalGold(query, "cirr", "reference", "target", subset=subset)


def fashion(query="q", category="dress"):
    return RetrievalGold(query, "fashioniq", "reference", "target", category=category)


def test_cosine_direction_not_vector_magnitude_and_lexical_ties():
    candidates = {"z": (50, 0), "a": (2, 0), "b": (0, 1), "c": (-1, 0)}
    assert rank_cosine((10, 0), candidates) == ("a", "z", "b", "c")
    assert rank_cosine((1e308, 1e308), {"b": (1e308, 1e308), "a": (1e-308, 0)}) == (
        "b",
        "a",
    )


@pytest.mark.parametrize("bad", [(), (0, 0), (math.nan, 1), (math.inf, 1), (True, 1)])
def test_invalid_embedding_is_failure(bad):
    with pytest.raises(ValueError):
        rank_cosine(bad, {"a": (1, 0)})
    with pytest.raises(ValueError):
        rank_cosine((1, 0), {"a": bad})


def test_invalid_pool_and_dimensions():
    with pytest.raises(ValueError):
        rank_cosine((1, 0), {})
    with pytest.raises(ValueError):
        rank_cosine((1, 0), {"": (1, 0)})
    with pytest.raises(ValueError):
        rank_cosine((1, 0), {"a": (1, 0, 0)})


def test_fusion_uses_both_modalities_and_rejects_cancellation():
    assert compose_vectors((2, 0), (0, 100), image_weight=0.5) == pytest.approx(
        (math.sqrt(0.5), math.sqrt(0.5))
    )
    assert compose_vectors((2, 0), (0, 100), image_weight=0) == (0, 1)
    assert compose_vectors((2, 0), (0, 100), image_weight=1) == (1, 0)
    with pytest.raises(ValueError):
        compose_vectors((1, 0), (-1, 0), image_weight=0.5)
    with pytest.raises(ValueError):
        compose_vectors((1, 0), (1, 0, 0), image_weight=0.5)


@pytest.mark.parametrize("weight", [-0.1, 1.1, math.inf, math.nan, True, "0.5"])
def test_fusion_weight_validation(weight):
    with pytest.raises(ValueError):
        compose_vectors((1, 0), (0, 1), image_weight=weight)


def test_cirr_removes_reference_before_global_cutoff():
    ranking = ("reference", "a", "b", "c", "d", "target")
    result = score_ranking(cirr(), candidate_ids=ranking, ranking=ranking)
    assert result.primary_hit == 1  # target becomes fifth, not sixth
    assert dict(result.recalls) == {1: 0, 5: 1, 10: 1, 50: 1}
    assert dict(result.subset_recalls) == {1: 1, 2: 1, 3: 1}


def test_cirr_subset_is_filtered_ranking_not_global_top_k():
    ranking = ("reference", "a", "b", "c", "d", "e", "target", "f")
    result = score_ranking(
        cirr(subset=("f", "target", "reference")), candidate_ids=ranking, ranking=ranking
    )
    assert result.primary_hit == 0
    assert dict(result.subset_recalls)[1] == 1


def test_fashion_keeps_reference_at_cutoff():
    ranking = ("reference", *(f"d{i}" for i in range(9)), "target")
    result = score_ranking(fashion(), candidate_ids=ranking, ranking=ranking)
    assert result.primary_hit == 0  # reference removal would incorrectly make it tenth
    assert dict(result.recalls) == {10: 0, 50: 1}
    assert result.subset_recalls == ()


@pytest.mark.parametrize(
    "ranking", [(), ("reference", "target", "target"), ("target",), ("reference", "unknown")]
)
def test_bad_ranking_is_not_scored_zero(ranking):
    with pytest.raises(ValueError):
        score_ranking(cirr(), candidate_ids=("reference", "target"), ranking=ranking)


def test_missing_gold_or_subset_member_fails():
    with pytest.raises(ValueError):
        score_ranking(cirr(), candidate_ids=("reference", "other"), ranking=("reference", "other"))
    with pytest.raises(ValueError):
        score_ranking(
            cirr(subset=("reference", "target", "absent")),
            candidate_ids=("reference", "target"),
            ranking=("reference", "target"),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_id": "", "dataset": "cirr", "reference_id": "r", "target_id": "t"},
        {"query_id": "q", "dataset": "circo", "reference_id": "r", "target_id": "t"},
        {"query_id": "q", "dataset": "cirr", "reference_id": "r", "target_id": "r"},
        {
            "query_id": "q",
            "dataset": "cirr",
            "reference_id": "r",
            "target_id": "t",
            "subset": ("r",),
        },
        {
            "query_id": "q",
            "dataset": "fashioniq",
            "reference_id": "r",
            "target_id": "t",
            "category": "other",
        },
        {
            "query_id": "q",
            "dataset": "fashioniq",
            "reference_id": "r",
            "target_id": "t",
            "category": "dress",
            "subset": ("r", "t"),
        },
    ],
)
def test_gold_contract_rejects_unsupported(kwargs):
    with pytest.raises(ValueError):
        RetrievalGold(**kwargs)


def test_fashion_macro_is_not_query_micro():
    # Dress has two successful queries, shirt one success, toptee one failure.
    pool = ("reference", *(f"d{i}" for i in range(9)), "target")
    good, bad = ("target", *pool[:-1]), pool
    gold = [fashion("d1"), fashion("d2"), fashion("s", "shirt"), fashion("t", "toptee")]
    report = summarize_rankings(
        gold,
        {"d1": good, "d2": good, "s": good, "t": bad},
        dict.fromkeys(("dress", "shirt", "toptee"), pool),
    )
    assert report["query_micro_recall"]["10"] == 0.75
    assert report["category_macro_recall"]["10"] == pytest.approx(2 / 3)
    assert report["primary_query_micro_hit"] == 0.75
    assert report["official_native_parity_established"] is False


def test_cirr_aggregate_and_incomplete_coverage():
    pool = ("reference", "target")
    assert summarize_rankings([cirr()], {"q": pool}, {"cirr": pool})["subset_recall"]["1"] == 1
    with pytest.raises(ValueError):
        summarize_rankings([], {}, {})
    with pytest.raises(ValueError):
        summarize_rankings([cirr(), cirr()], {"q": pool}, {"cirr": pool})
    with pytest.raises(ValueError):
        summarize_rankings([cirr()], {}, {"cirr": pool})
    with pytest.raises(ValueError):
        summarize_rankings([cirr(), fashion("f")], {"q": pool, "f": pool}, {"cirr": pool})
    with pytest.raises(ValueError):
        summarize_rankings(
            [fashion()], {"q": pool}, dict.fromkeys(("dress", "shirt", "toptee"), pool)
        )
    with pytest.raises(ValueError):
        summarize_rankings([cirr()], {"q": pool}, {"other": pool})


@pytest.mark.parametrize("case", ["eligible", "rejected", "failed"])
def test_rankings_supply_actual_binary_gate_observations(tmp_path, case):
    # Authored vectors only. Sorting/scoring work is real CPU work, never an
    # image encoder/model-call saving or a physical resource-ledger entry.
    pool = {"reference": (1, 0, 0), "target": (0, 1, 0), **{f"d{i}": (0, 0, 1) for i in range(8)}}
    old_rank = rank_cosine((0, 0, 1) if case == "eligible" else (0, 1, 0), pool)
    units = [f"q-{i}" for i in range(64)]
    reference = {
        u: score_ranking(cirr(u), candidate_ids=tuple(pool), ranking=old_rank).primary_hit
        for u in units
    }
    plan = make_plan(
        reference, candidate_digest=digest([case]), execution_digest=digest(pool), fixture_seed=11
    )
    journal = AuditJournal(
        tmp_path / "AUTHORED_ONLY.sqlite",
        run_id=case,
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=64,
    )
    queried = []

    def query(unit):
        queried.append(unit)
        ranking = (
            ()
            if case == "failed"
            else rank_cosine((0, 1, 0) if case == "eligible" else (0, 0, 1), pool)
        )
        return score_ranking(cirr(unit), candidate_ids=tuple(pool), ranking=ranking).primary_hit

    result = evaluate_candidate(
        plan,
        journal,
        query,
        contract=ContractCheck(ContractStatus.VALID, ()),
        execution_scope="FROZEN_TABLE",
    )
    expected = {
        "eligible": GateStatus.ELIGIBLE,
        "rejected": GateStatus.INELIGIBLE,
        "failed": GateStatus.INCONCLUSIVE,
    }[case]
    assert result.status == expected
    assert result.certificate_scope == "MECHANICAL_SEEDED_FIXTURE"
    assert 0 < len(queried) < 64
    if case == "eligible":

        def complete(unit):
            attempt = journal.reserve(plan, unit)
            score = query(unit)
            journal.settle(plan, unit, attempt, score=score)
            return score

        vector = complete_survivor_scores(tuple(units), journal.observations(plan), complete)
        assert vector.scores == (1,) * 64
        assert len(queried) == len(set(queried)) == 64
    elif case == "failed":
        assert journal.observations(plan) == {}
        assert journal.consumed_episodes() == 65
    journal.close()
