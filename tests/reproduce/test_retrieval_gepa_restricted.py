"""Authored scorer-worker GEPA boundary; no official data or real model."""

from __future__ import annotations

import json
import os

import pytest

from reproduce import retrieval_gepa_restricted as qualifier
from reproduce.retrieval_work_ledger import RetrievalWorkLedger

SEARCH = ("search-0", "search-1")
SELECTION = ("selection-0", "selection-1")
POOL = ("reference", "target", "other")
RANK = ("target", "reference", "other")


def test_failed_qualification_has_no_premature_worker_claim():
    report = qualifier.qualification_report_header("authored-test/v1")
    assert report["status"] == "FAILED_RETAINED"
    assert report["score_worker_is_real_landlock_child"] is False
    assert report["scientific_result"] is False


def _setup(tmp_path):
    store = tmp_path / "store"
    qualifier.create_authored_store(store, search_ids=SEARCH, selection_ids=SELECTION, pool=POOL)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ledger = RetrievalWorkLedger(tmp_path / "work.sqlite")
    scoring = qualifier.RestrictedGEPAQualifierScoring(
        store=store,
        scratch_root=scratch,
        ledger=ledger,
        search_ids=SEARCH,
        selection_ids=SELECTION,
    )
    return store, scratch, ledger, scoring


def test_authored_store_has_disjoint_role_inputs_and_gold(tmp_path):
    store, _scratch, ledger, _scoring = _setup(tmp_path)
    try:
        search_rows = [
            json.loads(line)
            for line in (store / "search/inputs/cirr.jsonl").read_text().splitlines()
        ]
        selection_rows = [
            json.loads(line)
            for line in (store / "selection/inputs/cirr.jsonl").read_text().splitlines()
        ]
        assert {row["id"] for row in search_rows} == set(SEARCH)
        assert {row["id"] for row in selection_rows} == set(SELECTION)
        assert all(
            set(row) == {"id", "reference_id", "modification", "category"} for row in search_rows
        )
        assert not (store / "final/gold/cirr.jsonl").exists()
        with pytest.raises(ValueError, match="disjoint"):
            qualifier.create_authored_store(
                tmp_path / "bad", search_ids=("q",), selection_ids=("q",), pool=POOL
            )
    finally:
        ledger.close()


def test_reference_native_and_audit_use_separate_scorer_receipts(tmp_path, monkeypatch):
    store, scratch, ledger, scoring = _setup(tmp_path)
    calls = []

    def scorer(data, work, dataset, stage, rankings):
        assert data == store and work.is_dir() and dataset == "cirr"
        calls.append((stage, tuple(rankings)))
        return {
            "worker_pid": os.getpid() + len(calls),
            "observations": {unit: {"primary_hit": 1} for unit in rankings},
        }

    monkeypatch.setattr(qualifier, "score_rankings_restricted", scorer)
    try:
        assert scoring.score("search", dict.fromkeys(SEARCH, RANK), purpose="reference")
        assert scoring.score("search", {SEARCH[0]: RANK}, purpose="native")
        assert scoring.score("selection", dict.fromkeys(SELECTION, RANK), purpose="native")
        assert scoring.audit_score(store, scratch, "cirr", "search", {SEARCH[1]: RANK})
        assert calls == [
            ("search", SEARCH),
            ("search", (SEARCH[0],)),
            ("selection", SELECTION),
            ("search", (SEARCH[1],)),
        ]
        assert (scoring.reference_units, scoring.native_search_units) == (2, 1)
        assert (scoring.native_selection_units, scoring.audit_units) == (2, 1)
        assert len(scoring.worker_pids) == 4
        assert ledger.summary()["attempts"] == 3
        assert ledger.summary()["unresolved"] == 0
        with pytest.raises(ValueError, match="full selection"):
            scoring.score("selection", {SELECTION[0]: RANK}, purpose="native")
    finally:
        ledger.close()


def test_failed_reference_is_charged_and_never_becomes_zero(tmp_path, monkeypatch):
    _store, _scratch, ledger, scoring = _setup(tmp_path)

    def failed(*_args):
        raise RuntimeError("restricted worker failed")

    monkeypatch.setattr(qualifier, "score_rankings_restricted", failed)
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            scoring.score("search", dict.fromkeys(SEARCH, RANK), purpose="reference")
        assert ledger.summary()["attempts"] == 1
        assert ledger.summary()["failed"] == 1
        assert ledger.summary()["unresolved"] == 0
        assert scoring.reference_units == 0 and scoring.worker_pids == []
    finally:
        ledger.close()


def test_later_failure_retains_confirmed_reference_and_failed_attempt(tmp_path, monkeypatch):
    _store, _scratch, native_ledger, scoring = _setup(tmp_path)
    audit_ledger = RetrievalWorkLedger(tmp_path / "audit-work.sqlite")
    calls = 0

    def scorer(_data, _work, _dataset, _stage, rankings):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("native worker failed")
        return {
            "worker_pid": os.getpid() + 1,
            "observations": {unit: {"primary_hit": 1} for unit in rankings},
        }

    monkeypatch.setattr(qualifier, "score_rankings_restricted", scorer)
    try:
        scoring.score("search", dict.fromkeys(SEARCH, RANK), purpose="reference")
        with pytest.raises(RuntimeError, match="native worker failed"):
            scoring.score("search", {SEARCH[0]: RANK}, purpose="native")
        receipt = qualifier.scenario_progress_receipt(
            scoring, native_ledger, audit_ledger, audit_rank_units=0
        )
        assert receipt["reference_actual_restricted_units"] == 2
        assert receipt["native_search_scored_units"] == 0
        assert receipt["scorer_worker_launches"] == 1
        assert receipt["native_score_work"]["attempts"] == 2
        assert receipt["native_score_work"]["completed"] == 1
        assert receipt["native_score_work"]["failed"] == 1
        assert receipt["native_score_work"]["forward_count_unmeasured_attempts"] == 2
    finally:
        native_ledger.close()
        audit_ledger.close()
