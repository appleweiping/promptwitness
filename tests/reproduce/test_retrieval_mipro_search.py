"""Authored native-interface checks; no actual optimizer or CIR experiment."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from promptwitness.incremental.contracts import ContractCheck, ContractStatus
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest, make_plan
from reproduce import retrieval_audit_bridge as bridge_module
from reproduce.mipro_search import mipro_prompt
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.retrieval_mipro_search import RetrievalMIPROGateEvaluator
from reproduce.retrieval_scoring import RetrievalGold, score_ranking
from reproduce.retrieval_work_ledger import RetrievalWorkLedger
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def native_modules(monkeypatch):
    class Result:
        def __init__(self, score, results):
            self.score, self.results = score, results

    class Pruned(Exception):
        pass

    dspy = ModuleType("dspy")
    evaluate = ModuleType("dspy.evaluate.evaluate")
    optuna = ModuleType("optuna")
    dspy.Prediction = lambda **kwargs: SimpleNamespace(**kwargs)
    evaluate.EvaluationResult = Result
    optuna.TrialPruned = Pruned
    for module in (dspy, evaluate, optuna):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return Pruned


def _program(instructions):
    predictor = SimpleNamespace(
        signature=SimpleNamespace(
            instructions=instructions,
            input_fields={"task_input": None},
            output_fields={"task_output": None},
        ),
        demos=[],
    )
    return SimpleNamespace(predictors=lambda: [predictor])


class NativeExampleShape:
    """Pinned DSPy Example is indexable but is not a Mapping."""

    def __init__(self, unit_id):
        self.unit_id = unit_id

    def __getitem__(self, key):
        if key != "unit_id":
            raise KeyError(key)
        return self.unit_id


def _fixture(
    tmp_path,
    monkeypatch,
    *,
    rank,
    reference_hit=0,
    score_failure=False,
    wrong_document=False,
    fail_completion=False,
):
    units = tuple(f"q{i}" for i in range(64))
    reference = ActualScoreVector(units, (reference_hit,) * len(units))
    seed = mipro_prompt(_program("authored seed"), "seed")
    child = _program("authored child")
    plan = make_plan(
        dict.fromkeys(units, reference_hit),
        candidate_digest=digest("authored child"),
        execution_digest=digest("authored execution"),
        fixture_seed=11,
    )
    journal = AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="authored-retrieval-mipro",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
    )
    ledger = RetrievalWorkLedger(tmp_path / "retrieval-work.sqlite")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ranked = []
    pool = ("reference", "target", *(f"other-{index}" for index in range(10)))
    bridge = None

    def rank_one(unit):
        ranked.append(unit)
        if fail_completion and bridge is not None and bridge._last_gate is not None:
            raise RuntimeError("authored completion failed")
        if rank == "malformed":
            return ("other-0", "other-0")
        hit = rank == "target" or (rank == "mixed" and int(unit[1:]) % 2 == 0)
        return ("target", "reference", *pool[2:]) if hit else ("reference", *pool[2:], "target")

    def scorer(_store, _scratch, _dataset, _stage, rankings):
        if score_failure:
            raise ValueError("authored scorer failed")
        unit, ranking = next(iter(rankings.items()))
        gold = RetrievalGold(
            unit,
            "cirr",
            "reference",
            "target",
            category="",
            subset=("reference", "target"),
        )
        hit = score_ranking(gold, candidate_ids=pool, ranking=ranking).primary_hit
        return {"observations": {unit: {"primary_hit": hit}}}

    monkeypatch.setattr(bridge_module, "score_rankings_restricted", scorer)
    bridge = RetrievalAuditBridge(
        store=tmp_path,
        scratch_root=scratch,
        dataset="cirr",
        plan=plan,
        journal=journal,
        work_ledger=ledger,
        unit_ids=units,
        rank_one=rank_one,
    )
    freeze_calls = []

    def freeze(program):
        freeze_calls.append(program)
        document = mipro_prompt(program, "child")
        if wrong_document:
            document["messages"][0]["content"] = "different after freeze"
        return "child", document, bridge

    evaluator = RetrievalMIPROGateEvaluator(
        reference_document=seed,
        reference_scores=reference,
        freeze=freeze,
        contract=ContractCheck(ContractStatus.VALID, ()),
        execution_scope="FROZEN_TABLE",
    )
    return evaluator, child, units, ranked, freeze_calls, journal, ledger


def test_reference_and_survivor_use_only_complete_actual_scores(
    tmp_path, monkeypatch, native_modules
):
    evaluator, child, units, ranked, freeze_calls, journal, ledger = _fixture(
        tmp_path, monkeypatch, rank="target"
    )
    try:
        seed_result = evaluator(
            _program("authored seed"),
            devset=[{"unit_id": units[0]}],
            callback_metadata={"metric_key": "eval_full"},
        )
        assert seed_result.score == 0
        assert seed_result.results[0][1].primary_hit == 0
        assert not ranked and not freeze_calls

        minibatch = [NativeExampleShape(unit) for unit in units[:8]]
        result = evaluator(
            child, devset=minibatch, callback_metadata={"metric_key": "eval_minibatch"}
        )
        assert result.score == 100 and len(result.results) == 8
        assert all(row[1].primary_hit == 1 for row in result.results)
        assert len(ranked) == 64 == len(set(ranked))
        assert journal.consumed_episodes() == 128
        assert ledger.summary()["attempts"] == 128
        evaluator(child, devset=minibatch, callback_metadata={"metric_key": "eval_full"})
        assert len(ranked) == 64  # same frozen candidate does not replay physical work
        assert len(freeze_calls) == 2
    finally:
        ledger.close()
        journal.close()


def test_mixed_scores_keep_native_requested_order(tmp_path, monkeypatch, native_modules):
    evaluator, child, units, ranked, _, journal, ledger = _fixture(
        tmp_path, monkeypatch, rank="mixed"
    )
    try:
        batch = [NativeExampleShape(unit) for unit in reversed(units[:8])]
        result = evaluator(child, devset=batch, callback_metadata={"metric_key": "eval_minibatch"})
        assert [row[0] is example for row, example in zip(result.results, batch, strict=True)] == [
            True
        ] * len(batch)
        assert [row[1].primary_hit for row in result.results] == [
            int(int(example.unit_id[1:]) % 2 == 0) for example in batch
        ]
        assert result.score == 50
        assert len(ranked) == 64
        assert ledger.summary()["attempts"] == 128
    finally:
        ledger.close()
        journal.close()


def test_completion_failure_never_returns_partial_native_vector(
    tmp_path, monkeypatch, native_modules
):
    evaluator, child, units, ranked, _, journal, ledger = _fixture(
        tmp_path, monkeypatch, rank="target", fail_completion=True
    )
    try:
        with pytest.raises(RuntimeError, match="completion failed"):
            evaluator(
                child,
                devset=[NativeExampleShape(units[0])],
                callback_metadata={"metric_key": "eval_minibatch"},
            )
        assert 0 < len(journal.observations(evaluator.freeze(child)[2].plan)) < 64
        assert ledger.summary()["failed"] == 1
        before = len(ranked)
        with pytest.raises(ValueError, match="attempt limit"):
            evaluator(
                child,
                devset=[NativeExampleShape(units[0])],
                callback_metadata={"metric_key": "eval_full"},
            )
        assert len(ranked) == before
    finally:
        ledger.close()
        journal.close()


def test_same_instruction_with_new_demo_is_not_the_reference(tmp_path, monkeypatch, native_modules):
    evaluator, _, units, ranked, freeze_calls, journal, ledger = _fixture(
        tmp_path, monkeypatch, rank="target"
    )
    changed = _program("authored seed")
    changed.predictors()[0].demos = [{"task_input": "fit only", "task_output": "example"}]
    try:
        result = evaluator(
            changed,
            devset=[NativeExampleShape(units[0])],
            callback_metadata={"metric_key": "eval_full"},
        )
        assert result.score == 100
        assert freeze_calls == [changed]
        assert len(ranked) == 64
    finally:
        ledger.close()
        journal.close()


@pytest.mark.parametrize(
    "case", ["pruned", "failed", "malformed", "wrong_document", "wrong_units", "unsupported"]
)
def test_retrieval_native_boundary_never_imputes_or_selects_incomplete_vector(
    tmp_path, monkeypatch, native_modules, case
):
    evaluator, child, units, ranked, _, journal, ledger = _fixture(
        tmp_path,
        monkeypatch,
        rank=case if case in ("pruned", "malformed") else "target",
        reference_hit=int(case == "pruned"),
        score_failure=case == "failed",
        wrong_document=case == "wrong_document",
    )
    if case == "unsupported":
        evaluator.execution_scope = "UNVERIFIED"
    batch = [{"unit_id": "foreign" if case == "wrong_units" else units[0]}]
    expected = native_modules if case == "pruned" else ScoringError
    try:
        with pytest.raises(expected):
            evaluator(child, devset=batch, callback_metadata={"metric_key": "eval_full"})
        assert (
            len(ranked)
            == {
                "pruned": 4,
                "failed": 1,
                "malformed": 1,
                "wrong_document": 0,
                "wrong_units": 0,
                "unsupported": 0,
            }[case]
        )
        assert journal.observations(evaluator.freeze(child)[2].plan) == (
            {} if case != "pruned" else dict.fromkeys(ranked, 0)
        )
    finally:
        ledger.close()
        journal.close()


def test_reference_vector_must_be_complete_binary(tmp_path):
    with pytest.raises(ScoringError, match="complete binary"):
        RetrievalMIPROGateEvaluator(
            reference_document={"id": "seed"},
            reference_scores=ActualScoreVector(("q0", "q0"), (0, 1)),
            freeze=lambda program: None,
            contract=ContractCheck(ContractStatus.VALID, ()),
            execution_scope="FROZEN_TABLE",
        )
