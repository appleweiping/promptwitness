import json

import pytest

pytest.importorskip("scipy")

from promptwitness.incremental import (
    AuditJournal,
    AuditPlan,
    BudgetExhausted,
    ContractCheck,
    ContractStatus,
    GateStatus,
    Stratum,
    evaluate_candidate,
    fixed_looks,
    make_plan,
)
from promptwitness.incremental.sampling import digest


def make(reference, seed=11, risk=None, candidate="candidate"):
    return make_plan(
        reference,
        candidate_digest=digest(candidate),
        execution_digest=digest("execution"),
        fixture_seed=seed,
        risk=risk,
    )


def journal(tmp_path, plan, limit=2048):
    return AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="test",
        reference_digest=plan.reference_digest,
        execution_digest=plan.execution_digest,
        reference_episodes=plan.population,
        max_episodes=limit,
    )


VALID = ContractCheck(ContractStatus.VALID, ())


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16, 19, 256, 512])
def test_allocation_fixed_monotone_census_and_homogeneous(size):
    reference = {str(i): i % 2 for i in range(size)}
    risk = {str(i): i / size for i in range(size)}
    plan = make(reference, risk=risk)
    assert len(plan.strata) <= 8
    assert tuple(map(sum, plan.allocations)) == fixed_looks(size)
    assert len(plan.allocations) <= 6
    assert plan.allocations[-1] == tuple(len(s.permutation) for s in plan.strata)
    assert AuditPlan.from_dict(plan.to_dict()) == plan
    assert all(reference[x] == s.old_correct for s in plan.strata for x in s.permutation)
    assert make(reference, risk=risk) == plan
    if size >= 16:
        assert make(reference, risk=risk, seed=23).sha256 != plan.sha256


def test_fresh_randomness_is_default():
    plan = make({str(i): 1 for i in range(40)}, seed=None)
    assert plan.randomization == "fresh_system_random"


@pytest.mark.parametrize(
    "reference,risk,bins",
    [
        ({}, None, 4),
        ({"x": True}, None, 4),
        ({"x": 1}, {"x": float("nan")}, 4),
        ({"x": 1}, {"foreign": 0.2}, 4),
        ({"x": 1}, None, 5),
    ],
)
def test_bad_plan_inputs(reference, risk, bins):
    with pytest.raises(ValueError):
        make_plan(
            reference,
            candidate_digest=digest("x"),
            execution_digest=digest("y"),
            risk=risk,
            bins_per_old_score=bins,
        )


def test_malformed_serialized_plans_and_strata():
    plan = make({"a": 1, "b": 0, "c": 1})
    for changes in (
        {"extra": 1},
        {"candidate_digest": "bad"},
        {"randomization": "adaptive"},
        {"allocations": [[9, 9]]},
        {"reference_digest": digest("wrong")},
    ):
        value = plan.to_dict() | changes
        with pytest.raises(ValueError):
            AuditPlan.from_dict(value)
    with pytest.raises(ValueError):
        Stratum(1, ("duplicate", "duplicate"))


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({str(i): 1 for i in range(20)}, GateStatus.ELIGIBLE),
        ({str(i): 0 for i in range(20)}, GateStatus.INELIGIBLE),
    ],
)
def test_gate_real_observation_census_and_resume(tmp_path, scores, expected):
    plan = make({str(i): 1 for i in range(20)})
    store = journal(tmp_path, plan)
    calls = []

    def evaluate(unit):
        calls.append(unit)
        return scores[unit]

    result = evaluate_candidate(
        plan, store, evaluate, contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert result.status == expected
    charged = store.consumed_episodes()
    assert result.candidate_slot == 0
    assert result.certificate_scope == "MECHANICAL_SEEDED_FIXTURE"
    store.close()
    store = journal(tmp_path, plan)
    resumed = evaluate_candidate(
        plan, store, evaluate, contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert resumed.status == result.status and store.consumed_episodes() == charged
    assert len(calls) == charged - plan.population
    assert store.load_plan(plan.candidate_digest) == plan
    assert store.report()["physical_model_calls_or_gpu_costs_inferred"] is False
    store.close()


def test_budget_exhaustion_and_unknown_contract_no_calls(tmp_path):
    plan = make({str(i): 1 for i in range(20)})
    store = journal(tmp_path, plan, limit=21)
    result = evaluate_candidate(
        plan, store, lambda x: 1, contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert result.status == GateStatus.INCONCLUSIVE and store.consumed_episodes() == 21
    before = store.consumed_episodes()
    unknown = evaluate_candidate(plan, store, lambda x: 1, contract=VALID)
    assert unknown.status == GateStatus.UNSUPPORTED and store.consumed_episodes() == before
    unsupported = evaluate_candidate(
        plan,
        store,
        lambda x: 1,
        contract=ContractCheck(ContractStatus.UNSUPPORTED, ("schema",)),
        execution_scope="FROZEN_TABLE",
    )
    assert unsupported.status == GateStatus.UNSUPPORTED
    store.close()


@pytest.mark.parametrize("outcome", [None, True, float("nan"), 0.5])
def test_failed_queries_never_imputed_or_discarded(tmp_path, outcome):
    plan = make({"a": 1, "b": 1})
    store = journal(tmp_path, plan)
    result = evaluate_candidate(
        plan, store, lambda x: outcome, contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert result.status == GateStatus.INCONCLUSIVE and store.consumed_episodes() == 3
    assert store.observations(plan) == {}
    again = evaluate_candidate(
        plan, store, lambda x: 1, contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert again.status == GateStatus.INCONCLUSIVE and store.consumed_episodes() == 3
    repaired = evaluate_candidate(
        plan,
        store,
        lambda x: 1,
        contract=VALID,
        execution_scope="FROZEN_TABLE",
        max_unit_attempts=2,
    )
    assert repaired.status == GateStatus.ELIGIBLE and store.consumed_episodes() == 5
    store.close()


def test_crash_reservation_no_replay_and_no_replan(tmp_path):
    plan = make({str(i): 1 for i in range(20)})
    store = journal(tmp_path, plan)
    store.freeze(plan)
    unit = plan.strata[0].permutation[0]
    store.reserve(plan, unit)
    store.close()
    store = journal(tmp_path, plan)
    calls = []
    result = evaluate_candidate(
        plan, store, lambda x: calls.append(x), contract=VALID, execution_scope="FROZEN_TABLE"
    )
    assert not calls and result.status == GateStatus.INCONCLUSIVE
    with pytest.raises(ValueError):
        store.freeze(make({str(i): 1 for i in range(20)}, seed=23))
    with pytest.raises(ValueError):
        store.reserve(plan, unit)
    with pytest.raises(ValueError):
        store.reserve(plan, "foreign")
    store.close()
    with pytest.raises(ValueError):
        journal(tmp_path, plan, limit=3000)


def test_all_candidate_slots_persist_and_not_recycled(tmp_path):
    reference = {"a": 1}
    plan = make(reference)
    store = journal(tmp_path, plan)
    for candidate in range(32):
        assert store.freeze(make(reference, candidate=str(candidate))) == candidate
    with pytest.raises(BudgetExhausted):
        store.freeze(make(reference, candidate="overflow"))
    result = evaluate_candidate(
        make(reference, candidate="overflow"),
        store,
        lambda x: 1,
        contract=VALID,
        execution_scope="FROZEN_TABLE",
    )
    assert result.status == GateStatus.INCONCLUSIVE and result.candidate_slot is None
    assert store.report()["reserved_candidate_slots"] == 32
    store.close()


def test_tampered_plan_persisted_hash_rejected(tmp_path):
    plan = make({"a": 1})
    store = journal(tmp_path, plan)
    store.freeze(plan)
    value = plan.to_dict() | {"execution_digest": digest("tampered")}
    store.connection.execute("UPDATE delta_candidates SET plan_json=?", (json.dumps(value),))
    store.connection.commit()
    with pytest.raises(ValueError):
        store.load_plan(plan.candidate_digest)
    store.close()
