"""Engineering review one: genuine failure and recovery branches, no method gains."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")

from promptwitness.incremental import (
    AuditJournal,
    AuditPlan,
    ContractStatus,
    CountInterval,
    ExecutionIdentity,
    FeatureVector,
    ParentTrace,
    TrainingRow,
    TransitionPredictor,
    check_contract,
    combine_counts,
    count_interval,
    extract_features,
    make_plan,
)
from promptwitness.incremental.budget import ResourceLedger, ResourceLimit
from promptwitness.incremental.optimizers import evaluate_dspy_native, evaluate_gepa_native
from promptwitness.incremental.sampling import Stratum, digest
from promptwitness.incremental.statistics import exact_tail
from promptwitness.models import Message, PromptDocument, ToolSpec


def plan():
    return make_plan(
        {"a": 1, "b": 0},
        candidate_digest=digest("c"),
        execution_digest=digest("e"),
        fixture_seed=11,
    )


def store(tmp_path, p=None):
    p = plan() if p is None else p
    return AuditJournal(
        tmp_path / "audit.sqlite",
        run_id="r",
        reference_digest=p.reference_digest,
        execution_digest=p.execution_digest,
        reference_episodes=p.population,
    )


@pytest.mark.parametrize("kwargs", [{"run_id": ""}, {"reference_digest": "bad"}])
def test_bad_journal_identity(tmp_path, kwargs):
    params = {
        "run_id": "r",
        "reference_digest": digest("r"),
        "execution_digest": digest("e"),
        "reference_episodes": 1,
    }
    with pytest.raises(ValueError):
        AuditJournal(tmp_path / "x.sqlite", **(params | kwargs))


def test_unfrozen_missing_reference_unknown_settle_and_observation_corruption(tmp_path):
    p = plan()
    journal = store(tmp_path, p)
    assert journal.load_plan(digest("absent")) is None
    with pytest.raises(ValueError):
        journal.freeze(replace(p, execution_digest=digest("different")))
    journal.freeze(p)
    with pytest.raises(ValueError):
        journal.reserve(p, "a", max_unit_attempts=4)
    attempt = journal.reserve(p, "a")
    with pytest.raises(ValueError):
        journal.settle(p, "a", attempt, score=True)
    with pytest.raises(ValueError):
        journal.settle(p, "a", attempt, score=None)
    journal.settle(p, "a", attempt, score=1)
    with pytest.raises(ValueError):
        journal.settle(p, "a", attempt, score=1)
    journal.connection.execute("UPDATE delta_queries SET score=3")
    journal.connection.commit()
    with pytest.raises(ValueError):
        journal.observations(p)
    journal.close()


def test_oracle_source_cannot_change_on_recovery(tmp_path):
    p = plan()
    journal = store(tmp_path, p)
    with pytest.raises(ValueError):
        journal.bind_outcome_source(p, digest("table"))
    journal.freeze(p)
    journal.bind_outcome_source(p, digest("table"))
    journal.bind_outcome_source(p, digest("table"))
    with pytest.raises(ValueError):
        journal.bind_outcome_source(p, digest("swapped labels"))
    journal.close()


@pytest.mark.parametrize("bad", [Stratum.__name__, None])
def test_serialized_plan_invalid_shapes(bad):
    value = plan().to_dict()
    for field in ("strata", "allocations"):
        with pytest.raises(ValueError):
            AuditPlan.from_dict(value | {field: bad})
    with pytest.raises(ValueError):
        AuditPlan.from_dict(value | {"strata": [bad]})
    with pytest.raises(ValueError):
        AuditPlan.from_dict(value | {"allocations": [bad]})


@pytest.mark.parametrize("old,members", [(True, ("a",)), (2, ("a",)), (1, ()), (1, ("",))])
def test_stratum_invalid(old, members):
    with pytest.raises(ValueError):
        Stratum(old, members)


def test_direct_plan_allocation_invariants():
    p = plan()
    for changes in (
        {"strata": ()},
        {"allocations": ()},
        {"allocations": ((1,), (2,))},
        {"allocations": ((2, 0), (0, 2))},
        {"strata": (Stratum(1, ("a",)), Stratum(0, ("a",)))},
    ):
        with pytest.raises(ValueError):
            replace(p, **changes)


def test_invalid_intervals_exact_reference_and_empty_regression_rational():
    with pytest.raises(ValueError):
        CountInterval(2, 1, 10, 2, 1)
    with pytest.raises(ValueError):
        exact_tail(4097, 0, 0, 0, upper=True)
    no_old_correct = combine_counts(((0, count_interval(10, 10, 3)),))
    assert no_old_correct.regression_estimate == Fraction(0)
    assert isinstance(no_old_correct.regression_estimate, Fraction)


def test_missing_science_library_and_nonfinite_tail_fail_closed(monkeypatch):
    import promptwitness.incremental.statistics as stats

    count_interval.cache_clear()

    def missing(name):
        raise ImportError("owned test fixture")

    monkeypatch.setattr(stats, "import_module", missing)
    with pytest.raises(ValueError):
        count_interval(17, 4, 1)
    monkeypatch.setattr(
        stats,
        "import_module",
        lambda name: SimpleNamespace(
            hypergeom=SimpleNamespace(sf=lambda *args: float("nan"), cdf=lambda *args: float("nan"))
        ),
    )
    with pytest.raises(ArithmeticError):
        count_interval(17, 4, 1)
    count_interval.cache_clear()


@pytest.mark.parametrize(
    "names,values", [((), ()), (("x", "x"), (1.0, 2.0)), (("x",), (float("inf"),))]
)
def test_feature_validation(names, values):
    with pytest.raises(ValueError):
        FeatureVector(names, values, (), True)


def test_feature_bad_version_multimodal_input_type_and_reference_trace_cost():
    p = PromptDocument("p", (Message("system", "x"),))
    with pytest.raises(ValueError):
        extract_features(p, p, 3)
    v = extract_features(p, p, "", trace=ParentTrace(allocated_seconds=0.1))
    with pytest.raises(ValueError):
        replace(v, version="unapproved")


def test_predictor_checkpoint_invalid_provenance_and_fit_rows(monkeypatch):
    p = PromptDocument("p", (Message("system", "x"),))
    v = extract_features(p, p, "input")
    model = TransitionPredictor()
    row = TrainingRow(v, 1, 1, "lineage", "source", "fit", digest("l"), kind="synthetic")
    with pytest.raises(ValueError):
        replace(row, lineage="")
    with pytest.raises(ValueError):
        replace(row, allocated_seconds=-1)
    import promptwitness.incremental.predictor as module

    def missing(name):
        raise ImportError("fixture")

    monkeypatch.setattr(module, "import_module", missing)
    with pytest.raises(ValueError):
        model.fit((row,))
    monkeypatch.undo()
    model.fit((row,))
    original = model.to_dict()
    for changes in (
        {"format": "bad"},
        {"status": "unknown"},
        {"feature_names": []},
        {"heads": {}},
        {"heads": {"regression": {}, "improvement": None}},
        {"heads": {"regression": {"coefficients": [], "intercept": 0}, "improvement": None}},
    ):
        with pytest.raises(ValueError):
            TransitionPredictor.from_dict(original | changes)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": ["string", "null"]},
        {"required": False},
        {"properties": {"a": {"unexpected": True}}},
        {"additionalProperties": {"unexpected": True}},
    ],
)
def test_schema_support_boundaries(schema):
    document = PromptDocument("p", (Message("system", "x"),), (ToolSpec("t", "d", {"x": schema}),))
    assert check_contract(document, document).status != ContractStatus.VALID


def test_identity_bad_blank_and_digest():
    fields = {"model_revision": "m", "tokenizer_revision": "t", "backend_version": "b"} | {
        key: digest(key)
        for key in (
            "backend_config_digest",
            "template_digest",
            "decoding_digest",
            "scorer_digest",
            "data_digest",
            "tool_environment_digest",
        )
    }
    with pytest.raises(ValueError):
        ExecutionIdentity(**(fields | {"backend_version": ""}))
    with pytest.raises(ValueError):
        ExecutionIdentity(**(fields | {"template_digest": "invalid"}))


@pytest.mark.parametrize(
    "payload",
    [
        SimpleNamespace(outputs=[1], scores=[1], trajectories=None),
        SimpleNamespace(outputs=[1], scores=[float("nan")], trajectories=[1]),
    ],
)
def test_gepa_native_failures(payload):
    adapter = SimpleNamespace(evaluate=lambda *args, **kwargs: payload)
    with pytest.raises(ValueError):
        evaluate_gepa_native(adapter, [1], {}, capture_traces=True)


def test_dspy_native_nonfinite_results():
    def evaluate(*args, **kwargs):
        return SimpleNamespace(results=[(1, 2, float("nan"))])

    evaluate.failure_score = float("nan")
    evaluate.max_errors = 1
    with pytest.raises(ValueError):
        evaluate_dspy_native(evaluate, None, [1])


def test_dspy_default_zero_failure_is_rejected_before_evaluation():
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(results=[(1, "missing", 0)])

    evaluate.failure_score = 0.0
    evaluate.max_errors = 1
    with pytest.raises(ValueError, match="fabricated zero"):
        evaluate_dspy_native(evaluate, None, [1])
    assert not calls


def test_native_missing_output_and_dspy_incomplete_vector_are_not_observations():
    adapter = SimpleNamespace(
        evaluate=lambda *args, **kwargs: SimpleNamespace(outputs=[None], scores=[0])
    )
    with pytest.raises(ValueError):
        evaluate_gepa_native(adapter, [1], {})

    def evaluate(*args, **kwargs):
        return SimpleNamespace(results=[])

    evaluate.failure_score = float("nan")
    evaluate.max_errors = 1
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_dspy_native(evaluate, None, [1])


def test_budget_limits_invalid_unknown_stage_and_status(tmp_path):
    for limit in ((True, 0, 0, 1), (1, 1, 1, float("inf"))):
        with pytest.raises(ValueError):
            ResourceLimit(*limit)
    params = {
        "historical_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        "historical_digest": digest("prior"),
        "global_limit": ResourceLimit(10, 100, 100, 8),
        "stage_limits": {"s": ResourceLimit(10, 100, 100, 8)},
        "gpu_uuid": "GPU-owned",
    }
    with pytest.raises(ValueError):
        ResourceLedger(tmp_path / "bad.sqlite", **(params | {"stage_limits": {}}))
    with pytest.raises(ValueError):
        ResourceLedger(tmp_path / "bad.sqlite", **(params | {"gpu_uuid": "not-a-device"}))
    resource = ResourceLedger(tmp_path / "resource.sqlite", **params)
    with pytest.raises(ValueError):
        resource.usage("unmeasured")
    with pytest.raises(ValueError):
        resource.reserve_call("a", "missing", digest("r"), input_cap=1, output_cap=1)
    with pytest.raises(ValueError):
        resource.reserve_call("", "s", digest("r"), input_cap=1, output_cap=1)
    with pytest.raises(ValueError):
        resource.settle_call("unknown", succeeded=True, input_tokens=None, output_tokens=None)
    with pytest.raises(ValueError):
        resource.settle_call("unknown", succeeded=False, input_tokens=1, output_tokens=None)
    resource.reserve_call("a", "s", digest("r"), input_cap=1, output_cap=1)
    resource.settle_call("a", succeeded=False, input_tokens=1, output_tokens=0)
    resource.close()
