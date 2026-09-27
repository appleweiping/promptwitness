from dataclasses import replace

import pytest

pytest.importorskip("sklearn")

from promptwitness.incremental import (
    ContractStatus,
    ExecutionIdentity,
    ParentTrace,
    TrainingRow,
    TransitionPredictor,
    check_contract,
    extract_features,
)
from promptwitness.incremental.optimizers import (
    complete_survivor_scores,
    evaluate_dspy_native,
    evaluate_gepa_native,
)
from promptwitness.incremental.sampling import digest
from promptwitness.models import ContentBlock, Message, PromptDocument, ToolSpec


def documents(schema=None):
    tool = ToolSpec("lookup", "look up", schema or {"city": {"type": "string"}}, ("city",))
    parent = PromptDocument("old", (Message("system", "answer {city}", message_id="s"),), (tool,))
    candidate = replace(
        parent,
        prompt_id="new",
        messages=(Message("system", "Return JSON for {city}", message_id="s"),),
    )
    return parent, candidate


def test_frozen_structure_text_ablations_trace_and_source_map():
    parent, candidate = documents()
    vector = extract_features(
        parent, candidate, "Boston", trace=ParentTrace(correct=1, input_tokens=9)
    )
    text = extract_features(
        parent, candidate, "Boston", trace=ParentTrace(correct=1, input_tokens=9), structured=False
    )
    assert vector.names[: len(text.names)] == text.names
    assert vector.values[: len(text.values)] == text.values
    assert len(vector.values) > len(text.values) and vector.source_map
    assert (
        vector.sha256
        == extract_features(
            parent, candidate, "Boston", trace=ParentTrace(correct=1, input_tokens=9)
        ).sha256
    )
    assert text.source_map == () and not text.structured
    missing = extract_features(parent, candidate, "Boston")
    assert (
        dict(zip(missing.names, missing.values, strict=True))["parent_trace.correct.missing"] == 1
    )
    with pytest.raises(TypeError):
        extract_features(parent, candidate, "Boston", candidate_output="not permitted")


@pytest.mark.parametrize(
    "bad",
    [
        dict(correct=True),
        dict(correct=2),
        dict(input_tokens=-1),
        dict(output_tokens=True),
        dict(allocated_seconds=float("inf")),
    ],
)
def test_invalid_parent_trace(bad):
    with pytest.raises(ValueError):
        ParentTrace(**bad)


def rows():
    parent, candidate = documents()
    records = []
    for i in range(12):
        features = extract_features(
            parent, candidate, "city " * i, trace=ParentTrace(correct=i % 2)
        )
        records.append(
            TrainingRow(
                features,
                i % 2,
                (i // 2) % 2,
                f"lineage{i}",
                f"group{i}",
                "fit",
                digest(i),
                kind="synthetic",
                allocated_seconds=i + 1,
            )
        )
    return tuple(records)


def test_unfitted_and_synthetic_fit_not_research_effect():
    predictor = TransitionPredictor()
    record = rows()[0]
    assert predictor.predict(record.features).status == "UNFITTED"
    assert predictor.predict(record.features).regression_probability is None
    assert TransitionPredictor.from_dict(predictor.to_dict()).status == "UNFITTED"
    predictor.fit(rows())
    prediction = predictor.predict(record.features)
    assert prediction.status == "FITTED_SYNTHETIC"
    assert 0 < prediction.regression_probability < 1
    assert 0 < prediction.improvement_probability < 1
    assert prediction.estimated_allocated_seconds >= 0
    restored = TransitionPredictor.from_dict(predictor.to_dict())
    assert restored.predict(record.features) == prediction
    mutated = predictor.to_dict()
    mutated["heads"]["regression"]["coefficients"][0] = float("nan")
    with pytest.raises(ValueError):
        TransitionPredictor.from_dict(mutated)


def test_fit_isolation_missing_head_smoothing_and_cost_missing():
    data = rows()
    model = TransitionPredictor()
    for kwargs in (
        {"held_out_lineages": frozenset({data[0].lineage})},
        {"held_out_sources": frozenset({data[0].source_group})},
    ):
        with pytest.raises(ValueError):
            model.fit(data, **kwargs)
    with pytest.raises(ValueError):
        replace(data[0], split="final")
    with pytest.raises(ValueError):
        replace(data[0], new_correct=True)
    subset = tuple(
        replace(row, new_correct=1, allocated_seconds=None) for row in data if row.old_correct == 1
    )
    model.fit(subset)
    p = model.predict(subset[0].features)
    assert 0 < p.regression_probability < 1 and p.improvement_probability is None
    assert p.estimated_allocated_seconds is None
    parent, candidate = documents()
    with pytest.raises(ValueError):
        model.predict(extract_features(parent, candidate, "x", structured=False))
    with pytest.raises(ValueError):
        model.fit(())


def test_contract_unknown_semantics_and_unchanged_external_interface():
    parent, candidate = documents()
    assert check_contract(parent, candidate).status == ContractStatus.VALID
    changed = replace(
        candidate, tools=(ToolSpec("lookup", "look up", {"city": {"type": "integer"}}, ("city",)),)
    )
    assert check_contract(parent, changed).status == ContractStatus.INVALID
    unsupported_parent, unsupported_candidate = documents(
        {"city": {"type": "string", "format": "email"}}
    )
    assert (
        check_contract(unsupported_parent, unsupported_candidate).status
        == ContractStatus.UNSUPPORTED
    )
    parts = replace(
        candidate, messages=(Message("user", "", content_parts=(ContentBlock("image", {}),)),)
    )
    assert check_contract(parent, parts).status == ContractStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "schema",
    [
        {"minimum": 5, "maximum": 2},
        {"enum": []},
        {"type": "string", "minLength": True},
        {"description": 3},
        {"required": ["a", "a"]},
        {"properties": []},
        {"items": False},
        {"additionalProperties": []},
        {"minItems": -1},
        {"maximum": float("inf")},
    ],
)
def test_malformed_supported_contracts(schema):
    if any(isinstance(x, float) and x == float("inf") for x in schema.values()):
        with pytest.raises(ValueError):
            documents({"city": schema})
        return
    parent, candidate = documents({"city": schema})
    assert check_contract(parent, candidate).status == ContractStatus.INVALID


def test_nested_contract_and_depth_fail_closed():
    parent, candidate = documents(
        {
            "city": {
                "type": "object",
                "properties": {"a": {"type": "array", "items": {"type": "integer"}}},
                "additionalProperties": {"type": "string"},
            }
        }
    )
    assert check_contract(parent, candidate).status == ContractStatus.VALID
    nested = {"type": "string"}
    for _ in range(34):
        nested = {"type": "array", "items": nested}
    parent, candidate = documents({"city": nested})
    assert check_contract(parent, candidate).status == ContractStatus.UNSUPPORTED


def test_complete_cache_identity_backend_and_replicate_invalidate():
    kwargs = {"model_revision": "m1", "tokenizer_revision": "t1", "backend_version": "b1"} | {
        name: digest(name)
        for name in [
            "backend_config_digest",
            "template_digest",
            "decoding_digest",
            "scorer_digest",
            "data_digest",
            "tool_environment_digest",
        ]
    }
    identity = ExecutionIdentity(**kwargs)
    key = identity.request_key({"messages": ["all context"]}, unit_id="u", replicate_id="observed1")
    assert (
        replace(identity, backend_version="b2").request_key(
            {"messages": ["all context"]}, unit_id="u", replicate_id="observed1"
        )
        != key
    )
    assert (
        identity.request_key(
            {"messages": ["all context"]}, unit_id="u", replicate_id="independent2"
        )
        != key
    )
    assert (
        identity.request_key(
            {"messages": ["changed context"]}, unit_id="u", replicate_id="observed1"
        )
        != key
    )
    with pytest.raises(ValueError):
        identity.request_key({}, unit_id="u", replicate_id="")


def test_native_vector_completion_only_queries_actual_missing_units():
    called = []

    def query(unit):
        called.append(unit)
        return 1

    vector = complete_survivor_scores(("a", "b"), {"a": 0}, query)
    assert vector.scores == (0, 1) and called == ["b"]
    with pytest.raises(ValueError):
        complete_survivor_scores(("a", "b"), {"a": None}, query)
    with pytest.raises(ValueError):
        complete_survivor_scores(("a",), {"foreign": 1}, query)


def test_actual_native_interface_shape_checks():
    from types import SimpleNamespace

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            return SimpleNamespace(
                outputs=list(batch),
                scores=[1.0] * len(batch),
                trajectories=list(batch) if capture_traces else None,
            )

    result = evaluate_gepa_native(Adapter(), [1, 2], {"system": "candidate"}, capture_traces=True)
    assert result.scores == [1.0, 1.0]
    with pytest.raises(ValueError):
        evaluate_gepa_native(
            type(
                "Broken",
                (),
                {"evaluate": lambda *args, **kwargs: SimpleNamespace(outputs=[], scores=[])},
            )(),
            [1],
            {},
        )

    def evaluate(program, **kwargs):
        return SimpleNamespace(results=[(x, "actual", 1.0) for x in kwargs["devset"]])

    evaluate.failure_score = float("nan")
    evaluate.max_errors = 1
    assert len(evaluate_dspy_native(evaluate, "program", [1, 2]).results) == 2
    with pytest.raises(ValueError):
        evaluate_dspy_native(lambda *args, **kwargs: SimpleNamespace(results=[]), "program", [1])
