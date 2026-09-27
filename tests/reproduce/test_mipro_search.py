"""Authored module-boundary tests, not installed native optimizer evidence."""

import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from reproduce import mipro_search as bridge
from reproduce.strict_scoring import ScoringError


@pytest.fixture
def native_modules(monkeypatch):
    class Result:
        def __init__(self, score, results):
            self.score, self.results = score, results

    class Pruned(Exception):
        pass

    class Evaluate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    native = ModuleType("dspy.teleprompt.mipro_optimizer_v2")
    native.MIPROv2 = type("MIPROv2", (), {"_optimize_prompt_parameters": lambda *args: None})
    native.Evaluate = Evaluate
    native.eval_candidate_program = lambda *args: "original"
    utils = ModuleType("dspy.teleprompt.utils")
    utils.create_minibatch = lambda rows, size, rng: (rng or random).sample(rows, size)
    dspy = ModuleType("dspy")
    dspy.Prediction = lambda **kwargs: SimpleNamespace(**kwargs)
    evaluate_module = ModuleType("dspy.evaluate.evaluate")
    evaluate_module.EvaluationResult = Result
    optuna = ModuleType("optuna")
    optuna.TrialPruned = Pruned
    for module in (native, utils, dspy, evaluate_module, optuna):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return native, Result, Pruned


def program(text="authored instruction", demos=None):
    predictor = SimpleNamespace(
        signature=SimpleNamespace(
            instructions=text,
            input_fields={"task_input": None},
            output_fields={"task_output": None},
        ),
        demos=[] if demos is None else demos,
    )
    return SimpleNamespace(predictors=lambda: [predictor])


def test_exact_native_text_demo_rendering_not_response_schema():
    document = bridge.mipro_prompt(
        program(demos=[{"task_input": "all native context", "task_output": "original answer"}]),
        "candidate",
    )
    assert document["messages"] == [
        {"id": "instruction", "role": "system", "content": "authored instruction"},
        {"id": "demo-0-input", "role": "user", "content": "all native context"},
        {"id": "demo-0-output", "role": "assistant", "content": "original answer"},
    ]
    with pytest.raises(ScoringError, match="stringified"):
        bridge.mipro_prompt(program(demos=[{"task_input": ["context"], "task_output": "a"}]), "c")
    bad = program()
    bad.predictors()[0].signature.output_fields = {"other": None}
    with pytest.raises(ScoringError, match="signature"):
        bridge.mipro_prompt(bad, "c")


def test_native_sampler_strict_scores_and_restore(native_modules):
    native, Result, _ = native_modules
    original = (
        native.Evaluate,
        native.eval_candidate_program,
        native.MIPROv2._optimize_prompt_parameters,
    )
    data = [{"unit_id": str(i)} for i in range(8)]
    captured = []

    def evaluate(candidate, *, devset, callback_metadata):
        captured.append((candidate, devset, callback_metadata))
        return Result(100.0, [(x, object(), 1.0) for x in devset])

    expected = random.Random(11).sample(data, 3)
    with bridge.strict_mipro_search(evaluate):
        strict = native.Evaluate(max_errors=99, failure_score=0)
        assert strict.kwargs["max_errors"] == 1
        assert strict.kwargs["failure_score"] != strict.kwargs["failure_score"]
        result = native.eval_candidate_program(3, data, "c", None, random.Random(11))
        assert result.score == 100
        assert captured == [("c", expected, {"metric_key": "eval_minibatch"})]
        native.eval_candidate_program(8, data, "c", None, random.Random(11))
        assert captured[-1][1] is data
        assert captured[-1][2] == {"metric_key": "eval_full"}
        with pytest.raises(ScoringError, match="one owned"), bridge.strict_mipro_search():
            pytest.fail("nested ownership admitted")
    assert (
        native.Evaluate,
        native.eval_candidate_program,
        native.MIPROv2._optimize_prompt_parameters,
    ) == original
    assert not bridge._ACTIVE


@pytest.mark.parametrize("kind", ["prune", "failure", "incomplete"])
def test_native_error_is_not_zero_and_symbols_restored(native_modules, kind):
    native, Result, Pruned = native_modules
    original = (
        native.Evaluate,
        native.eval_candidate_program,
        native.MIPROv2._optimize_prompt_parameters,
    )

    def fail(*args, **kwargs):
        if kind == "incomplete":
            return Result(0.0, [])
        raise Pruned("actual prune") if kind == "prune" else RuntimeError("actual failure")

    error = Pruned if kind == "prune" else RuntimeError if kind == "failure" else ScoringError
    with pytest.raises(error), bridge.strict_mipro_search(fail):
        native.eval_candidate_program(1, [{}], None, None)
    assert (
        native.Evaluate,
        native.eval_candidate_program,
        native.MIPROv2._optimize_prompt_parameters,
    ) == original
    assert not bridge._ACTIVE


@pytest.mark.parametrize("bad", [None, float("nan"), 0.2, True, "1"])
def test_missing_nonbinary_native_results_refused(native_modules, bad):
    _, Result, _ = native_modules
    example = {}
    with pytest.raises(ScoringError, match="actual binary"):
        bridge.validate_result(Result(0, [(example, object(), bad)]), [example])


def test_native_output_and_aggregate_identity_required(native_modules):
    _, Result, _ = native_modules
    a, b = {"unit_id": "same"}, {"unit_id": "same"}
    with pytest.raises(ScoringError, match="identity"):
        bridge.validate_result(Result(100, [(b, object(), 1.0)]), [a])
    with pytest.raises(ScoringError, match="missing"):
        bridge.validate_result(Result(100, [(a, None, 1.0)]), [a])
    with pytest.raises(ScoringError, match="aggregate"):
        bridge.validate_result(Result(0, [(a, object(), 1.0)]), [a])
