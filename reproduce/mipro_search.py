"""Adapt pinned MIPRO search to the existing actual incremental controller.

The pinned helper swallows Exception (including TrialPruned) into score=0.
The synchronous, single-owner boundary adapts its helper, Evaluate and objective
method during compile, retains its exact sampler and restores all three symbols.
No partial/predicted scores or unverified online assumptions are introduced.
"""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager

from promptwitness.incremental.gate import GateStatus
from reproduce.mipro_cadence import cadence_optimizer
from reproduce.strict_scoring import ScoringError

_ACTIVE = False


def mipro_prompt(program, identifier):
    """Faithful single-response rendering, including original native demos.

    This is the task_input -> task_output text interface already exercised by
    check_mipro_native.py, not DSPy's default response-schema chat formatting.
    The same renderer must be used by both the native and certifier controls.
    Other program signatures are unsupported, not coerced or stringified.
    """
    predictors = program.predictors()
    if len(predictors) != 1:
        raise ScoringError("MIPRO research rendering requires one original task predictor")
    predictor = predictors[0]
    signature = predictor.signature
    if list(signature.input_fields) != ["task_input"] or list(signature.output_fields) != [
        "task_output"
    ]:
        raise ScoringError("unsupported MIPRO task signature")
    if not isinstance(signature.instructions, str):
        raise ScoringError("original native instruction text required")
    messages = [{"id": "instruction", "role": "system", "content": signature.instructions}]
    for index, demo in enumerate(predictor.demos):
        if not all(isinstance(demo[field], str) for field in ("task_input", "task_output")):
            raise ScoringError("native demonstration text cannot be discarded or stringified")
        messages.extend(
            [
                {"id": f"demo-{index}-input", "role": "user", "content": demo["task_input"]},
                {
                    "id": f"demo-{index}-output",
                    "role": "assistant",
                    "content": demo["task_output"],
                },
            ]
        )
    return {"schema_version": 1, "id": identifier, "messages": messages}


def validate_result(result, batch):
    """A native result must contain every requested, actually scored output."""
    entries = getattr(result, "results", None)
    if not batch or not isinstance(entries, list) or len(entries) != len(batch):
        raise ScoringError("MIPRO native result lacks complete actual outcomes")
    for example, entry in zip(batch, entries, strict=True):
        if (
            not isinstance(entry, tuple)
            or len(entry) != 3
            or entry[0] is not example
            or entry[1] is None
            or isinstance(entry[2], bool)
            or not isinstance(entry[2], (int, float))
            or not math.isfinite(entry[2])
            or entry[2] not in (0, 1)
        ):
            raise ScoringError("MIPRO native output identity or actual binary score missing")
    expected = round(100 * sum(row[2] for row in entries) / len(entries), 2)
    if getattr(result, "score", None) != expected:
        raise ScoringError("MIPRO native aggregate differs from complete actual outcomes")
    return result


@contextmanager
def strict_mipro_search(evaluator=None):
    """Run the original synchronous compile with strict evaluation semantics.

    This is an explicit adapted boundary, not a patch to the upstream checkout.
    Original proposer/bootstrap/TPE/minibatch/selector methods are invoked.
    The explicit shared objective adaptation performs due full evaluation of
    actual earlier survivors before propagating a candidate's TrialPruned.
    It is shared with the strict native control, not an unchanged upstream
    objective. No scientific qualification follows just from this context.
    Failures no longer become zero; TrialPruned reaches Optuna.
    Supply MIPROGateEvaluator for the certifier, or leave evaluator=None for the
    corresponding strict native control. One owned compile per process only.
    """
    global _ACTIVE
    native = importlib.import_module("dspy.teleprompt.mipro_optimizer_v2")
    utils = importlib.import_module("dspy.teleprompt.utils")
    if _ACTIVE:
        raise ScoringError("one owned synchronous MIPRO compile per process required")
    original_evaluate, original_helper = native.Evaluate, native.eval_candidate_program
    original_optimize = native.MIPROv2._optimize_prompt_parameters
    _ACTIVE = True

    class StrictEvaluate(original_evaluate):
        def __init__(self, *args, **kwargs):
            # Existing research interface policy: a failed execution is not an
            # incorrect-answer zero, and its cancellation must propagate.
            kwargs.update(max_errors=1, failure_score=float("nan"))
            super().__init__(*args, **kwargs)

    def strict_helper(batch_size, trainset, candidate_program, evaluate, rng=None):
        full = batch_size >= len(trainset)
        batch = trainset if full else utils.create_minibatch(trainset, batch_size, rng)
        selected = evaluate if evaluator is None else evaluator
        result = selected(
            candidate_program,
            devset=batch,
            callback_metadata={"metric_key": "eval_full" if full else "eval_minibatch"},
        )
        return validate_result(result, batch)

    native.Evaluate, native.eval_candidate_program = StrictEvaluate, strict_helper
    native.MIPROv2._optimize_prompt_parameters = cadence_optimizer(native)
    try:
        yield
    finally:
        native.Evaluate, native.eval_candidate_program = original_evaluate, original_helper
        native.MIPROv2._optimize_prompt_parameters = original_optimize
        _ACTIVE = False


class MIPROGateEvaluator:
    """Original MIPRO evaluation calls, backed by actual controller receipts.

    resolve(program) must faithfully render the original instructions/demos,
    register or reuse their exact PromptDocument, and freeze the existing audit
    before any current-candidate outcome. This evaluator never manufactures a
    rendering or grants ONLINE_PINNED. Search examples carry only unit_id and
    original inputs; no gold is needed here. All surviving candidates get a
    full original search census before their requested native batch is returned.
    """

    def __init__(self, driver, execute, resolve):
        self.driver, self.execute, self.resolve = driver, execute, resolve
        self.controller = driver.controller
        self.controller._one("reference_complete")

    def _complete(self, identifier):
        controller = self.controller
        if identifier != controller.spec["seed_prompt"]["id"]:
            # A mapper may not turn an already queried, unfrozen candidate into
            # a certificate after seeing its current outcomes.
            self.driver._audit(identifier)
            results = [
                event["result"]
                for event in controller.events
                if event["kind"] == "gate_result" and event["identifier"] == identifier
            ]
            result = (
                results[-1] if results else self.driver.evaluate(identifier, self.execute).to_dict()
            )
            if result["status"] == GateStatus.INELIGIBLE.value:
                controller._record("native_mipro_pruned", identifier=identifier, gate=result)
                import optuna

                raise optuna.TrialPruned("original finite-suite gate INELIGIBLE, not a zero score")
            if result["status"] != GateStatus.ELIGIBLE.value:
                controller._record("native_mipro_stopped", identifier=identifier, gate=result)
                raise ScoringError("MIPRO gate inconclusive/unsupported; not a pruned zero trial")
            complete = [
                event
                for event in controller.events
                if event["kind"] == "survivor_complete" and event["identifier"] == identifier
            ]
            if not complete:
                self.driver.complete_survivor(identifier, self.execute)
        scored = controller._candidate_scores(identifier)
        if len(scored) != 1 or set(scored[0]) != set(controller.spec["search_ids"]):
            raise ScoringError("native MIPRO survivor needs the entire actual search vector")
        tag = (
            "reference"
            if identifier == controller.spec["seed_prompt"]["id"]
            else ("search:" + identifier)
        )
        responses = {}
        for event in controller.events:
            if event["kind"] != "request" or event["tag"] != tag:
                continue
            for response in event["message"]["responses"]:
                unit = response["id"]
                if unit in responses and responses[unit] != response:
                    raise ScoringError("native output differs from the original scoring request")
                responses[unit] = response
        if set(responses) != set(scored[0]) or any(
            response["status"] != "completed"
            or not isinstance(response["output"], str)
            or response["replicate"] != scored[0][unit]["replicate"]
            for unit, response in responses.items()
        ):
            raise ScoringError("native actual outputs missing or random units changed")
        return responses, scored[0]

    def __call__(self, program, *, devset, callback_metadata):
        import dspy
        from dspy.evaluate.evaluate import EvaluationResult

        units = [example["unit_id"] for example in devset]
        if (
            not units
            or len(set(units)) != len(units)
            or not set(units) <= set(self.controller.spec["search_ids"])
            or callback_metadata
            not in ({"metric_key": "eval_full"}, {"metric_key": "eval_minibatch"})
        ):
            raise ScoringError("native MIPRO requests original search IDs and evaluation metadata")
        identifier = self.resolve(program)
        if mipro_prompt(program, identifier) != self.controller._candidate(identifier)["candidate"]:
            raise ScoringError("native MIPRO instructions/demos differ from the frozen prompt")
        self.controller._record(
            "native_mipro_evaluation_requested",
            identifier=identifier,
            unit_ids=units,
            metadata=callback_metadata,
        )
        responses, scored = self._complete(identifier)
        entries = [
            (
                example,
                dspy.Prediction(task_output=responses[unit]["output"]),
                float(scored[unit]["score"]),
            )
            for example, unit in zip(devset, units, strict=True)
        ]
        result = validate_result(
            EvaluationResult(
                round(100 * sum(row[2] for row in entries) / len(entries), 2), entries
            ),
            devset,
        )
        self.controller._record(
            "native_mipro_evaluation_complete",
            identifier=identifier,
            unit_ids=units,
            native_score=result.score,
            full_actual_survivor_vector=True,
            cost_source=(
                "existing logical audit journal and actual physical ledger, "
                "not native nominal batch count"
            ),
        )
        return result
