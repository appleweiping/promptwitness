"""Adapt native MIPRO evaluation to a frozen, scorer-only retrieval audit.

The caller owns input-only ranker creation, reference-score provenance and
candidate freezing. This adapter cannot authenticate those callbacks or turn
an authored qualification into an online/scientific result. It does ensure
that a native selector sees only actual, complete survivor hit vectors.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from promptwitness.incremental.contracts import ContractCheck
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.optimizers import ActualScoreVector
from reproduce.mipro_search import mipro_prompt, validate_result
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.strict_scoring import ScoringError


class RetrievalMIPROGateEvaluator:
    """Return actual scorer hits for native MIPRO's requested search batch.

    ``freeze`` must persist the candidate document, plan and ranker identity
    before its first outcome. The seeded reference vector must already have
    been scored on the same complete search population. Neither guarantee is
    inferred from Python callbacks, so production admission remains external.
    Native minibatches may receive only their requested subset, but an eligible
    candidate first pays for and completes the full original search vector.
    """

    def __init__(
        self,
        *,
        reference_document: dict,
        reference_scores: ActualScoreVector,
        freeze: Callable[[Any], tuple[str, dict, RetrievalAuditBridge]],
        contract: ContractCheck,
        execution_scope: str,
    ) -> None:
        if (
            not isinstance(reference_document, dict)
            or not isinstance(reference_document.get("id"), str)
            or not reference_document["id"]
            or not isinstance(reference_scores, ActualScoreVector)
            or not reference_scores.unit_ids
            or len(set(reference_scores.unit_ids)) != len(reference_scores.unit_ids)
            or len(reference_scores.unit_ids) != len(reference_scores.scores)
            or any(
                type(score) is not int or score not in (0, 1) for score in reference_scores.scores
            )
        ):
            raise ScoringError("complete binary reference search vector required")
        self.reference_document = copy.deepcopy(reference_document)
        self.reference_scores = reference_scores
        self.freeze = freeze
        self.contract = contract
        self.execution_scope = execution_scope

    def __call__(self, program: Any, *, devset: list[Any], callback_metadata: dict) -> Any:
        import dspy
        from dspy.evaluate.evaluate import EvaluationResult

        units = []
        for example in devset:
            try:
                unit = example["unit_id"]
            except (KeyError, TypeError, AttributeError) as error:
                raise ScoringError("native retrieval examples require search unit IDs") from error
            if not isinstance(unit, str):
                raise ScoringError("native retrieval examples require search unit IDs")
            units.append(unit)
        if (
            not units
            or len(set(units)) != len(units)
            or not set(units) <= set(self.reference_scores.unit_ids)
            or callback_metadata
            not in ({"metric_key": "eval_full"}, {"metric_key": "eval_minibatch"})
        ):
            raise ScoringError("native retrieval request has invalid units or evaluation metadata")

        reference_render = mipro_prompt(program, self.reference_document["id"])
        if reference_render == self.reference_document:
            vector = self.reference_scores
        else:
            identifier, document, bridge = self.freeze(program)
            if (
                not isinstance(identifier, str)
                or not identifier
                or mipro_prompt(program, identifier) != document
                or not isinstance(bridge, RetrievalAuditBridge)
                or bridge.unit_ids != self.reference_scores.unit_ids
            ):
                raise ScoringError("native retrieval program differs from frozen audit candidate")
            verdict = bridge.evaluate(contract=self.contract, execution_scope=self.execution_scope)
            if verdict.status == GateStatus.INELIGIBLE:
                import optuna

                raise optuna.TrialPruned("actual retrieval audit INELIGIBLE, not a zero score")
            if verdict.status != GateStatus.ELIGIBLE:
                raise ScoringError("retrieval audit inconclusive/unsupported, not a zero score")
            vector = bridge.complete_survivor()
            if vector.unit_ids != self.reference_scores.unit_ids:
                raise ScoringError("native retrieval survivor order changed")

        scored = dict(zip(vector.unit_ids, vector.scores, strict=True))
        entries = [
            (
                example,
                dspy.Prediction(primary_hit=scored[unit]),
                float(scored[unit]),
            )
            for example, unit in zip(devset, units, strict=True)
        ]
        return validate_result(
            EvaluationResult(
                round(100 * sum(row[2] for row in entries) / len(entries), 2), entries
            ),
            devset,
        )
