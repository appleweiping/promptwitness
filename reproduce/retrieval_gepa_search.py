"""Freeze GEPA retrieval proposals before native minibatch evaluation.

Use this as GEPA's ``custom_candidate_proposer`` and ``acceptance_criterion``
with its original reflective engine, ``use_merge=False``, the default
``AllImprovements`` selection, and ``raise_on_exception=True``. GEPA's
acceptance hook runs *after* the child minibatch: freezing there would make the
audit plan depend on candidate outcomes. The proposal hook instead persists
the plan before the child is scored. This is an adapted native search path, not
an unchanged upstream GEPA run or a scientific admission certificate. Only a
fresh run is supported: upstream resume can add a different seed directly,
without passing the proposal/acceptance hooks. Upstream may catch and retry a
proposal-hook error; ``raise_on_exception=True`` is not a global fail-stop.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from promptwitness.incremental.contracts import ContractCheck
from promptwitness.incremental.gate import GateStatus
from promptwitness.incremental.optimizers import ActualScoreVector
from promptwitness.incremental.sampling import digest
from reproduce.retrieval_audit_bridge import RetrievalAuditBridge
from reproduce.strict_scoring import ScoringError


def _program_key(candidate: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if (
        not isinstance(candidate, dict)
        or not candidate
        or any(
            not isinstance(name, str) or not name or not isinstance(text, str)
            for name, text in candidate.items()
        )
    ):
        raise ScoringError("GEPA candidate requires named original text components")
    return tuple(sorted(candidate.items()))


def gepa_prompt(candidate: dict[str, str], identifier: str) -> dict:
    """Preserve every native component name and text in a stable document."""
    components = _program_key(candidate)
    if not isinstance(identifier, str) or not identifier:
        raise ScoringError("GEPA candidate identifier required")
    return {
        "schema_version": 1,
        "id": identifier,
        "messages": [
            {"id": f"component:{name}", "role": "system", "content": text}
            for name, text in components
        ],
    }


@dataclass(frozen=True, slots=True)
class _Recorded:
    identifier: str
    document: dict
    scores: ActualScoreVector


class RetrievalGEPAGateController:
    """Bind proposal-time frozen plans to GEPA's native acceptance hook.

    The caller supplies an actual complete seed vector and owns the scorer,
    ranker, proposal model, stage isolation, physical ledger, and online
    identity. ``freeze`` must persist a plan for the proposed child against
    the exact parent vector it receives. This controller checks that receipt,
    rejects incomplete outcomes, and completes every accepted search vector
    before GEPA may add the child to its pool. It does not authenticate the
    provenance of externally supplied score or proposal callbacks.
    """

    def __init__(
        self,
        *,
        seed_candidate: dict[str, str],
        seed_identifier: str,
        seed_scores: ActualScoreVector,
        propose: Callable[[dict[str, str], Mapping, list[str]], dict[str, str]],
        freeze: Callable[
            [dict[str, str], dict, ActualScoreVector],
            tuple[str, dict, RetrievalAuditBridge],
        ],
        contract: ContractCheck,
        execution_scope: str,
    ) -> None:
        key = _program_key(seed_candidate)
        if (
            not isinstance(seed_scores, ActualScoreVector)
            or not seed_scores.unit_ids
            or len(seed_scores.unit_ids) != len(seed_scores.scores)
            or len(set(seed_scores.unit_ids)) != len(seed_scores.unit_ids)
            or any(type(score) is not int or score not in (0, 1) for score in seed_scores.scores)
        ):
            raise ScoringError("complete binary seed search vector required")
        if not callable(propose) or not callable(freeze):
            raise ScoringError("proposal and pre-outcome freeze callbacks required")
        self.propose = propose
        self.freeze = freeze
        self.contract = contract
        self.execution_scope = execution_scope
        self._recorded = {
            key: _Recorded(
                seed_identifier, gepa_prompt(seed_candidate, seed_identifier), seed_scores
            )
        }
        self._pending: dict[
            tuple[tuple[str, str], ...],
            tuple[tuple[tuple[str, str], ...], _Recorded, RetrievalAuditBridge],
        ] = {}
        self._reasons: dict[int, str] = {}

    def propose_new_texts(
        self,
        parent_candidate: dict[str, str],
        reflective_dataset: Mapping,
        components_to_update: list[str],
        *,
        metadata: Mapping | None = None,
    ) -> dict[str, str]:
        """GEPA calls this before the child minibatch, including on retries."""
        del metadata
        parent_key = _program_key(parent_candidate)
        if parent_key not in self._recorded:
            raise ScoringError("GEPA parent lacks a complete actual search vector")
        texts = self.propose(parent_candidate, reflective_dataset, components_to_update)
        if (
            not isinstance(texts, dict)
            or any(
                name not in parent_candidate or not isinstance(text, str)
                for name, text in texts.items()
            )
            or set(texts) - set(components_to_update)
        ):
            raise ScoringError("GEPA proposal changed unknown or unselected components")
        candidate = dict(parent_candidate)
        candidate.update(texts)
        child_key = _program_key(candidate)
        if child_key == parent_key:
            return {}  # Native GEPA skips a byte-identical proposal.
        if child_key in self._pending:
            raise ScoringError("GEPA child already has an outstanding frozen proposal")

        parent = self._recorded[parent_key]
        identifier, document, bridge = self.freeze(
            candidate, copy.deepcopy(parent.document), parent.scores
        )
        if (
            not isinstance(identifier, str)
            or not identifier
            or document != gepa_prompt(candidate, identifier)
            or not isinstance(bridge, RetrievalAuditBridge)
            or bridge.unit_ids != parent.scores.unit_ids
            or bridge.plan.candidate_digest != digest(document)
            or bridge.plan.reference_digest
            != digest(dict(zip(parent.scores.unit_ids, parent.scores.scores, strict=True)))
            or bridge.journal.load_plan(bridge.plan.candidate_digest) != bridge.plan
        ):
            raise ScoringError("GEPA child plan was not frozen against its actual parent")
        self._pending[child_key] = (
            parent_key,
            _Recorded(identifier, copy.deepcopy(document), parent.scores),
            bridge,
        )
        return texts

    def should_accept(self, proposal: Any, state: Any) -> bool:
        """Native GEPA calls this after child minibatch, before full valset."""
        parents = getattr(proposal, "parent_program_ids", None)
        programs = getattr(state, "program_candidates", None)
        if (
            not isinstance(parents, list)
            or len(parents) != 1
            or type(parents[0]) is not int
            or not isinstance(programs, list)
            or not 0 <= parents[0] < len(programs)
        ):
            raise ScoringError("GEPA retrieval gate supports one native parent, no merge")
        parent_key = _program_key(programs[parents[0]])
        child_key = _program_key(proposal.candidate)
        pending = self._pending.get(child_key)
        if pending is None or pending[0] != parent_key:
            raise ScoringError("GEPA child was scored before its parent-bound plan was frozen")
        before = getattr(proposal, "subsample_scores_before", None)
        after = getattr(proposal, "subsample_scores_after", None)
        indices = getattr(proposal, "subsample_indices", None)
        before_eval = getattr(proposal, "eval_before", None)
        after_eval = getattr(proposal, "eval_after", None)
        if (
            not isinstance(indices, list)
            or not indices
            or not isinstance(before, list)
            or not isinstance(after, list)
            or len(indices) != len(before)
            or len(indices) != len(after)
            or before_eval is None
            or after_eval is None
            or getattr(before_eval, "scores", None) != before
            or getattr(after_eval, "scores", None) != after
            or not isinstance(getattr(before_eval, "outputs", None), list)
            or not isinstance(getattr(after_eval, "outputs", None), list)
            or len(before_eval.outputs) != len(indices)
            or len(after_eval.outputs) != len(indices)
            or any(output is None for output in (*before_eval.outputs, *after_eval.outputs))
            or any(
                type(value) not in (int, float) or value not in (0, 1)
                for value in (*before, *after)
            )
        ):
            raise ScoringError("GEPA minibatch requires complete actual binary scores")
        if sum(after) <= sum(before):
            self._reasons[id(proposal)] = "native GEPA minibatch did not strictly improve"
            self._pending.pop(child_key)
            return False

        _, child_record, bridge = pending
        verdict = bridge.evaluate(contract=self.contract, execution_scope=self.execution_scope)
        if verdict.status == GateStatus.INELIGIBLE:
            self._reasons[id(proposal)] = "retrieval audit INELIGIBLE"
            self._pending.pop(child_key)
            return False
        if verdict.status != GateStatus.ELIGIBLE:
            raise ScoringError("retrieval audit inconclusive/unsupported, not a zero score")
        complete = bridge.complete_survivor()
        if complete.unit_ids != child_record.scores.unit_ids:
            raise ScoringError("GEPA survivor search order changed")
        self._recorded[child_key] = _Recorded(
            child_record.identifier, child_record.document, complete
        )
        self._pending.pop(child_key)
        return True

    def reject_reason(self, proposal: Any, state: Any) -> str:
        del state
        return self._reasons.get(id(proposal), "retrieval gate rejected proposal")

    def search_scores(self, candidate: dict[str, str]) -> ActualScoreVector:
        """Expose only completed accepted vectors for test/owned orchestration."""
        return self._recorded[_program_key(candidate)].scores
