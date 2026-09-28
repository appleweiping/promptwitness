"""Connect frozen retrieval audit plans to the scorer-only process.

This is a logical scoring route, not a model-execution or physical-cost ledger.
The caller must freeze the candidate/encoder, store and selector unit order,
and charge every image, text, LLM, and GPU operation performed by ``rank_one``
before scientific online use. This bridge cannot authenticate that callback
or establish ONLINE_PINNED execution semantics.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from promptwitness.incremental.contracts import ContractCheck
from promptwitness.incremental.gate import GateResult, GateStatus, evaluate_candidate
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import ActualScoreVector, complete_survivor_scores
from promptwitness.incremental.sampling import AuditPlan
from reproduce.retrieval_role_scoring import DATASETS, score_rankings_restricted


class RetrievalAuditBridge:
    """Single-writer, fixed-plan primary-hit route for the search population.

    The existing journal reserves each logical query before rank generation and
    charges failures. An eligible candidate is completed with actual scores
    before its vector can enter a native selector. Only the gate result is
    returned during early evaluation; raw scorer feedback is not passed to a
    proposer. A new instance may resume by re-evaluating the same frozen plan.
    """

    def __init__(
        self,
        *,
        store: Path,
        scratch_root: Path,
        dataset: str,
        plan: AuditPlan,
        journal: AuditJournal,
        unit_ids: tuple[str, ...],
        rank_one: Callable[[str], tuple[str, ...]],
    ) -> None:
        if dataset not in DATASETS:
            raise ValueError("only CIRR/FashionIQ primary-hit audits are supported")
        if not scratch_root.is_dir():
            raise ValueError("existing private scorer scratch root required")
        planned = {unit for stratum in plan.strata for unit in stratum.permutation}
        if (
            not isinstance(unit_ids, tuple)
            or len(unit_ids) != len(planned)
            or any(not isinstance(unit, str) or not unit for unit in unit_ids)
            or len(set(unit_ids)) != len(unit_ids)
            or set(unit_ids) != planned
        ):
            raise ValueError("frozen selector population/order must match the audit plan")
        self.store = store
        self.scratch_root = scratch_root
        self.dataset = dataset
        self.plan = plan
        self.journal = journal
        self.rank_one = rank_one
        self.unit_ids = unit_ids
        self._last_gate: GateResult | None = None

    def _score(self, unit: str) -> int:
        if unit not in self.unit_ids:
            raise ValueError("query is outside the frozen audit population")
        ranking = self.rank_one(unit)
        if not isinstance(ranking, tuple):
            raise ValueError("ranker must return a complete immutable ranking")
        work = Path(tempfile.mkdtemp(prefix="retrieval-score-", dir=self.scratch_root))
        report = score_rankings_restricted(
            self.store, work, self.dataset, "search", {unit: ranking}
        )
        score = report["observations"][unit]["primary_hit"]
        if type(score) is not int or score not in (0, 1):
            raise ValueError("retrieval scorer did not return an actual binary primary hit")
        return score

    def evaluate(self, *, contract: ContractCheck, execution_scope: str) -> GateResult:
        """Audit fixed random prefixes; do not infer ONLINE_PINNED from this call."""
        result = evaluate_candidate(
            self.plan,
            self.journal,
            self._score,
            contract=contract,
            execution_scope=execution_scope,
        )
        self._last_gate = result
        return result

    def complete_survivor(self) -> ActualScoreVector:
        """Pay for every missing actual hit before returning a selector vector."""
        if (
            self._last_gate is None
            or self._last_gate.status != GateStatus.ELIGIBLE
            or self._last_gate.plan_digest != self.plan.sha256
        ):
            raise ValueError("only an eligible result for this frozen plan may be completed")
        if self.journal.has_unresolved(self.plan):
            raise ValueError("unresolved logical query; no silent replay")

        def query(unit: str) -> int:
            attempt = self.journal.reserve(self.plan, unit)
            try:
                score = self._score(unit)
            except Exception as error:
                self.journal.settle(
                    self.plan, unit, attempt, score=None, error_type=type(error).__name__
                )
                raise
            self.journal.settle(self.plan, unit, attempt, score=score)
            return score

        return complete_survivor_scores(self.unit_ids, self.journal.observations(self.plan), query)
