"""Budgeted paired certification with explicit execution and contract boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from .contracts import ContractCheck, ContractStatus
from .journal import AuditJournal, BudgetExhausted
from .sampling import AuditPlan
from .statistics import PairedBounds, combine_counts, count_interval


class GateStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED = "UNSUPPORTED"


def decide(bounds: PairedBounds) -> GateStatus:
    if bounds.delta_lower >= Fraction(-1, 100) and bounds.regression_upper <= Fraction(1, 20):
        return GateStatus.ELIGIBLE
    if bounds.delta_upper < Fraction(-1, 100) or bounds.regression_lower > Fraction(1, 20):
        return GateStatus.INELIGIBLE
    return GateStatus.INCONCLUSIVE


@dataclass(frozen=True, slots=True)
class GateResult:
    status: GateStatus
    reason: str
    bounds: PairedBounds | None
    plan_digest: str
    candidate_slot: int | None
    look: int | None
    certificate_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "bounds": None if self.bounds is None else self.bounds.to_dict(),
            "plan_digest": self.plan_digest,
            "candidate_slot": self.candidate_slot,
            "look": self.look,
            "certificate_scope": self.certificate_scope,
            "per_count_direction_allowance": "1/61440",
            "per_run_familywise_alpha": "1/20",
            "future_distribution_or_global_campaign_guarantee": False,
        }


def evaluate_candidate(
    plan: AuditPlan,
    journal: AuditJournal,
    evaluator: Callable[[str], int],
    *,
    contract: ContractCheck,
    execution_scope: str = "UNVERIFIED",
    max_unit_attempts: int = 1,
) -> GateResult:
    """Only frozen-table mechanism or separately verified online execution admitted.

    Passing ONLINE_PINNED is a caller's documented assumption, not a proof by this
    function. Missing outcomes stay unavailable. Native score completion is a
    separate compulsory step for an eligible candidate entering native selection.
    """
    if contract.status != ContractStatus.VALID or execution_scope not in (
        "FROZEN_TABLE",
        "ONLINE_PINNED",
    ):
        return GateResult(
            GateStatus.UNSUPPORTED,
            "contract or execution assumptions not established",
            None,
            plan.sha256,
            None,
            None,
            "NONE",
        )
    scope = (
        "MECHANICAL_SEEDED_FIXTURE"
        if plan.randomization == "seeded_fixture"
        else "CONDITIONAL_FINITE_FROZEN_TABLE"
        if execution_scope == "FROZEN_TABLE"
        else "CONDITIONAL_FINITE_PINNED_ONLINE"
    )
    try:
        slot = journal.freeze(plan)
    except BudgetExhausted as error:
        return GateResult(GateStatus.INCONCLUSIVE, str(error), None, plan.sha256, None, None, scope)
    if journal.has_unresolved(plan):
        return GateResult(
            GateStatus.INCONCLUSIVE,
            "unresolved in-flight query; no silent replay",
            None,
            plan.sha256,
            slot,
            None,
            scope,
        )
    observations = journal.observations(plan)
    for look, allocation in enumerate(plan.allocations):
        current_counts = []
        for h, target in enumerate(allocation):
            stratum = plan.strata[h]
            expected = stratum.permutation[:target]
            for unit in expected:
                if unit in observations:
                    continue
                try:
                    attempt = journal.reserve(plan, unit, max_unit_attempts=max_unit_attempts)
                except BudgetExhausted as error:
                    result = GateResult(
                        GateStatus.INCONCLUSIVE, str(error), None, plan.sha256, slot, look, scope
                    )
                    journal.record(plan.candidate_digest, result.to_dict())
                    return result
                try:
                    score = evaluator(unit)
                    if type(score) is not int or score not in (0, 1):
                        raise ValueError("evaluator returned unsupported/missing/nonbinary score")
                except Exception as error:
                    journal.settle(plan, unit, attempt, score=None, error_type=type(error).__name__)
                    result = GateResult(
                        GateStatus.INCONCLUSIVE,
                        "charged evaluation failure; not imputed zero",
                        None,
                        plan.sha256,
                        slot,
                        look,
                        scope,
                    )
                    journal.record(plan.candidate_digest, result.to_dict())
                    return result
                journal.settle(plan, unit, attempt, score=score)
                observations[unit] = score
            # Recovered later looks may already have more observations; fixed
            # earlier looks still use precisely their original designated prefix.
            transitions = sum(observations[unit] != stratum.old_correct for unit in expected)
            current_counts.append(
                (stratum.old_correct, count_interval(len(stratum.permutation), target, transitions))
            )
        bounds = combine_counts(tuple(current_counts))
        result = GateResult(
            decide(bounds),
            "fixed-look simultaneous finite-suite bounds",
            bounds,
            plan.sha256,
            slot,
            look,
            scope,
        )
        journal.record(plan.candidate_digest, result.to_dict())
        if result.status != GateStatus.INCONCLUSIVE:
            return result
    return result
