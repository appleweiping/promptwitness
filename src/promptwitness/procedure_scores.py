"""Original bounded scoring definitions for six explicitly named procedure families."""

from __future__ import annotations

from typing import Any

from .procedure_common_score import (
    ProcedureScoreLimitError as ProcedureScoreLimitError,
)
from .procedure_common_score import (
    ProcedureScoreLimits as ProcedureScoreLimits,
)
from .procedure_common_score import (
    ScoreWork,
    utf8_size,
)
from .procedure_data import ProcedureCase
from .procedure_math_score import score_countdown
from .procedure_route_score import score_path, score_travel
from .procedure_text_score import score_code, score_tom, score_tsv

PROCEDURE_SCORER_VERSION = "promptwitness.procedure-scores/v1"
PROCEDURE_METRIC_CAPABILITIES = {
    "countdown": {
        "primary": "full_solution_valid",
        "unsupported": ["search_procedure_correctness"],
    },
    "path_traversal": {"primary": "route_valid", "unsupported": []},
    "html_to_tsv": {"primary": "row_f1", "unsupported": []},
    "tom_tracking": {"primary": "declared_trace_exact_match", "unsupported": ["belief_entailment"]},
    "travel_planning": {"primary": "plan_valid", "unsupported": ["search_procedure_correctness"]},
    "pseudo_to_code": {"primary": None, "unsupported": ["executable_test_pass"]},
}


def score_procedure(
    case: ProcedureCase, prediction: str, *, limits: ProcedureScoreLimits | None = None
) -> dict[str, Any]:
    """Score one completed text answer, without I/O, reference execution, or inference.

    Callers receiving provider envelopes must first enforce response completeness
    (for example via ``task_scores.response_text``). Malformed in-budget answers
    receive diagnostic zero scores; resource violations raise a limit error and
    must not be relabeled as model correctness failures.
    """
    if not isinstance(case, ProcedureCase):
        raise ValueError("case must be a validated ProcedureCase")
    if not isinstance(prediction, str):
        raise ValueError("procedure prediction must be text")
    if limits is not None and not isinstance(limits, ProcedureScoreLimits):
        raise ValueError("limits must be ProcedureScoreLimits")
    budget = limits if limits is not None else ProcedureScoreLimits()
    utf8_size(prediction, budget.max_prediction_bytes)
    work = ScoreWork(budget)
    work.source(case.input)
    work.source(case.reference)
    # Admit the complete output, not only an extracted tag. Parsing selected
    # sections adds its own units to the same aggregate work/line counters.
    work.lines(prediction)
    if case.family == "countdown":
        report = score_countdown(case.input, case.reference, prediction, work)
    elif case.family == "path_traversal":
        report = score_path(case.input, case.reference, prediction, work)
    elif case.family == "html_to_tsv":
        report = score_tsv(case.input, case.reference, prediction, work)
    elif case.family == "tom_tracking":
        report = score_tom(case.reference, prediction, work)
    elif case.family == "travel_planning":
        report = score_travel(case.input, case.reference, prediction, work)
    elif case.family == "pseudo_to_code":
        report = score_code(prediction)
    else:  # Defensive even if an invalid object bypasses dataclass construction.
        raise ValueError("unsupported procedure family")
    report.update(
        scorer_version=PROCEDURE_SCORER_VERSION,
        score_limits=budget.to_dict(),
        work={"items": work.items, "lines": work.line_count, "case_text_bytes": work.source_bytes},
    )
    return report
