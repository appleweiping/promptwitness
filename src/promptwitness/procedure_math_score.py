"""Closed integer Countdown transitions; no expression evaluator or execution."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .procedure_common_score import ScoreWork, result, tagged

_EQUATION = re.compile(
    r"(0|[1-9][0-9]{0,9})\s*([+*/-])\s*(0|[1-9][0-9]{0,9})\s*=\s*(0|[1-9][0-9]{0,9})"
)


def _equations(lines: list[str], work: ScoreWork) -> list[tuple[int, str, int, int]] | None:
    work.use(len(lines))
    output = []
    for line in lines:
        match = _EQUATION.fullmatch(line.strip())
        if match is None:
            return None
        left, operation, right, declared = match.groups()
        output.append((int(left), operation, int(right), int(declared)))
    return output


def _validate(
    equations: list[tuple[int, str, int, int]], data: Mapping[str, Any]
) -> tuple[bool, int, str | None]:
    inventory = Counter(data["numbers"])
    required = len(data["numbers"]) - 1
    good = 0
    for left, operation, right, declared in equations:
        operands = Counter((left, right))
        if any(inventory[number] < count for number, count in operands.items()):
            return False, good, "operand_multiplicity"
        if operation == "+":
            actual = left + right
        elif operation == "-":
            actual = left - right
        elif operation == "*":
            actual = left * right
        elif right == 0:
            return False, good, "division_by_zero"
        elif left % right:
            return False, good, "non_integer_division"
        else:
            actual = left // right
        if actual != declared:
            return False, good, "incorrect_equation"
        if not data["min_intermediate"] <= actual <= data["max_intermediate"]:
            return False, good, "intermediate_out_of_range"
        inventory.subtract(operands)
        inventory[actual] += 1
        good += 1
    if len(equations) != required or sum(inventory.values()) != 1:
        return False, good, "incomplete_solution"
    if inventory[data["target"]] != 1:
        return False, good, "target_not_reached"
    return True, good, None


def score_countdown(
    data: Mapping[str, Any], reference: Mapping[str, Any], prediction: str, work: ScoreWork
) -> dict[str, Any]:
    gold = _equations(list(reference["solution"]), work)
    gold_valid = gold is not None and _validate(gold, data)[0]
    text = tagged(prediction, "Solution")
    equations = _equations(work.lines(text), work) if text is not None else None
    valid, good, reason = (
        _validate(equations, data) if equations is not None else (False, 0, "solution_format")
    )
    required = len(data["numbers"]) - 1
    return result(
        {
            "full_solution_valid": float(valid),
            "format_valid": float(equations is not None),
            "valid_step_fraction": min(good, required) / required,
            "reference_solution_exact_match": float(equations is not None and equations == gold),
            "reference_valid": float(gold_valid),
        },
        "full_solution_valid",
        [dict(left=a, operator=op, right=b, result=c) for a, op, b, c in equations]
        if equations is not None
        else None,
        diagnostics={"failure": reason, "valid_steps": good, "required_steps": required},
        unsupported={"search_procedure_correctness": "search_trace_semantics_not_implemented"},
    )
