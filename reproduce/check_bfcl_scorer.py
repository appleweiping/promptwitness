"""CPU qualification of the real pinned BFCL matcher on authored fixtures only.

No benchmark samples, archived model responses, final labels or model calls are
loaded here. This verifies a native runtime slice, not scientific admission.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from reproduce.bfcl_native import TASK_MODELS, BFCLChecker, load_bfcl_native
from reproduce.strict_scoring import ScoringError, UnsupportedScoring, score_bfcl


def authored_fixtures() -> list[dict[str, Any]]:
    """Predetermined examples from checker semantics, not benchmark records."""

    def tool(name: str, kind: str = "integer") -> dict[str, Any]:
        return {
            "name": name,
            "description": "Authored scorer fixture, not a dataset example",
            "parameters": {
                "type": "dict",
                "properties": {"value": {"type": kind}},
                "required": ["value"],
            },
        }

    tools = [tool("fixture.alpha"), tool("fixture.beta"), tool("fixture.unused")]
    alpha = {"name": "fixture.alpha", "arguments": {"value": 2}}
    beta = {"name": "fixture.beta", "arguments": {"value": 3}}
    gold = [{"fixture.alpha": {"value": [2]}}]
    parallel_gold = [*gold, {"fixture.beta": {"value": [3]}}]
    cases: list[dict[str, Any]] = []

    def add(
        name: str,
        calls: Any,
        expected: int,
        category: str = "simple_python",
        functions: list[dict[str, Any]] | None = None,
        answers: list[dict[str, Any]] | None = None,
        *,
        raw: bool = False,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        cases.append(
            {
                "case": name,
                "response": calls if raw else json.dumps(calls),
                "expected": expected,
                "category": category,
                "functions": copy.deepcopy(functions if functions is not None else tools[:1]),
                "answers": copy.deepcopy(answers if answers is not None else gold),
                "status": status,
                "error": error,
            }
        )

    add("simple_exact_dotted_name", [alpha], 1)
    add("simple_underscore_alias_not_allowed", [{**alpha, "name": "fixture_alpha"}], 0)
    add("simple_wrong_function", [beta], 0)
    add("simple_wrong_value", [{**alpha, "arguments": {"value": 4}}], 0)
    add("simple_string_not_integer", [{**alpha, "arguments": {"value": "2"}}], 0)
    add("simple_bool_not_integer", [{**alpha, "arguments": {"value": True}}], 0)
    add("simple_missing_required", [{**alpha, "arguments": {}}], 0)
    add("simple_unexpected_parameter", [{**alpha, "arguments": {"value": 2, "extra": 1}}], 0)
    add("simple_wrong_count", [alpha, alpha], 0)
    add("simple_empty_calls", [], 0)
    add("simple_empty_completed_text", "", 0, raw=True)
    add("simple_expression_not_JSON", "fixture.alpha(value=2)", 0, raw=True)
    add("simple_wrong_JSON_shape", {"fixture.alpha": {"value": 2}}, 0)
    add("simple_extra_call_field", [{**alpha, "extra": True}], 0)
    add(
        "native_float_accepts_integer",
        [alpha],
        1,
        functions=[tool("fixture.alpha", "float")],
        answers=[{"fixture.alpha": {"value": [2.0]}}],
    )
    add(
        "native_string_standardization",
        [{**alpha, "arguments": {"value": " RED-BRIDGE "}}],
        1,
        functions=[tool("fixture.alpha", "string")],
        answers=[{"fixture.alpha": {"value": ["red bridge"]}}],
    )
    array_tool = tool("fixture.alpha", "array")
    array_tool["parameters"]["properties"]["value"]["items"] = {"type": "integer"}
    for name, value, expected in (
        ("ordered", [1, 2], 1),
        ("wrong_order", [2, 1], 0),
        ("wrong_nested_type", [True, 2], 0),
    ):
        add(
            f"native_array_{name}",
            [{**alpha, "arguments": {"value": value}}],
            expected,
            functions=[array_tool],
            answers=[{"fixture.alpha": {"value": [[1, 2]]}}],
        )
    optional_tool = tool("fixture.alpha")
    optional_tool["parameters"]["properties"]["limit"] = {"type": "integer"}
    add(
        "native_optional_omission",
        [alpha],
        1,
        functions=[optional_tool],
        answers=[{"fixture.alpha": {"value": [2], "limit": ["", 5]}}],
    )
    add(
        "native_optional_not_allowed_omission",
        [alpha],
        0,
        functions=[optional_tool],
        answers=[{"fixture.alpha": {"value": [2], "limit": [5]}}],
    )
    add("multiple_selected_function", [beta], 1, "multiple", tools, [parallel_gold[1]])
    add("multiple_wrong_choice", [alpha], 0, "multiple", tools, [parallel_gold[1]])
    for category in ("parallel", "parallel_multiple"):
        functions = tools[:2] if category == "parallel" else tools
        add(f"{category}_permuted", [beta, alpha], 1, category, functions, parallel_gold)
        add(f"{category}_wrong_count", [alpha], 0, category, functions, parallel_gold)
        add(f"{category}_duplicate_call", [alpha, alpha], 0, category, functions, parallel_gold)
    for name, calls, raw, expected in (
        ("empty_calls", [], False, 1),
        ("refusal", "No appropriate function.", True, 1),
        ("invalid_JSON", "[not JSON", True, 1),
        ("decoded_call", [alpha], False, 0),
    ):
        add(f"irrelevance_{name}", calls, expected, "irrelevance", raw=raw)
    for status in ("failed", "cancelled"):
        add(f"{status}_is_error_not_zero", [alpha], 0, status=status, error="ScoringError")
    add("missing_text_is_error_not_zero", None, 0, raw=True, error="ScoringError")
    add("unsupported_category_is_error", [alpha], 0, "simple_java", error="UnsupportedScoring")
    add("broken_annotation_is_error_not_zero", [alpha], 0, answers=[{}], error="ScoringError")
    add("broken_annotation_empty_calls_is_error", [], 0, answers=[{}], error="ScoringError")
    add(
        "broken_annotation_invalid_JSON_is_error",
        "not JSON",
        0,
        answers=[{}],
        raw=True,
        error="ScoringError",
    )
    return cases


def mechanical_checks(checker: BFCLChecker) -> list[dict[str, Any]]:
    results = []
    for row in authored_fixtures():
        before = copy.deepcopy(row)
        actual_error: str | None
        actual: int | None
        try:
            score = score_bfcl(
                row["response"],
                row["status"],
                row["category"],
                row["functions"],
                row["answers"],
                checker,
            )
        except (ScoringError, UnsupportedScoring) as exc:
            actual_error = type(exc).__name__
            if actual_error != row["error"]:
                raise ScoringError(f"unexpected native fixture error: {row['case']}") from exc
            actual = None
        else:
            actual_error = None
            actual = score.value
            if row["error"] is not None or actual != row["expected"]:
                raise ScoringError(f"native fixture score mismatch: {row['case']}")
        if row != before:
            raise ScoringError(f"native scoring mutated fixture input: {row['case']}")
        results.append(
            {
                "case": row["case"],
                "expected": None if row["error"] else row["expected"],
                "actual": actual,
                "expected_error": row["error"],
                "actual_error": actual_error,
            }
        )
    return results


def check(source: Path, work_root: Path) -> dict[str, Any]:
    bindings = []
    fixtures = {}
    for model in TASK_MODELS:
        checker, metadata = load_bfcl_native(source, work_root, model)
        fixtures[model] = mechanical_checks(checker)
        bindings.append(metadata)
    return {
        "format": "promptwitness.native-BFCL-scorer-check/v1",
        "scope": "authored_mechanical_fixtures_only_no_dataset_or_model_responses",
        "evaluation_type": "mechanical_native_runtime_check_not_performance_evaluation",
        "native_bindings": bindings,
        "mechanical_checks": fixtures,
        "new_model_calls": 0,
        "new_allocated_GPU_hours": 0,
        "paid_API_USD": 0,
        "dataset_files_or_final_examples_read": False,
        "method_effect_or_pilot_measured": False,
        "BFCL_runtime_slice": "QUALIFIED_AUTHORED_FIXTURES_ONLY",
        "BFCL_real_GT_end_to_end": "PENDING_FIT_ONLY_AUDITOR_ACCESS",
        "all_scorers_gate": "PARTIAL_NOT_PASSED",
        "process_split_access_gate": "NOT_PASSED_BY_THIS_CPU_CHECK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="pinned gorilla source checkout")
    parser.add_argument("work_root", type=Path, help="task-owned scratch outside source")
    parser.add_argument("output", type=Path, help="new private JSON output; must not already exist")
    args = parser.parse_args()
    # Refuse accidental reuse before importing or executing any fixture.
    with args.output.open("x", encoding="utf-8") as stream:
        result = check(args.source, args.work_root)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "fixtures_per_model": {k: len(v) for k, v in result["mechanical_checks"].items()},
                "all_scorers_gate": result["all_scorers_gate"],
                "new_model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
