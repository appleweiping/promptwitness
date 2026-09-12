"""Independent arithmetic/graph oracles; no model or reference evaluator execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from promptwitness.procedure_baselines import (
    ProcedureBaselineLimits,
    solve_countdown,
    walk_graph_path,
)
from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_scores import score_procedure


def countdown(numbers: list[int] | None = None, target: int = 10) -> dict[str, Any]:
    return {
        "numbers": [1, 2, 3, 4] if numbers is None else numbers,
        "target": target,
        "min_intermediate": 1,
        "max_intermediate": 2000,
    }


def graph(
    edges: list[tuple[str, str, str]], source: str = "A", target: str = "C"
) -> dict[str, Any]:
    return {
        "edges": [dict(source=a, method=m, target=b) for a, m, b in edges],
        "source": source,
        "target": target,
        "problem_description": "An authored directed graph.",
        "context_nl": "Do not use this decoy text: A can go directly to C.",
        "question_nl": "Use only the typed graph.",
    }


def independently_check_equations(numbers: list[int], target: int, steps: tuple[str, ...]) -> None:
    bag = Counter(numbers)
    assert len(steps) == len(numbers) - 1
    for step in steps:
        left, operator, right, equals, result = step.split()
        assert equals == "="
        a, b, value = int(left), int(right), int(result)
        for operand in (a, b):
            assert bag[operand] > 0
            bag[operand] -= 1
        if operator == "+":
            calculated = Fraction(a) + b
        elif operator == "-":
            calculated = Fraction(a) - b
        elif operator == "*":
            calculated = Fraction(a) * b
        else:
            assert operator == "/"
            calculated = Fraction(a, b)
        assert calculated == value and 1 <= value <= 2000
        bag[value] += 1
    assert +bag == Counter({target: 1})


def subset_oracle(numbers: list[int]) -> set[int]:
    """Bottom-up labeled-subset DP, independently of DFS state/memo structure."""
    reach: dict[int, set[int]] = {1 << i: {value} for i, value in enumerate(numbers)}
    for mask in range(1, 1 << len(numbers)):
        values = reach.setdefault(mask, set())
        for left in range(1, mask):
            if left & mask != left:
                continue
            right = mask ^ left
            for a in reach[left]:
                for b in reach[right]:
                    outputs = (Fraction(a + b), Fraction(a - b), Fraction(a * b), Fraction(a, b))
                    values.update(int(v) for v in outputs if v.denominator == 1 and 1 <= v <= 2000)
    return reach[(1 << len(numbers)) - 1]


@pytest.mark.parametrize("numbers", [[1, 2, 3, 4], [2, 2, 2, 2], [1, 1, 1, 1]])
def test_search_matches_independent_subset_dp(numbers: list[int]) -> None:
    available = subset_oracle(numbers)
    for target in range(1, 26):
        result = solve_countdown(countdown(numbers, target))
        assert (result.status == "solved") == (target in available)
        if result.status == "solved":
            independently_check_equations(numbers, target, result.steps)
        else:
            assert result.status == "unsatisfiable"
            assert result.reason == "complete_search_no_solution"


def test_exact_trace_and_no_input_mutation() -> None:
    data = countdown()
    before = repr(data)
    result = solve_countdown(data)
    assert result.steps == ("1 + 2 = 3", "3 + 3 = 6", "4 + 6 = 10")
    assert result.prediction == "<Solution>\n1 + 2 = 3\n3 + 3 = 6\n4 + 6 = 10\n</Solution>"
    assert result.states == 4 and result.transitions == 3
    assert repr(data) == before
    assert solve_countdown(data) == result
    with pytest.raises(FrozenInstanceError):
        result.status = "wrong"  # type: ignore[misc]


def test_upper_bound_and_full_consumption() -> None:
    result = solve_countdown(countdown([2000, 2, 1, 1], 2000))
    assert result.status == "solved"
    independently_check_equations([2000, 2, 1, 1], 2000, result.steps)
    # Merely finding a target operand is insufficient; all four occurrences go in.
    result = solve_countdown(countdown([1, 1, 1, 1], 1))
    assert len(result.steps) == 3


@pytest.mark.parametrize("field", ["max_states", "max_transitions"])
def test_budget_stop_is_not_unsatisfiable(field: str) -> None:
    limits = ProcedureBaselineLimits(**{field: 1})
    result = solve_countdown(countdown(), limits=limits)
    assert result.status == "exhausted" and result.prediction is None and result.steps == ()
    assert (
        result.reason
        == {"max_states": "state_budget", "max_transitions": "transition_budget"}[field]
    )
    assert getattr(result, field.removeprefix("max_")) == 1


def test_exact_prediction_byte_budget() -> None:
    baseline = solve_countdown(countdown())
    assert baseline.prediction is not None
    size = len(baseline.prediction.encode())
    assert (
        solve_countdown(
            countdown(), limits=ProcedureBaselineLimits(max_prediction_bytes=size)
        ).status
        == "solved"
    )
    small = solve_countdown(
        countdown(), limits=ProcedureBaselineLimits(max_prediction_bytes=size - 1)
    )
    assert small.status == "exhausted" and small.reason == "prediction_budget"


@pytest.mark.parametrize(
    "field",
    ["max_states", "max_transitions", "max_input_bytes", "max_edges", "max_prediction_bytes"],
)
@pytest.mark.parametrize("value", [0, True, 1.5, 10**100])
def test_strict_limits(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        ProcedureBaselineLimits(**{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"target": True},
        {"numbers": [1, 2, 3]},
        {"numbers": [0, 1, 2, 3]},
        {"numbers": [1, 2, 3, 2001]},
        {"min_intermediate": True},
        {"max_intermediate": 2001},
        {"reference": "must not be admitted"},
    ],
)
def test_closed_countdown_input(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        solve_countdown({**countdown(), **change})


def test_input_byte_budget() -> None:
    with pytest.raises(ValueError, match="byte budget"):
        solve_countdown(countdown(), limits=ProcedureBaselineLimits(max_input_bytes=1))


def test_gold_changes_do_not_change_solver_or_result_identity() -> None:
    reference = dict(
        solution=["1 + 2 = 3"],
        solution_text="A",
        demonstration="B",
        search_steps=0,
        num_search_tokens=0,
    )
    first = ProcedureCase(
        "countdown", "one", "0.5k", countdown(), reference, {"kind": "authored", "label": "one"}
    )
    second = ProcedureCase(
        "countdown",
        "two",
        "8k",
        countdown(),
        {**reference, "solution": ["99 * 99 = 9801"]},
        {"kind": "authored", "label": "different"},
    )
    assert first.digest != second.digest
    actual = solve_countdown(first.input)
    assert actual == solve_countdown(second.input)
    assert actual.prediction is not None
    assert score_procedure(first, actual.prediction)["metrics"]["full_solution_valid"] == 1
    assert score_procedure(second, actual.prediction)["metrics"]["reference_valid"] == 0
    with pytest.raises(ValueError):
        solve_countdown(first)  # type: ignore[arg-type]


def test_graph_stops_at_destination_and_preserves_methods() -> None:
    data = graph([("C", "plane", "D"), ("A", "train", "B"), ("B", "ferry", "C")])
    result = walk_graph_path(data)
    assert result.steps == ("From A, take a train to B.", "From B, take a ferry to C.")
    assert result.states == 3 and result.transitions == 2
    assert "D." not in (result.prediction or "")
    reordered = {**data, "edges": list(reversed(data["edges"]))}
    assert walk_graph_path(reordered).steps == result.steps


@pytest.mark.parametrize(
    "edges,reason",
    [
        ([("B", "bus", "A")], "directed_dead_end"),
        ([("A", "bus", "B"), ("B", "train", "A")], "directed_cycle_before_target"),
        ([("A", "plane", "A")], "directed_cycle_before_target"),
    ],
)
def test_graph_proof_of_no_route(edges: list[tuple[str, str, str]], reason: str) -> None:
    result = walk_graph_path(graph(edges))
    assert result.status == "unsatisfiable" and result.reason == reason
    assert result.prediction is None and not result.steps


def test_zero_step_graph_route_and_empty_edge_table() -> None:
    result = walk_graph_path(graph([], source="A", target="A"))
    assert result.status == "solved" and result.steps == () and result.states == 1
    assert result.prediction == "<Route>\n\n</Route>"


def test_graph_budgets_and_unicode() -> None:
    data = graph([("😀", "ferry", "e\u0301")], source="😀", target="e\u0301")
    result = walk_graph_path(data)
    assert result.prediction is not None
    size = len(result.prediction.encode())
    assert (
        walk_graph_path(data, limits=ProcedureBaselineLimits(max_prediction_bytes=size)).status
        == "solved"
    )
    assert (
        walk_graph_path(data, limits=ProcedureBaselineLimits(max_prediction_bytes=size - 1)).reason
        == "prediction_budget"
    )
    assert walk_graph_path(data, limits=ProcedureBaselineLimits(max_states=1)).status == "exhausted"
    too_many = graph([("A", "bus", "B"), ("B", "bus", "C")])
    with pytest.raises(ValueError):
        walk_graph_path(too_many, limits=ProcedureBaselineLimits(max_edges=1))


@pytest.mark.parametrize("city", ["A\nB", "A\u2028B", "A</Route>B"])
def test_unrepresentable_route_is_explicit(city: str) -> None:
    assert walk_graph_path(graph([(city, "bus", "C")], source=city)).status == "unsupported"


@pytest.mark.parametrize("edges", [[("A", "car", "C")], [("A", "bus", "B"), ("A", "plane", "C")]])
def test_graph_rejects_unknown_methods_or_branching(edges: list[tuple[str, str, str]]) -> None:
    with pytest.raises(ValueError):
        walk_graph_path(graph(edges))


@pytest.fixture
def benchmark() -> Any:
    path = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_procedures.py"
    spec = spec_from_file_location("authored_procedure_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authored_case(inputs: dict[str, Any], family: str = "countdown") -> ProcedureCase:
    if family == "countdown":
        reference = dict(
            solution=["1 + 2 = 3"],
            solution_text="incomplete gold",
            demonstration="authored",
            search_steps=0,
            num_search_tokens=0,
        )
    else:
        reference = {"steps": [], "answer_nl": ""}
    return ProcedureCase(
        family, "authored", "0.5k", inputs, reference, {"kind": "authored", "label": "test"}
    )


def test_benchmark_all_case_denominator_and_bad_gold(benchmark: Any) -> None:
    cases = (authored_case(countdown()), authored_case(countdown([1, 1, 1, 1], 20)))
    report = benchmark.evaluate_cases("countdown", cases, ProcedureBaselineLimits())
    assert report["selected"] == 2 and report["scored"] == 2
    assert report["statuses"] == {"solved": 1, "unsatisfiable": 1}
    assert report["valid"] == 1 and report["valid_fraction_all_selected"] == 0.5
    assert report["reference_invalid"] == 2 and report["solved_but_score_invalid"] == 0
    for control in report["controls"].values():
        assert control == dict(attempted=2, scored=2, accepted=0, failed=0, unsupported=0)
    bounded = benchmark.evaluate_cases("countdown", cases, ProcedureBaselineLimits(max_states=1))
    assert bounded["statuses"] == {"exhausted": 2}
    assert bounded["valid_fraction_all_selected"] == 0.0


def test_benchmark_directed_path_controls(benchmark: Any) -> None:
    data = graph([("A", "train", "B"), ("B", "ferry", "C")])
    report = benchmark.evaluate_cases(
        "path_traversal", (authored_case(data, "path_traversal"),), ProcedureBaselineLimits()
    )
    assert report["valid"] == 1 and report["reference_invalid"] == 1
    assert all(row["accepted"] == 0 and row["scored"] == 1 for row in report["controls"].values())


def test_benchmark_keeps_unavailable_metric_separate(benchmark: Any) -> None:
    data = graph([("A\nB", "train", "C")], source="A\nB")
    report = benchmark.evaluate_cases(
        "path_traversal", (authored_case(data, "path_traversal"),), ProcedureBaselineLimits()
    )
    assert report["selected"] == 1 and report["scored"] == report["scoring_failed"] == 0
    assert report["scoring_unsupported"] == 1 and report["reference_diagnostic_unknown"] == 1
    assert report["valid_fraction_scored"] is None
    assert all(row["unsupported"] == 1 for row in report["controls"].values())


def test_benchmark_failed_solver_retained_and_redacted(
    benchmark: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import promptwitness.procedure_baselines as baselines

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("PRIVATE_INPUT_NEVER_PRINT")

    monkeypatch.setattr(baselines, "solve_countdown", failed)
    report = benchmark.evaluate_cases(
        "countdown", (authored_case(countdown()),), ProcedureBaselineLimits()
    )
    assert report["statuses"] == {"failed": 1} and report["selected"] == 1
    assert report["error_types"] == {"ValueError": 1}
    assert "PRIVATE_INPUT" not in str(report)


def test_benchmark_publication_exclusive_and_stdout_failure(
    benchmark: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import io
    import sys

    target = tmp_path / "report.json"
    benchmark.publish({"completed": True}, target)
    initial = target.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.publish({"completed": False}, target)
    assert target.read_bytes() == initial
    data = tmp_path / "data"
    data.mkdir()
    second = tmp_path / "second.json"
    monkeypatch.setattr(benchmark, "run", lambda _: {"completed": True, "model_calls": 0})
    monkeypatch.setattr(sys, "argv", ["benchmark", str(data), "--output", str(second)])
    closed = io.StringIO()
    closed.close()
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", closed)
        assert benchmark.main() == 1
    assert b'"completed": true' in second.read_bytes()


def test_preregistered_inventory_no_sampling(benchmark: Any) -> None:
    assert len(benchmark.DATASETS) == 6 and len(benchmark.DATA_SHA256) == 8
    assert ProcedureBaselineLimits().to_dict() == benchmark.LIMITS
    assert all(len(value) == 64 for value in benchmark.DATA_SHA256.values())
