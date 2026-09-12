"""Hand-authored oracle cases, never outputs from or imports of reference evaluators."""

import copy
import json
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_schema_data import travel_cities, travel_stays
from promptwitness.procedure_scores import (
    PROCEDURE_METRIC_CAPABILITIES,
    PROCEDURE_SCORER_VERSION,
    ProcedureScoreLimitError,
    ProcedureScoreLimits,
    score_procedure,
)


def case(family, inputs, reference):
    return ProcedureCase(
        family, "hand-oracle", "2k", inputs, reference, {"kind": "authored", "label": "test"}
    )


def countdown(numbers=(2, 2, 3, 4), target=24, solution=None):
    solution = solution or ["2 + 2 = 4", "4 + 4 = 8", "8 * 3 = 24"]
    return case(
        "countdown",
        {
            "numbers": list(numbers),
            "target": target,
            "min_intermediate": 1,
            "max_intermediate": 2000,
        },
        {
            "solution": solution,
            "solution_text": "\n".join(solution),
            "demonstration": "Authored example, not a reference search trace.",
            "search_steps": 0,
            "num_search_tokens": 0,
        },
    )


def solution(*lines):
    return "<Solution>\n" + "\n".join(lines) + "\n</Solution>"


def edge(source, target, method="bus"):
    return {"source": source, "method": method, "target": target}


def path(edges=None, steps=None, source="A", target="D"):
    edges = edges if edges is not None else [edge("A", "B"), edge("B", "C"), edge("C", "D")]
    return case(
        "path_traversal",
        {
            "edges": edges,
            "source": source,
            "target": target,
            "problem_description": "Follow directed typed edges.",
            "context_nl": "Authored map.",
            "question_nl": "Find the route.",
        },
        {"steps": edges if steps is None else steps, "answer_nl": "Off-wire declared reference."},
    )


def route(steps):
    return (
        "<Route>\n"
        + "\n".join(
            f"From {row['source']}, take a {row['method']} to {row['target']}." for row in steps
        )
        + "\n</Route>"
    )


def table(gold="id\tvalue\na\t1\na\t1\nb\t2\n", header=("id", "value")):
    return case(
        "html_to_tsv",
        {
            "html": "<table></table><script>never executed</script>",
            "header": list(header),
            "task_topic": "Authored table",
            "task_description": "Extract rows",
            "filtering_instruction": "",
            "website_id": "authored",
        },
        {"tsv": gold},
    )


def tom(
    trace=("- Step 0: Ada believes the key is in B.", "- Step 1: Ada believes the key is in C."),
):
    return case(
        "tom_tracking",
        {"story_components": "Ada, key, boxes", "story": "A declared story.", "question": "Where?"},
        {"solution": "\n".join(trace) or "No bullet trace.", "answer": ["C"], "trace": list(trace)},
    )


def travel(*, constraints=None, links=None, order=("A", "B", "C"), days=(2, 2, 2), total=4):
    constraints = (
        constraints
        if constraints is not None
        else [{"type": "duration", "city": name, "num_days": 2} for name in ("A", "B", "C")]
    )
    links = (
        links
        if links is not None
        else [[a, b] for a in ("A", "B", "C") for b in ("A", "B", "C") if a != b]
    )
    names, durations = "**".join(order), "**".join(map(str, days))
    return case(
        "travel_planning",
        {
            "problem": "Visit each city for its duration.",
            "original_question": "Authored travel problem.",
            "num_cities": len(travel_cities(constraints)),
            "total_days": total,
            "constraints": constraints,
            "cities": travel_cities(constraints),
            "flights": links,
        },
        {
            "ground_truth_cities": names,
            "ground_truth_durations": durations,
            "ground_truth_plan": "Off-wire declared schedule.",
            "solving_procedure": "",
            "estimated_output_tokens": 0,
            "stays": travel_stays(names, durations),
        },
    )


def plan(order=("A", "B", "C"), days=(2, 2, 2)):
    lines = []
    start = 1
    for index, (city, duration) in enumerate(zip(order, days, strict=True)):
        end = start + duration - 1
        lines.append(f"**Day {start}-{end}:** Visit {city} for {duration} days.")
        if index < len(order) - 1:
            lines.append(f"**Day {end}:** Fly from {city} to {order[index + 1]}.")
        start = end
    return "<Plan>\n" + "\n".join(lines) + "\n</Plan>"


def code():
    return case(
        "pseudo_to_code",
        {"pseudocode_lines": ["Print a number"]},
        {"code_lines": ["int main() { return 0; }"], "testcases": [[["input"], ["42"]]]},
    )


def test_all_six_capabilities_and_reports_are_explicit():
    cases = [countdown(), path(), table(), tom(), travel(), code()]
    assert set(PROCEDURE_METRIC_CAPABILITIES) == {item.family for item in cases}
    for item in cases:
        report = score_procedure(item, "")
        assert report["scorer_version"] == PROCEDURE_SCORER_VERSION
        assert report["score_limits"] == ProcedureScoreLimits().to_dict()
        assert all(0 <= score <= 1 for score in report["metrics"].values())
        json.dumps(report, allow_nan=False)


def test_countdown_exact_arithmetic_consumes_duplicate_multiplicities_without_mutation():
    item = countdown()
    before = item.to_dict()
    report = score_procedure(item, solution(*item.reference["solution"]))
    assert report["primary_score"] == 1
    assert report["metrics"]["reference_valid"] == 1
    assert report["diagnostics"]["valid_steps"] == 3
    assert item.to_dict() == before
    # A different legitimate reduction tree is also valid.
    alternative = score_procedure(item, solution("2 * 2 = 4", "3 * 4 = 12", "12 + 4 = 16"))
    assert alternative["primary_score"] == 0
    assert alternative["diagnostics"]["failure"] == "target_not_reached"


@pytest.mark.parametrize(
    "lines,reason,good",
    [
        (["2 + 2 = 4", "4 + 4 = 8"], "incomplete_solution", 2),
        (["2 + 2 = 4", "4 + 4 = 9", "9 * 3 = 27"], "incorrect_equation", 1),
        (["2 + 3 = 5", "2 + 2 = 4", "5 * 4 = 20"], "operand_multiplicity", 1),
        (["3 / 2 = 1", "2 + 4 = 6", "6 + 1 = 7"], "non_integer_division", 0),
        (["2 - 2 = 0", "3 + 4 = 7", "7 + 0 = 7"], "intermediate_out_of_range", 0),
        (["2 + 2 = 4", "4 + 4 = 8", "8 * 3 = 24", "24 + 24 = 48"], "operand_multiplicity", 3),
    ],
)
def test_countdown_single_constraint_mutations(lines, reason, good):
    report = score_procedure(countdown(), solution(*lines))
    assert report["primary_score"] == 0
    assert report["diagnostics"]["failure"] == reason
    assert report["metrics"]["valid_step_fraction"] == good / 3


def test_countdown_inclusive_upper_boundary_and_distinct_valid_solution():
    item = countdown(
        (1000, 1000, 2, 2), 2000, ["1000 + 1000 = 2000", "2 / 2 = 1", "2000 * 1 = 2000"]
    )
    assert score_procedure(item, solution(*item.reference["solution"]))["primary_score"] == 1
    other = solution("1000 * 2 = 2000", "1000 * 2 = 2000", "2000 / 2000 = 1")
    report = score_procedure(countdown((1000, 1000, 2, 2), 1), other)
    assert report["primary_score"] == 1
    assert report["metrics"]["reference_valid"] == 0
    over = score_procedure(
        item, solution("1000 * 1000 = 1000000", "2 / 2 = 1", "1000000 * 1 = 1000000")
    )
    assert over["diagnostics"]["failure"] == "intermediate_out_of_range"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('x')",
        "2 ** 2 = 4",
        "2 // 2 = 1",
        "(2 + 2) = 4",
        "02 + 2 = 4",
        "2 + 2 = 4.0",
        "2 + 2 = True",
        "٢ + 2 = 4",
        "9" * 10000 + " + 2 = 4",
    ],
)
def test_countdown_never_evaluates_general_expressions(expression):
    report = score_procedure(countdown(), solution(expression, "4 + 4 = 8", "8 * 3 = 24"))
    assert report["primary_score"] == 0
    assert report["metrics"]["format_valid"] == 0


@pytest.mark.parametrize(
    "response",
    [
        "",
        "<Solution>",
        "</Solution><Solution>",
        "<Solution></Solution><Solution></Solution>",
        "<Solution>2 + 2 = 4</Solution></Solution>",
    ],
)
def test_countdown_does_not_accept_unclosed_or_ambiguous_solution_tags(response):
    assert score_procedure(countdown(), response)["primary_score"] == 0


def test_path_complete_typed_edges_and_correct_prefix_denominator():
    item = path()
    steps = list(item.reference["steps"])
    full = score_procedure(item, route(steps))
    assert full["primary_score"] == 1
    for count in range(3):
        report = score_procedure(item, route(steps[:count]))
        assert report["primary_score"] == 0
        assert report["metrics"]["reference_prefix_fraction"] == count / 3
    wrong = score_procedure(item, route([edge("A", "B", "train"), *steps[1:]]))
    assert wrong["metrics"]["reference_prefix_fraction"] == 0
    assert wrong["diagnostics"]["failure"] == "unknown_directed_edge"


def test_path_disconnected_cycles_and_destination_suffix_are_invalid():
    steps = list(path().reference["steps"])
    assert (
        score_procedure(path(), route([steps[0], steps[2]]))["diagnostics"]["failure"]
        == "disconnected_route"
    )
    extra = score_procedure(path(), route([*steps, edge("D", "E")]))
    assert extra["primary_score"] == 0
    assert extra["metrics"]["reference_prefix_fraction"] == 1
    assert extra["diagnostics"]["failure"] == "route_continues_after_destination"
    cyclic = path([edge("A", "B"), edge("B", "A")])
    assert (
        score_procedure(cyclic, route(cyclic.reference["steps"]))["diagnostics"]["failure"]
        == "repeated_city"
    )


def test_path_zero_edge_route_and_city_delimiter_names():
    assert (
        score_procedure(path([], [], source="A", target="A"), "<Route></Route>")["primary_score"]
        == 1
    )
    name = "A, take a train to B"
    item = path([edge(name, "C to D")], source=name, target="C to D")
    assert score_procedure(item, route(item.reference["steps"]))["primary_score"] == 1


def test_multiline_city_names_are_an_explicit_grammar_limit_not_accuracy_zero():
    item = path([edge("A\nB", "D")], source="A\nB")
    report = score_procedure(item, route(item.reference["steps"]))
    assert report["primary_score"] is None
    assert report["unsupported_metrics"] == {"route_valid": "multiline_city_names_not_supported"}
    item = travel(
        constraints=[{"type": "duration", "city": "A\rB", "num_days": 2}],
        order=("A\rB",),
        days=(2,),
        links=[],
        total=2,
    )
    assert score_procedure(item, plan(("A\rB",), (2,)))["primary_score"] is None


@pytest.mark.parametrize(
    "response",
    [
        "<Route>not an edge</Route>",
        "<Route>From A, take a teleport to B.</Route>",
        "<Route>From A, take a bus to B</Route>",
        "From A, take a bus to B.",
    ],
)
def test_path_closed_grammar(response):
    report = score_procedure(path(), response)
    assert report["primary_score"] == 0
    assert report["metrics"]["format_valid"] == 0


def test_tsv_multiset_hand_calculation_does_not_reward_duplicate_inflation():
    # Gold a,a,b; prediction a,b,b,c. Intersection has two rows, not three.
    report = score_procedure(table(), "id\tvalue\na\t1\nb\t2\nb\t2\nc\t3\n")
    assert report["metrics"]["row_precision"] == float(Fraction(2, 4))
    assert report["metrics"]["row_recall"] == float(Fraction(2, 3))
    assert report["primary_score"] == float(Fraction(4, 7))
    assert score_procedure(table(), "id\tvalue\nb\t2\na\t1\na\t1")["primary_score"] == 1


@pytest.mark.parametrize(
    "response",
    [
        "value\tid\n1\ta",
        "id\tvalue\na\t1\tEXTRA",
        "id\tvalue\na",
        "id\tvalue\na\t1\nTRAILER",
        'id\tvalue\n"unclosed\t1',
        "```tsv\nid\tvalue\na\t1",
    ],
)
def test_tsv_requires_exact_header_arity_and_closed_fences(response):
    assert score_procedure(table(), response)["primary_score"] == 0


def test_tsv_exact_cells_quoted_unicode_inert_formulas_and_empty_tables():
    text = 'id\tvalue\n"a\tb"\t"café\nline"\nformula\t=HYPERLINK(""https://invalid.example"")\n'
    item = table(text)
    report = score_procedure(item, "```tsv\n" + text.rstrip("\n") + "\n```")
    assert report["primary_score"] == 1
    assert report["processed"]["rows"][0] == ["a\tb", "café\nline"]
    assert score_procedure(table("id\tvalue\n"), "id\tvalue\n")["primary_score"] == 1
    assert score_procedure(table("id\tvalue\n"), "id\tvalue\na\t1")["primary_score"] == 0
    assert score_procedure(table(), "id\tvalue\n")["primary_score"] == 0
    assert score_procedure(table("id\tvalue\na\t1.0"), "id\tvalue\na\t10")["primary_score"] == 0


def test_invalid_tsv_reference_is_not_a_model_accuracy_zero():
    report = score_procedure(table("WRONG\tHEADER\na\t1"), "id\tvalue\na\t1")
    assert report["primary_score"] is None
    assert report["score_status"] == "unsupported"
    assert report["metrics"]["reference_valid"] == 0


def test_tsv_crlf_escaped_quotes_and_independence_from_global_csv_limits(monkeypatch):
    import csv

    def forbidden(*args, **kwargs):
        raise AssertionError("global CSV parser policy must not influence this scorer")

    monkeypatch.setattr(csv, "reader", forbidden)
    text = 'id\tvalue\r\na\t"one ""quote"""\r\n'
    report = score_procedure(table(text), "```tsv\r\n" + text + "```\r\n")
    assert report["primary_score"] == 1
    assert report["processed"]["rows"] == [["a", 'one "quote"']]
    assert score_procedure(table(), 'id\tvalue\na\t"closed"garbage')["primary_score"] == 0


def test_tsv_aggregate_field_budget_is_admitted_before_table_allocation(monkeypatch):
    from promptwitness import procedure_text_score

    original = procedure_text_score._read_tsv
    calls = []

    def guarded(text, work):
        calls.append(len(text))
        if len(calls) > 1:
            raise AssertionError("over-budget prediction reached row allocation")
        return original(text, work)

    monkeypatch.setattr(procedure_text_score, "_read_tsv", guarded)
    with pytest.raises(ProcedureScoreLimitError, match="work"):
        score_procedure(
            table(), "id\tvalue\n" + "\t" * 1000, limits=ProcedureScoreLimits(max_work_items=200)
        )
    assert len(calls) == 1


def test_tsv_multiline_decoded_cell_has_its_own_byte_admission():
    # Physical lines fit 16 bytes, but the single decoded quoted cell does not.
    text = 'id\tvalue\na\t"12345678\n12345678\n12345678"'
    with pytest.raises(ProcedureScoreLimitError, match="cell"):
        score_procedure(table(), text, limits=ProcedureScoreLimits(max_line_bytes=16))


def test_tom_only_declared_bullet_trace_with_strict_length_and_prefix():
    item = tom()
    text = "Narrative ignored.\n" + "\n".join(item.reference["trace"])
    assert score_procedure(item, text)["primary_score"] == 1
    assert (
        score_procedure(item, text.replace("Ada believes", "Ada   believes"))["primary_score"] == 1
    )
    assert (
        score_procedure(item, item.reference["trace"][0])["metrics"][
            "declared_trace_prefix_fraction"
        ]
        == 0.5
    )
    extra = score_procedure(item, text + "\n- Step 2: Guess.")
    assert extra["primary_score"] == 0
    assert extra["metrics"]["declared_trace_prefix_fraction"] == 1
    assert extra["unsupported_metrics"]["belief_entailment"] == "semantic_judge_not_implemented"
    assert score_procedure(tom(()), "- arbitrary")["primary_score"] is None


@pytest.mark.parametrize(
    "mutation",
    ["does not believe", "believes not", "believes the key is on", "BELIEVES", "believes?"],
)
def test_tom_does_not_delete_semantically_meaningful_tokens(mutation):
    text = "\n".join(tom().reference["trace"]).replace("believes", mutation)
    assert score_procedure(tom(), text)["primary_score"] == 0


def test_travel_valid_alternative_is_not_required_to_match_gold_order():
    report = score_procedure(travel(), plan(("C", "A", "B")))
    assert report["primary_score"] == 1
    assert report["metrics"]["reference_plan_exact_match"] == 0
    assert report["metrics"]["reference_prefix_fraction"] == 0
    assert report["metrics"]["reference_valid"] == 1


@pytest.mark.parametrize(
    "old,new,failed",
    [
        ("Day 1-2", "Day 2-2", "durations_valid"),
        ("Visit B for 2", "Visit B for 3", "durations_valid"),
        ("Day 2-3", "Day 3-4", "calendar_valid"),
        ("Fly from B to C", "Fly from C to B", "direct_flights_valid"),
        ("**Day 2:**", "**Day 1:**", "direct_flights_valid"),
        ("Visit C", "Visit B", "city_coverage_valid"),
        ("Day 3-4", "Day 3-5", "calendar_valid"),
    ],
)
def test_travel_single_constraint_mutations(old, new, failed):
    report = score_procedure(travel(), plan().replace(old, new))
    assert report["primary_score"] == 0
    assert report["metrics"][failed] == 0


def test_travel_fixed_dates_directed_flights_and_single_city():
    constraints = [
        {"type": "duration", "city": city, "num_days": 2} for city in ("A", "B", "C")
    ] + [{"type": "fixed", "city": "B", "start_day": 2, "end_day": 3}]
    item = travel(constraints=constraints, links=[["A", "B"], ["B", "C"]])
    assert score_procedure(item, plan())["primary_score"] == 1
    reverse = score_procedure(item, plan(("C", "B", "A")))
    assert reverse["metrics"]["direct_flights_valid"] == 0
    fixed = score_procedure(travel(constraints=constraints), plan(("B", "A", "C")))
    assert fixed["metrics"]["fixed_schedules_valid"] == 0
    single = travel(
        constraints=[{"type": "duration", "city": "A", "num_days": 3}],
        links=[],
        order=("A",),
        days=(3,),
        total=3,
    )
    assert score_procedure(single, plan(("A",), (3,)))["primary_score"] == 1


def test_travel_names_are_bound_as_whole_fields_and_extra_stays_never_succeed():
    names = ("A and visit B", "C to D", "E for F")
    constraints = [{"type": "duration", "city": city, "num_days": 2} for city in names]
    item = travel(
        constraints=constraints, links=[[names[0], names[1]], [names[1], names[2]]], order=names
    )
    text = plan(names).replace(
        f"Visit {names[0]} for", f"Arriving in {names[0]} and visit {names[0]} for", 1
    )
    assert score_procedure(item, text)["primary_score"] == 1
    extra = score_procedure(travel(), plan(("A", "B", "C", "A"), (2, 2, 2, 2)))
    assert extra["primary_score"] == 0
    assert extra["metrics"]["reference_prefix_fraction"] == 1


@pytest.mark.parametrize(
    "response",
    [
        "<Plan></Plan>",
        "<Plan>nonsense</Plan>",
        "<Plan>**Day 1-2:** Visit A for 2 days.\n**Day 2:** Fly from A to B.</Plan>",
        "<Plan>**Day 1-2:** Arriving in A and visit B for 2 days.</Plan>",
    ],
)
def test_travel_format_errors_are_controlled(response):
    report = score_procedure(travel(), response)
    assert report["primary_score"] == 0
    assert report["metrics"]["format_valid"] == 0


def test_pseudo_code_is_not_executed_or_given_an_accuracy_surrogate(monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("no process invocation is authorized")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    report = score_procedure(
        code(), '```cpp\n#include <cstdlib>\nint main(){system("whoami");}\n```'
    )
    assert report["primary_metric"] is None
    assert report["primary_score"] is None
    assert report["score_status"] == "unsupported"
    assert report["diagnostics"]["execution_attempted"] is False
    assert report["metrics"] == {"code_fence_present": 1.0}


def test_score_limits_are_strict_and_frozen():
    limits = ProcedureScoreLimits()
    with pytest.raises(FrozenInstanceError):
        limits.max_lines = 1
    for field in limits.to_dict():
        for invalid in (True, 0, -1, 1.5, 10**100):
            with pytest.raises(ValueError):
                ProcedureScoreLimits(**{field: invalid})


def test_total_utf8_prediction_and_source_limits_precede_parsing():
    item = code()
    assert (
        score_procedure(item, "é", limits=ProcedureScoreLimits(max_prediction_bytes=2))[
            "score_status"
        ]
        == "unsupported"
    )
    with pytest.raises(ProcedureScoreLimitError):
        score_procedure(item, "é", limits=ProcedureScoreLimits(max_prediction_bytes=1))
    with pytest.raises(ProcedureScoreLimitError):
        score_procedure(item, "", limits=ProcedureScoreLimits(max_case_bytes=1))
    with pytest.raises(ValueError, match="Unicode"):
        score_procedure(item, "\ud800")


def test_aggregate_work_budget_exact_and_one_less():
    item = table()
    response = "id\tvalue\na\t1"
    baseline = score_procedure(item, response)
    count = baseline["work"]["items"]
    assert (
        score_procedure(item, response, limits=ProcedureScoreLimits(max_work_items=count))[
            "primary_score"
        ]
        == baseline["primary_score"]
    )
    with pytest.raises(ProcedureScoreLimitError):
        score_procedure(item, response, limits=ProcedureScoreLimits(max_work_items=count - 1))
    with pytest.raises(ProcedureScoreLimitError):
        score_procedure(item, response, limits=ProcedureScoreLimits(max_lines=2))
    with pytest.raises(ProcedureScoreLimitError):
        score_procedure(item, "x" * 65, limits=ProcedureScoreLimits(max_line_bytes=64))


def test_wrong_public_types_and_no_input_mutation():
    item = table()
    raw = copy.deepcopy(item.to_dict())
    for invalid in (None, {}, raw):
        with pytest.raises(ValueError, match="ProcedureCase"):
            score_procedure(invalid, "")
    with pytest.raises(ValueError, match="text"):
        score_procedure(item, {})
    with pytest.raises(ValueError, match="limits"):
        score_procedure(item, "", limits={})
    score_procedure(item, "id\tvalue")
    assert item.to_dict() == raw
