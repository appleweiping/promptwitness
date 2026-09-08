from __future__ import annotations

import json

import pytest

from promptwitness import (
    BenchmarkCase,
    evaluate_benchmark,
    load_benchmark_cases,
    score_prediction,
)
from promptwitness.cli import main


def test_benchmark_loader_and_report_are_task_aware_without_gold_in_digest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"a","task":"recall","prompt":"Who?","expected":"Ada","metadata":{"length":8}}\n'
        '{"case_id":"b","task":"summ","prompt":"Summarize","expected":"short"}\n',
        encoding="utf-8",
    )
    cases = load_benchmark_cases(path)
    assert (
        cases[0].prompt_digest
        == BenchmarkCase("a", "recall", "Who?", "Other", {"length": 8}).prompt_digest
    )
    report = evaluate_benchmark(cases, lambda case: "Ada" if case.case_id == "a" else "wrong")
    assert report.accuracy == 0.5
    assert report.by_task()["recall"]["accuracy"] == 1.0


def test_benchmark_cli_filters_tasks_and_retains_failures(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = tmp_path / "cases.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "report.json"
    cases.write_text(
        json.dumps(
            [
                {"case_id": "a", "task": "recall", "prompt": "p", "expected": "yes"},
                {"case_id": "b", "task": "summ", "prompt": "q", "expected": "short"},
            ]
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"a":"yes","b":"wrong"}', encoding="utf-8")
    assert (
        main(
            ["benchmark", str(cases), str(predictions), "--task", "recall", "--output", str(output)]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cases"] == 1 and payload["accuracy"] == 1.0
    assert main(["benchmark", str(cases), str(predictions), "--task", "missing"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("case_id", ""), ("task", " "), ("prompt", "")],
)
def test_benchmark_case_requires_non_empty_text(field, value) -> None:  # type: ignore[no-untyped-def]
    values = {"case_id": "id", "task": "task", "prompt": "prompt", "expected": "answer"}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        BenchmarkCase(**values)


def test_benchmark_case_validates_expected_and_metadata() -> None:
    with pytest.raises(TypeError, match="expected"):
        BenchmarkCase("id", "task", "prompt", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata"):
        BenchmarkCase("id", "task", "prompt", "answer", [])  # type: ignore[arg-type]
    case = BenchmarkCase("id", "task", "prompt", "answer", {"source": "fixture"})
    with pytest.raises(TypeError):
        case.metadata["new"] = "value"  # type: ignore[index]


def test_benchmark_report_tracks_failures_and_task_aggregates() -> None:
    cases = (
        BenchmarkCase("ok", "recall", "p", "yes"),
        BenchmarkCase("wrong", "recall", "q", "yes"),
        BenchmarkCase("boom", "summ", "r", "done"),
    )

    def answer(case: BenchmarkCase) -> str:
        if case.case_id == "boom":
            raise RuntimeError("provider unavailable")
        return "yes" if case.case_id == "ok" else "no"

    report = evaluate_benchmark(cases, answer)
    assert report.attempted == 2
    assert report.failed == 1
    assert report.accuracy == 0.5
    assert report.by_task()["summ"] == {
        "cases": 1,
        "attempted": 0,
        "failed": 1,
        "accuracy": None,
        "mean_score": None,
    }
    payload = report.to_dict()
    assert payload["results"][2]["error"] == "RuntimeError: provider unavailable"


def test_benchmark_evaluator_validates_inputs_and_strict_errors() -> None:
    case = BenchmarkCase("id", "task", "prompt", "answer")
    with pytest.raises(TypeError, match="callable"):
        evaluate_benchmark((case,), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strict"):
        evaluate_benchmark((case,), lambda _: "answer", strict=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        evaluate_benchmark((), lambda _: "answer")
    with pytest.raises(ValueError, match="failed"):
        evaluate_benchmark((case,), lambda _: 1, strict=True)  # type: ignore[return-value]


def test_benchmark_scoring_registry_preserves_continuous_scores() -> None:
    cases = (
        BenchmarkCase("exact", "facts", "p", "Ada", scorer="exact"),
        BenchmarkCase("contains", "facts", "p", "Ada", scorer="contains"),
        BenchmarkCase("tokens", "facts", "p", "red blue", scorer="token_f1", threshold=0.5),
        BenchmarkCase("json", "facts", "p", '{"a": 1}', scorer="json"),
    )
    assert score_prediction(cases[0], " ada ") == 1.0
    assert score_prediction(cases[1], "Ada Lovelace") == 1.0
    assert score_prediction(cases[2], "red green") == pytest.approx(0.5)
    assert score_prediction(cases[3], '{"a":1}') == 1.0
    report = evaluate_benchmark(
        cases, lambda case: {"tokens": "red green"}.get(case.case_id, case.expected)
    )
    assert report.accuracy == 1.0
    assert report.mean_score == pytest.approx(0.875)
    assert report.to_dict()["results"][2]["score"] == pytest.approx(0.5)


def test_benchmark_loader_round_trips_scorer_and_rejects_invalid_scoring(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scored.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "a",
                    "task": "facts",
                    "prompt": "p",
                    "expected": "red blue",
                    "scorer": "token_f1",
                    "threshold": 0.75,
                }
            ]
        ),
        encoding="utf-8",
    )
    case = load_benchmark_cases(path)[0]
    assert case.scorer == "token_f1" and case.threshold == 0.75
    assert case.prompt_digest != BenchmarkCase("a", "facts", "p", "red blue").prompt_digest
    with pytest.raises(ValueError, match="unknown benchmark scorer"):
        BenchmarkCase("id", "task", "p", "e", scorer="bleu")
    with pytest.raises(ValueError, match="between zero and one"):
        BenchmarkCase("id", "task", "p", "e", threshold=1.1)
    with pytest.raises(TypeError, match="real number"):
        BenchmarkCase("id", "task", "p", "e", threshold="1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BenchmarkCase"):
        score_prediction(object(), "answer")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prediction"):
        score_prediction(BenchmarkCase("id", "task", "p", "e"), 1)  # type: ignore[arg-type]


def test_benchmark_scoring_edge_cases_are_explicit() -> None:
    assert score_prediction(BenchmarkCase("empty", "t", "p", ""), "anything") == 0.0
    assert score_prediction(BenchmarkCase("both-empty", "t", "p", "", scorer="token_f1"), "") == 1.0
    assert (
        score_prediction(BenchmarkCase("one-empty", "t", "p", "word", scorer="token_f1"), "") == 0.0
    )
    assert (
        score_prediction(BenchmarkCase("no-overlap", "t", "p", "word", scorer="token_f1"), "other")
        == 0.0
    )
    assert score_prediction(BenchmarkCase("bad-json", "t", "p", "{", scorer="json"), "{") == 0.0


@pytest.mark.parametrize(
    ("text", "suffix", "message"),
    [
        ("{}", ".json", "array"),
        ("[]", ".json", "at least one"),
        ('[{"case_id":"a","task":"t","prompt":"p","expected":"e","extra":1}]', ".json", "unknown"),
        ("[1]", ".json", "object"),
        ('[{"case_id":"a","task":"t","prompt":"p"}]', ".json", "invalid"),
        (
            '[{"case_id":"a","task":"t","prompt":"p","expected":"e"},{"case_id":"a","task":"t","prompt":"q","expected":"e"}]',
            ".json",
            "unique",
        ),
    ],
)
def test_benchmark_loader_rejects_invalid_inputs(tmp_path, text, suffix, message) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / f"cases{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_benchmark_cases(path)


def test_benchmark_loader_wraps_io_and_json_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="cannot load"):
        load_benchmark_cases(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        load_benchmark_cases(malformed)
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        load_benchmark_cases(jsonl)


def test_benchmark_cli_rejects_bad_predictions_and_strict_failures(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"case_id":"a","task":"recall","prompt":"p","expected":"yes"}]',
        encoding="utf-8",
    )
    bad_predictions = tmp_path / "bad.json"
    bad_predictions.write_text("[]", encoding="utf-8")
    assert main(["benchmark", str(cases), str(bad_predictions)]) == 1
    predictions = tmp_path / "predictions.json"
    predictions.write_text('{"a":"no"}', encoding="utf-8")
    assert main(["benchmark", str(cases), str(predictions), "--strict"]) == 0
