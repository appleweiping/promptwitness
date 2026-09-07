from __future__ import annotations

import json

import pytest

from promptwitness import (
    BenchmarkCase,
    BenchmarkSuite,
    BenchmarkTask,
    load_benchmark_suite,
)
from promptwitness.cli import main


def _suite_payload() -> dict[str, object]:
    return {
        "format": "promptwitness.benchmark-suite.v1",
        "suite_id": "release-gate",
        "metadata": {"version": 1},
        "tasks": [
            {
                "name": "recall",
                "metadata": {"domain": "facts"},
                "cases": [
                    {
                        "case_id": "recall-1",
                        "task": "recall",
                        "prompt": "Who?",
                        "expected": "Ada",
                    }
                ],
            },
            {
                "name": "summarize",
                "cases": [
                    {
                        "case_id": "summ-1",
                        "task": "summarize",
                        "prompt": "Shorten this.",
                        "expected": "brief",
                        "metadata": {"length": "short"},
                    }
                ],
            },
        ],
    }


def test_suite_loader_and_digest_exclude_gold_answers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first = tmp_path / "first.json"
    first.write_text(json.dumps(_suite_payload()), encoding="utf-8")
    suite = load_benchmark_suite(first)
    assert isinstance(suite, BenchmarkSuite)
    assert [task.name for task in suite.tasks] == ["recall", "summarize"]
    assert [case.case_id for case in suite.cases] == ["recall-1", "summ-1"]

    changed = _suite_payload()
    changed["tasks"][0]["cases"][0]["expected"] = "Grace"  # type: ignore[index]
    second = tmp_path / "second.json"
    second.write_text(json.dumps(changed), encoding="utf-8")
    assert load_benchmark_suite(second).digest == suite.digest
    assert suite.to_dict()["format"] == "promptwitness.benchmark-suite.v1"


def test_suite_models_enforce_task_and_global_case_invariants() -> None:
    case = BenchmarkCase("id", "task", "prompt", "answer")
    other_case = BenchmarkCase("id", "other", "prompt", "answer")
    with pytest.raises(ValueError, match="match"):
        BenchmarkTask("other", (case,))
    with pytest.raises(ValueError, match="unique"):
        BenchmarkTask("task", (case, case))
    with pytest.raises(ValueError, match="unique"):
        BenchmarkSuite(
            "suite",
            (BenchmarkTask("task", (case,)), BenchmarkTask("other", (other_case,))),
        )
    with pytest.raises(ValueError, match="at least one"):
        BenchmarkSuite("suite", ())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root"),
        ({"suite_id": "x", "tasks": []}, "format"),
        ({"format": "wrong", "suite_id": "x", "tasks": []}, "unsupported"),
        (
            {
                "format": "promptwitness.benchmark-suite.v1",
                "suite_id": "x",
                "tasks": [{"name": "t", "cases": []}],
            },
            "at least one",
        ),
        (
            {
                "format": "promptwitness.benchmark-suite.v1",
                "suite_id": "x",
                "tasks": [{"name": "t", "cases": [{"case_id": "a"}]}],
            },
            "invalid",
        ),
    ],
)
def test_suite_loader_rejects_malformed_payloads(tmp_path, payload, message) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_benchmark_suite(path)


def test_suite_loader_rejects_unknown_fields_and_io(tmp_path) -> None:  # type: ignore[no-untyped-def]
    unknown = _suite_payload()
    unknown["extra"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_benchmark_suite(path)
    with pytest.raises(ValueError, match="cannot load"):
        load_benchmark_suite(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ("not-an-object", "object"),
        ({"name": "t", "cases": "not-an-array"}, "requires"),
        ({"name": "t", "cases": ["not-an-object"]}, "must be an object"),
        ({"name": "t", "cases": [], "extra": True}, "unknown"),
    ],
)
def test_suite_loader_rejects_invalid_task_shapes(tmp_path, task, message) -> None:  # type: ignore[no-untyped-def]
    payload = _suite_payload()
    payload["tasks"] = [task]
    path = tmp_path / "invalid-task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_benchmark_suite(path)


def test_benchmark_suite_cli_filters_tasks_and_reports_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(_suite_payload()), encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text('{"recall-1":"Ada","summ-1":"wrong"}', encoding="utf-8")
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "benchmark-suite",
                str(suite_path),
                str(predictions),
                "--task",
                "recall",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["suite_id"] == "release-gate"
    assert report["cases"] == 1
    assert report["selected_tasks"] == ["recall"]
    assert main(["benchmark-suite", str(suite_path), str(predictions), "--task", "missing"]) == 1
