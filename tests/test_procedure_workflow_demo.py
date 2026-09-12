"""Hand-counted lifecycle checks for the independently authored six-case demo."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from promptwitness.procedure_plan import load_procedure_suite
from promptwitness.task_runs import TaskRunStore


@pytest.fixture
def demo() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts/procedure_workflow_demo.py"
    spec = spec_from_file_location("original_procedure_demo", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_independent_cases_and_exact_generation_inventory(demo: Any) -> None:
    plan = load_procedure_suite(demo.EXAMPLES / "suite.json")
    assert len(plan.payload["tasks"]) == 6
    assert len(plan.payload["items"]) == 12
    assert list(plan.payload["budgets"]) == [1, 8192]
    assert {task["family"]: task["generation"]["max_tokens"] for task in plan.payload["tasks"]} == {
        "countdown": 128,
        "path_traversal": 160,
        "html_to_tsv": 192,
        "tom_tracking": 224,
        "travel_planning": 256,
        "pseudo_to_code": 288,
    }
    for item in plan.payload["items"]:
        assert item["record"]["source"]["kind"] == "authored"
        assert item["skip_reason"] == ("input_budget_exceeded" if item["budget"] == 1 else None)
        assert 1 < item["input_units"] <= 8192
        user_input = json.loads(item["messages"][1]["content"])
        assert (
            not {"reference", "source", "case_id"} & user_input.keys()
            or item["family"] == "path_traversal"
        )
        assert "reference" not in user_input and "case_id" not in user_input


def test_hand_counted_durable_lifecycle(demo: Any, tmp_path: Path) -> None:
    output = tmp_path / "new-run"
    report = demo.run_demo(output)
    assert report["passed"] is True and report["synthetic"] is True
    assert report["model_calls"] == report["network_calls"] == report["subprocesses"] == 0
    assert report["scripted_provider_calls"] == 7
    assert report["reservation_attempts"] == 8
    assert report["journal_events"] == 17
    assert report["stale_response_rejected"] is True
    assert report["successful_rerun_additional_calls"] == 0
    assert report["coverage"]["planned"] == 12
    assert report["coverage"]["skipped"] == report["coverage"]["succeeded"] == 6
    assert report["coverage"]["scored"] == 5
    assert report["coverage"]["unsupported_primary"] == 1
    assert report["coverage"]["score_coverage"] == 5 / 12
    assert report["six_family_execution_complete"] is False
    assert report["six_family_scoring_complete"] is False
    assert [stage["calls"] for stage in report["trace"]] == [0, 5, 7, 7]
    assert report["trace"][0]["statuses"] == {"skipped": 6, "running": 1, "ready": 5}
    assert report["trace"][1]["statuses"] == {
        "skipped": 6,
        "running": 1,
        "failed": 1,
        "succeeded": 4,
    }
    assert report["family_results"]["countdown"]["revision"] == 4
    assert report["family_results"]["path_traversal"]["revision"] == 5
    assert all(
        report["family_results"][family]["revision"] == 2
        for family in ("html_to_tsv", "tom_tracking", "travel_planning", "pseudo_to_code")
    )
    assert report["family_results"]["pseudo_to_code"]["primary_score"] is None
    assert report["family_results"]["pseudo_to_code"]["score_status"] == "unsupported"
    with sqlite3.connect(output / demo.DATABASE) as connection:
        assert connection.execute("SELECT count(*) FROM task_event").fetchone()[0] == 17
        assert (
            connection.execute(
                "SELECT count(*) FROM task_event WHERE kind='retry_authorized'"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT count(*) FROM task_event WHERE kind='failed'").fetchone()[0]
            == 1
        )
    with TaskRunStore(output / demo.DATABASE) as store:
        assert store.snapshot()["coverage"]["scored"] == 5
    # Aggregate stdout/report excludes the authored input/answer bodies.
    serialized = json.dumps(report)
    assert "Neri" not in serialized and "Mica" not in serialized and "#include" not in serialized


def test_deterministic_demo_repetition_and_existing_directory_protection(
    demo: Any, tmp_path: Path
) -> None:
    one = demo.run_demo(tmp_path / "one")
    two = demo.run_demo(tmp_path / "two")
    assert one == two
    database = tmp_path / "one" / demo.DATABASE
    before = database.read_bytes()
    with pytest.raises(ValueError):
        demo.run_demo(tmp_path / "one")
    assert database.read_bytes() == before


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_database_and_sidecar_output_aliases_rejected_before_creation(
    demo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    directory = tmp_path / "run"
    output = directory / (demo.DATABASE + suffix)
    monkeypatch.setattr(
        sys, "argv", ["demo", "--directory", str(directory), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="aliases"):
        demo.main()
    assert not directory.exists()


def test_hardlinked_output_source_remains_unchanged(
    demo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "valuable.txt"
    original.write_bytes(b"keep exactly")
    alias = tmp_path / "alias.json"
    os.link(original, alias)
    monkeypatch.setattr(
        sys, "argv", ["demo", "--directory", str(tmp_path / "run"), "--output", str(alias)]
    )
    with pytest.raises(ValueError):
        demo.main()
    assert original.read_bytes() == b"keep exactly"
    assert not (tmp_path / "run").exists()


def test_example_sources_cannot_be_output(demo: Any, tmp_path: Path) -> None:
    before = (demo.EXAMPLES / "suite.json").read_bytes()
    with pytest.raises(ValueError):
        demo._check_output(demo.EXAMPLES / "suite.json", tmp_path / "run")
    with pytest.raises(ValueError):
        demo._check_output(demo.EXAMPLES / "new-report.json", None)
    assert (demo.EXAMPLES / "suite.json").read_bytes() == before


def test_exclusive_report_survives_closed_stdout(
    demo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    output = directory / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["demo", "--directory", str(directory), "--output", str(output)]
    )
    closed = io.StringIO()
    closed.close()
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", closed)
        if sys.version_info >= (3, 14):
            import _colorize

            # Exercise POSIX-style stream probing even on a Windows test runner.
            context.setattr(_colorize, "can_colorize", lambda **_: closed.isatty())
        assert demo.main() == 1
    assert json.loads(output.read_bytes())["passed"] is True
    before = output.read_bytes()
    with pytest.raises(ValueError):
        demo.main()
    assert output.read_bytes() == before
