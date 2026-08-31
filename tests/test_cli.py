from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptwitness.cli import build_parser, main


def write_prompt(path: Path, *, content: str = "Hello", role: str = "user") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": path.stem,
                "messages": [{"role": role, "content": content}],
            }
        ),
        encoding="utf-8",
    )


def test_validate_json_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "valid.json"
    write_prompt(source)
    assert main(["validate", str(source), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_validate_policy_failure_and_never_gate(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    write_prompt(source, content="")
    assert main(["validate", str(source)]) == 2
    assert main(["validate", str(source), "--allow-empty-content"]) == 0
    assert main(["validate", str(source), "--fail-on", "never"]) == 0


def test_diff_writes_html_and_applies_threshold(tmp_path: Path) -> None:
    before, after, output = (
        tmp_path / "before.json",
        tmp_path / "after.json",
        tmp_path / "out/report.html",
    )
    write_prompt(before, content="Hello")
    write_prompt(after, content="Changed")
    assert main(["diff", str(before), str(after), "--format", "html", "--output", str(output)]) == 0
    assert "PromptWitness diff" in output.read_text(encoding="utf-8")
    assert main(["diff", str(before), str(after), "--fail-on", "warning"]) == 2


def test_diff_breaking_and_ignore_metadata(tmp_path: Path) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    write_prompt(before, role="user")
    write_prompt(after, role="system")
    assert main(["diff", str(before), str(after), "--format", "json"]) == 2


def test_input_failure_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{", encoding="utf-8")
    assert main(["validate", str(source)]) == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_parser_version_and_invalid_context() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as version:
        parser.parse_args(["--version"])
    assert version.value.code == 0
    with pytest.raises(SystemExit) as invalid:
        parser.parse_args(["diff", "a", "b", "--context-lines", "-1"])
    assert invalid.value.code == 1
