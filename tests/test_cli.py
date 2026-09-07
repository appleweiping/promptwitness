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


def test_convert_openai_and_validate_provider_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "openai.json"
    source.write_text(
        json.dumps(
            {
                "model": "example-model",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "native.json"
    assert (
        main(
            [
                "convert",
                str(source),
                "--from-format",
                "openai",
                "--id",
                "converted",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "converted"
    assert "adapter warning" in capsys.readouterr().err
    assert main(["validate", str(source), "--from-format", "openai"]) == 0


def test_matrix_renders_scenarios(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    write_prompt(prompt, content="Hello {{name}}")
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps([{"id": "a", "values": {"name": "Ada"}}]), encoding="utf-8")
    output = tmp_path / "matrix.json"
    assert main(["matrix", str(prompt), str(scenarios), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"][0]["messages"][0]["content"] == "Hello Ada"


def test_matrix_diff_cli_reports_scenario_changes(tmp_path: Path) -> None:
    prompt_before = tmp_path / "before-prompt.json"
    prompt_after = tmp_path / "after-prompt.json"
    write_prompt(prompt_before, content="Hello {{name}}")
    write_prompt(prompt_after, content="Hi {{name}}")
    after_payload = json.loads(prompt_after.read_text(encoding="utf-8"))
    after_payload["id"] = "before-prompt"
    prompt_after.write_text(json.dumps(after_payload), encoding="utf-8")
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps([{"id": "a", "values": {"name": "Ada"}}]), encoding="utf-8")
    before = tmp_path / "before-matrix.json"
    after = tmp_path / "after-matrix.json"
    assert main(["matrix", str(prompt_before), str(scenarios), "--artifact", str(before)]) == 0
    assert main(["matrix", str(prompt_after), str(scenarios), "--artifact", str(after)]) == 0
    output = tmp_path / "matrix-diff.json"
    assert main(["matrix-diff", str(before), str(after), "--output", str(output)]) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["changed"] is True
    assert payload["scenarios"][0]["scenario_id"] == "a"


def test_long_context_cli_scores_recorded_predictions(tmp_path: Path) -> None:
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "one",
                    "context": ["front needle", "back"],
                    "needle": "needle",
                    "query": "What?",
                    "expected": "yes",
                    "needle_index": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text('{"one":"yes"}', encoding="utf-8")
    output = tmp_path / "long-context.json"
    assert main(["long-context", str(cases), str(predictions), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["accuracy"] == 1.0


def test_cli_policy_smart_alignment_and_sarif(tmp_path: Path) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "before",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "after",
                "messages": [
                    {"role": "system", "content": "Rules"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "diff": {
                    "message_alignment": "smart",
                    "severity_overrides": {"message_added": "breaking"},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.sarif"
    assert (
        main(
            [
                "diff",
                str(before),
                str(after),
                "--policy",
                str(policy),
                "--format",
                "sarif",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    results = payload["runs"][0]["results"]
    assert len(results) == 1
    assert results[0]["ruleId"].endswith("message_added")
    assert results[0]["level"] == "error"


def test_cli_flags_can_tighten_or_relax_policy_booleans(tmp_path: Path) -> None:
    source = tmp_path / "prompt.json"
    write_prompt(source, content="sk-abcdefghijklmnop", role="user")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation": {
                    "require_system_first": True,
                    "scan_literal_secrets": False,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["validate", str(source), "--policy", str(policy)]) == 2
    assert (
        main(
            [
                "validate",
                str(source),
                "--policy",
                str(policy),
                "--no-require-system-first",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                str(source),
                "--policy",
                str(policy),
                "--no-require-system-first",
                "--secret-scan",
            ]
        )
        == 2
    )


def test_cli_refuses_to_overwrite_prompt_or_policy_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = tmp_path / "prompt.json"
    write_prompt(prompt)
    original_prompt = prompt.read_bytes()
    assert main(["validate", str(prompt), "--output", str(prompt)]) == 1
    assert prompt.read_bytes() == original_prompt

    alias = tmp_path / "alias.json"
    alias.hardlink_to(prompt)
    assert main(["validate", str(prompt), "--output", str(alias)]) == 1
    assert prompt.read_bytes() == original_prompt

    provider = tmp_path / "provider.json"
    provider.write_text('{"messages":[]}', encoding="utf-8")
    original_provider = provider.read_bytes()
    assert main(["convert", str(provider), "--output", str(provider)]) == 1
    assert provider.read_bytes() == original_provider

    policy = tmp_path / "policy.json"
    policy.write_text('{"schema_version":1}', encoding="utf-8")
    original_policy = policy.read_bytes()
    assert (
        main(
            [
                "validate",
                str(prompt),
                "--policy",
                str(policy),
                "--output",
                str(policy),
            ]
        )
        == 1
    )
    assert policy.read_bytes() == original_policy
    assert "must not overwrite" in capsys.readouterr().err
