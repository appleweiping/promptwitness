from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptwitness.diff import DiffOptions, MessageAlignment, compare_prompts
from promptwitness.models import ChangeKind, Message, PromptDocument, Severity
from promptwitness.policies import PolicyFormatError, load_policy, parse_policy
from promptwitness.reporting import render_sarif
from promptwitness.validation import validate_prompt


def document(prompt_id: str, *messages: Message) -> PromptDocument:
    return PromptDocument(prompt_id, messages)


def test_smart_alignment_isolates_a_message_insertion() -> None:
    before = document("before", Message("user", "Question"), Message("assistant", "Answer"))
    after = document(
        "after",
        Message("system", "Safety instruction"),
        Message("user", "Question"),
        Message("assistant", "Answer"),
    )
    positional = compare_prompts(before, after)
    smart = compare_prompts(before, after, DiffOptions(message_alignment=MessageAlignment.SMART))
    assert len(positional.changes) > 1
    assert [(change.kind, change.path) for change in smart.changes] == [
        (ChangeKind.MESSAGE_ADDED, "/messages/0")
    ]


def test_smart_alignment_keeps_true_edits_and_removals() -> None:
    before = document(
        "before",
        Message("system", "Rules"),
        Message("user", "Old question"),
        Message("assistant", "Answer"),
    )
    after = document("after", Message("user", "New question"), Message("assistant", "Answer"))
    report = compare_prompts(before, after, DiffOptions(message_alignment=MessageAlignment.SMART))
    assert [change.kind for change in report.changes[:2]] == [
        ChangeKind.MESSAGE_REMOVED,
        ChangeKind.MESSAGE_CONTENT,
    ]
    assert report.changes[1].path == "/messages/0/content"


def test_smart_alignment_boundaries_and_repeated_message_ties_are_stable() -> None:
    empty = document("empty")
    populated = document("populated", Message("user", "x"), Message("assistant", "y"))
    added = compare_prompts(
        empty,
        populated,
        DiffOptions(message_alignment=MessageAlignment.SMART),
    )
    removed = compare_prompts(
        populated,
        empty,
        DiffOptions(message_alignment=MessageAlignment.SMART),
    )
    assert [change.path for change in added.changes] == ["/messages/0", "/messages/1"]
    assert [change.path for change in removed.changes] == ["/messages/0", "/messages/1"]

    repeated_before = document("before", Message("user", "x"), Message("user", "x"))
    repeated_after = document(
        "after",
        Message("user", "x"),
        Message("user", "inserted"),
        Message("user", "x"),
    )
    options = DiffOptions(message_alignment=MessageAlignment.SMART)
    first = compare_prompts(repeated_before, repeated_after, options)
    assert first == compare_prompts(repeated_before, repeated_after, options)
    assert [(change.kind, change.path) for change in first.changes] == [
        (ChangeKind.MESSAGE_ADDED, "/messages/1")
    ]


def test_severity_overrides_apply_after_structural_comparison() -> None:
    report = compare_prompts(
        document("before", Message("user", "old")),
        document("after", Message("user", "new")),
        DiffOptions(severity_overrides={ChangeKind.MESSAGE_CONTENT: Severity.BREAKING}),
    )
    assert report.changes[0].severity is Severity.BREAKING
    assert not report.compatible


def test_policy_file_controls_validation_alignment_and_severity(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation": {
                    "allowed_roles": ["user", "critic"],
                    "allow_empty_content": True,
                    "scan_literal_secrets": False,
                },
                "diff": {
                    "message_alignment": "smart",
                    "content_change_severity": "info",
                    "severity_overrides": {"message_added": "breaking"},
                    "context_lines": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    policy = load_policy(path)
    assert policy.validation.allowed_roles == frozenset({"user", "critic"})
    assert policy.validation.allow_empty_content
    assert policy.diff.message_alignment is MessageAlignment.SMART
    assert policy.diff.severity_overrides[ChangeKind.MESSAGE_ADDED] is Severity.BREAKING


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "must be an object"),
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": 1, "unknown": {}}, "unknown policy"),
        ({"schema_version": 1, "validation": {"allowed_roles": []}}, "allowed_roles"),
        (
            {"schema_version": 1, "validation": {"require_system_first": 1}},
            "must be a boolean",
        ),
        (
            {"schema_version": 1, "diff": {"message_alignment": "magic"}},
            "unknown message_alignment",
        ),
        (
            {"schema_version": 1, "diff": {"severity_overrides": {"unknown": "info"}}},
            "unknown change kind",
        ),
        (
            {
                "schema_version": 1,
                "diff": {"severity_overrides": {"message_added": "fatal"}},
            },
            "unknown severity",
        ),
    ],
)
def test_policy_parser_rejects_invalid_documents(raw: object, message: str) -> None:
    with pytest.raises(PolicyFormatError, match=message):
        parse_policy(raw)


def test_policy_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"schema_version":1,"diff":{},"diff":{}}', encoding="utf-8")
    with pytest.raises(PolicyFormatError, match="duplicate"):
        load_policy(path)


def test_policy_loader_wraps_io_json_and_nonfinite_errors(tmp_path: Path) -> None:
    with pytest.raises(PolicyFormatError, match="cannot read"):
        load_policy(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(PolicyFormatError, match="line 1, column 2"):
        load_policy(malformed)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":1,"diff":{"context_lines":NaN}}', encoding="utf-8")
    with pytest.raises(PolicyFormatError, match="non-finite"):
        load_policy(nonfinite)
    overflow = tmp_path / "overflow.json"
    overflow.write_text(
        '{"schema_version":1,"diff":{"context_lines":1e400}}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyFormatError, match="finite float range"):
        load_policy(overflow)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"validation": []}, "validation policy"),
        ({"diff": []}, "diff policy"),
        ({"validation": {"unknown": True}}, "unknown validation"),
        ({"diff": {"unknown": True}}, "unknown diff"),
        ({"validation": {"allowed_roles": ["user", "user"]}}, "duplicates"),
        ({"diff": {"context_lines": True}}, "integer"),
        ({"diff": {"context_lines": -1}}, "non-negative"),
        ({"diff": {"message_alignment": 1}}, "must be a string"),
        ({"diff": {"severity_overrides": []}}, "must be an object"),
        ({"diff": {"added_message_severity": 1}}, "severity string"),
    ],
)
def test_policy_nested_validation_is_strict(body: dict[str, object], message: str) -> None:
    with pytest.raises(PolicyFormatError, match=message):
        parse_policy({"schema_version": 1, **body})


def test_sarif_diff_and_validation_are_stable() -> None:
    diff_report = compare_prompts(
        document("before", Message("user", "old")),
        document("after", Message("user", "new")),
    )
    first = render_sarif(diff_report, artifact_uri="prompts/current.json")
    assert first == render_sarif(diff_report, artifact_uri="prompts/current.json")
    payload = json.loads(first)
    assert payload["version"] == "2.1.0"
    result = payload["runs"][0]["results"][0]
    assert result["ruleId"] == "promptwitness.diff.message_content_changed"
    assert result["level"] == "warning"
    assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == (
        "prompts/current.json"
    )
    assert len(result["partialFingerprints"]["promptWitnessFingerprint/v1"]) == 64

    validation = validate_prompt(document("bad", Message("unsupported", "")))
    validation_payload = json.loads(render_sarif(validation))
    assert {result["level"] for result in validation_payload["runs"][0]["results"]} == {"error"}


def test_empty_sarif_and_invalid_artifact_uri() -> None:
    clean = validate_prompt(document("clean", Message("user", "hello")))
    assert json.loads(render_sarif(clean))["runs"][0]["results"] == []
    with pytest.raises(ValueError, match="artifact_uri"):
        render_sarif(clean, artifact_uri="")


def test_sarif_encodes_path_characters_and_rejects_controls() -> None:
    invalid = validate_prompt(document("bad", Message("unsupported", "")))
    payload = json.loads(render_sarif(invalid, artifact_uri=r"prompts\a b#c.json"))
    uri = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"][
        "uri"
    ]
    assert uri == "prompts/a%20b%23c.json"
    with pytest.raises(ValueError, match="control"):
        render_sarif(invalid, artifact_uri="bad\x00name.json")
