from __future__ import annotations

import json

import pytest

from promptwitness.diff import DiffOptions, compare_prompts
from promptwitness.models import (
    ChangeKind,
    FindingCode,
    Message,
    PromptDocument,
    Severity,
    ToolSpec,
)
from promptwitness.reporting import (
    diff_to_dict,
    render_html,
    render_json,
    render_markdown,
    validation_to_dict,
)
from promptwitness.validation import validate_prompt


def prompt(
    prompt_id: str,
    *messages: Message,
    tools: tuple[ToolSpec, ...] = (),
    metadata: dict[str, object] | None = None,
) -> PromptDocument:
    return PromptDocument(prompt_id, messages, tools, metadata or {})


def test_identical_documents_are_compatible() -> None:
    value = prompt("same", Message("user", "Hello"))
    report = compare_prompts(value, value)
    assert report.compatible
    assert report.changes == ()
    assert report.warning_count == report.breaking_count == 0
    assert "No changes" in render_markdown(report)


def test_message_and_variable_changes() -> None:
    before = prompt("v1", Message("user", "Hello {{name}}", "customer"), Message("assistant", "A"))
    after = prompt("v2", Message("system", "Welcome {{account}}", "agent"))
    report = compare_prompts(before, after)
    kinds = [change.kind for change in report.changes]
    assert kinds == [
        ChangeKind.MESSAGE_ROLE,
        ChangeKind.MESSAGE_NAME,
        ChangeKind.MESSAGE_CONTENT,
        ChangeKind.MESSAGE_REMOVED,
        ChangeKind.VARIABLE_ADDED,
        ChangeKind.VARIABLE_REMOVED,
    ]
    content = report.changes[2]
    assert content.severity is Severity.WARNING
    assert content.unified_diff[:2] == ("--- before", "+++ after")
    assert report.breaking_count == 4


def test_added_message_severity_is_configurable() -> None:
    before = prompt("v1")
    after = prompt("v2", Message("user", "Hi"))
    report = compare_prompts(before, after, DiffOptions(added_message_severity=Severity.INFO))
    assert report.changes[0].kind is ChangeKind.MESSAGE_ADDED
    assert report.changes[0].severity is Severity.INFO
    with pytest.raises(ValueError, match="non-negative"):
        DiffOptions(context_lines=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"added_message_severity": "warning"},
        {"content_change_severity": "breaking"},
        {"include_metadata": "false"},
        {"context_lines": True},
        {"context_lines": 1.5},
    ],
)
def test_diff_options_reject_wrong_runtime_types(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        DiffOptions(**kwargs)  # type: ignore[arg-type]


def test_tool_compatibility_matrix() -> None:
    old_lookup = ToolSpec(
        "lookup",
        "old description",
        {"id": {"type": "string"}, "legacy": {"type": "boolean"}},
        ("id", "legacy"),
    )
    new_lookup = ToolSpec(
        "lookup",
        "new description",
        {"id": {"type": "integer"}, "region": {"type": "string"}},
        ("id", "region"),
    )
    removed = ToolSpec("removed", "gone", {"q": {"type": "string"}})
    added = ToolSpec("added", "new", {"q": {"type": "string"}})
    report = compare_prompts(
        prompt("v1", tools=(old_lookup, removed)),
        prompt("v2", tools=(new_lookup, added)),
    )
    kinds = [change.kind for change in report.changes]
    assert kinds == [
        ChangeKind.TOOL_REMOVED,
        ChangeKind.TOOL_ADDED,
        ChangeKind.TOOL_DESCRIPTION,
        ChangeKind.TOOL_PARAMETER_REMOVED,
        ChangeKind.TOOL_PARAMETER_ADDED,
        ChangeKind.TOOL_PARAMETER_CHANGED,
        ChangeKind.TOOL_REQUIRED_ADDED,
        ChangeKind.TOOL_REQUIRED_REMOVED,
    ]
    assert report.breaking_count == 5
    assert report.warning_count == 1


def test_optional_parameter_and_required_removal() -> None:
    old = ToolSpec("tool", "d", {"keep": {"type": "string"}}, ("keep",))
    new = ToolSpec(
        "tool",
        "d",
        {"keep": {"type": "string"}, "optional": {"type": "number"}},
    )
    report = compare_prompts(prompt("a", tools=(old,)), prompt("b", tools=(new,)))
    assert report.changes[0].severity is Severity.INFO
    assert report.changes[1].kind is ChangeKind.TOOL_REQUIRED_REMOVED


def test_metadata_can_be_excluded() -> None:
    before = prompt("a", metadata={"owner": "one"})
    after = prompt("b", metadata={"owner": "two"})
    assert compare_prompts(before, after).changes[0].kind is ChangeKind.METADATA
    assert compare_prompts(before, after, DiffOptions(include_metadata=False)).changes == ()


def test_json_boolean_is_not_equal_to_number() -> None:
    before_tool = ToolSpec("x", "d", {"p": {"type": "integer", "default": True}})
    after_tool = ToolSpec("x", "d", {"p": {"type": "integer", "default": 1}})
    report = compare_prompts(
        prompt("a", tools=(before_tool,), metadata={"enabled": True}),
        prompt("b", tools=(after_tool,), metadata={"enabled": 1}),
    )
    assert [change.kind for change in report.changes] == [
        ChangeKind.TOOL_PARAMETER_CHANGED,
        ChangeKind.METADATA,
    ]


def test_report_paths_use_unambiguous_json_pointer_escaping() -> None:
    before_tool = ToolSpec("a/b~c", "d", {"x/y": {"type": "string"}})
    after_tool = ToolSpec("a/b~c", "d", {})
    report = compare_prompts(prompt("a", tools=(before_tool,)), prompt("b", tools=(after_tool,)))
    assert report.changes[0].path == "/tools/a~1b~0c/parameters/x~1y"


def test_json_and_markdown_diff_reports() -> None:
    report = compare_prompts(
        prompt("a", Message("user", "one")), prompt("b", Message("user", "two"))
    )
    payload = diff_to_dict(report)
    assert payload["schema_version"] == 1
    assert payload["report_type"] == "diff"
    assert json.loads(render_json(report))["summary"]["changes"] == 1
    markdown = render_markdown(report)
    assert "```diff" in markdown
    assert "message_content_changed" in markdown


def test_markdown_uses_a_safe_fence_for_prompt_backticks() -> None:
    report = compare_prompts(
        prompt("a", Message("user", "before")),
        prompt("b", Message("user", "```\nafter\n```")),
    )
    markdown = render_markdown(report)
    assert "````diff" in markdown
    assert markdown.rstrip().endswith("````")


def test_validation_reports_and_html_escape_content() -> None:
    report = validate_prompt(prompt("<unsafe>", Message("bad|role", "")))
    payload = validation_to_dict(report)
    assert payload["valid"] is False
    assert payload["findings"][0]["code"] == FindingCode.UNSUPPORTED_ROLE.value
    assert json.loads(render_json(report))["prompt_id"] == "<unsafe>"
    assert "bad&#124;role" in render_markdown(report)
    html = render_html(report)
    assert "&lt;unsafe&gt;" in html
    assert '<span class="badge fail">Invalid</span>' in html
    assert "<unsafe>" not in html


def test_clean_validation_html_has_empty_state() -> None:
    report = validate_prompt(prompt("clean", Message("user", "hello")))
    html = render_html(report)
    assert "No findings" in html
    assert '<span class="badge pass">Valid</span>' in html
