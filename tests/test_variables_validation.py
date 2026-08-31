from __future__ import annotations

import pytest

from promptwitness.models import FindingCode, Message, PromptDocument, Severity, ToolSpec
from promptwitness.validation import ValidationPolicy, validate_prompt
from promptwitness.variables import inspect_variables, render_template


def document(*messages: Message, tools: tuple[ToolSpec, ...] = ()) -> PromptDocument:
    return PromptDocument("test", messages, tools)


def test_variable_inventory_and_literal_rendering() -> None:
    inventory = inspect_variables("Hi {{ user }}; again {{user}} and {{ account.id }}")
    assert inventory.counts == {"user": 2, "account.id": 1}
    assert inventory.names == frozenset({"user", "account.id"})
    assert not inventory.malformed
    assert render_template("{{ x }} + {{x}}", {"x": "<literal>"}) == "<literal> + <literal>"


def test_render_strict_and_permissive() -> None:
    try:
        render_template("{{ known }} {{ missing }}", {"known": 1})
    except KeyError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("strict rendering should reject missing values")
    assert render_template("{{ known }} {{ missing }}", {"known": 1}, strict=False) == (
        "1 {{ missing }}"
    )


def test_malformed_template_cannot_render() -> None:
    assert inspect_variables("{{ open").malformed
    try:
        render_template("{{ open", {})
    except ValueError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("unbalanced template should fail")


@pytest.mark.parametrize("text", ["{{ 123 }}", "{{{ user }}}", "stray }}", "{{ bad name }}"])
def test_unsupported_placeholder_shapes_are_malformed(text: str) -> None:
    assert inspect_variables(text).malformed
    with pytest.raises(ValueError, match="malformed"):
        render_template(text, {})


def test_valid_document_has_no_findings() -> None:
    report = validate_prompt(document(Message("system", "Help {{ user }}."), Message("user", "Hi")))
    assert report.valid
    assert report.findings == ()
    assert report.warning_count == report.breaking_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_roles": "system"},
        {"allowed_roles": frozenset({"system", 1})},
        {"require_system_first": "false"},
        {"allow_empty_content": 1},
        {"report_repeated_variables": None},
        {"scan_literal_secrets": "true"},
    ],
)
def test_validation_policy_rejects_wrong_runtime_types(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        ValidationPolicy(**kwargs)  # type: ignore[arg-type]


def test_message_structure_findings() -> None:
    report = validate_prompt(
        document(
            Message("custom", " ", "bad name!"),
            Message("system", "{{ repeated }} and {{repeated}} plus {{ broken"),
        ),
        ValidationPolicy(require_system_first=True),
    )
    codes = [finding.code for finding in report.findings]
    assert FindingCode.SYSTEM_POSITION in codes
    assert FindingCode.UNSUPPORTED_ROLE in codes
    assert FindingCode.EMPTY_CONTENT in codes
    assert FindingCode.INVALID_MESSAGE_NAME in codes
    assert FindingCode.MALFORMED_TEMPLATE in codes
    assert FindingCode.REPEATED_VARIABLE in codes
    assert not report.valid
    assert report.breaking_count >= 4
    assert report.warning_count == 1


def test_empty_prompt_and_policy_exceptions() -> None:
    empty = validate_prompt(document(), ValidationPolicy(require_system_first=True))
    assert [item.code for item in empty.findings] == [
        FindingCode.NO_MESSAGES,
        FindingCode.SYSTEM_POSITION,
    ]
    allowed = validate_prompt(
        document(Message("custom", "")),
        ValidationPolicy(
            allowed_roles=frozenset({"custom"}),
            allow_empty_content=True,
            report_repeated_variables=False,
        ),
    )
    assert allowed.valid


def test_literal_secret_detection_can_be_disabled() -> None:
    prompt = document(Message("user", "Use api_key = abcdefghijklmnop for this request"))
    report = validate_prompt(prompt)
    assert report.findings[0].code is FindingCode.SECRET_LITERAL
    assert report.findings[0].severity is Severity.BREAKING
    assert validate_prompt(prompt, ValidationPolicy(scan_literal_secrets=False)).valid


def test_tool_findings_are_deterministic() -> None:
    tool = ToolSpec(
        "lookup",
        "",
        {
            "a_not_object": "string",
            "b_missing_type": {},
            "c_bad_type": {"type": "date"},
            "d_empty_enum": {"type": "string", "enum": []},
            "e_valid": {"type": "integer", "enum": [1, 2]},
        },
    )
    report = validate_prompt(document(Message("user", "Go"), tools=(tool,)))
    assert [finding.code for finding in report.findings] == [
        FindingCode.EMPTY_TOOL_DESCRIPTION,
        FindingCode.INVALID_PARAMETER_SCHEMA,
        FindingCode.INVALID_PARAMETER_SCHEMA,
        FindingCode.INVALID_PARAMETER_SCHEMA,
        FindingCode.INVALID_PARAMETER_SCHEMA,
    ]
    assert report.warning_count == 1
    assert report.breaking_count == 4
