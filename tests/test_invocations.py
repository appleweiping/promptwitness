import json

import pytest

from promptwitness import ToolSpec, validate_tool_arguments
from promptwitness.cli import main


def test_validate_tool_arguments_reports_required_extra_and_nested_findings() -> None:
    tool = ToolSpec(
        "lookup",
        "Look up an order",
        {
            "order_id": {"type": "string", "minLength": 3},
            "options": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1}},
                "required": ["limit"],
                "additionalProperties": False,
            },
        },
        ("order_id", "options"),
    )
    report = validate_tool_arguments(
        tool,
        {"order_id": "x", "options": {"limit": 0, "unused": True}, "extra": 1},
    )
    assert not report.valid
    assert [issue.path for issue in report.issues] == [
        "/extra",
        "/order_id",
        "/options/limit",
        "/options/unused",
    ]


def test_validate_tool_arguments_resolves_local_refs() -> None:
    tool = ToolSpec(
        "search",
        "Search",
        {
            "query": {
                "$ref": "#/$defs/query",
                "$defs": {"query": {"type": "string", "minLength": 2}},
            }
        },
        ("query",),
    )
    assert validate_tool_arguments(tool, {"query": "ok"}).valid
    assert not validate_tool_arguments(tool, {"query": ""}).valid


def test_validate_tool_arguments_covers_schema_combinators_and_constraints() -> None:
    tool = ToolSpec(
        "complex",
        "Complex",
        {
            "choice": {"type": "string", "enum": ["a"], "const": "a"},
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 2, "maximum": 4},
            },
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "^x$", "maxLength": 1}},
                "additionalProperties": {"type": "string"},
                "required": ["value"],
            },
            "union": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "exclusive": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "combined": {"allOf": [{"type": "integer"}, {"minimum": 2}]},
        },
    )
    invalid = validate_tool_arguments(
        tool,
        {
            "choice": "b",
            "items": [1, 5, 7],
            "nested": {"value": "y", "extra": 3},
            "union": [],
            "exclusive": True,
            "combined": 1,
        },
    )
    assert not invalid.valid
    assert any(issue.path == "/items/0" for issue in invalid.issues)
    valid = validate_tool_arguments(
        tool,
        {
            "choice": "a",
            "items": [2, 4],
            "nested": {"value": "x", "extra": "ok"},
            "union": "text",
            "exclusive": 3,
            "combined": 3,
        },
    )
    assert valid.valid


def test_validate_tool_arguments_rejects_bad_declarations_and_input_types() -> None:
    with pytest.raises(TypeError):
        validate_tool_arguments("not-a-tool", {})  # type: ignore[arg-type]
    tool = ToolSpec("bad", "Bad", {"value": "not-a-schema"})
    report = validate_tool_arguments(tool, {"value": 1})
    assert report.issues[0].message == "parameter schema must be an object"
    with pytest.raises(TypeError):
        validate_tool_arguments(tool, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only local"):
        validate_tool_arguments(
            ToolSpec("external", "External", {"value": {"$ref": "https://example.com/schema"}}),
            {"value": 1},
        )
    with pytest.raises(ValueError, match="cyclic"):
        validate_tool_arguments(
            ToolSpec(
                "cycle",
                "Cycle",
                {
                    "value": {
                        "$defs": {"x": {"$ref": "#/$defs/y"}, "y": {"$ref": "#/$defs/x"}},
                        "$ref": "#/$defs/x",
                    }
                },
            ),
            {"value": 1},
        )


def test_validate_tool_arguments_can_leave_refs_unresolved() -> None:
    tool = ToolSpec("raw", "Raw", {"value": {"$ref": "#/$defs/value"}})
    assert validate_tool_arguments(tool, {"value": 3}, resolve_refs=False).valid


def test_check_call_cli_returns_gate_status(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    prompt = tmp_path / "prompt.json"
    arguments = tmp_path / "arguments.json"
    prompt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "demo",
                "messages": [{"role": "user", "content": "Use the tool."}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Look up",
                        "parameters": {"id": {"type": "string", "minLength": 2}},
                        "required": ["id"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    arguments.write_text('{"id":"x"}', encoding="utf-8")
    assert main(["check-call", str(prompt), "lookup", str(arguments)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["issues"][0]["path"] == "/id"
