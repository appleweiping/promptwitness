from __future__ import annotations

import json

import pytest

from promptwitness.models import Message, PromptDocument, ToolSpec
from promptwitness.parser import PromptFormatError, load_prompt, parse_prompt


def raw_prompt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": "support-v1",
        "messages": [{"role": "user", "content": "Hello {{ customer }}"}],
        "tools": [
            {
                "name": "lookup",
                "description": "Find an order",
                "parameters": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            }
        ],
        "metadata": {"owner": "support"},
    }


def test_parse_complete_prompt() -> None:
    document = parse_prompt(raw_prompt())
    assert document.prompt_id == "support-v1"
    assert document.messages[0].name is None
    assert document.tools[0].required == ("order_id",)
    assert document.tool_map() == {"lookup": document.tools[0]}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be an object"),
        ({"schema_version": 2, "id": "x", "messages": []}, "schema_version"),
        ({"schema_version": True, "id": "x", "messages": []}, "schema_version"),
        ({"schema_version": 1, "id": 3, "messages": []}, "id must"),
        ({"schema_version": 1, "id": "x", "messages": {}}, "messages must"),
        ({"schema_version": 1, "id": "x", "messages": [], "tools": {}}, "tools must"),
        (
            {"schema_version": 1, "id": "x", "messages": [], "metadata": []},
            "metadata must",
        ),
        ({"schema_version": 1, "id": "x", "messages": [], "extra": 1}, "unknown"),
    ],
)
def test_rejects_invalid_document_shapes(value: object, message: str) -> None:
    with pytest.raises(PromptFormatError, match=message):
        parse_prompt(value)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("bad", "message 0 must"),
        ({"role": "user"}, "requires string"),
        ({"role": "user", "content": "x", "extra": 1}, "unknown fields"),
        ({"role": "user", "content": "x", "name": 2}, "name must"),
    ],
)
def test_rejects_invalid_messages(message: object, expected: str) -> None:
    raw = raw_prompt()
    raw["messages"] = [message]
    with pytest.raises(PromptFormatError, match=expected):
        parse_prompt(raw)


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("bad", "tool 0 must"),
        ({"name": "x", "unknown": 1}, "unknown fields"),
        ({"name": 2}, "requires string"),
        ({"name": "x", "parameters": []}, "parameters must"),
        ({"name": "x", "required": "p"}, "required must"),
        (
            {"name": "x", "parameters": {}, "required": ["missing"]},
            "requires undeclared",
        ),
        (
            {"name": "x", "parameters": {"p": {}}, "required": ["p", "p"]},
            "duplicate required",
        ),
    ],
)
def test_rejects_invalid_tools(tool: object, expected: str) -> None:
    raw = raw_prompt()
    raw["tools"] = [tool]
    with pytest.raises(PromptFormatError, match=expected):
        parse_prompt(raw)


def test_model_invariants() -> None:
    with pytest.raises(ValueError, match="role"):
        Message(" ", "x")
    with pytest.raises(TypeError, match="content"):
        Message("user", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        ToolSpec(" ", "")
    duplicate = ToolSpec("same", "")
    with pytest.raises(ValueError, match="unique"):
        PromptDocument("x", (), (duplicate, duplicate))
    with pytest.raises(ValueError, match="prompt_id"):
        PromptDocument(" ", ())
    with pytest.raises(TypeError, match="role"):
        Message(1, "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name"):
        Message("user", "x", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parameters"):
        ToolSpec("x", "", {1: {}})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="messages"):
        PromptDocument("x", [])  # type: ignore[arg-type]


def test_load_prompt_errors_are_contextual(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "missing.json"  # type: ignore[operator]
    with pytest.raises(PromptFormatError, match="cannot read"):
        load_prompt(missing)
    invalid = tmp_path / "invalid.json"  # type: ignore[operator]
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(PromptFormatError, match="line 1, column 2"):
        load_prompt(invalid)

    invalid_utf8 = tmp_path / "invalid-utf8.json"  # type: ignore[operator]
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(PromptFormatError, match="cannot read"):
        load_prompt(invalid_utf8)


def test_load_prompt_from_json(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "prompt.json"  # type: ignore[operator]
    path.write_text(json.dumps(raw_prompt()), encoding="utf-8")
    assert load_prompt(path).prompt_id == "support-v1"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"schema_version":1,"id":"a","id":"b","messages":[]}', "duplicate"),
        (
            '{"schema_version":1,"id":"a","messages":[],"metadata":{"score":NaN}}',
            "non-finite",
        ),
        (
            '{"schema_version":1,"id":"a","messages":[],"metadata":{"score":Infinity}}',
            "non-finite",
        ),
    ],
)
def test_load_rejects_non_standard_or_ambiguous_json(
    tmp_path: pytest.TempPathFactory, content: str, message: str
) -> None:
    path = tmp_path / "strict.json"  # type: ignore[operator]
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PromptFormatError, match=message):
        load_prompt(path)


def test_parse_wraps_model_validation_as_format_error() -> None:
    raw = raw_prompt()
    raw["id"] = " "
    with pytest.raises(PromptFormatError, match="prompt_id"):
        parse_prompt(raw)


def test_nested_json_values_are_immutable_and_detached() -> None:
    raw = raw_prompt()
    document = parse_prompt(raw)
    metadata = raw["metadata"]
    assert isinstance(metadata, dict)
    metadata["owner"] = "changed"
    assert document.metadata["owner"] == "support"
    with pytest.raises(TypeError):
        document.metadata["owner"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        document.tools[0].parameters["order_id"] = {}  # type: ignore[index]


def test_direct_models_reject_non_json_payloads() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        PromptDocument("x", (), metadata={"score": float("nan")})
    with pytest.raises(TypeError, match="non-JSON"):
        ToolSpec("x", "", {"p": object()})
