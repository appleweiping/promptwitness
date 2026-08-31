from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptwitness.adapters import (
    AdapterError,
    AdapterFormat,
    adapt_prompt,
    load_adapted_prompt,
    prompt_to_dict,
    render_prompt_json,
)


def test_native_auto_detection_and_id_override() -> None:
    raw = {"schema_version": 1, "id": "old", "messages": []}
    result = adapt_prompt(raw, prompt_id="new")
    assert result.source_format is AdapterFormat.NATIVE
    assert result.document.prompt_id == "new"
    assert result.warnings == ()
    assert json.loads(render_prompt_json(result.document))["id"] == "new"


def test_openai_adapter_preserves_messages_tools_and_reports_loss() -> None:
    raw = {
        "id": "support",
        "model": "example-model",
        "messages": [
            {"role": "system", "content": "Help {{ name }}."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "input_text", "text": "Second"},
                ],
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up an item",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "minLength": 1}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "metadata": {"owner": "support"},
    }
    result = adapt_prompt(raw, AdapterFormat.OPENAI)
    assert result.document.prompt_id == "support"
    assert result.document.messages[1].content == "First\nSecond"
    assert result.document.tools[0].required == ("id",)
    assert result.document.tools[0].parameters["id"]["minLength"] == 1
    assert result.document.metadata["promptwitness_adapter"] == "openai"
    assert any("top-level" in warning and "model" in warning for warning in result.warnings)
    assert any("cache_control" in warning for warning in result.warnings)
    assert any("additionalProperties" in warning for warning in result.warnings)
    assert any("strict" in warning for warning in result.warnings)
    assert any("boundaries were flattened" in warning for warning in result.warnings)


def test_anthropic_adapter_handles_system_blocks_and_input_schema() -> None:
    raw = {
        "name": "anthropic-example",
        "system": [{"type": "text", "text": "Be precise."}],
        "messages": [{"role": "user", "content": "Question"}],
        "tools": [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }
    result = adapt_prompt(raw)
    assert result.source_format is AdapterFormat.ANTHROPIC
    assert [message.role for message in result.document.messages] == ["system", "user"]
    assert result.document.tools[0].name == "search"


def test_anthropic_adapter_reports_message_and_tool_extras() -> None:
    result = adapt_prompt(
        {
            "system": "Rules",
            "messages": [{"role": "user", "content": "Hi", "cache_control": {}}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": {},
                }
            ],
        }
    )
    assert any("message 0" in warning and "cache_control" in warning for warning in result.warnings)
    assert any(
        "tool 'search'" in warning and "cache_control" in warning for warning in result.warnings
    )


def test_langchain_adapter_maps_roles_and_audits_declared_variables() -> None:
    raw = {
        "name": "chain",
        "input_variables": ["declared_only"],
        "messages": [
            {"type": "system", "prompt": {"template": "Help {{ user }}"}},
            {"_type": "human", "template": "Question"},
            {
                "role": "ai",
                "content": "Answer",
                "name": "reviewer",
                "additional_kwargs": {"refusal": None},
            },
        ],
    }
    result = adapt_prompt(raw, AdapterFormat.LANGCHAIN)
    assert [message.role for message in result.document.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert any("input_variables differ" in warning for warning in result.warnings)
    assert result.document.messages[2].name == "reviewer"
    assert any("additional_kwargs" in warning for warning in result.warnings)


def test_adapters_report_nested_text_and_langchain_prompt_loss() -> None:
    openai = adapt_prompt(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        },
        AdapterFormat.OPENAI,
    )
    assert any(
        "content block 0" in warning and "cache_control" in warning for warning in openai.warnings
    )

    langchain = adapt_prompt(
        {
            "messages": [
                {
                    "type": "human",
                    "prompt": {
                        "template": "Hello {{ name }}",
                        "input_variables": ["name"],
                        "template_format": "jinja2",
                    },
                }
            ]
        }
    )
    assert langchain.source_format is AdapterFormat.LANGCHAIN
    assert any(
        "prompt fields" in warning and "input_variables" in warning and "template_format" in warning
        for warning in langchain.warnings
    )

    empty = adapt_prompt(
        {"messages": [{"type": "human", "content": ""}]},
        AdapterFormat.LANGCHAIN,
    )
    assert empty.document.messages[0].content == ""


@pytest.mark.parametrize(
    "raw",
    [
        {"id": "one", "name": "two", "messages": []},
        {"messages": [{"role": "user", "type": "ai", "content": "x"}]},
        {"messages": [{"type": "human", "content": "x", "template": "y"}]},
    ],
)
def test_adapters_reject_conflicting_aliases(raw: object) -> None:
    with pytest.raises(AdapterError, match="conflicting"):
        adapt_prompt(raw, AdapterFormat.LANGCHAIN)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "cannot detect"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "cannot be"),
        (
            {
                "messages": [],
                "tools": [{"type": "retrieval", "function": {}}],
            },
            "type 'function'",
        ),
        (
            {
                "system": "x",
                "messages": [],
                "tools": [
                    {
                        "name": "bad",
                        "input_schema": {"type": "array", "properties": {}},
                    }
                ],
            },
            "type 'object'",
        ),
    ],
)
def test_adapters_reject_unrepresentable_payloads(raw: object, message: str) -> None:
    with pytest.raises(AdapterError, match=message):
        adapt_prompt(raw)


def test_load_adapter_rejects_ambiguous_json_and_invalid_utf8(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"messages":[],"messages":[]}', encoding="utf-8")
    with pytest.raises(AdapterError, match="duplicate"):
        load_adapted_prompt(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"messages":[],"temperature":NaN}', encoding="utf-8")
    with pytest.raises(AdapterError, match="non-finite"):
        load_adapted_prompt(nonfinite)
    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"messages":[],"temperature":1e400}', encoding="utf-8")
    with pytest.raises(AdapterError, match="finite float range"):
        load_adapted_prompt(overflow)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(AdapterError, match="cannot read"):
        load_adapted_prompt(invalid)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(AdapterError, match="line 1, column 2"):
        load_adapted_prompt(malformed)
    with pytest.raises(AdapterError, match="cannot read"):
        load_adapted_prompt(tmp_path / "missing.json")


def test_prompt_to_dict_thaws_nested_immutable_values() -> None:
    result = adapt_prompt(
        {
            "schema_version": 1,
            "id": "native",
            "messages": [{"role": "user", "content": "x", "name": "caller"}],
            "metadata": {"nested": [1, {"ok": True}]},
        }
    )
    payload = prompt_to_dict(result.document)
    assert payload["metadata"] == {"nested": [1, {"ok": True}]}
    assert payload["messages"][0]["name"] == "caller"


def test_adapter_api_and_model_failures_are_contextual() -> None:
    with pytest.raises(TypeError, match="AdapterFormat"):
        adapt_prompt({}, "openai")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prompt_id"):
        adapt_prompt({"messages": []}, AdapterFormat.OPENAI, prompt_id=" ")
    with pytest.raises(AdapterError, match="reserved"):
        adapt_prompt(
            {
                "messages": [],
                "metadata": {"promptwitness_adapter": "forged"},
            },
            AdapterFormat.OPENAI,
        )
    with pytest.raises(AdapterError, match="name must"):
        adapt_prompt(
            {"messages": [{"role": "user", "content": "x", "name": 1}]},
            AdapterFormat.OPENAI,
        )
    duplicate_tools = {
        "messages": [],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "same",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "same",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }
    with pytest.raises(AdapterError, match="unique"):
        adapt_prompt(duplicate_tools, AdapterFormat.OPENAI)
