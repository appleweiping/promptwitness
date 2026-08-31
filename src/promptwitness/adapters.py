"""Loss-aware adapters for common provider prompt/request shapes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from .models import Message, PromptDocument, ToolSpec
from .parser import PromptFormatError, parse_prompt
from .variables import inspect_variables


class AdapterFormat(str, Enum):
    """Supported source formats for prompt conversion."""

    AUTO = "auto"
    NATIVE = "native"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LANGCHAIN = "langchain"


class AdapterError(ValueError):
    """Raised when a provider payload cannot be represented without ambiguity."""


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Converted document plus explicit information-loss warnings."""

    document: PromptDocument
    source_format: AdapterFormat
    warnings: tuple[str, ...] = ()


def load_adapted_prompt(
    path: str | Path,
    source_format: AdapterFormat = AdapterFormat.AUTO,
    *,
    prompt_id: str | None = None,
) -> AdapterResult:
    """Load strict JSON and adapt it to PromptWitness's native model."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AdapterError(f"cannot read {source}: {error}") from error
    try:
        raw = json.loads(
            content,
            parse_constant=_reject_non_finite,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise AdapterError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}"
        ) from error
    return adapt_prompt(raw, source_format, prompt_id=prompt_id)


def adapt_prompt(
    raw: Any,
    source_format: AdapterFormat = AdapterFormat.AUTO,
    *,
    prompt_id: str | None = None,
) -> AdapterResult:
    """Convert one JSON-compatible provider payload to a prompt document."""

    if not isinstance(source_format, AdapterFormat):
        raise TypeError("source_format must be an AdapterFormat")
    if prompt_id is not None and (not isinstance(prompt_id, str) or not prompt_id.strip()):
        raise ValueError("prompt_id must be a non-empty string or None")
    selected = _detect_format(raw) if source_format is AdapterFormat.AUTO else source_format
    try:
        if selected is AdapterFormat.NATIVE:
            document = parse_prompt(raw)
            if prompt_id is not None:
                document = PromptDocument(
                    prompt_id, document.messages, document.tools, document.metadata
                )
            return AdapterResult(document, selected)
        if selected is AdapterFormat.OPENAI:
            return _adapt_openai(raw, prompt_id)
        if selected is AdapterFormat.ANTHROPIC:
            return _adapt_anthropic(raw, prompt_id)
        if selected is AdapterFormat.LANGCHAIN:
            return _adapt_langchain(raw, prompt_id)
    except AdapterError:
        raise
    except (PromptFormatError, TypeError, ValueError) as error:
        raise AdapterError(str(error)) from error
    raise AssertionError(f"unsupported adapter format: {selected}")


def prompt_to_dict(document: PromptDocument) -> dict[str, Any]:
    """Convert a native document to its strict schema-version-1 wire shape."""

    return {
        "schema_version": 1,
        "id": document.prompt_id,
        "messages": [
            {
                **{"role": message.role, "content": message.content},
                **({"name": message.name} if message.name is not None else {}),
            }
            for message in document.messages
        ],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": _thaw_json(tool.parameters),
                "required": list(tool.required),
            }
            for tool in document.tools
        ],
        "metadata": _thaw_json(document.metadata),
    }


def render_prompt_json(document: PromptDocument) -> str:
    """Render a converted prompt as deterministic native JSON."""

    return json.dumps(prompt_to_dict(document), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _detect_format(raw: Any) -> AdapterFormat:
    if not isinstance(raw, dict):
        raise AdapterError("provider prompt must be a JSON object")
    if "schema_version" in raw:
        return AdapterFormat.NATIVE
    if "system" in raw or _tools_use_key(raw.get("tools"), "input_schema"):
        return AdapterFormat.ANTHROPIC
    if "input_variables" in raw or _looks_like_langchain_messages(raw.get("messages")):
        return AdapterFormat.LANGCHAIN
    if isinstance(raw.get("messages"), list):
        return AdapterFormat.OPENAI
    raise AdapterError(
        "cannot detect prompt format; choose native, openai, anthropic, or langchain explicitly"
    )


def _adapt_openai(raw: Any, prompt_id: str | None) -> AdapterResult:
    payload = _object(raw, "OpenAI payload")
    messages_raw = _array(payload.get("messages"), "OpenAI messages")
    warnings = _ignored_top_level(
        payload, {"id", "name", "messages", "tools", "metadata"}, "OpenAI"
    )
    messages: list[Message] = []
    for index, item in enumerate(messages_raw):
        message = _object(item, f"OpenAI message {index}")
        role = _string(message.get("role"), f"OpenAI message {index} role")
        content = _text_content(
            message.get("content"),
            f"OpenAI message {index} content",
            warnings,
        )
        name = message.get("name")
        if name is not None and not isinstance(name, str):
            raise AdapterError(f"OpenAI message {index} name must be a string or null")
        extras = set(message) - {"role", "content", "name"}
        if extras:
            warnings.append(
                f"OpenAI message {index} fields were not represented: {', '.join(sorted(extras))}"
            )
        messages.append(Message(role, content, name))
    tools = _adapt_openai_tools(payload.get("tools", []), warnings)
    metadata = _metadata(payload, AdapterFormat.OPENAI)
    return AdapterResult(
        PromptDocument(
            _prompt_id(payload, prompt_id, "openai-prompt"), tuple(messages), tools, metadata
        ),
        AdapterFormat.OPENAI,
        tuple(warnings),
    )


def _adapt_anthropic(raw: Any, prompt_id: str | None) -> AdapterResult:
    payload = _object(raw, "Anthropic payload")
    messages_raw = _array(payload.get("messages"), "Anthropic messages")
    warnings = _ignored_top_level(
        payload, {"id", "name", "system", "messages", "tools", "metadata"}, "Anthropic"
    )
    messages: list[Message] = []
    if "system" in payload:
        messages.append(
            Message(
                "system",
                _text_content(payload["system"], "Anthropic system", warnings),
            )
        )
    for index, item in enumerate(messages_raw):
        message = _object(item, f"Anthropic message {index}")
        role = _string(message.get("role"), f"Anthropic message {index} role")
        content = _text_content(
            message.get("content"),
            f"Anthropic message {index} content",
            warnings,
        )
        extras = set(message) - {"role", "content"}
        if extras:
            warnings.append(
                f"Anthropic message {index} fields were not represented: "
                f"{', '.join(sorted(extras))}"
            )
        messages.append(Message(role, content))
    tools_raw = _array(payload.get("tools", []), "Anthropic tools")
    tools: list[ToolSpec] = []
    for index, item in enumerate(tools_raw):
        tool = _object(item, f"Anthropic tool {index}")
        name = _string(tool.get("name"), f"Anthropic tool {index} name")
        description = tool.get("description", "")
        if not isinstance(description, str):
            raise AdapterError(f"Anthropic tool {index} description must be a string")
        schema = _object(tool.get("input_schema"), f"Anthropic tool {index} input_schema")
        tools.append(_tool_from_object_schema(name, description, schema, warnings, "Anthropic"))
        extras = set(tool) - {"name", "description", "input_schema"}
        if extras:
            warnings.append(
                f"Anthropic tool {name!r} fields were not represented: {', '.join(sorted(extras))}"
            )
    metadata = _metadata(payload, AdapterFormat.ANTHROPIC)
    return AdapterResult(
        PromptDocument(
            _prompt_id(payload, prompt_id, "anthropic-prompt"),
            tuple(messages),
            tuple(tools),
            metadata,
        ),
        AdapterFormat.ANTHROPIC,
        tuple(warnings),
    )


def _adapt_langchain(raw: Any, prompt_id: str | None) -> AdapterResult:
    payload = _object(raw, "LangChain payload")
    messages_raw = _array(payload.get("messages"), "LangChain messages")
    warnings = _ignored_top_level(
        payload, {"id", "name", "messages", "input_variables", "metadata"}, "LangChain"
    )
    messages: list[Message] = []
    role_names = {"human": "user", "ai": "assistant"}
    for index, item in enumerate(messages_raw):
        message = _object(item, f"LangChain message {index}")
        role_keys = [key for key in ("role", "type", "_type") if key in message]
        if not role_keys:
            raise AdapterError(f"LangChain message {index} requires one of role, type, or _type")
        raw_roles = [_string(message[key], f"LangChain message {index} {key}") for key in role_keys]
        normalized_roles = [role_names.get(role, role) for role in raw_roles]
        if len(set(normalized_roles)) != 1:
            raise AdapterError(f"LangChain message {index} has conflicting role fields")
        raw_role = raw_roles[0]
        role = role_names.get(
            raw_role,
            raw_role,
        )
        content_keys = [key for key in ("content", "template", "prompt") if key in message]
        if not content_keys:
            raise AdapterError(
                f"LangChain message {index} requires content, template, or prompt.template"
            )
        if len(content_keys) != 1:
            raise AdapterError(f"LangChain message {index} has conflicting content fields")
        content_key = content_keys[0]
        if content_key != "prompt":
            content = _content_string(
                message[content_key], f"LangChain message {index} {content_key}"
            )
        else:
            prompt = _object(message.get("prompt"), f"LangChain message {index} prompt")
            content = _content_string(prompt.get("template"), f"LangChain message {index} template")
            prompt_extras = set(prompt) - {"template"}
            if prompt_extras:
                warnings.append(
                    f"LangChain message {index} prompt fields were not represented: "
                    f"{', '.join(sorted(prompt_extras))}"
                )
        name = message.get("name")
        if name is not None and not isinstance(name, str):
            raise AdapterError(f"LangChain message {index} name must be a string or null")
        represented = {*role_keys, content_key, "name"}
        extras = set(message) - represented
        if extras:
            warnings.append(
                f"LangChain message {index} fields were not represented: "
                f"{', '.join(sorted(extras))}"
            )
        messages.append(Message(role, content, name))
    declared = payload.get("input_variables")
    if declared is not None:
        declared_names = set(_string_array(declared, "LangChain input_variables"))
        observed_names: set[str] = set()
        for compiled_message in messages:
            observed_names.update(inspect_variables(compiled_message.content).names)
        if declared_names != observed_names:
            warnings.append(
                "LangChain declared input_variables differ from variables observed "
                "in message templates"
            )
    metadata = _metadata(payload, AdapterFormat.LANGCHAIN)
    return AdapterResult(
        PromptDocument(
            _prompt_id(payload, prompt_id, "langchain-prompt"), tuple(messages), (), metadata
        ),
        AdapterFormat.LANGCHAIN,
        tuple(warnings),
    )


def _adapt_openai_tools(raw: Any, warnings: list[str]) -> tuple[ToolSpec, ...]:
    tools_raw = _array(raw, "OpenAI tools")
    tools: list[ToolSpec] = []
    for index, item in enumerate(tools_raw):
        wrapper = _object(item, f"OpenAI tool {index}")
        if wrapper.get("type") != "function":
            raise AdapterError(f"OpenAI tool {index} must have type 'function'")
        function = _object(wrapper.get("function"), f"OpenAI tool {index} function")
        name = _string(function.get("name"), f"OpenAI tool {index} name")
        description = function.get("description", "")
        if not isinstance(description, str):
            raise AdapterError(f"OpenAI tool {index} description must be a string")
        schema = _object(function.get("parameters"), f"OpenAI tool {index} parameters")
        tools.append(_tool_from_object_schema(name, description, schema, warnings, "OpenAI"))
        extras = set(function) - {"name", "description", "parameters"}
        extras.update(set(wrapper) - {"type", "function"})
        if extras:
            warnings.append(
                f"OpenAI tool {name!r} fields were not represented: {', '.join(sorted(extras))}"
            )
    return tuple(tools)


def _tool_from_object_schema(
    name: str,
    description: str,
    schema: dict[str, Any],
    warnings: list[str],
    provider: str,
) -> ToolSpec:
    schema_type = schema.get("type", "object")
    if schema_type != "object":
        raise AdapterError(f"{provider} tool {name!r} parameters must have type 'object'")
    properties = _object(schema.get("properties", {}), f"{provider} tool {name!r} properties")
    required = _string_array(schema.get("required", []), f"{provider} tool {name!r} required")
    extras = set(schema) - {"type", "properties", "required"}
    if extras:
        warnings.append(
            f"{provider} tool {name!r} object-schema fields were not represented: "
            f"{', '.join(sorted(extras))}"
        )
    try:
        return ToolSpec(name, description, properties, tuple(required))
    except (TypeError, ValueError) as error:
        raise AdapterError(f"{provider} tool {name!r}: {error}") from error


def _text_content(value: Any, label: str, warnings: list[str]) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise AdapterError(f"{label} must be a string or an array of text blocks")
    parts: list[str] = []
    for index, item in enumerate(value):
        block = _object(item, f"{label} block {index}")
        block_type = block.get("type")
        if block_type not in {"text", "input_text", "output_text"}:
            raise AdapterError(
                f"{label} block {index} type {block_type!r} cannot be represented as text"
            )
        parts.append(_content_string(block.get("text"), f"{label} block {index} text"))
        extras = set(block) - {"type", "text"}
        if extras:
            warnings.append(
                f"{label} block {index} fields were not represented: {', '.join(sorted(extras))}"
            )
    if len(parts) > 1:
        warnings.append(f"{label} text-block boundaries were flattened with newline separators")
    return "\n".join(parts)


def _metadata(payload: dict[str, Any], source_format: AdapterFormat) -> dict[str, Any]:
    raw = payload.get("metadata", {})
    if not isinstance(raw, dict):
        raise AdapterError("provider metadata must be an object")
    metadata = dict(raw)
    if "promptwitness_adapter" in metadata:
        raise AdapterError("metadata key 'promptwitness_adapter' is reserved")
    metadata["promptwitness_adapter"] = source_format.value
    return metadata


def _prompt_id(payload: dict[str, Any], explicit: str | None, fallback: str) -> str:
    if explicit is not None:
        return explicit
    if "id" in payload and "name" in payload:
        identifier = _string(payload["id"], "prompt id")
        name = _string(payload["name"], "prompt name")
        if identifier != name:
            raise AdapterError("provider payload has conflicting id and name fields")
        return identifier
    value = payload.get("id", payload.get("name", fallback))
    return _string(value, "prompt id")


def _ignored_top_level(payload: dict[str, Any], represented: set[str], provider: str) -> list[str]:
    extras = set(payload) - represented
    return (
        [f"{provider} top-level fields were not represented: {', '.join(sorted(extras))}"]
        if extras
        else []
    )


def _tools_use_key(raw: Any, key: str) -> bool:
    return isinstance(raw, list) and any(isinstance(item, dict) and key in item for item in raw)


def _looks_like_langchain_messages(raw: Any) -> bool:
    return isinstance(raw, list) and any(
        isinstance(item, dict)
        and (
            "prompt" in item
            or "template" in item
            or ("role" not in item and ("type" in item or "_type" in item))
        )
        for item in raw
    )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AdapterError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{label} must be a non-empty string")
    return value


def _content_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise AdapterError(f"{label} must be a string")
    return value


def _string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AdapterError(f"{label} must be an array of strings")
    if len(set(value)) != len(value):
        raise AdapterError(f"{label} must not contain duplicates")
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _reject_non_finite(value: str) -> None:
    raise AdapterError(f"non-finite JSON number {value!r} is not allowed")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise AdapterError(f"JSON number {value!r} is outside the finite float range")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AdapterError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output
