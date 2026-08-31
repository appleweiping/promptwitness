"""Strict parsing for PromptWitness's portable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Message, PromptDocument, ToolSpec


class PromptFormatError(ValueError):
    """Raised for an invalid prompt specification."""


def load_prompt(path: str | Path) -> PromptDocument:
    """Load one UTF-8 JSON prompt document."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PromptFormatError(f"cannot read {source}: {error}") from error
    try:
        raw = json.loads(
            content,
            parse_constant=_reject_non_finite,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise PromptFormatError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}"
        ) from error
    return parse_prompt(raw)


def parse_prompt(raw: Any) -> PromptDocument:
    """Parse a JSON-compatible object and reject ambiguous fields."""

    if not isinstance(raw, dict):
        raise PromptFormatError("prompt document must be an object")
    allowed = {"schema_version", "id", "messages", "tools", "metadata"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise PromptFormatError(f"unknown document fields: {', '.join(sorted(unexpected))}")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise PromptFormatError("schema_version must be 1")
    prompt_id = raw.get("id")
    messages = raw.get("messages")
    tools = raw.get("tools", [])
    metadata = raw.get("metadata", {})
    if not isinstance(prompt_id, str):
        raise PromptFormatError("id must be a string")
    if not isinstance(messages, list):
        raise PromptFormatError("messages must be an array")
    if not isinstance(tools, list):
        raise PromptFormatError("tools must be an array")
    if not isinstance(metadata, dict):
        raise PromptFormatError("metadata must be an object")
    try:
        return PromptDocument(
            prompt_id=prompt_id,
            messages=tuple(_parse_message(item, index) for index, item in enumerate(messages)),
            tools=tuple(_parse_tool(item, index) for index, item in enumerate(tools)),
            metadata=metadata,
        )
    except PromptFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise PromptFormatError(str(error)) from error


def _parse_message(raw: Any, index: int) -> Message:
    if not isinstance(raw, dict):
        raise PromptFormatError(f"message {index} must be an object")
    unexpected = set(raw) - {"role", "content", "name"}
    if unexpected:
        raise PromptFormatError(
            f"message {index} has unknown fields: {', '.join(sorted(unexpected))}"
        )
    role, content, name = raw.get("role"), raw.get("content"), raw.get("name")
    if not isinstance(role, str) or not isinstance(content, str):
        raise PromptFormatError(f"message {index} requires string role and content")
    if name is not None and not isinstance(name, str):
        raise PromptFormatError(f"message {index} name must be a string or null")
    try:
        return Message(role=role, content=content, name=name)
    except (TypeError, ValueError) as error:
        raise PromptFormatError(f"message {index}: {error}") from error


def _parse_tool(raw: Any, index: int) -> ToolSpec:
    if not isinstance(raw, dict):
        raise PromptFormatError(f"tool {index} must be an object")
    unexpected = set(raw) - {"name", "description", "parameters", "required"}
    if unexpected:
        raise PromptFormatError(f"tool {index} has unknown fields: {', '.join(sorted(unexpected))}")
    name = raw.get("name")
    description = raw.get("description", "")
    parameters = raw.get("parameters", {})
    required = raw.get("required", [])
    if not isinstance(name, str) or not isinstance(description, str):
        raise PromptFormatError(f"tool {index} requires string name and description")
    if not isinstance(parameters, dict):
        raise PromptFormatError(f"tool {index} parameters must be an object")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise PromptFormatError(f"tool {index} required must be an array of strings")
    try:
        return ToolSpec(name, description, parameters, tuple(required))
    except ValueError as error:
        raise PromptFormatError(str(error)) from error


def _reject_non_finite(value: str) -> None:
    raise PromptFormatError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PromptFormatError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output
