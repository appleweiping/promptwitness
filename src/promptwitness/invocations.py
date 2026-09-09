"""Deterministic validation of arguments before dispatching a prompt tool."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import ToolSpec
from .schemas import resolve_local_refs


@dataclass(frozen=True, slots=True)
class ToolArgumentIssue:
    """One argument validation finding addressed by a JSON pointer."""

    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True, slots=True)
class ToolArgumentReport:
    """Validation result for one named tool invocation."""

    tool: str
    issues: tuple[ToolArgumentIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_tool_arguments(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    resolve_refs: bool = True,
) -> ToolArgumentReport:
    """Validate one JSON object against a :class:`ToolSpec` declaration.

    This is intentionally a bounded subset rather than a claim of complete
    JSON Schema compliance. Declaration errors (external or cyclic references,
    malformed local references) raise ``ValueError``; user argument mismatches
    are returned as stable, machine-readable findings.
    """

    if not isinstance(tool, ToolSpec):
        raise TypeError("tool must be a ToolSpec")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be an object")
    issues: list[ToolArgumentIssue] = []
    for name in tool.required:
        if name not in arguments:
            issues.append(ToolArgumentIssue(_pointer(name), "required argument is missing"))
    for name in arguments:
        if name not in tool.parameters:
            issues.append(ToolArgumentIssue(_pointer(name), "additional argument is not declared"))
    for name, schema in tool.parameters.items():
        if name not in arguments:
            continue
        if not isinstance(schema, Mapping):
            issues.append(ToolArgumentIssue(_pointer(name), "parameter schema must be an object"))
            continue
        active = resolve_local_refs(schema) if resolve_refs else schema
        for path, message in _validate(arguments[name], active, _pointer(name)):
            issues.append(ToolArgumentIssue(path, message))
    return ToolArgumentReport(tool.name, tuple(issues))


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> tuple[tuple[str, str], ...]:
    findings: list[tuple[str, str]] = []
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        return ((path, f"expected type {expected!r}, got {_type_name(value)!r}"),)
    if "const" in schema and not _json_equal(value, schema["const"]):
        findings.append((path, "value does not match const"))
    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, (list, tuple)) or not any(_json_equal(value, item) for item in enum)
    ):
        findings.append((path, "value is not in enum"))
    for keyword, message in (("anyOf", "anyOf"), ("oneOf", "oneOf")):
        options = schema.get(keyword)
        if isinstance(options, (list, tuple)) and options:
            matches = sum(
                isinstance(option, Mapping) and not _validate(value, option, path)
                for option in options
            )
            if (keyword == "anyOf" and matches == 0) or (keyword == "oneOf" and matches != 1):
                findings.append((path, f"value does not match {message}"))
    all_of = schema.get("allOf")
    if isinstance(all_of, (list, tuple)):
        for option in all_of:
            if isinstance(option, Mapping):
                findings.extend(_validate(value, option, path))
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, (list, tuple)):
            for name in required:
                if isinstance(name, str) and name not in value:
                    findings.append((_pointer(path, name), "required property is missing"))
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = _pointer(path, str(name))
            if name in properties and isinstance(properties[name], Mapping):
                findings.extend(_validate(item, properties[name], child))
            elif additional is False:
                findings.append((child, "additional property is not allowed"))
            elif isinstance(additional, Mapping):
                findings.extend(_validate(item, additional, child))
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            findings.append((path, f"array has fewer than {minimum} items"))
        if isinstance(maximum, int) and len(value) > maximum:
            findings.append((path, f"array has more than {maximum} items"))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                findings.extend(_validate(item, item_schema, _pointer(path, str(index))))
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            findings.append((path, f"string is shorter than {minimum} characters"))
        if isinstance(maximum, int) and len(value) > maximum:
            findings.append((path, f"string is longer than {maximum} characters"))
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            findings.append((path, "string does not match pattern"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            findings.append((path, f"number is below minimum {minimum}"))
        if isinstance(maximum, (int, float)) and value > maximum:
            findings.append((path, f"number is above maximum {maximum}"))
    return tuple(findings)


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality keeps booleans distinct from numbers and thaws schema arrays."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_json_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def _matches_type(value: Any, expected: Any) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    if not isinstance(expected_types, (list, tuple)) or not all(
        isinstance(item, str) for item in expected_types
    ):
        return False
    return any(
        (kind == "null" and value is None)
        or (kind == "boolean" and isinstance(value, bool))
        or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
        or (kind == "string" and isinstance(value, str))
        or (kind == "array" and isinstance(value, list))
        or (kind == "object" and isinstance(value, dict))
        for kind in expected_types
    )


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _pointer(*parts: str) -> str:
    if parts and parts[0].startswith("/"):
        path, parts = parts[0], parts[1:]
    else:
        path = ""
    for part in parts:
        escaped = part.replace("~", "~0").replace("/", "~1")
        path += "/" + escaped
    return path or "/"


__all__ = ["ToolArgumentIssue", "ToolArgumentReport", "validate_tool_arguments"]
