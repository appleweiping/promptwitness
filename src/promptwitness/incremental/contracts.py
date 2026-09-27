"""Fail-closed static tool-contract support, separate from behavior certificates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from promptwitness.models import PromptDocument


class ContractStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class ContractCheck:
    status: ContractStatus
    reasons: tuple[str, ...]


_SUPPORTED = {
    "type",
    "description",
    "title",
    "enum",
    "const",
    "properties",
    "required",
    "items",
    "additionalProperties",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
}
_TYPES = {"null", "boolean", "integer", "number", "string", "array", "object"}


def _check_schema(schema: Any, path: str, depth: int = 0) -> ContractCheck:
    if depth > 32:
        return ContractCheck(ContractStatus.UNSUPPORTED, (f"{path}: schema depth exceeds 32",))
    if not isinstance(schema, Mapping):
        return ContractCheck(ContractStatus.INVALID, (f"{path}: object schema required",))
    unknown = set(schema) - _SUPPORTED
    if unknown:
        return ContractCheck(
            ContractStatus.UNSUPPORTED, (f"{path}: unsupported keywords {sorted(unknown)}",)
        )
    types = schema.get("type")
    if types is not None and (not isinstance(types, str) or types not in _TYPES):
        return ContractCheck(
            ContractStatus.UNSUPPORTED, (f"{path}: single supported type required",)
        )
    for key in ("title", "description"):
        if key in schema and not isinstance(schema[key], str):
            return ContractCheck(ContractStatus.INVALID, (f"{path}/{key}: string required",))
    if "enum" in schema and (not isinstance(schema["enum"], (list, tuple)) or not schema["enum"]):
        return ContractCheck(ContractStatus.INVALID, (f"{path}/enum: nonempty array required",))
    for lower, upper in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minimum", "maximum"),
    ):
        for key in (lower, upper):
            if key not in schema:
                continue
            value = schema[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return ContractCheck(
                    ContractStatus.INVALID, (f"{path}/{key}: finite bound required",)
                )
            if key not in ("minimum", "maximum") and (type(value) is not int or value < 0):
                return ContractCheck(
                    ContractStatus.INVALID, (f"{path}/{key}: nonnegative integer required",)
                )
        if lower in schema and upper in schema and schema[lower] > schema[upper]:
            return ContractCheck(ContractStatus.INVALID, (f"{path}: inverted bounds",))
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return ContractCheck(ContractStatus.INVALID, (f"{path}/properties: object required",))
    required = schema.get("required", ())
    if (
        not isinstance(required, (tuple, list))
        or any(not isinstance(x, str) for x in required)
        or len(set(required)) != len(required)
    ):
        return ContractCheck(ContractStatus.INVALID, (f"{path}/required: unique names required",))
    for name, child in properties.items():
        result = _check_schema(child, f"{path}/properties/{name}", depth + 1)
        if result.status != ContractStatus.VALID:
            return result
    if "items" in schema:
        result = _check_schema(schema["items"], f"{path}/items", depth + 1)
        if result.status != ContractStatus.VALID:
            return result
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, Mapping):
        result = _check_schema(additional, f"{path}/additionalProperties", depth + 1)
        if result.status != ContractStatus.VALID:
            return result
    elif type(additional) is not bool:
        return ContractCheck(
            ContractStatus.INVALID, (f"{path}/additionalProperties: boolean or schema required",)
        )
    return ContractCheck(ContractStatus.VALID, ())


def check_contract(parent: PromptDocument, candidate: PromptDocument) -> ContractCheck:
    """Unknown semantics never pass. External tool contracts are immutable here."""
    for document in (parent, candidate):
        if any(message.content_parts for message in document.messages):
            return ContractCheck(
                ContractStatus.UNSUPPORTED, ("multimodal research semantics unsupported",)
            )
        for tool in document.tools:
            for name, schema in tool.parameters.items():
                result = _check_schema(schema, f"tools/{tool.name}/{name}")
                if result.status != ContractStatus.VALID:
                    return result
    old, new = parent.tool_map(), candidate.tool_map()
    if set(old) != set(new) or any(
        old[name].parameters != new[name].parameters or old[name].required != new[name].required
        for name in old
    ):
        return ContractCheck(ContractStatus.INVALID, ("external tool signature/schema changed",))
    return ContractCheck(ContractStatus.VALID, ())
