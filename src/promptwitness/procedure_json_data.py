"""Bounded, deterministic JSON primitives for inert procedure datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

MAX_CASE_BYTES = 16 * 1024 * 1024
MAX_NODES = 1_000_000
MAX_FILE_NODES = 4_000_000
MAX_DEPTH = 32
MAX_INTEGER = 2**53 - 1


class ProcedureDataError(ValueError):
    """Invalid or over-budget local procedure data (no source text in errors)."""


def object_fields(value: Any, fields: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields.split()):
        raise ProcedureDataError("procedure object has missing or unknown fields")
    return value


def sequence(value: Any, *, maximum: int = 100_000, minimum: int = 0) -> Any:
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise ProcedureDataError("procedure array has invalid type or length")
    return value


def integer(value: Any, *, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProcedureDataError("procedure integer is invalid")
    return value


def string(value: Any, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ProcedureDataError("procedure text is invalid")
    return value


def sha256(value: Any) -> str:
    result = string(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ProcedureDataError("procedure SHA-256 is invalid")
    return result


def canonical_bytes(
    value: Any, *, limit: int = MAX_CASE_BYTES, node_limit: int | None = None
) -> bytes:
    """Check nodes before copying; encode strings in small escaped UTF-8 chunks.

    This is a serialized-data budget, not a process memory or CPU sandbox.
    Only concrete JSON containers are consumed, never arbitrary iterables.
    """
    integer(limit, minimum=1)
    maximum_nodes = MAX_NODES if node_limit is None else integer(node_limit, minimum=1)
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > MAX_DEPTH or nodes > maximum_nodes:
            raise ProcedureDataError("procedure JSON exceeds structural budget")
        if isinstance(item, Mapping):
            if 2 * len(item) > maximum_nodes - nodes - len(pending):
                raise ProcedureDataError("procedure JSON exceeds structural budget")
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ProcedureDataError("procedure JSON keys must be strings")
                pending.extend(((key, depth + 1), (nested, depth + 1)))
        elif isinstance(item, (list, tuple)):
            if len(item) > maximum_nodes - nodes - len(pending):
                raise ProcedureDataError("procedure JSON exceeds structural budget")
            pending.extend((nested, depth + 1) for nested in item)
        elif isinstance(item, str) or item is None or type(item) is bool:
            continue
        elif type(item) is int:
            if abs(item) > MAX_INTEGER:
                raise ProcedureDataError("procedure JSON integer exceeds portable range")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ProcedureDataError("procedure JSON number must be finite")
        else:
            raise ProcedureDataError("procedure data must contain JSON values only")

    result = bytearray()

    def append(part: str) -> None:
        try:
            encoded = part.encode("utf-8")
        except UnicodeError:
            raise ProcedureDataError("procedure JSON has invalid Unicode") from None
        if len(result) + len(encoded) > limit:
            raise ProcedureDataError("procedure JSON exceeds byte budget")
        result.extend(encoded)

    def emit(item: Any) -> None:
        if isinstance(item, str):
            append('"')
            for offset in range(0, len(item), 4096):
                append(json.dumps(item[offset : offset + 4096], ensure_ascii=False)[1:-1])
            append('"')
        elif isinstance(item, Mapping):
            append("{")
            for index, key in enumerate(sorted(item)):
                if index:
                    append(",")
                emit(key)
                append(":")
                emit(item[key])
            append("}")
        elif isinstance(item, (list, tuple)):
            append("[")
            for index, nested in enumerate(item):
                if index:
                    append(",")
                emit(nested)
            append("]")
        else:
            append(json.dumps(item, allow_nan=False))

    emit(value)
    return bytes(result)


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(raw: bytes, *, limit: int) -> Any:
    """Strict UTF-8 JSON; reject duplicate keys, constants and excessive nesting."""
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise ProcedureDataError("procedure JSON exceeds byte budget")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise ProcedureDataError("procedure JSON must be UTF-8") from None
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise ProcedureDataError("procedure JSON exceeds structural budget")
        elif char in "]}":
            depth -= 1

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ProcedureDataError("procedure JSON has duplicate keys")
            result[key] = item
        return result

    def constant(_: str) -> Any:
        raise ProcedureDataError("procedure JSON number must be finite")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ProcedureDataError("procedure JSON is invalid") from None
    canonical_bytes(value, limit=limit, node_limit=MAX_FILE_NODES)
    return value
