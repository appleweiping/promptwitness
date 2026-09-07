"""Small, deterministic resolver for local JSON Schema references."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def resolve_local_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Expand local ``$ref`` pointers without network access.

    Only fragment references into the supplied schema are accepted. Cycles,
    missing targets, and external URLs fail explicitly so a compatibility
    report never silently compares an unresolved or remotely fetched schema.
    Sibling keys next to ``$ref`` override the referenced object, matching the
    useful draft-2020-12 behavior for local tool contracts.
    """

    if not isinstance(schema, Mapping):
        raise TypeError("schema must be an object")
    return cast(dict[str, Any], _resolve(schema, schema, ()))


def _resolve(value: Any, root: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    if isinstance(value, list):
        return [_resolve(item, root, stack) for item in value]
    if not isinstance(value, Mapping):
        return value
    ref = value.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError("only local JSON Schema references beginning with '#' are supported")
        if ref in stack:
            raise ValueError(f"cyclic JSON Schema reference {ref!r}")
        target = _pointer(root, ref)
        if not isinstance(target, Mapping):
            raise ValueError(f"JSON Schema reference {ref!r} does not resolve to an object")
        resolved = _resolve(target, root, (*stack, ref))
        merged = dict(resolved)
        merged.update(
            {
                key: _resolve(item, root, (*stack, ref))
                for key, item in value.items()
                if key != "$ref"
            }
        )
        return merged
    return {key: _resolve(item, root, stack) for key, item in value.items()}


def _pointer(root: Mapping[str, Any], ref: str) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise ValueError(f"invalid local JSON Schema reference {ref!r}")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"missing JSON Schema reference target {ref!r}")
        current = current[part]
    return current
