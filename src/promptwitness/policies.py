"""Strict, reviewable policy-file loading for validation and diff behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from .diff import DiffOptions, MessageAlignment
from .models import ChangeKind, Severity
from .validation import ValidationPolicy


class PolicyFormatError(ValueError):
    """Raised when a policy file is malformed or ambiguous."""


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """Validation and compatibility policy parsed from one versioned file."""

    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    diff: DiffOptions = field(default_factory=DiffOptions)


def load_policy(path: str | Path) -> PolicyBundle:
    """Load a UTF-8 policy JSON document with duplicate-key protection."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PolicyFormatError(f"cannot read policy {source}: {error}") from error
    try:
        raw = json.loads(
            content,
            parse_constant=_reject_non_finite,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise PolicyFormatError(
            f"invalid policy JSON at line {error.lineno}, column {error.colno}"
        ) from error
    return parse_policy(raw)


def parse_policy(raw: Any) -> PolicyBundle:
    """Parse a JSON-compatible schema-version-1 policy object."""

    if not isinstance(raw, dict):
        raise PolicyFormatError("policy document must be an object")
    unexpected = set(raw) - {"schema_version", "validation", "diff"}
    if unexpected:
        raise PolicyFormatError(f"unknown policy fields: {', '.join(sorted(unexpected))}")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise PolicyFormatError("policy schema_version must be 1")
    try:
        return PolicyBundle(
            validation=_parse_validation(raw.get("validation", {})),
            diff=_parse_diff(raw.get("diff", {})),
        )
    except PolicyFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise PolicyFormatError(str(error)) from error


def _parse_validation(raw: Any) -> ValidationPolicy:
    if not isinstance(raw, dict):
        raise PolicyFormatError("validation policy must be an object")
    allowed = {
        "allowed_roles",
        "require_system_first",
        "allow_empty_content",
        "report_repeated_variables",
        "scan_literal_secrets",
    }
    unexpected = set(raw) - allowed
    if unexpected:
        raise PolicyFormatError(
            f"unknown validation policy fields: {', '.join(sorted(unexpected))}"
        )
    roles = raw.get("allowed_roles", ["system", "user", "assistant", "tool"])
    if (
        not isinstance(roles, list)
        or not roles
        or not all(isinstance(role, str) and role for role in roles)
    ):
        raise PolicyFormatError("allowed_roles must be a non-empty array of strings")
    if len(set(roles)) != len(roles):
        raise PolicyFormatError("allowed_roles must not contain duplicates")
    options = {
        name: _boolean(raw, name, default)
        for name, default in (
            ("require_system_first", False),
            ("allow_empty_content", False),
            ("report_repeated_variables", True),
            ("scan_literal_secrets", True),
        )
    }
    return ValidationPolicy(allowed_roles=frozenset(roles), **options)


def _parse_diff(raw: Any) -> DiffOptions:
    if not isinstance(raw, dict):
        raise PolicyFormatError("diff policy must be an object")
    allowed = {
        "added_message_severity",
        "content_change_severity",
        "include_metadata",
        "context_lines",
        "message_alignment",
        "severity_overrides",
    }
    unexpected = set(raw) - allowed
    if unexpected:
        raise PolicyFormatError(f"unknown diff policy fields: {', '.join(sorted(unexpected))}")
    context_lines = raw.get("context_lines", 2)
    if isinstance(context_lines, bool) or not isinstance(context_lines, int):
        raise PolicyFormatError("context_lines must be an integer")
    alignment = raw.get("message_alignment", MessageAlignment.POSITIONAL.value)
    if not isinstance(alignment, str):
        raise PolicyFormatError("message_alignment must be a string")
    try:
        message_alignment = MessageAlignment(alignment)
    except ValueError as error:
        raise PolicyFormatError(f"unknown message_alignment: {alignment}") from error
    overrides_raw = raw.get("severity_overrides", {})
    if not isinstance(overrides_raw, dict):
        raise PolicyFormatError("severity_overrides must be an object")
    overrides: dict[ChangeKind, Severity] = {}
    for raw_kind, raw_severity in overrides_raw.items():
        try:
            kind = ChangeKind(raw_kind)
        except ValueError as error:
            raise PolicyFormatError(
                f"unknown change kind in severity_overrides: {raw_kind}"
            ) from error
        overrides[kind] = _severity(raw_severity, f"severity_overrides.{raw_kind}")
    return DiffOptions(
        added_message_severity=_severity(
            raw.get("added_message_severity", Severity.WARNING.value),
            "added_message_severity",
        ),
        content_change_severity=_severity(
            raw.get("content_change_severity", Severity.WARNING.value),
            "content_change_severity",
        ),
        include_metadata=_boolean(raw, "include_metadata", True),
        context_lines=context_lines,
        message_alignment=message_alignment,
        severity_overrides=overrides,
    )


def _severity(value: Any, name: str) -> Severity:
    if not isinstance(value, str):
        raise PolicyFormatError(f"{name} must be a severity string")
    try:
        return Severity(value)
    except ValueError as error:
        raise PolicyFormatError(f"unknown severity for {name}: {value}") from error


def _boolean(raw: dict[str, Any], name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if type(value) is not bool:
        raise PolicyFormatError(f"{name} must be a boolean")
    return value


def _reject_non_finite(value: str) -> None:
    raise PolicyFormatError(f"non-finite policy number {value!r} is not allowed")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise PolicyFormatError(f"policy number {value!r} is outside the finite float range")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PolicyFormatError(f"duplicate policy object key {key!r}")
        output[key] = value
    return output
