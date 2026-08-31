"""Static validation for prompt documents.

Validation deliberately examines structure and obvious operational hazards. It
does not claim that a prompt is safe, correct, or effective for a particular
model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from .models import (
    Finding,
    FindingCode,
    PromptDocument,
    Severity,
    ValidationReport,
)
from .paths import json_pointer
from .variables import inspect_variables

_MESSAGE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:api[_-]?key|access[_-]?token)\s*[:=]\s*[\"']?[A-Za-z0-9_./+-]{12,}", re.I),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Controls opinionated validation rules.

    The defaults accept the common chat roles and require non-empty content,
    but do not require a system message because many valid prompts omit one.
    """

    allowed_roles: frozenset[str] = field(
        default_factory=lambda: frozenset({"system", "user", "assistant", "tool"})
    )
    require_system_first: bool = False
    allow_empty_content: bool = False
    report_repeated_variables: bool = True
    scan_literal_secrets: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_roles, frozenset) or not all(
            isinstance(role, str) for role in self.allowed_roles
        ):
            raise TypeError("allowed_roles must be a frozenset of strings")
        for name in (
            "require_system_first",
            "allow_empty_content",
            "report_repeated_variables",
            "scan_literal_secrets",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")


def validate_prompt(
    document: PromptDocument, policy: ValidationPolicy | None = None
) -> ValidationReport:
    """Return deterministic findings ordered by document position."""

    active = policy or ValidationPolicy()
    findings: list[Finding] = []
    if not document.messages:
        findings.append(
            Finding(
                FindingCode.NO_MESSAGES,
                json_pointer("messages"),
                Severity.BREAKING,
                "prompt has no messages",
            )
        )
    if active.require_system_first and (
        not document.messages or document.messages[0].role != "system"
    ):
        findings.append(
            Finding(
                FindingCode.SYSTEM_POSITION,
                json_pointer("messages", 0),
                Severity.BREAKING,
                "the first message must have the system role",
            )
        )

    for index, message in enumerate(document.messages):
        if message.role not in active.allowed_roles:
            findings.append(
                Finding(
                    FindingCode.UNSUPPORTED_ROLE,
                    json_pointer("messages", index, "role"),
                    Severity.BREAKING,
                    f"unsupported role {message.role!r}",
                )
            )
        if not message.content.strip() and not active.allow_empty_content:
            findings.append(
                Finding(
                    FindingCode.EMPTY_CONTENT,
                    json_pointer("messages", index, "content"),
                    Severity.BREAKING,
                    "message content is empty",
                )
            )
        if message.role == "system" and index > 0:
            findings.append(
                Finding(
                    FindingCode.SYSTEM_POSITION,
                    json_pointer("messages", index, "role"),
                    Severity.WARNING,
                    "system message appears after the first position",
                )
            )
        if message.name is not None and not _MESSAGE_NAME.fullmatch(message.name):
            findings.append(
                Finding(
                    FindingCode.INVALID_MESSAGE_NAME,
                    json_pointer("messages", index, "name"),
                    Severity.BREAKING,
                    "message name must use 1-64 letters, digits, underscores, or hyphens",
                )
            )

        inventory = inspect_variables(message.content)
        if inventory.malformed:
            findings.append(
                Finding(
                    FindingCode.MALFORMED_TEMPLATE,
                    json_pointer("messages", index, "content"),
                    Severity.BREAKING,
                    "template has malformed or unsupported variable syntax",
                )
            )
        if active.report_repeated_variables:
            for name, count in sorted(inventory.counts.items()):
                if count > 1:
                    findings.append(
                        Finding(
                            FindingCode.REPEATED_VARIABLE,
                            json_pointer("messages", index, "content"),
                            Severity.INFO,
                            f"variable {name!r} occurs {count} times",
                        )
                    )
        if active.scan_literal_secrets and _contains_secret(message.content):
            findings.append(
                Finding(
                    FindingCode.SECRET_LITERAL,
                    json_pointer("messages", index, "content"),
                    Severity.BREAKING,
                    "content resembles a literal credential; replace it with a variable",
                )
            )

    for index, tool in enumerate(document.tools):
        if not tool.description.strip():
            findings.append(
                Finding(
                    FindingCode.EMPTY_TOOL_DESCRIPTION,
                    json_pointer("tools", index, "description"),
                    Severity.WARNING,
                    "tool description is empty",
                )
            )
        for parameter, schema in sorted(tool.parameters.items()):
            problem = _parameter_schema_problem(schema)
            if problem is not None:
                findings.append(
                    Finding(
                        FindingCode.INVALID_PARAMETER_SCHEMA,
                        json_pointer("tools", index, "parameters", parameter),
                        Severity.BREAKING,
                        problem,
                    )
                )

    return ValidationReport(document.prompt_id, tuple(findings))


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _SECRET_PATTERNS)


def _parameter_schema_problem(schema: object) -> str | None:
    if not isinstance(schema, Mapping):
        return "parameter schema must be an object"
    schema_type = schema.get("type")
    if schema_type is None:
        return "parameter schema must declare a type"
    if not isinstance(schema_type, str) or schema_type not in _JSON_SCHEMA_TYPES:
        return f"unsupported JSON Schema type {schema_type!r}"
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, tuple) or not enum):
        return "enum must be a non-empty array"
    return None
