"""Immutable prompt and diff models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any, path: str) -> Any:
    """Return an immutable JSON value and reject non-portable Python values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            frozen[key] = _freeze_json(nested, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


class Severity(str, Enum):
    """Compatibility impact assigned to a change or validation finding."""

    INFO = "info"
    WARNING = "warning"
    BREAKING = "breaking"


class ChangeKind(str, Enum):
    """Stable machine-readable change categories."""

    MESSAGE_ADDED = "message_added"
    MESSAGE_REMOVED = "message_removed"
    MESSAGE_ROLE = "message_role_changed"
    MESSAGE_NAME = "message_name_changed"
    MESSAGE_CONTENT = "message_content_changed"
    VARIABLE_ADDED = "variable_added"
    VARIABLE_REMOVED = "variable_removed"
    TOOL_ADDED = "tool_added"
    TOOL_REMOVED = "tool_removed"
    TOOL_DESCRIPTION = "tool_description_changed"
    TOOL_PARAMETER_ADDED = "tool_parameter_added"
    TOOL_PARAMETER_REMOVED = "tool_parameter_removed"
    TOOL_PARAMETER_CHANGED = "tool_parameter_changed"
    TOOL_REQUIRED_ADDED = "tool_required_added"
    TOOL_REQUIRED_REMOVED = "tool_required_removed"
    METADATA = "metadata_changed"


class FindingCode(str, Enum):
    """Stable validation categories suitable for CI allowlists."""

    NO_MESSAGES = "no_messages"
    UNSUPPORTED_ROLE = "unsupported_role"
    EMPTY_CONTENT = "empty_content"
    SYSTEM_POSITION = "system_message_position"
    MALFORMED_TEMPLATE = "malformed_template"
    REPEATED_VARIABLE = "repeated_variable"
    INVALID_MESSAGE_NAME = "invalid_message_name"
    EMPTY_TOOL_DESCRIPTION = "empty_tool_description"
    INVALID_PARAMETER_SCHEMA = "invalid_parameter_schema"
    SECRET_LITERAL = "secret_literal"


@dataclass(frozen=True, slots=True)
class Message:
    """One ordered chat message template."""

    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise TypeError("message role must be a string")
        if not self.role.strip():
            raise ValueError("message role must not be empty")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("message name must be a string or None")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A JSON-schema-like function interface used by a prompt."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("tool name must be a string")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not isinstance(self.description, str):
            raise TypeError("tool description must be a string")
        if not isinstance(self.parameters, Mapping) or not all(
            isinstance(name, str) for name in self.parameters
        ):
            raise TypeError("tool parameters must be an object with string keys")
        if not isinstance(self.required, tuple) or not all(
            isinstance(name, str) for name in self.required
        ):
            raise TypeError("tool required parameters must be a tuple of strings")
        object.__setattr__(self, "parameters", _freeze_json(self.parameters, "tool parameters"))
        if len(set(self.required)) != len(self.required):
            raise ValueError(f"tool {self.name!r} has duplicate required parameters")
        missing = set(self.required) - self.parameters.keys()
        if missing:
            raise ValueError(
                f"tool {self.name!r} requires undeclared parameters: {', '.join(sorted(missing))}"
            )


@dataclass(frozen=True, slots=True)
class PromptDocument:
    """A versionable chat prompt specification."""

    prompt_id: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str):
            raise TypeError("prompt_id must be a string")
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, Message) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of Message values")
        if not isinstance(self.tools, tuple) or not all(
            isinstance(tool, ToolSpec) for tool in self.tools
        ):
            raise TypeError("tools must be a tuple of ToolSpec values")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be an object")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))
        tool_names = [tool.name for tool in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("tool names must be unique")

    def tool_map(self) -> dict[str, ToolSpec]:
        """Return tools keyed by their stable names."""

        return {tool.name: tool for tool in self.tools}


@dataclass(frozen=True, slots=True)
class Change:
    """One structural or textual difference."""

    kind: ChangeKind
    path: str
    severity: Severity
    summary: str
    before: Any = None
    after: Any = None
    unified_diff: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _freeze_json(self.before, "change.before"))
        object.__setattr__(self, "after", _freeze_json(self.after, "change.after"))


@dataclass(frozen=True, slots=True)
class DiffReport:
    """Versioned comparison result for two prompt documents."""

    before_id: str
    after_id: str
    changes: tuple[Change, ...]

    @property
    def breaking_count(self) -> int:
        return sum(change.severity is Severity.BREAKING for change in self.changes)

    @property
    def warning_count(self) -> int:
        return sum(change.severity is Severity.WARNING for change in self.changes)

    @property
    def compatible(self) -> bool:
        return self.breaking_count == 0


@dataclass(frozen=True, slots=True)
class Finding:
    """One prompt validation result."""

    code: FindingCode
    path: str
    severity: Severity
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validation results for a prompt document."""

    prompt_id: str
    findings: tuple[Finding, ...]

    @property
    def breaking_count(self) -> int:
        return sum(finding.severity is Severity.BREAKING for finding in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(finding.severity is Severity.WARNING for finding in self.findings)

    @property
    def valid(self) -> bool:
        return self.breaking_count == 0
