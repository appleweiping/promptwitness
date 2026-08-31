"""Deterministic structural comparison for prompt documents."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import Change, ChangeKind, DiffReport, Message, PromptDocument, Severity, ToolSpec
from .paths import json_pointer
from .variables import inspect_variables


@dataclass(frozen=True, slots=True)
class DiffOptions:
    """Compatibility policy for behavior-oriented prompt changes."""

    added_message_severity: Severity = Severity.WARNING
    content_change_severity: Severity = Severity.WARNING
    include_metadata: bool = True
    context_lines: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.added_message_severity, Severity):
            raise TypeError("added_message_severity must be a Severity")
        if not isinstance(self.content_change_severity, Severity):
            raise TypeError("content_change_severity must be a Severity")
        if type(self.include_metadata) is not bool:
            raise TypeError("include_metadata must be a boolean")
        if isinstance(self.context_lines, bool) or not isinstance(self.context_lines, int):
            raise TypeError("context_lines must be an integer")
        if self.context_lines < 0:
            raise ValueError("context_lines must be non-negative")


def compare_prompts(
    before: PromptDocument,
    after: PromptDocument,
    options: DiffOptions | None = None,
) -> DiffReport:
    """Compare documents using positional messages and named tools.

    Message position is the only stable identity in schema version 1. Inserting
    a message can therefore produce multiple changes; the report keeps this
    behavior explicit instead of guessing which repeated role was intended.
    """

    active = options or DiffOptions()
    changes: list[Change] = []
    changes.extend(_compare_messages(before.messages, after.messages, active))
    changes.extend(_compare_variables(before, after))
    changes.extend(_compare_tools(before.tool_map(), after.tool_map()))
    if active.include_metadata and not _json_equal(before.metadata, after.metadata):
        changes.append(
            Change(
                ChangeKind.METADATA,
                json_pointer("metadata"),
                Severity.INFO,
                "prompt metadata changed",
                before.metadata,
                after.metadata,
            )
        )
    return DiffReport(before.prompt_id, after.prompt_id, tuple(changes))


def _compare_messages(
    before: tuple[Message, ...], after: tuple[Message, ...], options: DiffOptions
) -> list[Change]:
    changes: list[Change] = []
    common = min(len(before), len(after))
    for index in range(common):
        old, new = before[index], after[index]
        if old.role != new.role:
            changes.append(
                Change(
                    ChangeKind.MESSAGE_ROLE,
                    json_pointer("messages", index, "role"),
                    Severity.BREAKING,
                    f"message role changed from {old.role!r} to {new.role!r}",
                    old.role,
                    new.role,
                )
            )
        if old.name != new.name:
            changes.append(
                Change(
                    ChangeKind.MESSAGE_NAME,
                    json_pointer("messages", index, "name"),
                    Severity.BREAKING,
                    "message name changed",
                    old.name,
                    new.name,
                )
            )
        if old.content != new.content:
            changes.append(
                Change(
                    ChangeKind.MESSAGE_CONTENT,
                    json_pointer("messages", index, "content"),
                    options.content_change_severity,
                    "message content changed",
                    old.content,
                    new.content,
                    _unified_content_diff(old.content, new.content, options.context_lines),
                )
            )
    for index in range(common, len(before)):
        changes.append(
            Change(
                ChangeKind.MESSAGE_REMOVED,
                json_pointer("messages", index),
                Severity.BREAKING,
                f"{before[index].role!r} message removed",
                _message_value(before[index]),
                None,
            )
        )
    for index in range(common, len(after)):
        changes.append(
            Change(
                ChangeKind.MESSAGE_ADDED,
                json_pointer("messages", index),
                options.added_message_severity,
                f"{after[index].role!r} message added",
                None,
                _message_value(after[index]),
            )
        )
    return changes


def _compare_variables(before: PromptDocument, after: PromptDocument) -> list[Change]:
    old_names = _document_variables(before)
    new_names = _document_variables(after)
    changes: list[Change] = []
    for name in sorted(new_names - old_names):
        changes.append(
            Change(
                ChangeKind.VARIABLE_ADDED,
                json_pointer("variables", name),
                Severity.BREAKING,
                f"new render input {name!r} is required",
                None,
                name,
            )
        )
    for name in sorted(old_names - new_names):
        changes.append(
            Change(
                ChangeKind.VARIABLE_REMOVED,
                json_pointer("variables", name),
                Severity.WARNING,
                f"render input {name!r} is no longer used",
                name,
                None,
            )
        )
    return changes


def _compare_tools(before: dict[str, ToolSpec], after: dict[str, ToolSpec]) -> list[Change]:
    changes: list[Change] = []
    for name in sorted(before.keys() - after.keys()):
        changes.append(
            Change(
                ChangeKind.TOOL_REMOVED,
                json_pointer("tools", name),
                Severity.BREAKING,
                f"tool {name!r} was removed",
                _tool_value(before[name]),
                None,
            )
        )
    for name in sorted(after.keys() - before.keys()):
        changes.append(
            Change(
                ChangeKind.TOOL_ADDED,
                json_pointer("tools", name),
                Severity.INFO,
                f"tool {name!r} was added",
                None,
                _tool_value(after[name]),
            )
        )
    for name in sorted(before.keys() & after.keys()):
        changes.extend(_compare_tool(before[name], after[name]))
    return changes


def _compare_tool(before: ToolSpec, after: ToolSpec) -> list[Change]:
    path = ("tools", before.name)
    changes: list[Change] = []
    if before.description != after.description:
        changes.append(
            Change(
                ChangeKind.TOOL_DESCRIPTION,
                json_pointer(*path, "description"),
                Severity.INFO,
                "tool description changed",
                before.description,
                after.description,
            )
        )
    old_parameters, new_parameters = before.parameters, after.parameters
    for parameter in sorted(old_parameters.keys() - new_parameters.keys()):
        changes.append(
            Change(
                ChangeKind.TOOL_PARAMETER_REMOVED,
                json_pointer(*path, "parameters", parameter),
                Severity.BREAKING,
                f"tool parameter {parameter!r} was removed",
                old_parameters[parameter],
                None,
            )
        )
    for parameter in sorted(new_parameters.keys() - old_parameters.keys()):
        severity = Severity.BREAKING if parameter in after.required else Severity.INFO
        changes.append(
            Change(
                ChangeKind.TOOL_PARAMETER_ADDED,
                json_pointer(*path, "parameters", parameter),
                severity,
                f"{'required' if severity is Severity.BREAKING else 'optional'} tool parameter "
                f"{parameter!r} was added",
                None,
                new_parameters[parameter],
            )
        )
    for parameter in sorted(old_parameters.keys() & new_parameters.keys()):
        if not _json_equal(old_parameters[parameter], new_parameters[parameter]):
            changes.append(
                Change(
                    ChangeKind.TOOL_PARAMETER_CHANGED,
                    json_pointer(*path, "parameters", parameter),
                    Severity.BREAKING,
                    f"schema for tool parameter {parameter!r} changed",
                    old_parameters[parameter],
                    new_parameters[parameter],
                )
            )
    for parameter in sorted(set(after.required) - set(before.required)):
        changes.append(
            Change(
                ChangeKind.TOOL_REQUIRED_ADDED,
                json_pointer(*path, "required", parameter),
                Severity.BREAKING,
                f"tool parameter {parameter!r} became required",
                False,
                True,
            )
        )
    for parameter in sorted(set(before.required) - set(after.required)):
        changes.append(
            Change(
                ChangeKind.TOOL_REQUIRED_REMOVED,
                json_pointer(*path, "required", parameter),
                Severity.WARNING,
                f"tool parameter {parameter!r} is no longer required",
                True,
                False,
            )
        )
    return changes


def _document_variables(document: PromptDocument) -> frozenset[str]:
    names: set[str] = set()
    for message in document.messages:
        names.update(inspect_variables(message.content).names)
    return frozenset(names)


def _unified_content_diff(before: str, after: str, context: int) -> tuple[str, ...]:
    return tuple(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=context,
        )
    )


def _message_value(message: Message) -> dict[str, str | None]:
    return {"role": message.role, "content": message.content, "name": message.name}


def _tool_value(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "required": list(tool.required),
    }


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` ambiguity."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and left == right
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and len(left) == len(right)
            and all(_json_equal(old, new) for old, new in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right
