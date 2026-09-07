"""Auditable prompt rendering matrices for variable scenarios."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .models import Message, PromptDocument
from .variables import inspect_variables, render_template


@dataclass(frozen=True, slots=True)
class Scenario:
    """One named JSON-compatible variable assignment."""

    scenario_id: str
    values: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if not isinstance(self.values, Mapping) or not all(
            isinstance(key, str) for key in self.values
        ):
            raise TypeError("scenario values must be an object with string keys")
        if not isinstance(self.tags, tuple) or not all(
            isinstance(tag, str) and tag for tag in self.tags
        ):
            raise TypeError("scenario tags must be a tuple of non-empty strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("scenario tags must be unique")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class RenderedScenario:
    """One rendered scenario with a reproducible content digest."""

    scenario_id: str
    prompt_id: str
    messages: tuple[Message, ...]
    variables: tuple[str, ...]
    digest: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatrixDiff:
    """Regression comparison for one scenario ID."""

    scenario_id: str
    before_digest: str | None
    after_digest: str | None

    @property
    def changed(self) -> bool:
        """Whether the scenario was added, removed, or rendered differently."""
        return self.before_digest != self.after_digest


@dataclass(frozen=True, slots=True)
class MatrixArtifact:
    """Persisted rendering matrix with an authenticated row inventory."""

    prompt_id: str
    rows: tuple[RenderedScenario, ...]
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("matrix prompt_id must be a non-empty string")
        if not isinstance(self.rows, tuple) or not all(
            isinstance(row, RenderedScenario) for row in self.rows
        ):
            raise TypeError("matrix rows must be a tuple of RenderedScenario values")
        if len({row.scenario_id for row in self.rows}) != len(self.rows):
            raise ValueError("matrix scenario IDs must be unique")
        if any(row.prompt_id != self.prompt_id for row in self.rows):
            raise ValueError("matrix rows must use the artifact prompt ID")
        expected = _matrix_digest(self.prompt_id, self.rows)
        if self.digest != expected:
            raise ValueError("matrix artifact digest does not match its rows")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation including rendered content."""
        return {
            "format": 1,
            "prompt_id": self.prompt_id,
            "digest": self.digest,
            "rows": [
                {
                    "scenario_id": row.scenario_id,
                    "messages": [
                        {"role": message.role, "name": message.name, "content": message.content}
                        for message in row.messages
                    ],
                    "variables": list(row.variables),
                    "digest": row.digest,
                    "tags": list(row.tags),
                }
                for row in self.rows
            ],
        }

    def save(self, path: str) -> None:
        """Write a stable UTF-8 matrix artifact."""
        from pathlib import Path

        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> MatrixArtifact:
        """Load and authenticate a matrix artifact."""
        from pathlib import Path

        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read matrix artifact {path}: {error}") from error
        if not isinstance(raw, dict) or raw.get("format") != 1:
            raise ValueError("unsupported matrix artifact format")
        prompt_id = raw.get("prompt_id")
        digest = raw.get("digest")
        rows_value = raw.get("rows")
        if (
            not isinstance(prompt_id, str)
            or not isinstance(digest, str)
            or not isinstance(rows_value, list)
        ):
            raise ValueError("invalid matrix artifact fields")
        rows: list[RenderedScenario] = []
        for index, value in enumerate(rows_value, start=1):
            if not isinstance(value, dict):
                raise ValueError(f"matrix row {index} must be an object")
            messages_value = value.get("messages")
            variables = value.get("variables")
            tags = value.get("tags")
            if not isinstance(value.get("scenario_id"), str) or not isinstance(
                messages_value, list
            ):
                raise ValueError(f"invalid matrix row {index}")
            if not isinstance(variables, list) or not all(
                isinstance(item, str) for item in variables
            ):
                raise ValueError(f"invalid matrix row variables {index}")
            if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
                raise ValueError(f"invalid matrix row tags {index}")
            messages: list[Message] = []
            for message in messages_value:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(f"invalid matrix row message {index}")
                name = message.get("name")
                if name is not None and not isinstance(name, str):
                    raise ValueError(f"invalid matrix row message name {index}")
                messages.append(Message(message["role"], message["content"], name))
            rows.append(
                RenderedScenario(
                    value["scenario_id"],
                    prompt_id,
                    tuple(messages),
                    tuple(variables),
                    value.get("digest", ""),
                    tuple(tags),
                )
            )
            if rows[-1].digest != _scenario_digest(prompt_id, rows[-1]):
                raise ValueError(f"matrix row {index} digest does not match rendered content")
        return cls(prompt_id, tuple(rows), digest)


def render_matrix(
    document: PromptDocument,
    scenarios: Iterable[Scenario],
    *,
    strict: bool = True,
) -> tuple[RenderedScenario, ...]:
    """Render a prompt over unique scenarios in input order.

    The source document is not mutated. Digests cover the prompt ID, rendered
    message roles/names/content, variable names, and scenario tags, making a
    matrix row suitable as a cache key without storing variable secrets.
    """
    if not isinstance(document, PromptDocument):
        raise TypeError("document must be a PromptDocument")
    materialized = tuple(scenarios)
    if not all(isinstance(scenario, Scenario) for scenario in materialized):
        raise TypeError("scenarios must contain Scenario values")
    ids = [scenario.scenario_id for scenario in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
    names = tuple(
        sorted(
            {
                name
                for message in document.messages
                for name in inspect_variables(message.content).names
            }
        )
    )
    result: list[RenderedScenario] = []
    for scenario in materialized:
        rendered = tuple(
            Message(
                message.role,
                render_template(message.content, scenario.values, strict=strict),
                message.name,
            )
            for message in document.messages
        )
        digest = _scenario_digest(
            document.prompt_id,
            RenderedScenario(
                scenario.scenario_id, document.prompt_id, rendered, names, "", scenario.tags
            ),
        )
        result.append(
            RenderedScenario(
                scenario.scenario_id, document.prompt_id, rendered, names, digest, scenario.tags
            )
        )
    return tuple(result)


def save_matrix(rows: Iterable[RenderedScenario], path: str) -> MatrixArtifact:
    """Authenticate and persist already rendered rows."""
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("matrix must contain at least one rendered scenario")
    prompt_ids = {row.prompt_id for row in materialized}
    if len(prompt_ids) != 1:
        raise ValueError("matrix rows must use one prompt ID")
    artifact = MatrixArtifact(
        next(iter(prompt_ids)), materialized, _matrix_digest(next(iter(prompt_ids)), materialized)
    )
    artifact.save(path)
    return artifact


def _matrix_digest(prompt_id: str, rows: tuple[RenderedScenario, ...]) -> str:
    payload = {
        "prompt_id": prompt_id,
        "rows": [
            {"scenario_id": row.scenario_id, "digest": row.digest, "tags": list(row.tags)}
            for row in rows
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _scenario_digest(prompt_id: str, row: RenderedScenario) -> str:
    """Hash rendered content and non-secret matrix metadata."""
    body = {
        "prompt": prompt_id,
        "messages": [
            {"role": message.role, "name": message.name, "content": message.content}
            for message in row.messages
        ],
        "variables": list(row.variables),
        "tags": list(row.tags),
    }
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compare_matrices(
    before: Iterable[RenderedScenario],
    after: Iterable[RenderedScenario],
) -> tuple[MatrixDiff, ...]:
    """Compare rendered scenarios by ID in stable order."""
    before_items = tuple(before)
    after_items = tuple(after)
    before_map = {item.scenario_id: item.digest for item in before_items}
    after_map = {item.scenario_id: item.digest for item in after_items}
    if len(before_map) != len(before_items) or len(after_map) != len(after_items):
        raise ValueError("rendered scenario IDs must be unique")
    return tuple(
        MatrixDiff(identifier, before_map.get(identifier), after_map.get(identifier))
        for identifier in sorted(set(before_map) | set(after_map))
    )
