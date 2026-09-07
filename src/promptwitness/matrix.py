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
        digest_body = {
            "prompt": document.prompt_id,
            "messages": [
                {"role": message.role, "name": message.name, "content": message.content}
                for message in rendered
            ],
            "variables": list(names),
            "tags": list(scenario.tags),
        }
        digest = hashlib.sha256(
            json.dumps(
                digest_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        result.append(
            RenderedScenario(
                scenario.scenario_id, document.prompt_id, rendered, names, digest, scenario.tags
            )
        )
    return tuple(result)


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
