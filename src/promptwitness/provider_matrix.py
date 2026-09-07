"""Deterministic multi-provider replay matrices for prompt experiments."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution import ExecutionReport, Provider, ScenarioExecutor
from .matrix import Scenario
from .models import PromptDocument
from .providers import ReplayProvider


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One named provider in a matrix."""

    name: str
    provider: Provider
    retries: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or any(char.isspace() for char in self.name)
        ):
            raise ValueError("provider name must be a non-empty token")
        if not callable(self.provider):
            raise TypeError("provider must be callable")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError("retries must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProviderRun:
    """One provider's ordered execution report."""

    name: str
    report: ExecutionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_id": self.report.prompt_id,
            "scenarios": len(self.report.rows),
            "succeeded": self.report.succeeded,
            "failed": self.report.failed,
            "rows": [
                {
                    "scenario_id": row.scenario_id,
                    "prompt_digest": row.prompt_digest,
                    "output_digest": row.output_digest,
                    "attempts": row.attempts,
                    "error": row.error,
                }
                for row in self.report.rows
            ],
        }


@dataclass(frozen=True, slots=True)
class ProviderMatrixReport:
    """All provider runs and pairwise output-digest agreement diagnostics."""

    prompt_id: str
    runs: tuple[ProviderRun, ...]
    pairwise_agreement: float | None

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(run.name for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "prompt_id": self.prompt_id,
            "providers": list(self.providers),
            "pairwise_agreement": self.pairwise_agreement,
            "runs": [run.to_dict() for run in self.runs],
        }


class ProviderMatrix:
    """Execute multiple named providers over one rendered scenario matrix."""

    def __init__(self, providers: Iterable[ProviderSpec]) -> None:
        self.providers = tuple(providers)
        if not self.providers:
            raise ValueError("provider matrix requires at least one provider")
        names = [item.name for item in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")

    def run(
        self,
        document: PromptDocument,
        scenarios: Iterable[Scenario],
        *,
        workers: int = 1,
        scenario_workers: int = 1,
        strict: bool = True,
    ) -> ProviderMatrixReport:
        """Run providers concurrently while retaining deterministic provider order."""

        if not isinstance(document, PromptDocument):
            raise TypeError("document must be a PromptDocument")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if (
            isinstance(scenario_workers, bool)
            or not isinstance(scenario_workers, int)
            or scenario_workers < 1
        ):
            raise ValueError("scenario_workers must be a positive integer")
        scenario_values = tuple(scenarios)
        if not scenario_values:
            raise ValueError("provider matrix requires at least one scenario")

        def execute(spec: ProviderSpec) -> ProviderRun:
            report = ScenarioExecutor().run(
                document,
                scenario_values,
                spec.provider,
                workers=scenario_workers,
                retries=spec.retries,
                strict=strict,
            )
            return ProviderRun(spec.name, report)

        if workers == 1:
            runs = tuple(execute(spec) for spec in self.providers)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                runs = tuple(executor.map(execute, self.providers))
        return ProviderMatrixReport(document.prompt_id, runs, _agreement(runs))


def load_replay_providers(path: str | Path) -> tuple[ProviderSpec, ...]:
    """Load strict, output-safe replay provider definitions from JSON."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load provider matrix: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "promptwitness.replay-providers.v1"
    ):
        raise ValueError("unsupported provider matrix format")
    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("provider matrix providers must be a non-empty array")
    result: list[ProviderSpec] = []
    for index, raw in enumerate(raw_providers, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"provider {index} must be an object")
        unknown = set(raw) - {"name", "responses", "retries"}
        if unknown:
            raise ValueError(f"provider {index} has unknown fields: {', '.join(sorted(unknown))}")
        responses = raw.get("responses")
        if not isinstance(responses, Mapping) or not all(isinstance(key, str) for key in responses):
            raise ValueError(f"provider {index} responses must map string digests to JSON values")
        try:
            result.append(
                ProviderSpec(str(raw["name"]), ReplayProvider(responses), raw.get("retries", 0))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid provider {index}: {error}") from error
    return tuple(result)


def _agreement(runs: Iterable[ProviderRun]) -> float | None:
    values = tuple(runs)
    if len(values) < 2:
        return None
    rows: dict[str, list[str]] = {}
    for run in values:
        for row in run.report.rows:
            if row.output_digest is not None:
                rows.setdefault(row.scenario_id, []).append(row.output_digest)
    comparable = [digests for digests in rows.values() if len(digests) >= 2]
    if not comparable:
        return None
    pairs = 0
    matching = 0
    for digests in comparable:
        counts = Counter(digests)
        total = len(digests) * (len(digests) - 1) // 2
        pairs += total
        matching += sum(value * (value - 1) // 2 for value in counts.values())
    return matching / pairs if pairs else None
