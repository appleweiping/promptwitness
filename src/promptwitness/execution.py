"""Deterministic local execution of rendered prompt scenarios."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .matrix import RenderedScenario, Scenario, render_matrix
from .models import PromptDocument


@dataclass(frozen=True, slots=True)
class ExecutionRow:
    """One provider result linked to a rendered scenario digest."""

    scenario_id: str
    prompt_digest: str
    output: Any = None
    output_digest: str | None = None
    attempts: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the provider returned a digest-authenticated result."""
        return self.error is None and self.output_digest is not None


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Ordered provider results and retry/error accounting."""

    prompt_id: str
    rows: tuple[ExecutionRow, ...]

    @property
    def succeeded(self) -> int:
        return sum(row.succeeded for row in self.rows)

    @property
    def failed(self) -> int:
        return len(self.rows) - self.succeeded

    @property
    def complete(self) -> bool:
        return self.failed == 0


Provider = Callable[[RenderedScenario], Any]


class ScenarioExecutor:
    """Run one provider over a scenario matrix with ordered retries."""

    def run(
        self,
        document: PromptDocument,
        scenarios: Iterable[Scenario],
        provider: Provider,
        *,
        workers: int = 1,
        retries: int = 0,
        strict: bool = True,
        fail_fast: bool = False,
    ) -> ExecutionReport:
        """Render and execute scenarios without exposing variable assignments."""
        if not callable(provider):
            raise TypeError("provider must be callable")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        rendered = render_matrix(document, scenarios, strict=strict)

        def invoke(row: RenderedScenario) -> ExecutionRow:
            last_error: Exception | None = None
            for attempt in range(1, retries + 2):
                try:
                    output = provider(row)
                    encoded = json.dumps(
                        output,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    return ExecutionRow(
                        row.scenario_id,
                        row.digest,
                        output,
                        hashlib.sha256(encoded).hexdigest(),
                        attempt,
                    )
                except Exception as error:  # provider failures are reported per scenario
                    last_error = error
            if fail_fast and last_error is not None:
                raise ValueError(
                    f"provider failed for scenario {row.scenario_id!r}: {last_error}"
                ) from last_error
            return ExecutionRow(
                row.scenario_id,
                row.digest,
                attempts=retries + 1,
                error=(
                    f"{type(last_error).__name__}: {last_error}"
                    if last_error
                    else "provider failed"
                ),
            )

        if workers == 1:
            rows = tuple(invoke(row) for row in rendered)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = tuple(executor.map(invoke, rendered))
        return ExecutionReport(document.prompt_id, rows)
