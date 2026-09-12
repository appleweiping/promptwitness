"""Input-only, deterministic integer-search and directed-path baselines.

No reference answer, ProcedureCase, filesystem, provider or executable-expression
API is accepted. These small authored algorithms are not language models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .procedure_json_data import canonical_bytes, integer, object_fields, sequence, string

BASELINE_VERSION = "promptwitness.procedure-baselines/v1"
_METHODS = ("bus", "train", "plane", "ferry")
_LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"


@dataclass(frozen=True)
class ProcedureBaselineLimits:
    """Deterministic work/encoded-data caps, not elapsed-time or RSS guarantees."""

    max_states: int = 50_000
    max_transitions: int = 250_000
    max_input_bytes: int = 8 * 1024 * 1024
    max_edges: int = 100_000
    max_prediction_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_states", 1_000_000),
            ("max_transitions", 5_000_000),
            ("max_input_bytes", 16 * 1024 * 1024),
            ("max_edges", 100_000),
            ("max_prediction_bytes", 4 * 1024 * 1024),
        ):
            integer(getattr(self, name), minimum=1, maximum=ceiling)

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "max_states",
                "max_transitions",
                "max_input_bytes",
                "max_edges",
                "max_prediction_bytes",
            )
        }


@dataclass(frozen=True)
class ProcedureBaselineResult:
    algorithm: str
    status: str
    prediction: str | None
    reason: str | None
    input_sha256: str
    states: int
    transitions: int
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.algorithm not in ("countdown-integer-dfs/v1", "functional-directed-walk/v1"):
            raise ValueError("unknown procedural baseline algorithm")
        if self.status not in ("solved", "unsatisfiable", "exhausted", "unsupported"):
            raise ValueError("unknown procedural baseline status")
        if len(self.input_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.input_sha256
        ):
            raise ValueError("baseline input digest is invalid")
        integer(self.states, maximum=1_000_000)
        integer(self.transitions, maximum=5_000_000)
        sequence(self.steps)
        for step in self.steps:
            string(step)
        if self.status == "solved":
            string(self.prediction)
            if self.reason is not None:
                raise ValueError("solved result cannot have an unresolved reason")
        elif self.prediction is not None or self.steps or not isinstance(self.reason, str):
            raise ValueError("unresolved baseline cannot publish a partial solution")
        object.__setattr__(self, "steps", tuple(self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BASELINE_VERSION,
            "algorithm": self.algorithm,
            "status": self.status,
            "prediction": self.prediction,
            "reason": self.reason,
            "input_sha256": self.input_sha256,
            "states": self.states,
            "transitions": self.transitions,
            "steps": list(self.steps),
        }


class _Exhausted(Exception):
    pass


class _Work:
    def __init__(self, limits: ProcedureBaselineLimits) -> None:
        self.limits = limits
        self.states = 0
        self.transitions = 0

    def state(self) -> None:
        if self.states >= self.limits.max_states:
            raise _Exhausted("state_budget")
        self.states += 1

    def transition(self) -> None:
        if self.transitions >= self.limits.max_transitions:
            raise _Exhausted("transition_budget")
        self.transitions += 1

    def result(
        self,
        algorithm: str,
        digest: str,
        status: str,
        reason: str | None = None,
        steps: tuple[str, ...] = (),
        tag: str = "Solution",
    ) -> ProcedureBaselineResult:
        prediction = None
        if status == "solved":
            # Admission precedes joining potentially large route strings.
            size = len(f"<{tag}>\n\n</{tag}>".encode())
            size += sum(len(step.encode("utf-8")) for step in steps) + max(0, len(steps) - 1)
            if size > self.limits.max_prediction_bytes:
                status, reason, steps = "exhausted", "prediction_budget", ()
            else:
                prediction = f"<{tag}>\n" + "\n".join(steps) + f"\n</{tag}>"
        return ProcedureBaselineResult(
            algorithm, status, prediction, reason, digest, self.states, self.transitions, steps
        )


def _admit(inputs: Mapping[str, Any], limits: ProcedureBaselineLimits | None) -> tuple[_Work, str]:
    if limits is not None and not isinstance(limits, ProcedureBaselineLimits):
        raise ValueError("limits must be ProcedureBaselineLimits")
    actual = limits if limits is not None else ProcedureBaselineLimits()
    encoded = canonical_bytes(inputs, limit=actual.max_input_bytes)
    return _Work(actual), hashlib.sha256(encoded).hexdigest()


def solve_countdown(
    inputs: Mapping[str, Any], *, limits: ProcedureBaselineLimits | None = None
) -> ProcedureBaselineResult:
    """Use all four operand occurrences with exact, positive integers <= 2000.

    A depth-first search expands sorted multisets in a fixed order. Only proven
    dead multisets are memoized. A budget stop is unresolved, never a proof that
    the target is unreachable. Leaf checks still consume one state admission.
    """
    work, digest = _admit(inputs, limits)
    data = object_fields(inputs, "numbers target min_intermediate max_intermediate")
    values = tuple(
        integer(value, minimum=1, maximum=2000)
        for value in sequence(data["numbers"], minimum=4, maximum=4)
    )
    target = integer(data["target"], minimum=1, maximum=2000)
    if (
        type(data["min_intermediate"]) is not int
        or data["min_intermediate"] != 1
        or type(data["max_intermediate"]) is not int
        or data["max_intermediate"] != 2000
    ):
        raise ValueError("Countdown requires the positive-integer 1..2000 policy")
    dead: set[tuple[int, ...]] = set()

    def search(state: tuple[int, ...]) -> tuple[str, ...] | None:
        state = tuple(sorted(state))
        if state in dead:
            return None
        work.state()
        if len(state) == 1:
            if state[0] == target:
                return ()
            dead.add(state)
            return None
        for left_index, left in enumerate(state):
            for right_index in range(left_index + 1, len(state)):
                right = state[right_index]
                remainder = tuple(
                    value
                    for index, value in enumerate(state)
                    if index not in (left_index, right_index)
                )
                # Division is considered only if exact; no floats, eval or AST.
                for a, operation, b in (
                    (left, "+", right),
                    (left, "*", right),
                    (left, "-", right),
                    (right, "-", left),
                    (left, "/", right),
                    (right, "/", left),
                ):
                    work.transition()
                    if operation == "+":
                        value = a + b
                    elif operation == "*":
                        value = a * b
                    elif operation == "-":
                        value = a - b
                    elif a % b:
                        continue
                    else:
                        value = a // b
                    if not 1 <= value <= 2000:
                        continue
                    suffix = search((*remainder, value))
                    if suffix is not None:
                        return (f"{a} {operation} {b} = {value}", *suffix)
        dead.add(state)
        return None

    algorithm = "countdown-integer-dfs/v1"
    try:
        solution = search(values)
    except _Exhausted as exc:
        return work.result(algorithm, digest, "exhausted", str(exc))
    if solution is None:
        return work.result(algorithm, digest, "unsatisfiable", "complete_search_no_solution")
    return work.result(algorithm, digest, "solved", steps=solution)


def walk_graph_path(
    inputs: Mapping[str, Any], *, limits: ProcedureBaselineLimits | None = None
) -> ProcedureBaselineResult:
    """Follow the unique labeled outgoing edge; stop at the first destination.

    One outgoing edge per source is an explicit input contract, not an inferred
    property of natural-language context. Cycle/dead-end proofs apply only to
    this directed functional graph. Unrepresentable line/tag names are explicit.
    """
    work, digest = _admit(inputs, limits)
    data = object_fields(inputs, "edges source target problem_description context_nl question_nl")
    for name in ("source", "target", "problem_description", "context_nl", "question_nl"):
        string(data[name])
    outgoing: dict[str, tuple[str, str]] = {}
    unsupported = False
    for edge in sequence(data["edges"], maximum=work.limits.max_edges):
        object_fields(edge, "source method target")
        source, target = string(edge["source"]), string(edge["target"])
        if edge["method"] not in _METHODS:
            raise ValueError("graph method is unsupported")
        if source in outgoing:
            raise ValueError("graph requires exactly one outgoing edge per source")
        outgoing[source] = (edge["method"], target)
        for city in (source, target):
            unsupported |= (
                any(char in city for char in _LINE_BREAKS)
                or "<Route>" in city
                or "</Route>" in city
            )
    algorithm = "functional-directed-walk/v1"
    if unsupported:
        return work.result(algorithm, digest, "unsupported", "route_text_cannot_represent_city")
    current = data["source"]
    visited: set[str] = set()
    steps: list[str] = []
    output_bytes = len(b"<Route>\n\n</Route>")
    try:
        while True:
            work.state()
            if current == data["target"]:
                return work.result(algorithm, digest, "solved", steps=tuple(steps), tag="Route")
            if current in visited:
                return work.result(
                    algorithm, digest, "unsatisfiable", "directed_cycle_before_target"
                )
            visited.add(current)
            if current not in outgoing:
                return work.result(algorithm, digest, "unsatisfiable", "directed_dead_end")
            work.transition()
            method, target = outgoing[current]
            size = (
                len(current.encode("utf-8"))
                + len(target.encode("utf-8"))
                + len(method)
                + len("From , take a  to .")
                + int(bool(steps))
            )
            if output_bytes + size > work.limits.max_prediction_bytes:
                raise _Exhausted("prediction_budget")
            steps.append(f"From {current}, take a {method} to {target}.")
            output_bytes += size
            current = target
    except _Exhausted as exc:
        return work.result(algorithm, digest, "exhausted", str(exc))
