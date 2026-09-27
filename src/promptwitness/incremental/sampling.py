"""Outcome-free, persisted stratified SRSWOR plans and fixed monotone looks."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .statistics import MAX_POPULATION, integer


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def fixed_looks(population: int) -> tuple[int, ...]:
    integer(population, "population", 1)
    return tuple(
        sorted(
            {
                (population * numerator + denominator - 1) // denominator
                for numerator, denominator in ((1, 16), (1, 8), (1, 4), (1, 2), (3, 4), (1, 1))
            }
        )
    )


@dataclass(frozen=True, slots=True)
class Stratum:
    old_correct: int
    permutation: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.old_correct) is not int or self.old_correct not in (0, 1):
            raise ValueError("reference correctness must be binary")
        if (
            not isinstance(self.permutation, tuple)
            or not self.permutation
            or any(not isinstance(x, str) or not x for x in self.permutation)
        ):
            raise ValueError("strata require a nonempty immutable identity permutation")
        if len(set(self.permutation)) != len(self.permutation):
            raise ValueError("duplicate stratum identity")


@dataclass(frozen=True, slots=True)
class AuditPlan:
    candidate_digest: str
    execution_digest: str
    reference_digest: str
    strata: tuple[Stratum, ...]
    allocations: tuple[tuple[int, ...], ...]
    randomization: str

    def __post_init__(self) -> None:
        for value in (self.candidate_digest, self.execution_digest, self.reference_digest):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("SHA-256 identity required")
        if not isinstance(self.strata, tuple) or not 1 <= len(self.strata) <= 8:
            raise ValueError("one to eight strata required")
        identities = [x for s in self.strata for x in s.permutation]
        if len(set(identities)) != len(identities) or len(identities) > MAX_POPULATION:
            raise ValueError("duplicate identities or unsupported population")
        looks = fixed_looks(len(identities))
        if not isinstance(self.allocations, tuple) or len(self.allocations) != len(looks):
            raise ValueError("fixed look schedule required")
        previous = (0,) * len(self.strata)
        for look, allocation in zip(looks, self.allocations, strict=True):
            if not isinstance(allocation, tuple) or len(allocation) != len(self.strata):
                raise ValueError("one immutable allocation per stratum required")
            if (
                any(
                    type(n) is not int or not previous[h] <= n <= len(self.strata[h].permutation)
                    for h, n in enumerate(allocation)
                )
                or sum(allocation) != look
            ):
                raise ValueError("allocation is not fixed, monotone and within capacity")
            previous = allocation
        if self.randomization not in ("fresh_system_random", "seeded_fixture"):
            raise ValueError("unsupported randomization source")
        expected_reference = {x: s.old_correct for s in self.strata for x in s.permutation}
        if digest(expected_reference) != self.reference_digest:
            raise ValueError("reference identity does not match homogeneous strata")

    @property
    def population(self) -> int:
        return sum(len(s.permutation) for s in self.strata)

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.delta.audit-plan/v1.1",
            "candidate_digest": self.candidate_digest,
            "execution_digest": self.execution_digest,
            "reference_digest": self.reference_digest,
            "randomization": self.randomization,
            "strata": [
                {"old_correct": s.old_correct, "permutation": list(s.permutation)}
                for s in self.strata
            ],
            "allocations": [list(a) for a in self.allocations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuditPlan:
        expected = {
            "format",
            "candidate_digest",
            "execution_digest",
            "reference_digest",
            "randomization",
            "strata",
            "allocations",
        }
        if set(value) != expected or value["format"] != "promptwitness.delta.audit-plan/v1.1":
            raise ValueError("unknown plan format/fields")
        if not isinstance(value["strata"], list) or not isinstance(value["allocations"], list):
            raise ValueError("serialized strata and allocations must be arrays")
        strata = []
        for row in value["strata"]:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"old_correct", "permutation"}
                or not isinstance(row["permutation"], list)
            ):
                raise ValueError("unknown stratum fields")
        if any(not isinstance(allocation, list) for allocation in value["allocations"]):
            raise ValueError("serialized allocations must be arrays")
        for row in value["strata"]:
            strata.append(Stratum(row["old_correct"], tuple(row["permutation"])))
        return cls(
            value["candidate_digest"],
            value["execution_digest"],
            value["reference_digest"],
            tuple(strata),
            tuple(tuple(a) for a in value["allocations"]),
            value["randomization"],
        )


def make_plan(
    reference: Mapping[str, int],
    *,
    candidate_digest: str,
    execution_digest: str,
    risk: Mapping[str, float] | None = None,
    bins_per_old_score: int = 4,
    fixture_seed: int | None = None,
) -> AuditPlan:
    """Freeze allocation and independent random permutations without candidate outcomes.

    Risk is an outcome-free prior, never a replacement for an observed score.
    Default proportional allocations intentionally separate the first certifier
    from learning an allocation policy. Within-stratum means use design weights.
    """
    if (
        not reference
        or len(reference) > MAX_POPULATION
        or any(
            not isinstance(key, str) or not key or type(score) is not int or score not in (0, 1)
            for key, score in reference.items()
        )
    ):
        raise ValueError("bounded nonempty binary reference vector required")
    integer(bins_per_old_score, "bins_per_old_score", 1)
    if bins_per_old_score > 4:
        raise ValueError("at most four risk bins per old score")
    if risk is not None and (
        set(risk) != set(reference)
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in risk.values()
        )
    ):
        raise ValueError("one finite prior probability per reference unit required")
    if fixture_seed is not None:
        integer(fixture_seed, "fixture_seed")
    rng = random.SystemRandom()
    strata = []
    for old in (0, 1):
        members = sorted(
            (key for key in reference if reference[key] == old),
            key=lambda key: (0 if risk is None else risk[key], key),
        )
        bins = min(bins_per_old_score if risk is not None else 1, len(members))
        for h in range(bins):
            group = members[h * len(members) // bins : (h + 1) * len(members) // bins]
            if fixture_seed is None:
                rng.shuffle(group)
            else:
                # Reproducible mechanical fixture only, NOT an SRSWOR source.
                # The seeded_fixture label forbids a live statistical claim.
                group.sort(key=lambda key: digest([fixture_seed, old, h, key]))
            strata.append(Stratum(old, tuple(group)))
    allocations = []
    counts = [0] * len(strata)
    looks = set(fixed_looks(len(reference)))
    for total in range(1, len(reference) + 1):
        available = [h for h, s in enumerate(strata) if counts[h] < len(s.permutation)]
        # Rational integer cross-products avoid floating allocation ties.
        chosen = available[0]
        for h in available[1:]:
            if (counts[h] + 1) * len(strata[chosen].permutation) < (counts[chosen] + 1) * len(
                strata[h].permutation
            ):
                chosen = h
        counts[chosen] += 1
        if total in looks:
            allocations.append(tuple(counts))
    return AuditPlan(
        candidate_digest,
        execution_digest,
        digest(dict(reference)),
        tuple(strata),
        tuple(allocations),
        "fresh_system_random" if fixture_seed is None else "seeded_fixture",
    )
