"""Discrete finite-suite inference, not a future-distribution safety guarantee."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol, cast

TAIL_ALLOWANCE = Fraction(1, 61440)
MAX_POPULATION = 4096


class _HypergeometricTails(Protocol):
    def sf(self, x: int, population: int, successes: int, sampled: int) -> float: ...

    def cdf(self, x: int, population: int, successes: int, sampled: int) -> float: ...


def integer(value: Any, name: str, low: int = 0) -> int:
    """Reject booleans and ambiguous/coerced integer counts."""
    if type(value) is not int or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")
    return value


@dataclass(frozen=True, slots=True)
class CountInterval:
    lower: int
    upper: int
    population: int
    sampled: int
    observed: int

    def __post_init__(self) -> None:
        for field in ("lower", "upper", "population", "sampled", "observed"):
            integer(getattr(self, field), field)
        if not (
            self.observed <= self.sampled <= self.population
            and self.observed
            <= self.lower
            <= self.upper
            <= self.population - self.sampled + self.observed
        ):
            raise ValueError("inconsistent finite-population count bounds")


def exact_tail(
    population: int, successes: int, sampled: int, observed: int, *, upper: bool
) -> Fraction:
    """Integer-combination reference; supported small/medium finite populations."""
    for value, name in (
        (population, "population"),
        (successes, "successes"),
        (sampled, "sampled"),
        (observed, "observed"),
    ):
        integer(value, name)
    if population > MAX_POPULATION or successes > population or sampled > population:
        raise ValueError("population outside supported reference range")
    low, high = max(0, sampled - (population - successes)), min(sampled, successes)
    indices = range(max(low, observed), high + 1) if upper else range(low, min(high, observed) + 1)
    numerator = sum(
        math.comb(successes, x) * math.comb(population - successes, sampled - x) for x in indices
    )
    return Fraction(numerator, math.comb(population, sampled))


@lru_cache(maxsize=32768)
def count_interval(
    population: int, sampled: int, observed: int, allowance: Fraction = TAIL_ALLOWANCE
) -> CountInterval:
    """Invert one-sided tails over integer K, retaining strictly p > allowance.

    SciPy's maintained distribution (Boost CDF/SF) is crossed against exact
    arithmetic in tests. Near-threshold numerical ambiguity uses integer sums.
    The bounded implementation refuses N > 4096 instead of extrapolating a claim.
    """
    integer(population, "population")
    integer(sampled, "sampled")
    integer(observed, "observed")
    if population > MAX_POPULATION or not observed <= sampled <= population:
        raise ValueError("invalid or unsupported hypergeometric counts")
    if not isinstance(allowance, Fraction) or not 0 < allowance < Fraction(1, 2):
        raise ValueError("tail allowance must be a rational in (0, 1/2)")
    if sampled == 0:
        return CountInterval(0, population, population, 0, 0)
    if sampled == population:
        return CountInterval(observed, observed, population, sampled, observed)
    try:
        hypergeom = cast(_HypergeometricTails, import_module("scipy.stats").hypergeom)
    except ImportError as error:
        raise ValueError("install promptwitness[research] for finite-suite inference") from error
    threshold = float(allowance)

    def retained(k: int, *, upper: bool) -> bool:
        value = float(
            hypergeom.sf(observed - 1, population, k, sampled)
            if upper
            else hypergeom.cdf(observed, population, k, sampled)
        )
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ArithmeticError("non-finite hypergeometric tail")
        if abs(value - threshold) <= max(1e-14, 1e-9 * threshold):
            return exact_tail(population, k, sampled, observed, upper=upper) > allowance
        return value > threshold

    feasible_low, feasible_high = observed, population - sampled + observed
    low, high = feasible_low, feasible_high
    while low < high:
        middle = (low + high) // 2
        if retained(middle, upper=True):
            high = middle
        else:
            low = middle + 1
    lower = low
    low, high = feasible_low, feasible_high
    while low < high:
        middle = (low + high + 1) // 2
        if retained(middle, upper=False):
            low = middle
        else:
            high = middle - 1
    return CountInterval(lower, low, population, sampled, observed)


@dataclass(frozen=True, slots=True)
class PairedBounds:
    delta_lower: Fraction
    delta_upper: Fraction
    regression_lower: Fraction
    regression_upper: Fraction
    delta_estimate: Fraction | None
    regression_estimate: Fraction | None
    population: int
    queried: int

    def to_dict(self) -> dict[str, Any]:
        return {
            field: (None if (value := getattr(self, field)) is None else float(value))
            for field in (
                "delta_lower",
                "delta_upper",
                "regression_lower",
                "regression_upper",
                "delta_estimate",
                "regression_estimate",
            )
        } | {
            "population": self.population,
            "queried": self.queried,
            "rational_bounds": {
                field: str(getattr(self, field))
                for field in ("delta_lower", "delta_upper", "regression_lower", "regression_upper")
            },
        }


def combine_counts(strata: tuple[tuple[int, CountInterval], ...]) -> PairedBounds:
    """Each stratum has one known homogeneous old score; R uses whole-suite N."""
    if not strata or len(strata) > 8:
        raise ValueError("one to eight nonempty strata required")
    if any(
        type(old) is not int or old not in (0, 1) or interval.population == 0
        for old, interval in strata
    ):
        raise ValueError("nonempty homogeneous binary-reference strata required")
    population = sum(interval.population for _, interval in strata)
    lower = sum(i.lower if old == 0 else -i.upper for old, i in strata)
    upper = sum(i.upper if old == 0 else -i.lower for old, i in strata)
    reg_low = sum(i.lower for old, i in strata if old == 1)
    reg_high = sum(i.upper for old, i in strata if old == 1)
    fully_represented = all(i.sampled > 0 for _, i in strata)
    estimate = (
        sum(
            (
                (1 if old == 0 else -1) * Fraction(i.population * i.observed, i.sampled)
                for old, i in strata
            ),
            Fraction(0),
        )
        / population
        if fully_represented
        else None
    )
    reg_estimate = (
        sum(
            (Fraction(i.population * i.observed, i.sampled) for old, i in strata if old == 1),
            Fraction(0),
        )
        / population
        if fully_represented
        else None
    )
    return PairedBounds(
        Fraction(lower, population),
        Fraction(upper, population),
        Fraction(reg_low, population),
        Fraction(reg_high, population),
        estimate,
        reg_estimate,
        population,
        sum(i.sampled for _, i in strata),
    )
