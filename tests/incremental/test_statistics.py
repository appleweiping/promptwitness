"""Finite-population mathematical checks, not scientific method effects."""

from fractions import Fraction
from itertools import combinations, product
from math import comb

import pytest

pytest.importorskip("scipy")

from promptwitness.incremental.gate import GateStatus, decide
from promptwitness.incremental.statistics import (
    TAIL_ALLOWANCE,
    CountInterval,
    combine_counts,
    count_interval,
    exact_tail,
)


@pytest.mark.parametrize("allowance", [TAIL_ALLOWANCE, Fraction(1, 10), Fraction(1, 20)])
def test_exhaustive_small_population_tail_inversion_and_coverage(allowance):
    # Every feasible total K and observed x at each n, exact rational weights.
    for population in range(1, 13):
        for sampled in range(population + 1):
            for observed in range(sampled + 1):
                interval = count_interval(population, sampled, observed, allowance)
                candidates = range(observed, population - sampled + observed + 1)
                lower = min(
                    k
                    for k in candidates
                    if exact_tail(population, k, sampled, observed, upper=True) > allowance
                )
                upper = max(
                    k
                    for k in candidates
                    if exact_tail(population, k, sampled, observed, upper=False) > allowance
                )
                assert (interval.lower, interval.upper) == (lower, upper)
            for successes in range(population + 1):
                missed_low = missed_high = Fraction(0)
                for x in range(
                    max(0, sampled - population + successes), min(sampled, successes) + 1
                ):
                    probability = Fraction(
                        comb(successes, x) * comb(population - successes, sampled - x),
                        comb(population, sampled),
                    )
                    interval = count_interval(population, sampled, x, allowance)
                    if interval.lower > successes:
                        missed_low += probability
                    if interval.upper < successes:
                        missed_high += probability
                assert missed_low <= allowance and missed_high <= allowance


def test_fixed_error_budget_union_not_campaign_claim():
    assert Fraction(1, 20) == 32 * 6 * 8 * 2 * TAIL_ALLOWANCE
    assert 180 * Fraction(1, 20) != Fraction(1, 20)


@pytest.mark.parametrize("args", [(True, 0, 0), (-1, 0, 0), (10, 11, 0), (10, 3, 4), (4097, 0, 0)])
def test_invalid_counts_fail_closed(args):
    with pytest.raises(ValueError):
        count_interval(*args)


@pytest.mark.parametrize("bad", [0.01, Fraction(0), Fraction(1, 2)])
def test_invalid_allowance(bad):
    with pytest.raises(ValueError):
        count_interval(8, 3, 1, bad)


def test_census_unsampled_empty_and_unconditional_regression():
    assert count_interval(0, 0, 0) == CountInterval(0, 0, 0, 0, 0)
    assert count_interval(9, 0, 0).upper == 9
    assert count_interval(9, 9, 2).lower == count_interval(9, 9, 2).upper == 2
    bounds = combine_counts(((1, count_interval(8, 8, 1)), (0, count_interval(2, 2, 1))))
    assert bounds.delta_lower == bounds.delta_upper == 0
    assert bounds.regression_lower == bounds.regression_upper == Fraction(1, 10)
    assert bounds.regression_lower != Fraction(1, 8)
    assert decide(bounds) == GateStatus.INELIGIBLE
    missing = combine_counts(((1, count_interval(8, 0, 0)), (0, count_interval(2, 1, 1))))
    assert missing.delta_estimate is None and missing.regression_estimate is None


def test_weighted_fixed_look_estimate_design_unbiased_by_enumeration():
    old_correct = (0, 1, 0, 0)
    old_incorrect = (1, 0, 1)
    population_delta = Fraction(sum(old_incorrect) - sum(1 - x for x in old_correct), 7)
    estimates = []
    for a, b in product(combinations(old_correct, 2), combinations(old_incorrect, 1)):
        estimates.append(
            combine_counts(
                (
                    (1, count_interval(4, 2, sum(1 - x for x in a))),
                    (0, count_interval(3, 1, sum(b))),
                )
            ).delta_estimate
        )
    assert sum(estimates) / len(estimates) == population_delta


def test_rational_census_thresholds():
    eligible = combine_counts(((1, count_interval(100, 100, 1)),))
    assert decide(eligible) == GateStatus.ELIGIBLE
    worse = combine_counts(((1, count_interval(100, 100, 2)),))
    assert decide(worse) == GateStatus.INELIGIBLE
    uncertain = combine_counts(((1, count_interval(100, 10, 0)),))
    assert decide(uncertain) == GateStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    "strata", [(), ((True, CountInterval(0, 0, 1, 1, 0)),), ((0, CountInterval(0, 0, 0, 0, 0)),)]
)
def test_bad_strata(strata):
    with pytest.raises(ValueError):
        combine_counts(strata)


def test_medium_population_matches_direct_distribution_inversion():
    from scipy.stats import hypergeom

    for population, sampled, observed in [
        (256, 192, 0),
        (512, 256, 11),
        (4096, 3072, 80),
        (512, 128, 128),
    ]:
        interval = count_interval(population, sampled, observed)
        feasible = range(observed, population - sampled + observed + 1)
        lower = min(
            k
            for k in feasible
            if hypergeom.sf(observed - 1, population, k, sampled) > float(TAIL_ALLOWANCE)
        )
        upper = max(
            k
            for k in feasible
            if hypergeom.cdf(observed, population, k, sampled) > float(TAIL_ALLOWANCE)
        )
        assert (interval.lower, interval.upper) == (lower, upper)
