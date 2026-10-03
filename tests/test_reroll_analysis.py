import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "graft"))

from reroll_analysis import fmt, load_pool, permutation_tests, pooled_test

EXCHANGEABLE_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EXCHANGEABLE_B = [1.5, 4.5, 6.5, 9.5]  # same mean, similar spread
SHIFTED_B = [-3, -2, -1, 0]  # every value below every value of A
SPREAD_A = list(range(20))
TIGHT_B = [9.4, 9.5, 9.5, 9.6, 9.5]  # same centre as SPREAD_A, almost no spread


def test_exchangeable_labels_give_large_p_values():
    res = permutation_tests(EXCHANGEABLE_A, EXCHANGEABLE_B)
    mean, var = res["mean_difference"], res["log_variance_ratio"]
    assert mean.exact and mean.n_relabelings == math.comb(14, 4)
    assert mean.observed == pytest.approx(0.0)
    assert mean.p_two_sided == 1.0  # zero difference: every relabeling is at least as extreme
    assert var.p_two_sided > 0.3
    assert mean.null_mean == pytest.approx(0.0, abs=1e-12)  # relabeling: E[mean_a - mean_b] = 0


def test_shifted_means_give_the_smallest_attainable_p_value():
    res = permutation_tests(EXCHANGEABLE_A, SHIFTED_B)
    mean = res["mean_difference"]
    # only the observed labelling is as extreme: p_upper = 1 / C(14, 4), doubled for two sides
    assert mean.p_upper == pytest.approx(1 / math.comb(14, 4))
    assert mean.p_two_sided == pytest.approx(2 / math.comb(14, 4))
    assert mean.p_two_sided < 0.01


def test_different_spreads_give_a_small_variance_p_value():
    res = permutation_tests(SPREAD_A, TIGHT_B)
    var = res["log_variance_ratio"]
    assert var.observed > 0  # edits (a) more spread than re-rolls (b)
    assert var.p_two_sided < 0.01
    assert res["mean_difference"].p_two_sided > 0.5  # same centre


def test_random_relabelings_agree_with_enumeration_and_are_seeded():
    exact = permutation_tests(SPREAD_A, TIGHT_B)["log_variance_ratio"]
    sampled = permutation_tests(SPREAD_A, TIGHT_B, max_enumerate=10, n_random=5000, seed=3)
    again = permutation_tests(SPREAD_A, TIGHT_B, max_enumerate=10, n_random=5000, seed=3)
    var = sampled["log_variance_ratio"]
    assert not var.exact and var.n_relabelings == 5000
    assert var.p_two_sided == again["log_variance_ratio"].p_two_sided
    assert var.p_two_sided >= 2 / 5001  # (1 + count) / (1 + draws), doubled
    assert var.p_two_sided < 0.01 and exact.p_two_sided < 0.01
    loose = permutation_tests(EXCHANGEABLE_A, EXCHANGEABLE_B, max_enumerate=10, n_random=5000,
                              seed=3)
    assert loose["log_variance_ratio"].p_two_sided > 0.3


def test_zero_variance_group_counts_as_most_extreme():
    res = permutation_tests([1, 2, 3, 4, 5, 6], [3, 3, 3])
    var = res["log_variance_ratio"]
    assert math.isinf(var.observed) and var.n_infinite >= 1
    assert math.isfinite(var.z())  # clipped to the extreme finite null value


def test_pooled_test_combines_pools():
    differ = permutation_tests(SPREAD_A, TIGHT_B)["log_variance_ratio"]
    same = permutation_tests(EXCHANGEABLE_A, EXCHANGEABLE_B)["log_variance_ratio"]
    strong = pooled_test([differ, differ], n_random=20000, seed=1)
    weak = pooled_test([same, same], n_random=20000, seed=1)
    assert strong["observed_sum_z"] == pytest.approx(2 * differ.z())
    assert strong["p_two_sided"] < 0.01
    assert weak["p_two_sided"] > 0.3


def test_load_pool_requires_identical_base_reads(tmp_path):
    token = {"base_correct": [1, 0, 1], "per_edit": {}, "summary": {"base_accuracy": 2 / 3}}
    rerolls = {"base_correct": [1, 1, 1], "reroll_reads": [], "summary": {}}
    (tmp_path / "m-t-token.json").write_text(json.dumps(token), encoding="utf-8")
    (tmp_path / "m-t-token-rerolls.json").write_text(json.dumps(rerolls), encoding="utf-8")
    with pytest.raises(ValueError, match="base_correct differs"):
        load_pool(tmp_path, "m", "t")


def test_table_rounding_is_half_up():
    assert fmt(0.2025, 1, 100) == "20.3"
    assert fmt(0.0089, 3) == "0.009"
    assert fmt(0.575, 1, 100) == "57.5"
