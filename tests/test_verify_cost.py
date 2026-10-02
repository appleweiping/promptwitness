import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "graft"))

from verify_cost import regret, sign_flip, verify_all  # noqa: E402


def test_regret_breaks_ties_uniformly():
    target = [0.3, 0.2, 0.1, 0.0, -0.1]
    # perfect ranking: zero regret
    assert abs(regret([5, 4, 3, 2, 1], target)) < 1e-12
    # all tied: expected top-3 = mean of everything
    flat = regret([0, 0, 0, 0, 0], target)
    assert abs(flat - (0.2 - 0.1)) < 1e-12
    # one clear winner, the rest tied: winner + 2/4 of the mean of the others
    mixed = regret([1, 0, 0, 0, 0], target)
    assert abs(mixed - (0.2 - (0.3 + 2 * 0.05) / 3)) < 1e-12


def test_verify_all_accepts_only_positive_predictions():
    assert verify_all([0.0, -0.1], [0.5, 0.2]) == 0.0
    assert verify_all([0.2, 0.2, 0.1], [0.4, -0.2, 0.9]) == 0.1  # tie at the top: mean of tied targets


def test_sign_flip_extremes():
    assert sign_flip([1.0, 1.0, 1.0, 1.0]) == 2 / 16  # all-positive: only the two extreme sign patterns
    assert sign_flip([1.0, -1.0, 1.0, -1.0]) == 1.0
