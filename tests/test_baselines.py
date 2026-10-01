"""Checks on the memorization baselines (Park et al. Eqs. 4-5).

    python tests/test_baselines.py     # or: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.baselines import memorization_accuracy


def test_matches_closed_form():
    n, l = 16, np.array([1, 10, 50, 500], dtype=float)
    q = (n - 1) / n

    np.testing.assert_allclose(
        memorization_accuracy(l, n, shots=1), 1 - q**l, rtol=1e-12
    )
    np.testing.assert_allclose(
        memorization_accuracy(l, n, shots=2),
        np.clip((1 - q**l) - l * (1 / n) * q ** (l - 1), 0, 1),
        rtol=1e-12,
    )


def test_matches_empirical_sampling():
    """Eq. 4 should match uniform-with-replacement sampling, its stated model."""
    rng = np.random.default_rng(0)
    n, l, trials = 10, 20, 40000

    draws = rng.integers(0, n, size=(trials, l))
    # Probability that an arbitrary fixed node appears at least once / twice.
    hits = (draws == 0).sum(axis=1)
    empirical1 = float((hits >= 1).mean())
    empirical2 = float((hits >= 2).mean())

    assert abs(memorization_accuracy([l], n, shots=1)[0] - empirical1) < 0.01
    assert abs(memorization_accuracy([l], n, shots=2)[0] - empirical2) < 0.01


def test_monotonic_and_bounded():
    l = np.arange(1, 2000, dtype=float)
    for shots in (1, 2):
        p = memorization_accuracy(l, 16, shots=shots)
        assert np.all(p >= 0) and np.all(p <= 1)
        assert np.all(np.diff(p) >= -1e-12), "should be non-decreasing in context"

    # The 2-shot requirement is strictly harder.
    assert np.all(
        memorization_accuracy(l, 16, shots=2) <= memorization_accuracy(l, 16, shots=1)
    )


def test_rejects_bad_shots():
    try:
        memorization_accuracy([10], 16, shots=3)
    except ValueError:
        return
    raise AssertionError("shots=3 should raise")


if __name__ == "__main__":
    test_matches_closed_form()
    test_matches_empirical_sampling()
    test_monotonic_and_bounded()
    test_rejects_bad_shots()
    print("memorization baseline tests PASSED")
