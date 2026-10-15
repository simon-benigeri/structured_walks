"""Tests for the cyclic-order scoring behind Step 0.6 (Fig. 3)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis import chance_level, cycle_overlap, cyclic_order_score
from analysis.circular import angular_order


def _circle(n, order=None):
    """Points evenly spaced on a circle, laid out in the given cyclic order."""
    order = list(range(n)) if order is None else list(order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.zeros((n, 2))
    for position, node in enumerate(order):
        points[node] = [np.cos(angles[position]), np.sin(angles[position])]
    return points


def test_a_circle_scores_perfectly_on_its_own_order():
    points = _circle(7)
    assert cyclic_order_score(points, range(7)) == 1.0


def test_score_ignores_rotation_and_reflection():
    """A circle traced from any start, in either direction, is the same circle."""
    rotated = _circle(7, order=[3, 4, 5, 6, 0, 1, 2])
    reflected = _circle(7, order=list(reversed(range(7))))
    assert cyclic_order_score(rotated, range(7)) == 1.0
    assert cyclic_order_score(reflected, range(7)) == 1.0


def test_a_different_cycle_scores_below_one():
    # Lay the points out in one order, score against an unrelated one.
    points = _circle(7, order=[0, 2, 4, 6, 1, 3, 5])
    assert cyclic_order_score(points, range(7)) < 1.0


def test_two_orderings_can_be_scored_independently():
    """The Fig. 3 claim needs one plane to score high on exactly one ordering."""
    ring = list(range(7))
    calendar = [0, 3, 6, 2, 5, 1, 4]
    points = _circle(7, order=calendar)
    assert cyclic_order_score(points, calendar) == 1.0
    assert cyclic_order_score(points, ring) < 1.0


def test_overlap_counts_shared_adjacencies():
    # [0,1,2,3] and [0,1,3,2] share the pairs {0,1} and {2,3}: 2 of 4.
    assert cycle_overlap([0, 1, 3, 2], [0, 1, 2, 3]) == 0.5
    assert cycle_overlap([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_chance_level_is_the_bar_to_beat():
    """With few nodes, chance is high enough that it has to be reported."""
    assert chance_level(7) == 2 / 6
    assert chance_level(50) < chance_level(7)


def test_angular_order_is_invariant_to_translation():
    """Angles are taken around the centroid, so an offset must not matter."""
    points = _circle(7)
    assert angular_order(points + 100.0) == angular_order(points)


def test_scoring_only_uses_the_two_requested_components():
    """score_planes slices a 4-component projection; extra columns must not leak."""
    points = np.hstack([_circle(7), np.random.RandomState(0).randn(7, 2) * 50])
    assert cyclic_order_score(points[:, [0, 1]], range(7)) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print("circular order tests PASSED")
