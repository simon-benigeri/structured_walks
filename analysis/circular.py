"""Scoring how well a 2D layout reproduces a known cyclic ordering.

Step 0.6 reproduces Fig. 3, whose claim is about *which* cyclic order appears in
*which* principal components: the calendar order of weekday names in PCs 1-2,
and the in-context ring order in PCs 3-4. That needs a number per component
plane rather than an eyeball judgement, which is what this module provides.
"""

import numpy as np


def _cycle_pairs(order) -> set:
    """Unordered adjacent pairs of a cyclic ordering."""
    order = list(order)
    n = len(order)
    return {frozenset((order[i], order[(i + 1) % n])) for i in range(n)}


def cycle_overlap(observed, target) -> float:
    """Fraction of the target cycle's adjacencies present in `observed`.

    Insensitive to rotation and reflection, which is what we want: a circle
    traced from any starting point, in either direction, is the same circle.

    Args:
        observed: A cyclic ordering of point indices.
        target: The cyclic ordering to score against.

    Returns:
        Score in [0, 1]; 1.0 means every adjacency is reproduced.
    """
    target_pairs = _cycle_pairs(target)
    if not target_pairs:
        return float("nan")
    return len(_cycle_pairs(observed) & target_pairs) / len(target_pairs)


def angular_order(projected: np.ndarray) -> list[int]:
    """Order points by angle around their centroid."""
    centered = np.asarray(projected, dtype=float)[:, :2]
    centered = centered - centered.mean(axis=0)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    return [int(i) for i in np.argsort(angles)]


def cyclic_order_score(projected: np.ndarray, order) -> float:
    """Score a component plane against a target cyclic ordering.

    Args:
        projected: Array of shape (n, >=2); only the first two columns are used.
        order: Target cycle as point indices, e.g. range(7) for calendar order.

    Returns:
        Score in [0, 1]. Compare against chance_level(n), not against 0.
    """
    return cycle_overlap(angular_order(projected), order)


def chance_level(num_points: int) -> float:
    """Expected cyclic_order_score for a random layout of this many points.

    A random cyclic permutation preserves each target adjacency with
    probability 2/(n-1), so small graphs have a high floor -- for 7 weekdays
    chance is already 0.33, and a "recovered" circle needs to clear that.
    """
    if num_points < 3:
        return float("nan")
    return 2.0 / (num_points - 1)
