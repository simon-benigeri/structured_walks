"""Memorization baselines for rule-following accuracy (Park et al., Sec. 4.1).

A model could score well on rule-following purely by copying a node's observed
neighbors out of the context, with no graph representation at all. Under that
"memorization solution" the accuracy is 1 once the node has been seen and 0
before, so the expected accuracy is just the probability of having seen it.

Assuming nodes are sampled uniformly with replacement, for a context of length
l over n nodes:

    p_seen1(l) = 1 - ((n-1)/n)^l                                   (Eq. 4)
    p_seen2(l) = p_seen1(l) - l (1/n) ((n-1)/n)^(l-1)              (Eq. 5)

where p_seen2 is the stricter variant requiring two observations before a node
counts as an in-context exemplar.

The paper's conclusion is that observed accuracy ascends *later* than these
curves, so memorization cannot explain it.
"""

import numpy as np


def memorization_accuracy(
    context_lengths: np.ndarray,
    num_nodes: int,
    shots: int = 1,
) -> np.ndarray:
    """Expected rule-following accuracy under the memorization solution.

    Args:
        context_lengths: Array of context lengths l.
        num_nodes: Number of nodes n in the graph.
        shots: 1 for Eq. 4, 2 for the stricter Eq. 5.

    Returns:
        Array of expected accuracies in [0, 1], same shape as context_lengths.

    Note:
        Both equations assume uniform sampling with replacement. Our grid walks
        are random walks, which are correlated, so these are approximations in
        this setting -- the same approximation the paper makes.
    """
    if shots not in (1, 2):
        raise ValueError(f"shots must be 1 or 2, got {shots}")

    n = float(num_nodes)
    l = np.asarray(context_lengths, dtype=float)
    q = (n - 1.0) / n

    p1 = 1.0 - q**l
    if shots == 1:
        return p1

    p2 = p1 - l * (1.0 / n) * q ** (l - 1.0)
    # Tiny negatives can appear from floating point at very small l.
    return np.clip(p2, 0.0, 1.0)
