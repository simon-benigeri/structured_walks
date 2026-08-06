"""Biased drift walk strategy on a grid graph."""

import random
import networkx as nx


def biased_drift_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
    bias_direction: str = "right",
    bias_strength: float = 0.6,
) -> list[int]:
    """Perform a biased random walk on a grid graph.

    Like a random walk but with a directional bias.

    Args:
        graph: A grid graph with 'pos' attributes on nodes.
        num_steps: Number of steps.
        start_node: Starting node. If None, chosen uniformly at random.
        bias_direction: One of "right", "left", "up", "down".
        bias_strength: Probability weight given to the biased direction.

    Returns:
        List of visited node indices.
    """
    # TODO: Implement biased drift using grid positions
    raise NotImplementedError("Biased drift walk not yet implemented")
