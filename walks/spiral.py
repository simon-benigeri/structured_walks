"""Tight spiral walk strategy on a grid graph."""

import random
import networkx as nx


def spiral_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
) -> list[int]:
    """Perform a spiral walk on a grid graph.

    Attempts to turn consistently in one rotational direction.
    Falls back to random neighbor selection when the preferred
    direction is unavailable.

    Args:
        graph: A grid graph with 'pos' attributes on nodes.
        num_steps: Number of steps.
        start_node: Starting node. If None, chosen uniformly at random.

    Returns:
        List of visited node indices.
    """
    # TODO: Implement spiral logic using grid positions
    # For now, falls back to random walk
    raise NotImplementedError("Spiral walk not yet implemented")
