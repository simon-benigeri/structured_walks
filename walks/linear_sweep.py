"""Linear sweep walk strategy on a grid graph."""

import random
import networkx as nx


def linear_sweep_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
    continue_prob: float = 0.95,
) -> list[int]:
    """Perform a linear sweep walk on a grid graph.

    Picks a direction and moves in that direction with high probability,
    occasionally turning.

    Args:
        graph: A grid graph with 'pos' attributes on nodes.
        num_steps: Number of steps.
        start_node: Starting node. If None, chosen uniformly at random.
        continue_prob: Probability of continuing in the current direction.

    Returns:
        List of visited node indices.
    """
    # TODO: Implement linear sweep using grid positions and direction tracking
    raise NotImplementedError("Linear sweep walk not yet implemented")
