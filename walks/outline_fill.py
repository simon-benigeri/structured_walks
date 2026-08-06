"""Outline-then-fill walk strategy (Experiment 3)."""

import networkx as nx


def outline_fill_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
    jump_distance: int = 5,
) -> list[int]:
    """Jump to a distant node, then walk back filling in intermediates.

    Models 'explain the endpoints, then fill in the middle.'

    Args:
        graph: The graph to walk on.
        num_steps: Number of steps.
        start_node: Starting node. If None, chosen uniformly at random.
        jump_distance: Target shortest-path distance for the initial jump.

    Returns:
        List of visited node indices.
    """
    # TODO: Implement outline-fill using shortest paths
    raise NotImplementedError("Outline-fill walk not yet implemented")
