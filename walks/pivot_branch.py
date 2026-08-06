"""Pivot-and-branch walk strategy (Experiment 3)."""

import networkx as nx


def pivot_branch_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
    num_branches: int = 3,
    branch_length: int = 3,
) -> list[int]:
    """Hold a node fixed, take short walks in different directions.

    Models 'explore scenarios from a central point.'

    Args:
        graph: The graph to walk on.
        num_steps: Number of steps.
        start_node: Starting (pivot) node. If None, chosen uniformly at random.
        num_branches: Number of branches to explore from the pivot.
        branch_length: Steps per branch.

    Returns:
        List of visited node indices.
    """
    # TODO: Implement pivot-and-branch
    raise NotImplementedError("Pivot-branch walk not yet implemented")
