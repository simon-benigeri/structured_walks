"""Uniform random walk on a graph."""

import random
import networkx as nx


def random_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
) -> list[int]:
    """Perform a uniform random walk on the graph.

    At each step, move to a uniformly random neighbor of the current node.

    Args:
        graph: The graph to walk on.
        num_steps: Number of steps (tokens emitted = num_steps + 1 including start).
        start_node: Starting node. If None, chosen uniformly at random.

    Returns:
        List of visited node indices, length = num_steps + 1.
    """
    nodes = list(graph.nodes())
    if start_node is None:
        start_node = random.choice(nodes)

    path = [start_node]
    current = start_node

    for _ in range(num_steps):
        neighbors = list(graph.neighbors(current))
        current = random.choice(neighbors)
        path.append(current)

    return path
