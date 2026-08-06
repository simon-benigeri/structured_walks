"""Teleport-and-resume walk strategy (Experiment 3)."""

import random
import networkx as nx


def teleport_walk(
    graph: nx.Graph,
    num_steps: int,
    start_node: int | None = None,
    teleport_prob: float = 0.05,
) -> list[int]:
    """Random walk with occasional teleportation to a random node.

    Like PageRank's random surfer model. Models restarts and topic shifts.

    Args:
        graph: The graph to walk on.
        num_steps: Number of steps.
        start_node: Starting node. If None, chosen uniformly at random.
        teleport_prob: Probability of teleporting at each step.

    Returns:
        List of visited node indices.
    """
    nodes = list(graph.nodes())
    if start_node is None:
        start_node = random.choice(nodes)

    path = [start_node]
    current = start_node

    for _ in range(num_steps):
        if random.random() < teleport_prob:
            current = random.choice(nodes)
        else:
            neighbors = list(graph.neighbors(current))
            current = random.choice(neighbors)
        path.append(current)

    return path
