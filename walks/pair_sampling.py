"""Random neighboring-pair sampling (the paper's ring protocol).

Park et al. treat rings differently from grids: "For the ring, we add edges
between neighboring nodes and simply sample random pairs of neighboring tokens
on the graph." That is edge sampling, not a walk -- consecutive pairs are
independent, so the sequence has no trajectory structure.
"""

import random

import networkx as nx


def random_pair_sequence(
    graph: nx.Graph,
    num_tokens: int,
    start_node: int | None = None,
) -> list[int]:
    """Emit a sequence of independently sampled neighboring pairs.

    Args:
        graph: The graph to sample edges from.
        num_tokens: Target sequence length. Pairs are emitted whole, so the
            result may be one token longer when num_tokens is odd.
        start_node: If given, the first pair is an edge incident to this node.
            The paper uses this to guarantee every node appears somewhere in
            the batch of prompts.

    Returns:
        List of node indices, consecutive entries forming sampled edges.
    """
    edges = list(graph.edges())
    if not edges:
        raise ValueError("Graph has no edges to sample.")

    sequence: list[int] = []

    if start_node is not None:
        neighbors = list(graph.neighbors(start_node))
        if not neighbors:
            raise ValueError(f"Node {start_node} has no neighbors.")
        sequence.extend([start_node, random.choice(neighbors)])

    while len(sequence) < num_tokens:
        u, v = random.choice(edges)
        if random.random() < 0.5:
            u, v = v, u
        sequence.extend([u, v])

    return sequence
