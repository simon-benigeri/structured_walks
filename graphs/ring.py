"""Ring (cycle) graph construction."""

import networkx as nx


def make_ring_graph(n: int) -> nx.Graph:
    """Create a ring graph with n nodes.

    Args:
        n: Number of nodes in the cycle.

    Returns:
        A NetworkX cycle graph with integer node labels 0..n-1.
    """
    return nx.cycle_graph(n)
