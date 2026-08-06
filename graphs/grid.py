"""Square grid graph construction."""

import networkx as nx
from typing import Optional


def make_grid_graph(m: int, n: Optional[int] = None) -> nx.Graph:
    """Create an m x n square grid graph (no periodic boundaries).

    Args:
        m: Number of rows.
        n: Number of columns. Defaults to m (square grid).

    Returns:
        A NetworkX graph with integer node labels 0..m*n-1.
        Node attribute 'pos' stores (row, col) grid coordinates.
    """
    if n is None:
        n = m

    G = nx.grid_2d_graph(m, n)

    mapping = {}
    for (r, c) in G.nodes():
        mapping[(r, c)] = r * n + c

    G = nx.relabel_nodes(G, mapping)

    for node_id in G.nodes():
        r, c = divmod(node_id, n)
        G.nodes[node_id]["pos"] = (r, c)

    return G
