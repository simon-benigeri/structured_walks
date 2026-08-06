"""Hexagonal (honeycomb) lattice graph construction."""

import networkx as nx


def make_hexagonal_graph(rows: int = 3, cols: int = 5) -> nx.Graph:
    """Create a hexagonal lattice graph.

    Args:
        rows: Number of rows in the honeycomb lattice.
        cols: Number of columns in the honeycomb lattice.

    Returns:
        A NetworkX graph with integer node labels.
    """
    G = nx.hexagonal_lattice_graph(rows, cols)
    mapping = {old: i for i, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    return G
