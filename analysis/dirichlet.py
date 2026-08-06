"""Dirichlet energy computation on graph representations."""

import numpy as np
import networkx as nx


def dirichlet_energy(
    mean_activations: np.ndarray,
    graph: nx.Graph,
) -> float:
    """Compute the Dirichlet energy of representations on a graph.

    E_G(H) = sum_{i,j} A_{i,j} ||h_i - h_j||^2

    Low energy means neighboring nodes in the graph have similar representations.

    Args:
        mean_activations: Array of shape (num_nodes, hidden_dim).
        graph: The ground-truth graph (nodes must be 0-indexed integers).

    Returns:
        Scalar Dirichlet energy value.
    """
    energy = 0.0
    for i, j in graph.edges():
        diff = mean_activations[i] - mean_activations[j]
        energy += np.dot(diff, diff)

    # Each edge counted once; the formula sums over ordered pairs (i,j) where A_{i,j}=1,
    # which for undirected graphs counts each edge twice.
    return 2.0 * energy
