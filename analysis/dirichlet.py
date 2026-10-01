"""Dirichlet energy computation on graph representations."""

import numpy as np
import networkx as nx


def dirichlet_energy(
    mean_activations: np.ndarray,
    graph: nx.Graph,
    present: np.ndarray | None = None,
    per_edge: bool = False,
) -> float:
    """Compute the Dirichlet energy of representations on a graph.

    E_G(H) = sum_{i,j} A_{i,j} ||h_i - h_j||^2   (Eq. 3 of Park et al.)

    Low energy means neighboring nodes in the graph have similar representations.

    Args:
        mean_activations: Array of shape (num_nodes, hidden_dim).
        graph: The ground-truth graph (nodes must be 0-indexed integers).
        present: Optional boolean mask of shape (num_nodes,). Edges with an
            absent endpoint are skipped, since an absent concept's row is a
            zero vector rather than a representation.
        per_edge: If True, divide by the number of edges actually summed. Use
            this when comparing across context lengths, where the number of
            present nodes varies.

    Returns:
        Scalar Dirichlet energy value, or NaN if no edge could be evaluated.
    """
    energy = 0.0
    counted = 0
    for i, j in graph.edges():
        if present is not None and not (present[i] and present[j]):
            continue
        diff = mean_activations[i] - mean_activations[j]
        energy += np.dot(diff, diff)
        counted += 1

    if counted == 0:
        return float("nan")

    # Each edge is visited once here; Eq. 3 sums over ordered pairs (i,j) with
    # A_{i,j}=1, which counts each undirected edge twice.
    energy *= 2.0

    if per_edge:
        return float(energy / counted)
    return float(energy)
