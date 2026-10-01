"""Checks on the data-generating protocol from Park et al. Sec. 2 and App. A.

    python tests/test_protocol.py     # or: pytest tests/
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np

from graphs import make_grid_graph, make_ring_graph, make_hexagonal_graph
from graphs.tokens import assign_tokens, CONCEPT_TOKENS
from walks import random_walk, random_pair_sequence
from activation.mean_activations import accumulate_window


def test_pair_sampling_emits_edges():
    """Rings sample random neighboring pairs, so every pair must be an edge."""
    random.seed(0)
    graph = make_ring_graph(10)
    seq = random_pair_sequence(graph, 200, start_node=3)

    assert seq[0] == 3, "must start at the requested node"
    pairs = [(seq[i], seq[i + 1]) for i in range(0, len(seq) - 1, 2)]
    assert all(graph.has_edge(u, v) for u, v in pairs)

    # Unlike a walk, consecutive pairs are independent: the second element of
    # one pair is usually not adjacent to the first element of the next.
    joins = [(pairs[i][1], pairs[i + 1][0]) for i in range(len(pairs) - 1)]
    adjacent = sum(graph.has_edge(u, v) for u, v in joins)
    assert adjacent < 0.8 * len(joins), "pairs look like a walk, not independent samples"


def test_random_walk_steps_along_edges():
    random.seed(0)
    graph = make_grid_graph(4)
    walk = random_walk(graph, 300, start_node=5)
    assert walk[0] == 5
    assert all(graph.has_edge(a, b) for a, b in zip(walk, walk[1:]))


def test_token_assignment_is_random_but_seeded():
    a = assign_tokens(16, seed=0)
    b = assign_tokens(16, seed=0)
    c = assign_tokens(16, seed=1)

    assert a == b, "same seed must give the same arrangement"
    assert a != c, "different seeds must give different arrangements"
    assert len(set(a)) == 16
    assert set(a) <= set(CONCEPT_TOKENS)
    assert a != CONCEPT_TOKENS[:16], "arrangement should not be the list order"


def test_batch_of_prompts_covers_every_node():
    """App. A: one prompt per starting node guarantees full coverage."""
    random.seed(0)
    graph = make_grid_graph(4)
    n = graph.number_of_nodes()
    vocab = list(range(n))
    window, end = 50, 50

    sums = np.zeros((n, 3))
    counts = np.zeros(n)

    for start in range(n):
        walk = random_walk(graph, 400, start_node=start)
        acts = np.ones((len(walk), 3))
        accumulate_window(acts, walk, vocab, end, window, sums, counts)

    assert np.all(counts > 0), "pooling over start nodes must cover every concept"

    # A single prompt generally does not.
    single_sums = np.zeros((n, 3))
    single_counts = np.zeros(n)
    walk = random_walk(graph, 400, start_node=0)
    accumulate_window(np.ones((len(walk), 3)), walk, vocab, end, window,
                      single_sums, single_counts)
    assert np.any(single_counts == 0), "a 50-token window should miss some nodes"


def test_hex_default_is_30_nodes():
    """Fig. 4(c) uses a 30-node honeycomb."""
    assert make_hexagonal_graph(3, 3).number_of_nodes() == 30


def test_graphs_are_connected_and_zero_indexed():
    for graph in (make_grid_graph(4), make_ring_graph(10), make_hexagonal_graph(3, 3)):
        assert nx.is_connected(graph)
        assert set(graph.nodes()) == set(range(graph.number_of_nodes()))


if __name__ == "__main__":
    test_pair_sampling_emits_edges()
    test_random_walk_steps_along_edges()
    test_token_assignment_is_random_but_seeded()
    test_batch_of_prompts_covers_every_node()
    test_hex_default_is_30_nodes()
    test_graphs_are_connected_and_zero_indexed()
    print("protocol tests PASSED")
