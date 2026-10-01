"""The two rule-following accuracy implementations must agree.

rule_following_accuracy takes full logits; the _from_node_probs variant takes
probabilities already reduced to the node-token columns. They compute the same
quantity, so they must not drift apart.

    python tests/test_accuracy_equivalence.py     # or: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.special

from graphs import make_grid_graph
from walks import random_walk
from analysis.accuracy import (
    rule_following_accuracy,
    rule_following_accuracy_from_node_probs,
)


def test_accuracy_implementations_agree():
    rng = np.random.default_rng(0)
    vocab_size = 500

    graph = make_grid_graph(4)
    num_nodes = graph.number_of_nodes()
    # Node token IDs are deliberately non-contiguous and unsorted, so the test
    # would catch any assumption that column order follows token ID order.
    node_token_ids = list(rng.choice(vocab_size, size=num_nodes, replace=False))
    node_token_ids = [int(t) for t in node_token_ids]

    walk = random_walk(graph, 200)
    token_sequence = [node_token_ids[n] for n in walk]
    seq_len = len(token_sequence)

    logits = rng.normal(size=(seq_len, vocab_size)) * 3.0
    node_probs = scipy.special.softmax(logits, axis=-1)[:, node_token_ids]

    from_logits = rule_following_accuracy(
        logits, token_sequence, graph, node_token_ids
    )
    from_node_probs = rule_following_accuracy_from_node_probs(
        node_probs, token_sequence, graph, node_token_ids
    )

    assert from_logits.shape == from_node_probs.shape
    np.testing.assert_array_equal(
        np.isnan(from_logits), np.isnan(from_node_probs)
    )
    np.testing.assert_allclose(
        from_logits, from_node_probs, rtol=1e-10, atol=1e-12
    )
    # Only the final position should be NaN: every other current token is a node.
    assert np.isnan(from_logits).sum() == 1


def test_accuracy_is_a_probability():
    rng = np.random.default_rng(1)
    graph = make_grid_graph(3)
    node_token_ids = list(range(graph.number_of_nodes()))
    walk = random_walk(graph, 50)
    token_sequence = [node_token_ids[n] for n in walk]

    logits = rng.normal(size=(len(token_sequence), 40))
    acc = rule_following_accuracy(logits, token_sequence, graph, node_token_ids)
    finite = acc[~np.isnan(acc)]

    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)


if __name__ == "__main__":
    test_accuracy_implementations_agree()
    test_accuracy_is_a_probability()
    print("accuracy equivalence tests PASSED")
