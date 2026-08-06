"""Rule-following accuracy measurement."""

import numpy as np
import networkx as nx


def rule_following_accuracy(
    logits: np.ndarray,
    token_sequence: list[int],
    graph: nx.Graph,
    node_token_ids: list[int],
) -> np.ndarray:
    """Compute rule-following accuracy at each timestep.

    At each position, sums the model's predicted probability over all
    valid next nodes (neighbors of the current node in the graph).

    Args:
        logits: Array of shape (seq_len, vocab_size) — raw logits from the model.
        token_sequence: List of token IDs of length seq_len.
        graph: The ground-truth graph.
        node_token_ids: List mapping graph node index -> token ID.
            node_token_ids[i] is the token ID for node i.

    Returns:
        Array of shape (seq_len,) with accuracy at each position.
        Positions where the current token isn't a graph node get NaN.
    """
    import scipy.special

    seq_len = logits.shape[0]
    probs = scipy.special.softmax(logits, axis=-1)

    token_to_node = {tid: i for i, tid in enumerate(node_token_ids)}
    accuracies = np.full(seq_len, np.nan)

    for t in range(seq_len - 1):
        tok = token_sequence[t]
        if tok not in token_to_node:
            continue

        current_node = token_to_node[tok]
        neighbors = list(graph.neighbors(current_node))
        neighbor_token_ids = [node_token_ids[n] for n in neighbors]

        accuracies[t] = probs[t].take(neighbor_token_ids).sum()

    return accuracies
