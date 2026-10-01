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


def rule_following_accuracy_from_node_probs(
    node_probs: np.ndarray,
    token_sequence: list[int],
    graph: nx.Graph,
    node_token_ids: list[int],
) -> np.ndarray:
    """Rule-following accuracy from probabilities already reduced to node tokens.

    Same quantity as rule_following_accuracy, but takes the
    (seq_len, num_nodes) slice produced by extract_activations, so the full
    vocabulary distribution never has to reach the host.

    Args:
        node_probs: Array of shape (seq_len, num_nodes). Column i holds the
            next-token probability of node i's token.
        token_sequence: List of token IDs of length seq_len.
        graph: The ground-truth graph.
        node_token_ids: List mapping graph node index -> token ID.

    Returns:
        Array of shape (seq_len,) with accuracy at each position. Positions
        where the current token isn't a graph node get NaN.
    """
    seq_len = node_probs.shape[0]
    token_to_node = {tid: i for i, tid in enumerate(node_token_ids)}
    accuracies = np.full(seq_len, np.nan)

    for t in range(seq_len - 1):
        tok = token_sequence[t]
        if tok not in token_to_node:
            continue

        neighbors = list(graph.neighbors(token_to_node[tok]))
        accuracies[t] = node_probs[t, neighbors].sum()

    return accuracies


def windowed_mean(values: np.ndarray, end: int, window_size: int) -> float:
    """Mean of `values` over the window_size entries ending at `end`, NaNs skipped."""
    start = max(0, end - window_size)
    window = values[start:end]
    if window.size == 0 or np.all(np.isnan(window)):
        return np.nan
    return float(np.nanmean(window))
