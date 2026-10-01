"""Windowed mean activation computation per concept."""

import numpy as np


def accumulate_window(
    activations: np.ndarray,
    token_sequence: list[int],
    vocab: list[int],
    end: int,
    window_size: int | None,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Add one prompt's windowed activations into running sums and counts.

    Park et al. pool across a batch of prompts, one per starting node, so that
    every concept is observed at least once. Accumulating in place lets us run
    those prompts one at a time and discard each one's activations, instead of
    holding the whole batch in memory.

    Args:
        activations: Array of shape (seq_len, hidden_dim) for one prompt.
        token_sequence: Token IDs for that prompt.
        vocab: Concept token IDs, index i being node i.
        end: Context length to evaluate at.
        window_size: Tokens to look back over, or None to use the whole prefix
            (the appendix's N_w = N_c case).
        sums: Array of shape (num_concepts, hidden_dim), modified in place.
        counts: Array of shape (num_concepts,), modified in place.
    """
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    end = min(end, activations.shape[0])
    start = 0 if window_size is None else max(0, end - window_size)

    for t in range(start, end):
        idx = token_to_idx.get(token_sequence[t])
        if idx is not None:
            sums[idx] += activations[t]
            counts[idx] += 1


def compute_mean_activations(
    activations: np.ndarray,
    token_sequence: list[int],
    vocab: list[int],
    window_size: int = 50,
    end: int | None = None,
    return_counts: bool = False,
):
    """Compute windowed mean activations for each unique concept.

    Averages each concept's activations over the window_size tokens preceding
    `end`, which is the context length being analyzed.

    Args:
        activations: Array of shape (seq_len, hidden_dim) from one layer.
        token_sequence: List of token IDs of length seq_len.
        vocab: List of unique concept token IDs (the graph nodes).
        window_size: Number of preceding tokens to average over.
        end: Context length to evaluate at — the window is the window_size
            tokens ending here. Defaults to the full sequence length.
        return_counts: If True, also return the per-concept occurrence counts.

    Returns:
        Array of shape (num_concepts, hidden_dim) — mean activation per concept,
        and the counts array if return_counts is set.

        Concepts absent from the window get a zero vector, which is NOT a
        meaningful representation: a zero row among residual-stream vectors is a
        large outlier that dominates PCA and inflates Dirichlet energy. Callers
        must use the counts to mask absent concepts. With 16 nodes and a
        50-token window this is common, not an edge case.
    """
    seq_len, hidden_dim = activations.shape
    num_concepts = len(vocab)
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}

    if end is None:
        end = seq_len
    end = min(end, seq_len)

    mean_acts = np.zeros((num_concepts, hidden_dim))
    counts = np.zeros(num_concepts)

    start = max(0, end - window_size)
    for t in range(start, end):
        tok = token_sequence[t]
        if tok in token_to_idx:
            idx = token_to_idx[tok]
            mean_acts[idx] += activations[t]
            counts[idx] += 1

    nonzero = counts > 0
    mean_acts[nonzero] /= counts[nonzero, np.newaxis]

    if return_counts:
        return mean_acts, counts
    return mean_acts
