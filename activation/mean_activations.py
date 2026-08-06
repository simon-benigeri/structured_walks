"""Windowed mean activation computation per concept."""

import numpy as np


def compute_mean_activations(
    activations: np.ndarray,
    token_sequence: list[int],
    vocab: list[int],
    window_size: int = 50,
) -> np.ndarray:
    """Compute windowed mean activations for each unique concept.

    For each timestep t, looks at the preceding window_size tokens and
    computes the mean activation for each concept that appears in that window.

    Args:
        activations: Array of shape (seq_len, hidden_dim) from one layer.
        token_sequence: List of token IDs of length seq_len.
        vocab: List of unique concept token IDs (the graph nodes).
        window_size: Number of preceding tokens to average over.

    Returns:
        Array of shape (num_concepts, hidden_dim) — mean activation per concept.
        Concepts not seen in the window get a zero vector.
    """
    seq_len, hidden_dim = activations.shape
    num_concepts = len(vocab)
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}

    mean_acts = np.zeros((num_concepts, hidden_dim))
    counts = np.zeros(num_concepts)

    start = max(0, seq_len - window_size)
    for t in range(start, seq_len):
        tok = token_sequence[t]
        if tok in token_to_idx:
            idx = token_to_idx[tok]
            mean_acts[idx] += activations[t]
            counts[idx] += 1

    nonzero = counts > 0
    mean_acts[nonzero] /= counts[nonzero, np.newaxis]

    return mean_acts
