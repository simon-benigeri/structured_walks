"""Activation extraction from LLMs via NNsight."""

import os
import torch
import numpy as np
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG


load_dotenv()


def setup_model(
    model_name: str = "meta-llama/Llama-3.1-8B",
    remote: bool = True,
) -> LanguageModel:
    """Load model via NNsight.

    Args:
        model_name: HuggingFace model identifier.
        remote: If True, use NDIF remote inference (no local GPU needed).

    Returns:
        An NNsight LanguageModel instance.
    """
    api_key = os.getenv("NDIF_API_KEY")
    if api_key:
        CONFIG.set_default_api_key(api_key)

    model = LanguageModel(model_name)
    return model


def extract_activations(
    model: LanguageModel,
    token_ids: list[int],
    layers: list[int] | None = None,
    remote: bool = True,
) -> dict:
    """Extract residual-stream activations and logits for a token sequence.

    Args:
        model: NNsight LanguageModel instance.
        token_ids: List of token IDs to feed to the model.
        layers: Which layers to extract from. If None, extracts all layers.
        remote: Whether to run via NDIF remote inference.

    Returns:
        Dict with keys:
            'activations': dict mapping layer_idx -> np.ndarray of shape (seq_len, hidden_dim)
            'logits': np.ndarray of shape (seq_len, vocab_size)
    """
    # TODO: Determine correct attribute paths for the target model
    # For Llama-3.1: model.model.layers[i] for transformer blocks
    # For GPT-2: model.transformer.h[i] for transformer blocks
    raise NotImplementedError(
        "Implement after confirming model architecture attribute paths via NNsight"
    )
