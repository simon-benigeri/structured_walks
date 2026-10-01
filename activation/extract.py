"""Activation extraction from LLMs via NNsight."""

import os
import torch
import numpy as np
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG


load_dotenv()


def setup_model(
    model_name: str = "meta-llama/Llama-3.1-8B",
    remote: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
) -> LanguageModel:
    """Load model via NNsight.

    Args:
        model_name: HuggingFace model identifier.
        remote: If True, prepare for NDIF remote inference and leave weights
            unloaded. If False, dispatch weights onto the local GPU.
        dtype: Weight dtype for local loading.
        device_map: Accelerate device map. Defaults to pinning everything to
            cuda:0 -- "auto" may place some layers on CPU, which makes a
            long-context forward pass unusably slow.

    Returns:
        An NNsight LanguageModel instance.
    """
    if remote:
        api_key = os.getenv("NDIF_API_KEY")
        if api_key:
            # Writes into the nnsight package directory, so tolerate a
            # read-only install.
            try:
                CONFIG.set_default_api_key(api_key)
            except OSError:
                CONFIG.API.APIKEY = api_key
        return LanguageModel(model_name)

    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    # transformers>=5 renamed the `torch_dtype` argument to `dtype`.
    try:
        return LanguageModel(
            model_name, device_map=device_map, dtype=dtype, dispatch=True
        )
    except TypeError:
        return LanguageModel(
            model_name, device_map=device_map, torch_dtype=dtype, dispatch=True
        )


def describe_placement(model: LanguageModel) -> str:
    """Summarize which devices the model's parameters live on."""
    from collections import Counter

    counts = Counter(str(p.device) for p in model._model.parameters())
    return ", ".join(f"{dev}: {n} tensors" for dev, n in sorted(counts.items()))


def get_layers(model: LanguageModel):
    """Return the list of transformer block envoys for a model.

    Handles the two layouts we use: Llama/Mistral/Qwen (`model.model.layers`)
    and GPT-2 (`model.transformer.h`).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"Could not locate transformer blocks on {type(model).__name__}. "
        "Inspect the module tree with print(model) and extend get_layers()."
    )


def _as_hidden_states(output) -> torch.Tensor:
    """Pull the hidden-state tensor out of a decoder block's output.

    Older transformers returned a tuple `(hidden_states, ...)`; transformers>=5
    returns the tensor directly for most decoder layers.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def extract_activations(
    model: LanguageModel,
    inputs,
    layers: list[int] | None = None,
    remote: bool = False,
    node_token_ids: list[int] | None = None,
    return_logits: bool = False,
) -> dict:
    """Extract residual-stream activations for a token sequence.

    A single forward pass is enough for a full context-length sweep: the
    activation at position t depends only on the prefix up to t.

    Args:
        model: NNsight LanguageModel instance.
        inputs: Anything NNsight can trace — a prompt string, a list of token
            IDs, or a dict of tokenizer outputs.
        layers: Which layers to extract from. If None, extracts all layers.
        remote: Whether to run via NDIF remote inference.
        node_token_ids: If given, next-token probabilities are reduced to just
            these token IDs inside the trace. This keeps the full
            (seq_len, vocab_size) tensor off the host, which matters at long
            context: 8K tokens x 128K vocab is ~4 GB in float32.
        return_logits: If True, also return the full logits array. Memory
            hungry; prefer node_token_ids for rule-following accuracy.

    Returns:
        Dict with keys:
            'activations': dict mapping layer_idx -> np.ndarray of shape
                (seq_len, hidden_dim)
            'node_probs': np.ndarray of shape (seq_len, len(node_token_ids)),
                present only when node_token_ids is given
            'logits': np.ndarray of shape (seq_len, vocab_size), present only
                when return_logits is True
    """
    blocks = get_layers(model)
    if layers is None:
        layers = list(range(len(blocks)))

    trace_kwargs = {"remote": True} if remote else {}

    # Containers are created before the trace and only mutated inside it.
    # nnsight writes variables assigned in the trace body back into the
    # enclosing frame, so comprehensions (which get their own frame) and
    # rebinding inside the block are unreliable.
    saved_hidden = {}
    saved = {}

    with model.trace(inputs, **trace_kwargs):
        for layer in layers:
            saved_hidden[layer] = blocks[layer].output.save()

        if node_token_ids is not None or return_logits:
            logits = model.lm_head.output
            if node_token_ids is not None:
                probs = torch.softmax(logits[0].float(), dim=-1)
                saved["probs"] = probs[:, node_token_ids].save()
            if return_logits:
                saved["logits"] = logits.save()

    if not saved_hidden:
        raise RuntimeError(
            "The trace produced no saved activations. Check that nnsight is "
            f"version 0.7+ (found {getattr(__import__('nnsight'), '__version__', '?')})."
        )

    activations = {}
    for layer, value in saved_hidden.items():
        hidden = _as_hidden_states(value).detach().cpu().float().numpy()
        activations[layer] = hidden[0] if hidden.ndim == 3 else hidden

    results = {"activations": activations}

    if "probs" in saved:
        results["node_probs"] = saved["probs"].detach().cpu().float().numpy()

    if "logits" in saved:
        logits_np = saved["logits"].detach().cpu().float().numpy()
        results["logits"] = logits_np[0] if logits_np.ndim == 3 else logits_np

    return results
