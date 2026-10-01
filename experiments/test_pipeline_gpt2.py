"""Test the full pipeline locally with GPT-2 (no NDIF needed).

GPT-2 is small enough to run on CPU. This validates:
1. Graph construction + token assignment
2. Random walk generation
3. Activation extraction via NNsight (local)
4. Mean activation computation
5. Mean activations at a given context length
6. PCA visualization
7. Dirichlet energy
8. Rule-following accuracy (both implementations agree)

We don't expect GPT-2 to actually learn the graph structure in-context
(it has a 1024 token context window and is much weaker than Llama-3.1-8B),
but this confirms the pipeline is wired correctly end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from nnsight import LanguageModel

from graphs import make_grid_graph, make_ring_graph
from graphs.tokens import CONCEPT_TOKENS, verify_single_token
from walks.random_walk import random_walk
from activation.mean_activations import compute_mean_activations
from analysis.pca import pca_visualization
from analysis.dirichlet import dirichlet_energy
from analysis.accuracy import (
    rule_following_accuracy,
    rule_following_accuracy_from_node_probs,
    windowed_mean,
)


def main():
    print("=" * 60)
    print("Pipeline Test: GPT-2 (local, CPU)")
    print("=" * 60)

    # --- Step 1: Load GPT-2 via NNsight ---
    print("\n[1/8] Loading GPT-2...")
    model = LanguageModel("openai-community/gpt2")
    tokenizer = model.tokenizer
    print(f"  Model loaded. Vocab size: {tokenizer.vocab_size}")

    # --- Step 2: Build graph and verify tokens ---
    print("\n[2/8] Building 4x4 grid graph...")
    graph = make_grid_graph(4)
    num_nodes = graph.number_of_nodes()
    labels = CONCEPT_TOKENS[:num_nodes]

    token_check = verify_single_token(tokenizer, labels)
    bad_tokens = [w for w, ok in token_check.items() if not ok]
    if bad_tokens:
        print(f"  WARNING: These words are multi-token in GPT-2: {bad_tokens}")
        print("  (This is fine for testing — Llama tokenizer may differ)")
    else:
        print(f"  All {num_nodes} concept tokens are single-token in GPT-2 ✓")

    # GPT-2 uses byte-pair encoding where tokens in context have a leading space (Ġ).
    # Encode with the space prefix to match how they appear in the joined string.
    node_token_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in labels]
    print(f"  Concepts: {labels}")
    print(f"  Token IDs (first 5): {list(zip(labels[:5], node_token_ids[:5]))}")

    # --- Step 3: Generate random walk ---
    print("\n[3/8] Generating random walk (800 steps)...")
    num_steps = 800  # GPT-2 context is 1024, keep it under
    walk_nodes = random_walk(graph, num_steps)
    walk_tokens = [labels[node] for node in walk_nodes]
    print(f"  Walk length: {len(walk_tokens)} tokens")
    print(f"  First 20: {walk_tokens[:20]}")

    # --- Step 4: Extract activations ---
    print("\n[4/8] Extracting activations from GPT-2...")

    # Tokenize the walk into IDs
    # GPT-2 needs spaces before tokens for proper tokenization
    input_text = " ".join(walk_tokens)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    seq_len = input_ids.shape[1]
    print(f"  Input sequence length (after tokenization): {seq_len} tokens")

    # GPT-2 has 12 layers, hidden_dim=768
    target_layer = 10  # Use a late layer

    # Extract activations using NNsight tracing
    with model.trace(input_text) as tracer:
        hidden = model.transformer.h[target_layer].output[0].save()
        logits = model.lm_head.output.save()

    hidden_np = hidden.detach().cpu().numpy()
    logits_raw = logits.detach().cpu().numpy()
    print(f"  Raw hidden shape: {hidden_np.shape}")
    print(f"  Raw logits shape: {logits_raw.shape}")

    # Remove batch dimension if present
    if hidden_np.ndim == 3:
        activations = hidden_np[0]
    else:
        activations = hidden_np

    if logits_raw.ndim == 3:
        logits_np = logits_raw[0]
    else:
        logits_np = logits_raw
    print(f"  Activations shape: {activations.shape}")
    print(f"  Logits shape: {logits_np.shape}")

    # --- Step 5: Compute mean activations ---
    print("\n[5/8] Computing mean activations (window=50)...")

    # Map each position back to its concept token ID
    # Since GPT-2 may tokenize differently with spaces, we need to align
    # For simplicity, use the encoded IDs directly
    token_id_seq = input_ids.squeeze(0).tolist()

    mean_acts = compute_mean_activations(
        activations, token_id_seq, node_token_ids, window_size=50
    )
    print(f"  Mean activations shape: {mean_acts.shape}")
    nonzero_concepts = np.sum(np.any(mean_acts != 0, axis=1))
    print(f"  Concepts with nonzero activations: {nonzero_concepts}/{num_nodes}")

    # --- Step 6: PCA visualization ---
    print("\n[6/8] PCA visualization...")
    os.makedirs("outputs", exist_ok=True)

    fig, projected = pca_visualization(
        mean_acts, labels,
        title="GPT-2 Layer 10 — 4x4 Grid (pipeline test)"
    )
    fig.savefig("outputs/test_pca_gpt2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/test_pca_gpt2.png")

    # --- Step 7: Dirichlet energy ---
    print("\n[7/8] Computing Dirichlet energy...")
    energy = dirichlet_energy(mean_acts, graph)
    print(f"  Dirichlet energy: {energy:.4f}")

    # --- Step 8: Rule-following accuracy ---
    # Computed both ways: from full logits, and from probabilities reduced to the
    # node-token columns (what we use on Llama to keep a multi-GB vocabulary
    # tensor off the host). The two must agree.
    print("\n[8/8] Rule-following accuracy...")
    node_probs = scipy.special.softmax(logits_np, axis=-1)[:, node_token_ids]

    acc_full = rule_following_accuracy(
        logits_np, token_id_seq, graph, node_token_ids
    )
    acc_reduced = rule_following_accuracy_from_node_probs(
        node_probs, token_id_seq, graph, node_token_ids
    )
    np.testing.assert_allclose(acc_full, acc_reduced, rtol=1e-6, atol=1e-8)
    print("  Full-logit and reduced implementations agree ✓")
    print(f"  Mean accuracy over last 50 tokens: {windowed_mean(acc_full, len(token_id_seq), 50):.4f}")
    print(f"  (chance ≈ {np.mean([graph.degree(n) for n in graph.nodes()]) / tokenizer.vocab_size:.2e})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Pipeline test PASSED ✓")
    print("=" * 60)
    print("\nNote: GPT-2 is not expected to learn the grid structure.")
    print("This test confirms the code is wired correctly end-to-end.")
    print("For the real reproduction, run experiments/reproduce.py on a GPU node")
    print("with Llama-3.1-8B and a 2000-8000 token context.")


if __name__ == "__main__":
    main()
