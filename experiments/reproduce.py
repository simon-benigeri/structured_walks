"""Part 0: Reproduce core results from Park et al. (2025).

This script runs through Steps 0.1-0.7 of the research plan:
1. Build a grid graph and assign concept tokens to nodes
2. Generate a random walk sequence
3. Feed sequence through the model, extract activations
4. Compute mean activations per concept at various context lengths
5. PCA visualization — check for grid geometry emergence
6. Dirichlet energy curve
7. Rule-following accuracy curve
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from graphs import make_grid_graph
from graphs.tokens import CONCEPT_TOKENS
from walks import random_walk
from activation import extract_activations, compute_mean_activations
from analysis import pca_visualization, dirichlet_energy, rule_following_accuracy


def main():
    # --- Configuration ---
    GRID_SIZE = 4  # 4x4 grid = 16 nodes
    NUM_STEPS = 5000
    WINDOW_SIZE = 50
    TARGET_LAYER = 26  # Layer where grid structure is most visible (per paper)

    # --- Step 0.2: Build graph and assign tokens ---
    graph = make_grid_graph(GRID_SIZE)
    num_nodes = graph.number_of_nodes()
    concept_labels = CONCEPT_TOKENS[:num_nodes]

    print(f"Graph: {GRID_SIZE}x{GRID_SIZE} grid, {num_nodes} nodes")
    print(f"Concepts: {concept_labels}")

    # --- Generate random walk ---
    walk_nodes = random_walk(graph, NUM_STEPS)
    walk_tokens = [concept_labels[node] for node in walk_nodes]

    print(f"Walk length: {len(walk_tokens)} tokens")

    # --- Step 0.1: Extract activations ---
    # TODO: Uncomment once activation extraction is implemented
    # model = setup_model()
    # tokenizer = model.tokenizer
    # token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in walk_tokens]
    # results = extract_activations(model, token_ids, layers=[TARGET_LAYER])
    # activations = results['activations'][TARGET_LAYER]
    # logits = results['logits']

    print("\n[TODO] Activation extraction not yet implemented.")
    print("Once NNsight access is confirmed, fill in activation/extract.py")
    print("Then this script will produce PCA plots, energy curves, and accuracy curves.")


if __name__ == "__main__":
    main()
