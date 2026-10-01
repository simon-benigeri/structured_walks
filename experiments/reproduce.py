"""Part 0: Reproduce core results from Park et al. (2025).

Covers Steps 0.1-0.5 of the research plan:
1. Build a grid graph and assign concept tokens to nodes
2. Generate a random walk sequence
3. Feed the sequence through the model, extract activations (one forward pass)
4. Compute mean activations per concept at a range of context lengths
5. PCA visualization -- check for grid geometry emergence
6. Dirichlet energy vs context length
7. Rule-following accuracy vs context length, plus the transition point

Steps 0.6 (semantic priors) and 0.7 (scaling) live in separate scripts.

Run on a GPU node; see scripts/run_reproduce.sbatch.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from graphs import make_grid_graph
from graphs.tokens import CONCEPT_TOKENS, verify_single_token
from walks import random_walk
from activation import (
    setup_model,
    extract_activations,
    compute_mean_activations,
    describe_placement,
)
from analysis import (
    pca_visualization,
    dirichlet_energy,
    rule_following_accuracy_from_node_probs,
    windowed_mean,
    find_transition_point,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--window-size", type=int, default=50)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 16, 20, 26, 31],
        help="Layers to extract. The paper's grid geometry is clearest at 26.",
    )
    p.add_argument(
        "--target-layer",
        type=int,
        default=26,
        help="Layer used for the PCA snapshots. Must be in --layers.",
    )
    p.add_argument("--stride", type=int, default=100, help="Context-length spacing.")
    p.add_argument("--outdir", default="outputs/reproduce")
    p.add_argument("--remote", action="store_true", help="Use NDIF instead of local GPU.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cache",
        metavar="PATH",
        help="Save raw activations here so the analysis can be redone without a GPU.",
    )
    p.add_argument(
        "--from-cache",
        metavar="PATH",
        help="Load activations from a previous --cache run and skip the model entirely.",
    )
    return p.parse_args()


def load_cache(path: str):
    """Load activations saved by a previous run."""
    blob = np.load(path, allow_pickle=False)
    layers = sorted(
        int(k.split("_")[-1]) for k in blob.files if k.startswith("act_layer_")
    )
    acts = {l: blob[f"act_layer_{l}"].astype(np.float32) for l in layers}
    meta = {
        "token_id_seq": blob["token_id_seq"].tolist(),
        "node_probs": blob["node_probs"],
        "node_token_ids": blob["node_token_ids"].tolist(),
        "labels": [str(s) for s in blob["labels"]],
        "walk_nodes": blob["walk_nodes"],
        "grid_size": int(blob["grid_size"]),
    }
    return acts, meta


def save_cache(path, acts, token_id_seq, node_probs, node_token_ids, labels,
               walk_nodes, grid_size):
    """Persist activations as float16 -- plenty for this analysis, half the size."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        token_id_seq=np.array(token_id_seq),
        node_probs=node_probs,
        node_token_ids=np.array(node_token_ids),
        labels=np.array(labels),
        walk_nodes=np.array(walk_nodes),
        grid_size=grid_size,
        **{f"act_layer_{l}": v.astype(np.float16) for l, v in acts.items()},
    )


def main():
    args = parse_args()
    if args.target_layer not in args.layers:
        args.layers = sorted(set(args.layers) | {args.target_layer})

    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    cached = None
    if args.from_cache:
        print(f"[1/7] Loading cached activations from {args.from_cache} (no GPU needed)...")
        acts, cached = load_cache(args.from_cache)
        args.grid_size = cached["grid_size"]
        args.layers = sorted(acts)
        if args.target_layer not in acts:
            args.target_layer = max(acts)
            print(f"  Cache lacks the requested layer; using {args.target_layer}.")
    else:
        # --- Step 0.1: Load model ---
        print(f"[1/7] Loading {args.model} (remote={args.remote})...")
        model = setup_model(args.model, remote=args.remote)
        tokenizer = model.tokenizer
        if not args.remote:
            print(f"  Parameter placement: {describe_placement(model)}")

    # --- Step 0.2: Build graph and assign tokens ---
    graph = make_grid_graph(args.grid_size)
    num_nodes = graph.number_of_nodes()
    labels = CONCEPT_TOKENS[:num_nodes]
    if len(labels) < num_nodes:
        raise ValueError(
            f"Need {num_nodes} concept tokens but CONCEPT_TOKENS has {len(CONCEPT_TOKENS)}."
        )

    print(f"[2/7] Graph: {args.grid_size}x{args.grid_size} grid, {num_nodes} nodes")

    if cached is not None:
        labels = cached["labels"]
        node_token_ids = cached["node_token_ids"]
        token_id_seq = cached["token_id_seq"]
        node_probs = cached["node_probs"]
        walk_nodes = cached["walk_nodes"]
        print(f"[3/7] Cached walk: {len(walk_nodes)} steps")
        print(f"[4/7] Cached layers {args.layers}, {acts[args.target_layer].shape}")
    else:
        bad = [w for w, ok in verify_single_token(tokenizer, labels).items() if not ok]
        if bad:
            print(f"  WARNING: multi-token in this tokenizer: {bad}")

        # Mid-sequence occurrences carry a leading space, so that is the form
        # whose ID we need for both alignment and the output distribution.
        node_token_ids = [
            tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in labels
        ]
        if len(set(node_token_ids)) != num_nodes:
            raise ValueError("Concept tokens collide after space-prefixed encoding.")

        # --- Generate random walk ---
        walk_nodes = random_walk(graph, args.num_steps)
        walk_tokens = [labels[n] for n in walk_nodes]
        print(f"[3/7] Walk: {len(walk_tokens)} tokens, first 10: {walk_tokens[:10]}")

        input_text = " ".join(walk_tokens)
        token_id_seq = tokenizer(input_text)["input_ids"]
        num_specials = len(token_id_seq) - len(walk_tokens)
        if num_specials not in (0, 1):
            raise ValueError(
                f"Expected one token per walk step (plus at most a BOS), got "
                f"{len(token_id_seq)} tokens for {len(walk_tokens)} steps. "
                "Check that every concept word is single-token for this model."
            )

        # --- Step 0.1: Extract activations (single forward pass) ---
        print(f"[4/7] Extracting layers {args.layers} over {len(token_id_seq)} positions...")
        results = extract_activations(
            model,
            input_text,
            layers=args.layers,
            remote=args.remote,
            node_token_ids=node_token_ids,
        )
        acts = results["activations"]
        node_probs = results["node_probs"]

        if acts[args.target_layer].shape[0] != len(token_id_seq):
            raise ValueError(
                f"Activation length {acts[args.target_layer].shape[0]} != tokenized "
                f"length {len(token_id_seq)}; positions would be misaligned."
            )
        print(f"  Activations: {acts[args.target_layer].shape}, node_probs: {node_probs.shape}")

        if args.cache:
            save_cache(args.cache, acts, token_id_seq, node_probs, node_token_ids,
                       labels, walk_nodes, args.grid_size)
            print(f"  Cached activations to {args.cache}")

    seq_len = acts[args.target_layer].shape[0]

    # --- Step 0.5: Per-position rule-following accuracy ---
    per_step_acc = rule_following_accuracy_from_node_probs(
        node_probs, token_id_seq, graph, node_token_ids
    )

    # --- Steps 0.3-0.5: Sweep context lengths ---
    print(f"[5/7] Sweeping context lengths (stride={args.stride})...")
    context_lengths = np.arange(args.window_size, seq_len + 1, args.stride)
    energies = {layer: [] for layer in args.layers}
    accuracies = []
    coverage = []

    for end in context_lengths:
        for layer in args.layers:
            H, counts = compute_mean_activations(
                acts[layer], token_id_seq, node_token_ids,
                window_size=args.window_size, end=int(end), return_counts=True,
            )
            present = counts > 0
            # Per-edge, because the number of evaluable edges changes with how
            # many concepts the window happens to cover.
            energies[layer].append(
                dirichlet_energy(H, graph, present=present, per_edge=True)
            )
        coverage.append(int(present.sum()))
        accuracies.append(windowed_mean(per_step_acc, int(end), args.window_size))

    accuracies = np.array(accuracies)
    coverage = np.array(coverage)
    energies = {layer: np.array(v) for layer, v in energies.items()}
    print(
        f"  Concept coverage in the {args.window_size}-token window: "
        f"min {coverage.min()}/{num_nodes}, mean {coverage.mean():.1f}/{num_nodes}; "
        f"{(coverage < num_nodes).mean():.0%} of context lengths miss at least one"
    )

    # --- Step 0.3: PCA snapshots, short vs long context ---
    print("[6/7] PCA snapshots...")
    snapshots = [c for c in (200, 500, 1000, 2000, seq_len) if c <= seq_len]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(5 * len(snapshots), 5))
    axes = np.atleast_1d(axes)
    for ax, end in zip(axes, snapshots):
        H, counts = compute_mean_activations(
            acts[args.target_layer], token_id_seq, node_token_ids,
            window_size=args.window_size, end=int(end), return_counts=True,
        )
        present = counts > 0
        title = f"context={end}"
        if not present.all():
            title += f" ({present.sum()}/{num_nodes} nodes)"
        pca_visualization(H, labels, title=title, ax=ax, present=present)
    fig.suptitle(f"{args.model} layer {args.target_layer} — {args.grid_size}x{args.grid_size} grid")
    fig.savefig(f"{args.outdir}/pca_by_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Steps 0.4-0.5: Energy and accuracy curves ---
    print("[7/7] Energy and accuracy curves...")
    fig, (ax_e, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    for layer in args.layers:
        ax_e.plot(context_lengths, energies[layer], label=f"layer {layer}")
    ax_e.set_xlabel("context length (tokens)")
    ax_e.set_ylabel(r"Dirichlet energy $E_\mathcal{G}$ per edge")
    ax_e.set_xscale("log")
    ax_e.legend()
    ax_e.grid(alpha=0.3)

    ax_a.plot(context_lengths, accuracies, color="k")
    ax_a.set_xlabel("context length (tokens)")
    ax_a.set_ylabel("rule-following accuracy")
    ax_a.set_xscale("log")
    ax_a.grid(alpha=0.3)

    valid = ~np.isnan(accuracies)
    transition = None
    if valid.sum() >= 6:
        transition = find_transition_point(context_lengths[valid], accuracies[valid])
        ax_a.axvline(transition["transition_point"], ls="--", color="tab:red",
                     label=f"transition ≈ {transition['transition_point']}")
        ax_a.legend()
        print(
            f"  Transition at {transition['transition_point']} tokens "
            f"(slopes {transition['slow_slope']:.3g} -> {transition['fast_slope']:.3g})"
        )

    fig.savefig(f"{args.outdir}/energy_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        f"{args.outdir}/metrics.npz",
        context_lengths=context_lengths,
        accuracies=accuracies,
        coverage=coverage,
        per_step_accuracy=per_step_acc,
        walk_nodes=np.array(walk_nodes),
        **{f"energy_layer_{l}": v for l, v in energies.items()},
    )
    print(f"\nDone. Wrote plots and metrics.npz to {args.outdir}/")
    if transition is not None:
        print("Check: energy should bottom out near the accuracy inflection.")


if __name__ == "__main__":
    main()
