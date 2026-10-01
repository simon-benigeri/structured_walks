"""Part 0: Reproduce core results from Park et al. (2025), arXiv:2501.00070.

Covers Steps 0.1-0.5 of the research plan, following the paper's protocol:

- Concepts are randomly arranged on the graph (Sec. 2).
- Grids and hex lattices use random walks; rings sample random neighboring
  pairs instead (Sec. 2).
- Mean activations are pooled over a batch of prompts, one per starting node,
  so every concept is observed at least once (App. A, the N_w requirement).
- Mean activations use the most recent N_w = 50 tokens (Sec. 3.1).
- Dirichlet energy follows Eq. 3; rule-following accuracy and the 1-/2-shot
  memorization baselines follow Sec. 4.1 and Eqs. 4-5.

Steps 0.6 (semantic priors) and 0.7 (scaling) live in separate scripts.

Run on a GPU node; see scripts/run_reproduce.sbatch.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from graphs import make_grid_graph, make_ring_graph, make_hexagonal_graph
from graphs.tokens import assign_tokens, verify_single_token
from walks import random_walk, random_pair_sequence
from activation import (
    setup_model,
    extract_activations,
    accumulate_window,
    describe_placement,
)
from analysis import (
    pca_visualization,
    dirichlet_energy,
    rule_following_accuracy_from_node_probs,
    windowed_mean,
    find_transition_point,
    memorization_accuracy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument(
        "--graph",
        choices=["grid", "ring", "hex"],
        default="grid",
        help="Fig. 4 uses a 4x4 grid, a 10-node ring, and a 30-node hex lattice.",
    )
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--ring-nodes", type=int, default=10)
    # 3x3 is 30 nodes, matching Fig. 4(c).
    p.add_argument("--hex-rows", type=int, default=3)
    p.add_argument("--hex-cols", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--window-size", type=int, default=50, help="N_w from Sec. 3.1.")
    p.add_argument(
        "--window-mode",
        choices=["fixed", "cumulative"],
        default="fixed",
        help="'fixed' uses the last N_w tokens (Sec. 3.1). 'cumulative' uses the "
             "whole prefix, which is how App. A can be read (N_w = N_c).",
    )
    p.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20, 26, 31])
    p.add_argument("--target-layer", type=int, default=26)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument(
        "--prompts",
        type=int,
        default=0,
        help="Prompts in the batch, each starting at a different node. "
             "0 (default) means one per node, as the paper specifies.",
    )
    p.add_argument("--outdir", default="outputs/reproduce")
    p.add_argument("--remote", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", metavar="PATH", help="Save pooled sums for CPU re-analysis.")
    p.add_argument("--from-cache", metavar="PATH", help="Re-analyze a cached run, no GPU.")
    return p.parse_args()


def build_graph(args):
    """Construct the graph and a short description for plot titles."""
    if args.graph == "grid":
        return make_grid_graph(args.grid_size), f"{args.grid_size}x{args.grid_size} grid"
    if args.graph == "ring":
        return make_ring_graph(args.ring_nodes), f"{args.ring_nodes}-node ring"
    return (
        make_hexagonal_graph(args.hex_rows, args.hex_cols),
        f"{args.hex_rows}x{args.hex_cols} hex lattice",
    )


def generate_sequence(graph, args, start_node):
    """Rings sample random neighboring pairs; everything else walks (Sec. 2)."""
    if args.graph == "ring":
        return random_pair_sequence(graph, args.num_steps + 1, start_node=start_node)
    return random_walk(graph, args.num_steps, start_node=start_node)


def main():
    args = parse_args()
    if args.target_layer not in args.layers:
        args.layers = sorted(set(args.layers) | {args.target_layer})

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.from_cache:
        print(f"[1/7] Loading pooled activations from {args.from_cache} (no GPU)...")
        blob = np.load(args.from_cache, allow_pickle=False)
        for key in ("graph", "grid_size", "ring_nodes", "hex_rows", "hex_cols",
                    "window_size", "window_mode", "num_steps", "seed"):
            if key in blob.files:
                value = blob[key]
                setattr(args, key, value.item() if value.ndim == 0 else str(value))
        graph, graph_desc = build_graph(args)
        num_nodes = graph.number_of_nodes()
        labels = [str(s) for s in blob["labels"]]
        context_lengths = blob["context_lengths"]
        layers = sorted(int(k.split("_")[-1]) for k in blob.files if k.startswith("sums_layer_"))
        if args.target_layer not in layers:
            args.target_layer = max(layers)
            print(f"  Cache lacks the requested layer; using {args.target_layer}.")
        sums = {l: blob[f"sums_layer_{l}"].astype(np.float64) for l in layers}
        counts = blob["counts"].astype(np.float64)
        accuracies = blob["accuracies"]
        print(f"[2/7] Graph: {graph_desc}, {num_nodes} nodes")
        print(f"[3/7] Cached: {int(blob['num_prompts'])} prompts, "
              f"{len(context_lengths)} context lengths")
        print(f"[4/7] Cached layers {layers}")
    else:
        # --- Step 0.1: Load model ---
        print(f"[1/7] Loading {args.model} (remote={args.remote})...")
        model = setup_model(args.model, remote=args.remote)
        tokenizer = model.tokenizer
        if not args.remote:
            print(f"  Parameter placement: {describe_placement(model)}")

        # --- Step 0.2: Build graph, randomly arrange concepts ---
        graph, graph_desc = build_graph(args)
        num_nodes = graph.number_of_nodes()
        labels = assign_tokens(num_nodes, seed=args.seed)
        layers = args.layers

        print(f"[2/7] Graph: {graph_desc}, {num_nodes} nodes")
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

        num_prompts = args.prompts or num_nodes
        window = None if args.window_mode == "cumulative" else args.window_size
        context_lengths = np.arange(args.window_size, args.num_steps + 1, args.stride)

        hidden_dim = None
        sums = None
        counts = np.zeros((len(context_lengths), num_nodes))
        acc_sums = np.zeros(len(context_lengths))
        acc_counts = np.zeros(len(context_lengths))

        print(f"[3/7] {num_prompts} prompts of {args.num_steps} steps, one per "
              f"starting node (pooled so every concept is observed)")
        print(f"[4/7] Extracting layers {layers}, window={args.window_mode}"
              f"({args.window_size})...")

        for p_idx in range(num_prompts):
            start_node = p_idx % num_nodes
            seq_nodes = generate_sequence(graph, args, start_node)
            text = " ".join(labels[n] for n in seq_nodes)
            token_id_seq = tokenizer(text)["input_ids"]

            if len(token_id_seq) - len(seq_nodes) not in (0, 1):
                raise ValueError(
                    f"Expected one token per step (plus at most a BOS), got "
                    f"{len(token_id_seq)} for {len(seq_nodes)} steps."
                )

            results = extract_activations(
                model, text, layers=layers, remote=args.remote,
                node_token_ids=node_token_ids,
            )
            acts = results["activations"]
            node_probs = results["node_probs"]

            if acts[args.target_layer].shape[0] != len(token_id_seq):
                raise ValueError("Activations and tokens are misaligned.")

            if sums is None:
                hidden_dim = acts[args.target_layer].shape[1]
                sums = {
                    l: np.zeros((len(context_lengths), num_nodes, hidden_dim))
                    for l in layers
                }

            per_step_acc = rule_following_accuracy_from_node_probs(
                node_probs, token_id_seq, graph, node_token_ids
            )

            for ci, end in enumerate(context_lengths):
                for layer in layers:
                    accumulate_window(
                        acts[layer], token_id_seq, node_token_ids, int(end),
                        window, sums[layer][ci], counts[ci],
                    )
                value = windowed_mean(per_step_acc, int(end), args.window_size)
                if not np.isnan(value):
                    acc_sums[ci] += value
                    acc_counts[ci] += 1

            # Activations for this prompt are no longer needed; pooling is done.
            del acts, results
            if (p_idx + 1) % 4 == 0 or p_idx == num_prompts - 1:
                print(f"  prompt {p_idx + 1}/{num_prompts}")

        accuracies = np.where(acc_counts > 0, acc_sums / np.maximum(acc_counts, 1), np.nan)

        if args.cache:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            np.savez_compressed(
                args.cache,
                labels=np.array(labels),
                context_lengths=context_lengths,
                counts=counts,
                accuracies=accuracies,
                num_prompts=num_prompts,
                graph=args.graph, grid_size=args.grid_size,
                ring_nodes=args.ring_nodes, hex_rows=args.hex_rows,
                hex_cols=args.hex_cols, window_size=args.window_size,
                window_mode=args.window_mode, num_steps=args.num_steps,
                seed=args.seed,
                **{f"sums_layer_{l}": v.astype(np.float32) for l, v in sums.items()},
            )
            print(f"  Cached pooled sums to {args.cache}")

    # --- Steps 0.3-0.4: Means, energy, coverage ---
    print("[5/7] Pooled means, Dirichlet energy...")
    means = {}
    present = counts > 0
    for layer, s in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            means[layer] = s / counts[:, :, None]
        means[layer] = np.nan_to_num(means[layer])

    coverage = present.sum(axis=1)
    print(
        f"  Concept coverage: min {coverage.min()}/{num_nodes}, "
        f"mean {coverage.mean():.1f}/{num_nodes}; "
        f"{(coverage < num_nodes).mean():.0%} of context lengths miss at least one"
    )

    energies = {}
    for layer in sums:
        energies[layer] = np.array([
            dirichlet_energy(means[layer][ci], graph, present=present[ci], per_edge=True)
            for ci in range(len(context_lengths))
        ])

    # --- Step 0.3: PCA snapshots ---
    print("[6/7] PCA snapshots...")
    targets = [c for c in (200, 500, 1000, 2000, int(context_lengths[-1]))
               if c <= context_lengths[-1]]
    snap_idx = [int(np.abs(context_lengths - c).argmin()) for c in targets]
    fig, axes = plt.subplots(1, len(snap_idx), figsize=(5 * len(snap_idx), 5))
    axes = np.atleast_1d(axes)
    for ax, ci in zip(axes, snap_idx):
        title = f"context={context_lengths[ci]}"
        if not present[ci].all():
            title += f" ({present[ci].sum()}/{num_nodes} nodes)"
        pca_visualization(
            means[args.target_layer][ci], labels, title=title, ax=ax,
            present=present[ci],
        )
    fig.suptitle(f"{args.model} layer {args.target_layer} — {graph_desc}")
    fig.savefig(f"{args.outdir}/pca_by_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Steps 0.4-0.5: Curves ---
    print("[7/7] Energy and accuracy curves...")
    fig, (ax_e, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    for layer in sorted(energies):
        ax_e.plot(context_lengths, energies[layer], label=f"layer {layer}")
    ax_e.set_xlabel("context length (tokens)")
    ax_e.set_ylabel(r"Dirichlet energy $E_\mathcal{G}$ per edge")
    ax_e.set_xscale("log")
    ax_e.legend()
    ax_e.grid(alpha=0.3)

    ax_a.plot(context_lengths, accuracies, color="k", label="Llama (observed)")
    mem1 = memorization_accuracy(context_lengths, num_nodes, shots=1)
    mem2 = memorization_accuracy(context_lengths, num_nodes, shots=2)
    ax_a.plot(context_lengths, mem1, ls=":", color="tab:blue", label="1-shot memorization")
    ax_a.plot(context_lengths, mem2, ls=":", color="tab:green", label="2-shot memorization")
    ax_a.set_xlabel("context length (tokens)")
    ax_a.set_ylabel("rule-following accuracy")
    ax_a.set_xscale("log")
    ax_a.grid(alpha=0.3)

    finite = np.flatnonzero(~np.isnan(accuracies))
    if finite.size:
        i = finite[0]
        print(
            f"  At {context_lengths[i]} tokens: observed {accuracies[i]:.3f} vs "
            f"1-shot {mem1[i]:.3f}, 2-shot {mem2[i]:.3f}"
        )

    valid = ~np.isnan(accuracies)
    transition = None
    if valid.sum() >= 6:
        transition = find_transition_point(context_lengths[valid], accuracies[valid])
        point = transition["transition_point"]
        kind = "transition" if transition["is_transition"] else "saturation knee"
        ax_a.axvline(point, ls="--", color="tab:red", label=f"{kind} ≈ {point}")
        print(
            f"  Breakpoint at {point} tokens: slopes "
            f"{transition['slow_slope']:.3g} -> {transition['fast_slope']:.3g} "
            f"({transition['shape']})"
        )
        if not transition["is_transition"]:
            print(
                "  NOTE: the second segment is shallower, so this is a saturating "
                "curve, not the paper's slow-then-fast phase transition."
            )

    ax_a.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{args.outdir}/energy_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        f"{args.outdir}/metrics.npz",
        context_lengths=context_lengths,
        accuracies=accuracies,
        memorization_1shot=mem1,
        memorization_2shot=mem2,
        coverage=coverage,
        labels=np.array(labels),
        **{f"energy_layer_{l}": v for l, v in energies.items()},
    )
    print(f"\nDone. Wrote plots and metrics.npz to {args.outdir}/")
    for layer in sorted(energies):
        e = energies[layer]
        print(f"  layer {layer}: energy {e[0]:.3g} -> min {np.nanmin(e):.3g} "
              f"at context {context_lengths[np.nanargmin(e)]}")


if __name__ == "__main__":
    main()
