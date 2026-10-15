"""Step 0.6: Semantic priors coexisting with in-context structure (Fig. 3).

Park et al. relabel a ring with the days of the week, which carry a strong
pretrained cyclic order, and arrange them so the ring order is *not* the
calendar order. The result is that both orderings survive, in different
principal components: the calendar circle in PCs 1-2, the in-context ring in
PCs 3-4. The in-context structure does not overwrite the prior, it is
subordinate to it.

Protocol follows experiments/reproduce.py -- random pair sampling for rings
(Sec. 2), a batch of prompts with one per starting node (App. A), and mean
activations over the last N_w = 50 tokens (Sec. 3.1) -- so the only thing that
differs from the Fig. 4(b) ring run is the node labels.

Run on a GPU node; see scripts/run_all.sbatch weekday.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from graphs import make_ring_graph
from graphs.tokens import WEEKDAY_TOKENS, verify_single_token
from walks import random_pair_sequence
from activation import (
    setup_model,
    extract_activations,
    accumulate_window,
    describe_placement,
)
from analysis import chance_level, cycle_overlap, cyclic_order_score
from analysis.pca import pca_visualization

# Fig. 3 reads PCs 1-2 for the prior and PCs 3-4 for the in-context ring.
PRIOR_PLANE = (0, 1)
CONTEXT_PLANE = (2, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--window-size", type=int, default=50, help="N_w from Sec. 3.1.")
    p.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20, 26, 31])
    p.add_argument("--target-layer", type=int, default=26)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--outdir", default="outputs/weekday")
    p.add_argument("--remote", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", metavar="PATH", help="Save pooled sums for CPU re-analysis.")
    p.add_argument("--from-cache", metavar="PATH", help="Re-analyze a cached run, no GPU.")
    return p.parse_args()


def arrange_weekdays(seed: int, max_overlap: float = 0.5):
    """Place the weekdays on a 7-ring so ring order differs from calendar order.

    A random permutation can land on a rotation or reflection of the calendar
    order, which would make the two orderings indistinguishable and the whole
    experiment vacuous, so keep drawing until they disagree.

    Returns:
        (labels, ring_cycle, calendar_cycle). labels[i] is the weekday at node
        i; the cycles are node indices in ring and calendar order.
    """
    n = len(WEEKDAY_TOKENS)
    ring_cycle = list(range(n))
    rng = random.Random(seed)

    for _ in range(1000):
        labels = rng.sample(WEEKDAY_TOKENS, n)
        calendar_cycle = sorted(range(n), key=lambda node: WEEKDAY_TOKENS.index(labels[node]))
        if cycle_overlap(ring_cycle, calendar_cycle) <= max_overlap:
            return labels, ring_cycle, calendar_cycle

    raise RuntimeError("Could not find an arrangement distinct from calendar order.")


def score_planes(means, labels, present, ring_cycle, calendar_cycle):
    """Score both component planes against both orderings, per context length."""
    scores = {
        "prior_in_pc12": [], "context_in_pc12": [],
        "prior_in_pc34": [], "context_in_pc34": [],
    }
    for ci in range(means.shape[0]):
        # Scoring uses node indices, so a missing node would shift every row of
        # the projection. With 7 nodes in a 50-token window this is rare.
        if not present[ci].all():
            for key in scores:
                scores[key].append(np.nan)
            continue
        projected = PCA(n_components=4).fit_transform(means[ci])
        for plane, tag in ((PRIOR_PLANE, "pc12"), (CONTEXT_PLANE, "pc34")):
            pair = projected[:, list(plane)]
            scores[f"prior_in_{tag}"].append(cyclic_order_score(pair, calendar_cycle))
            scores[f"context_in_{tag}"].append(cyclic_order_score(pair, ring_cycle))

    return {k: np.array(v) for k, v in scores.items()}


def main():
    args = parse_args()
    if args.target_layer not in args.layers:
        args.layers = sorted(set(args.layers) | {args.target_layer})

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    num_nodes = len(WEEKDAY_TOKENS)
    graph = make_ring_graph(num_nodes)
    labels, ring_cycle, calendar_cycle = arrange_weekdays(args.seed)

    if args.from_cache:
        print(f"[1/5] Loading pooled activations from {args.from_cache} (no GPU)...")
        blob = np.load(args.from_cache, allow_pickle=False)
        labels = [str(s) for s in blob["labels"]]
        ring_cycle = [int(i) for i in blob["ring_cycle"]]
        calendar_cycle = [int(i) for i in blob["calendar_cycle"]]
        context_lengths = blob["context_lengths"]
        layers = sorted(int(k.split("_")[-1]) for k in blob.files
                        if k.startswith("sums_layer_"))
        if args.target_layer not in layers:
            args.target_layer = max(layers)
            print(f"  Cache lacks the requested layer; using {args.target_layer}.")
        sums = {l: blob[f"sums_layer_{l}"].astype(np.float64) for l in layers}
        counts = blob["counts"].astype(np.float64)
        args.window_size = int(blob["window_size"])
        # Otherwise the plot titles would report the CLI default model rather
        # than whichever one produced the cache.
        if "model" in blob.files:
            args.model = str(blob["model"])
    else:
        print(f"[1/5] Loading {args.model} (remote={args.remote})...")
        model = setup_model(args.model, remote=args.remote)
        tokenizer = model.tokenizer
        if not args.remote:
            print(f"  Parameter placement: {describe_placement(model)}")

        multi = [w for w, ok in verify_single_token(tokenizer, WEEKDAY_TOKENS).items()
                 if not ok]
        if multi:
            raise ValueError(
                f"Weekdays {multi} are multi-token for {args.model}, which breaks "
                "the one-token-per-step alignment this protocol depends on."
            )

        node_token_ids = [
            tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in labels
        ]
        layers = args.layers
        context_lengths = np.arange(args.window_size, args.num_steps + 1, args.stride)

        sums = None
        counts = np.zeros((len(context_lengths), num_nodes))

        print(f"[2/5] {num_nodes} prompts of {args.num_steps} steps, one per "
              f"starting node; extracting layers {layers}...")

        for start_node in range(num_nodes):
            seq_nodes = random_pair_sequence(
                graph, args.num_steps + 1, start_node=start_node
            )
            # Leading space on every word, including the first, so each step is
            # exactly one token (see experiments/reproduce.py).
            text = "".join(" " + labels[n] for n in seq_nodes)
            token_id_seq = tokenizer(text)["input_ids"]
            if len(token_id_seq) - len(seq_nodes) not in (0, 1):
                raise ValueError(
                    f"Expected one token per step (plus at most a BOS), got "
                    f"{len(token_id_seq)} for {len(seq_nodes)} steps."
                )

            results = extract_activations(
                model, text, layers=layers, remote=args.remote,
            )
            acts = results["activations"]

            if sums is None:
                hidden_dim = acts[args.target_layer].shape[1]
                sums = {l: np.zeros((len(context_lengths), num_nodes, hidden_dim))
                        for l in layers}

            for ci, end in enumerate(context_lengths):
                for layer in layers:
                    accumulate_window(
                        acts[layer], token_id_seq, node_token_ids, int(end),
                        args.window_size, sums[layer][ci], counts[ci],
                    )

            del acts, results
            print(f"  prompt {start_node + 1}/{num_nodes}")

        if args.cache:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            np.savez_compressed(
                args.cache,
                labels=np.array(labels),
                ring_cycle=np.array(ring_cycle),
                calendar_cycle=np.array(calendar_cycle),
                context_lengths=context_lengths,
                counts=counts,
                model=args.model,
                window_size=args.window_size,
                num_steps=args.num_steps,
                seed=args.seed,
                **{f"sums_layer_{l}": v.astype(np.float32) for l, v in sums.items()},
            )
            print(f"  Cached pooled sums to {args.cache}")

    print(f"[3/5] {num_nodes}-node ring labeled with weekdays")
    print(f"  ring order:     {' -> '.join(labels)}")
    print(f"  calendar order: {' -> '.join(WEEKDAY_TOKENS)}")
    print(f"  adjacency overlap between the two orders: "
          f"{cycle_overlap(ring_cycle, calendar_cycle):.2f} "
          f"(chance {chance_level(num_nodes):.2f})")

    present = counts > 0
    means = {}
    for layer, s in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            means[layer] = np.nan_to_num(s / counts[:, :, None])

    # --- Fig. 3: the two planes at the longest context ---
    print("[4/5] PCA of both component planes...")
    target = means[args.target_layer]
    last = len(context_lengths) - 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (plane, name) in zip(axes, ((PRIOR_PLANE, "pretrained calendar order"),
                                       (CONTEXT_PLANE, "in-context ring order"))):
        pca_visualization(
            target[last], labels, n_components=4, ax=ax,
            components=plane, present=present[last],
            title=f"PC{plane[0] + 1}-{plane[1] + 1} — expect {name}",
        )
    fig.suptitle(f"{args.model} layer {args.target_layer} — weekdays on a "
                 f"{num_nodes}-node ring, context {context_lengths[last]}")
    fig.savefig(f"{args.outdir}/weekday_planes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Does each ordering live where the paper says? ---
    print("[5/5] Cyclic order scores by context length...")
    scores = score_planes(target, labels, present, ring_cycle, calendar_cycle)
    floor = chance_level(num_nodes)

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "prior_in_pc12": ("tab:blue", "-", "calendar order in PCs 1-2"),
        "context_in_pc34": ("tab:red", "-", "ring order in PCs 3-4"),
        "context_in_pc12": ("tab:blue", ":", "ring order in PCs 1-2"),
        "prior_in_pc34": ("tab:red", ":", "calendar order in PCs 3-4"),
    }
    for key, (color, ls, label) in styles.items():
        ax.plot(context_lengths, scores[key], color=color, ls=ls, label=label)
    ax.axhline(floor, color="gray", ls="--", label=f"chance ({floor:.2f})")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("cyclic order agreement")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(f"{args.outdir}/weekday_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        f"{args.outdir}/metrics.npz",
        context_lengths=context_lengths,
        labels=np.array(labels),
        ring_cycle=np.array(ring_cycle),
        calendar_cycle=np.array(calendar_cycle),
        chance=floor,
        **scores,
    )

    print(f"\nDone. Wrote plots and metrics.npz to {args.outdir}/")
    print(f"  At context {context_lengths[last]}, layer {args.target_layer}:")
    for key, (_, _, label) in styles.items():
        print(f"    {label:<28} {scores[key][last]:.2f}")
    prior_ok = scores["prior_in_pc12"][last] > scores["context_in_pc12"][last]
    context_ok = scores["context_in_pc34"][last] > scores["prior_in_pc34"][last]
    if prior_ok and context_ok:
        print("  Matches Fig. 3: prior dominates PCs 1-2, in-context ring PCs 3-4.")
    else:
        print("  NOTE: the orderings did not separate as Fig. 3 describes. Try "
              "sweeping --target-layer, since the split is layer-dependent.")


if __name__ == "__main__":
    main()
