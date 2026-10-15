"""Step 0.7: Phase-transition point scales as a power law with graph size (Fig. 8).

Runs the Step 0.3-0.5 pipeline on square grids of increasing side length, reads
the accuracy curve out of each run, locates the transition, and fits

    transition_point ~ m^alpha        (m = grid side length)

on a log-log plot. The paper reads the exponent as evidence for its bond-
percolation account of the transition.

Each grid size is a full reproduce.py run, so this delegates rather than
duplicating the pipeline, and skips sizes whose metrics.npz already exists.

    python experiments/scaling.py                      # 3x3 .. 6x6
    python experiments/scaling.py --sizes 3 4 5        # fewer sizes
    python experiments/scaling.py --analyze-only       # refit existing runs

Run on a GPU node unless --analyze-only; see scripts/run_all.sbatch scaling.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from analysis import find_transition_point

REPRODUCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Grid side lengths. Fig. 8 uses 3-6, i.e. 9 to 36 nodes.",
    )
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20, 26, 31])
    p.add_argument("--target-layer", type=int, default=26)
    p.add_argument("--stride", type=int, default=100)
    # Small grids transition early, so the breakpoint needs resolution down
    # there; these are forwarded to reproduce.py unchanged.
    p.add_argument("--context-spacing", choices=["log", "linear"], default="log")
    p.add_argument("--context-points", type=int, default=60)
    p.add_argument("--min-context", type=int, default=10)
    p.add_argument("--accuracy-window", type=int, default=1)
    p.add_argument(
        "--prompts",
        type=int,
        default=0,
        help="Prompts per grid size. 0 keeps the paper's batch size of one per "
             "node, which for a 3x3 grid is only 9 samples per context length "
             "and leaves the breakpoint jittery. Raising this averages the same "
             "quantity over more samples, at linear GPU cost.",
    )
    p.add_argument("--outdir", default="outputs/scaling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Fit existing runs without touching the GPU.",
    )
    p.add_argument("--force", action="store_true", help="Re-run sizes that already have output.")
    return p.parse_args()


def run_one(args, m: int) -> str:
    """Run (or reuse) the reproduction pipeline for one grid size.

    Returns the directory holding that size's metrics.npz.
    """
    outdir = os.path.join(args.outdir, f"grid{m}")
    metrics = os.path.join(outdir, "metrics.npz")

    if os.path.exists(metrics) and not args.force:
        print(f"  {m}x{m}: reusing {metrics}")
        return outdir
    if args.analyze_only:
        print(f"  {m}x{m}: no metrics.npz and --analyze-only, skipping")
        return outdir

    cmd = [
        sys.executable, REPRODUCE,
        "--model", args.model,
        "--graph", "grid", "--grid-size", str(m),
        "--num-steps", str(args.num_steps),
        "--stride", str(args.stride),
        "--context-spacing", args.context_spacing,
        "--context-points", str(args.context_points),
        "--min-context", str(args.min_context),
        "--accuracy-window", str(args.accuracy_window),
        "--target-layer", str(args.target_layer),
        "--layers", *[str(l) for l in args.layers],
        "--outdir", outdir,
        "--cache", os.path.join(outdir, "pooled.npz"),
        "--seed", str(args.seed),
    ]
    if args.prompts:
        cmd += ["--prompts", str(args.prompts)]
    print(f"  {m}x{m}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return outdir


def transition_for(outdir: str) -> dict | None:
    """Locate the transition in one run's accuracy curve."""
    metrics = os.path.join(outdir, "metrics.npz")
    if not os.path.exists(metrics):
        return None

    blob = np.load(metrics, allow_pickle=False)
    lengths = blob["context_lengths"]
    accuracies = blob["accuracies"]
    valid = ~np.isnan(accuracies)
    if valid.sum() < 6:
        return None
    return find_transition_point(lengths[valid], accuracies[valid])


def fit_power_law(sizes: np.ndarray, points: np.ndarray) -> dict:
    """Fit points ~ sizes^alpha by least squares in log space."""
    log_m = np.log(sizes.astype(float))
    log_t = np.log(points.astype(float))
    alpha, log_c = np.polyfit(log_m, log_t, 1)

    predicted = alpha * log_m + log_c
    residual = log_t - predicted
    total = log_t - log_t.mean()
    r2 = 1.0 - residual @ residual / (total @ total) if total.any() else float("nan")

    return {"alpha": alpha, "coefficient": np.exp(log_c), "r2": r2}


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/3] Grid sizes {args.sizes} "
          f"({', '.join(f'{m * m} nodes' for m in args.sizes)})")
    outdirs = {m: run_one(args, m) for m in args.sizes}

    print("[2/3] Locating each transition...")
    sizes, points, saturating = [], [], []
    for m in args.sizes:
        result = transition_for(outdirs[m])
        if result is None:
            print(f"  {m}x{m}: no usable accuracy curve, excluded")
            continue
        point = float(result["transition_point"])
        sizes.append(m)
        points.append(point)
        kind = "transition" if result["is_transition"] else "saturation knee"
        if not result["is_transition"]:
            saturating.append(m)
        print(f"  {m}x{m} ({m * m} nodes): {kind} at {point:.0f} tokens "
              f"(slopes {result['slow_slope']:.3g} -> {result['fast_slope']:.3g})")

    if len(sizes) < 3:
        print("\nNeed at least 3 grid sizes with usable curves to fit a power law.")
        return

    sizes = np.array(sizes)
    points = np.array(points)
    fit = fit_power_law(sizes, points)

    print("[3/3] Power-law fit...")
    print(f"  transition ~ {fit['coefficient']:.3g} * m^{fit['alpha']:.3f} "
          f"(R^2 = {fit['r2']:.3f})")
    if saturating:
        print(f"  WARNING: grids {saturating} produced a saturating curve rather "
              "than a slow-then-fast transition, so their breakpoint is a knee, "
              "not the paper's transition. The exponent is unreliable.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes, points, s=70, zorder=5, color="k", label="measured")
    grid = np.linspace(sizes.min(), sizes.max(), 100)
    ax.plot(grid, fit["coefficient"] * grid ** fit["alpha"], ls="--", color="tab:red",
            label=rf"fit: $\alpha$ = {fit['alpha']:.2f}, $R^2$ = {fit['r2']:.2f}")
    for m, point in zip(sizes, points):
        marker = " (knee)" if m in saturating else ""
        ax.annotate(f"{m}x{m}{marker}", (m, point), fontsize=8,
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid side length $m$")
    ax.set_ylabel("transition point (context length, tokens)")
    ax.set_title(f"{args.model} — transition scaling with graph size")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(f"{args.outdir}/transition_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        f"{args.outdir}/scaling.npz",
        sizes=sizes,
        num_nodes=sizes ** 2,
        transition_points=points,
        alpha=fit["alpha"],
        coefficient=fit["coefficient"],
        r2=fit["r2"],
        saturating=np.array(saturating, dtype=int),
    )
    print(f"\nDone. Wrote transition_scaling.png and scaling.npz to {args.outdir}/")


if __name__ == "__main__":
    main()
