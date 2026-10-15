#!/bin/bash
#SBATCH --account=p53147
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=reproduce_all
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

# Run every Part 0 reproduction configuration in one go.
#
#   mkdir -p logs && sbatch scripts/run_all.sbatch           # Steps 0.3-0.6
#   mkdir -p logs && sbatch scripts/run_all.sbatch scaling   # Step 0.7
#   mkdir -p logs && sbatch scripts/run_all.sbatch grid4     # just one
#   bash scripts/run_all.sbatch grid4 ring10                 # inside an srun shell
#
# Overrides (LAYERS/TARGET_LAYER matter for smaller models: GPT-2 has 12 layers,
# Llama-3.1-8B has 32):
#   MODEL=openai-community/gpt2 STEPS=200 LAYERS="4 10" TARGET_LAYER=10 \
#       bash scripts/run_all.sbatch grid4
#
# Weights must already be in the HF cache (scripts/prefetch_model.py), since
# gengpu compute nodes have no internet access.

set -eo pipefail

module load python-miniconda3/4.12.0 2>/dev/null || true

PY="${CONDA_PREFIX_OVERRIDE:-$HOME/.conda/envs/structured_walks}/bin/python"
if [ ! -x "$PY" ]; then
    echo "No interpreter at $PY -- set CONDA_PREFIX_OVERRIDE to the env prefix." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
STEPS="${STEPS:-5000}"
LAYERS="${LAYERS:-8 16 20 26 31}"
TARGET_LAYER="${TARGET_LAYER:-26}"

# scaling is left out of the default sweep: it re-runs four grids on its own and
# the plan calls the power law nice-to-have rather than blocking. Ask for it by
# name once the Fig. 4/5 configs look right.
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(grid4 ring10 hex30 grid5 ring50 weekday)
fi

# Checked up front, so a typo costs nothing instead of surfacing after the
# tests and the model load.
for name in "${CONFIGS[@]}"; do
    case "$name" in
        grid4|grid5|ring10|ring50|hex30|weekday|scaling) ;;
        *) echo "Unknown config '$name' (expected grid4, grid5, ring10, ring50,"\
                "hex30, weekday, or scaling)" >&2
           exit 2 ;;
    esac
done

echo "host:   $(hostname)"
echo "configs: ${CONFIGS[*]}"
echo "python: $PY"
echo "model:  $MODEL  (steps=$STEPS)"
"$PY" -c "import torch, transformers, nnsight, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| nnsight', nnsight.__version__, '| accelerate', accelerate.__version__)"
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    "$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
elif [ -n "$ALLOW_CPU" ]; then
    echo "GPU:    none -- continuing on CPU because ALLOW_CPU is set"
else
    cat >&2 <<'MSG'
No GPU visible, so this would run an 8B model on CPU. If you are on a login
node, stop: submit the job instead, or get an allocation first.

    mkdir -p logs && sbatch scripts/run_all.sbatch
    srun --account=p53147 --partition=gengpu --gres=gpu:1 \
         --time=02:00:00 --mem=48G --pty bash -l

Set ALLOW_CPU=1 to override (only sensible for a small model such as
MODEL=openai-community/gpt2).
MSG
    exit 3
fi

# Fail fast on protocol/analysis regressions before spending GPU time.
echo
echo "=== tests ==="
for t in tests/test_protocol.py tests/test_baselines.py \
         tests/test_accuracy_equivalence.py tests/test_circular.py \
         tests/test_scales.py; do
    "$PY" "$t"
done

# Graph sizes follow Park et al. (arXiv:2501.00070): 4x4 and 5x5 grids, 10- and
# 50-node rings, a 30-node hex lattice, and 3x3-6x6 grids for the scaling fit.
run_config() {
    local name="$1"
    local common=(--model "$MODEL" --num-steps "$STEPS" --layers $LAYERS
                  --target-layer "$TARGET_LAYER")

    case "$name" in
        grid4)  # Figs. 1, 4(a)
            "$PY" experiments/reproduce.py "${common[@]}" \
                --graph grid --grid-size 4 \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        grid5)  # Fig. 5(b)
            "$PY" experiments/reproduce.py "${common[@]}" \
                --graph grid --grid-size 5 \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        ring10) # Fig. 4(b)
            "$PY" experiments/reproduce.py "${common[@]}" \
                --graph ring --ring-nodes 10 \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        ring50) # Fig. 5(a). 50 prompts of $STEPS tokens -- the longest config.
            "$PY" experiments/reproduce.py "${common[@]}" \
                --graph ring --ring-nodes 50 \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        hex30)  # Fig. 4(c)
            "$PY" experiments/reproduce.py "${common[@]}" \
                --graph hex --hex-rows 3 --hex-cols 3 \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        weekday) # Step 0.6, Fig. 3
            "$PY" experiments/semantic_priors.py "${common[@]}" \
                --outdir "outputs/$name" --cache "outputs/$name/pooled.npz" ;;
        scaling) # Step 0.7, Fig. 8. Runs four grids itself, so it is the slowest.
            "$PY" experiments/scaling.py "${common[@]}" \
                --sizes 3 4 5 6 --outdir "outputs/$name" ;;
        *) return 127 ;;
    esac
}

failed=()
for name in "${CONFIGS[@]}"; do
    echo
    echo "================ $name ================"
    start=$SECONDS
    # Keep going if one config fails, so the others still produce results.
    if run_config "$name"; then
        echo "--- $name finished in $((SECONDS - start))s"
    else
        echo "--- $name FAILED after $((SECONDS - start))s" >&2
        failed+=("$name")
    fi
done

echo
echo "================ summary ================"
for name in "${CONFIGS[@]}"; do
    if [ "$name" = scaling ]; then
        target="outputs/$name/scaling.npz"
    else
        target="outputs/$name/metrics.npz"
    fi
    if [ -f "$target" ]; then
        echo "  $name: outputs/$name/ (plots + $(basename "$target"))"
    else
        echo "  $name: no output"
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "All configs completed."
