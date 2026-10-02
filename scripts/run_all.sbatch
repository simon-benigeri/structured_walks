#!/bin/bash
#SBATCH --account=p53147
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=reproduce_all
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

# Run every Part 0 reproduction configuration in one go.
#
#   mkdir -p logs && sbatch scripts/run_all.sbatch           # all configs
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

echo "host:   $(hostname)"
echo "python: $PY"
echo "model:  $MODEL  (steps=$STEPS)"
"$PY" -c "import torch, transformers, nnsight, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| nnsight', nnsight.__version__, '| accelerate', accelerate.__version__)"
"$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU run)')"

# Fail fast on protocol/analysis regressions before spending GPU time.
echo
echo "=== tests ==="
for t in tests/test_protocol.py tests/test_baselines.py tests/test_accuracy_equivalence.py; do
    "$PY" "$t"
done

# Each config mirrors a figure from Park et al. (arXiv:2501.00070).
config_args() {
    case "$1" in
        grid4)  echo "--graph grid --grid-size 4" ;;            # Figs. 1, 4(a)
        ring10) echo "--graph ring --ring-nodes 10" ;;          # Fig. 4(b)
        hex30)  echo "--graph hex --hex-rows 3 --hex-cols 3" ;; # Fig. 4(c)
        grid5)  echo "--graph grid --grid-size 5" ;;            # Fig. 5(b)
        *)      return 1 ;;
    esac
}

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(grid4 ring10 hex30 grid5)
fi

failed=()
for name in "${CONFIGS[@]}"; do
    if ! args="$(config_args "$name")"; then
        echo "Unknown config '$name' (expected grid4, ring10, hex30, or grid5)" >&2
        exit 2
    fi

    echo
    echo "================ $name ================"
    start=$SECONDS
    # Keep going if one config fails, so the others still produce results.
    if "$PY" experiments/reproduce.py \
            --model "$MODEL" \
            $args \
            --num-steps "$STEPS" \
            --layers $LAYERS \
            --target-layer "$TARGET_LAYER" \
            --outdir "outputs/$name" \
            --cache "outputs/$name/pooled.npz"; then
        echo "--- $name finished in $((SECONDS - start))s"
    else
        echo "--- $name FAILED after $((SECONDS - start))s" >&2
        failed+=("$name")
    fi
done

echo
echo "================ summary ================"
for name in "${CONFIGS[@]}"; do
    if [ -f "outputs/$name/metrics.npz" ]; then
        echo "  $name: outputs/$name/ (plots + metrics.npz + pooled.npz)"
    else
        echo "  $name: no output"
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "All configs completed."
