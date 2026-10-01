#!/bin/bash
#SBATCH --account=p53147
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=reproduce
#SBATCH --output=logs/reproduce_%j.out
#SBATCH --error=logs/reproduce_%j.err

# Part 0 reproduction on a single GPU. Submit from the repo root:
#   mkdir -p logs && sbatch scripts/run_reproduce.sbatch
# Weights must already be in the HF cache (see scripts/prefetch_model.py),
# since gengpu compute nodes have no internet access.

set -eo pipefail

# Call the env's interpreter by absolute path. `module load` prepends the base
# miniconda bin to PATH, which shadows an activated env, so `python` can be the
# wrong interpreter even when the prompt says otherwise.
module load python-miniconda3/4.12.0
PY="${CONDA_PREFIX_OVERRIDE:-$HOME/.conda/envs/structured_walks}/bin/python"
if [ ! -x "$PY" ]; then
    echo "No interpreter at $PY -- set CONDA_PREFIX_OVERRIDE to the env prefix." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/structured_walks}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Without this, stdout is block-buffered into the Slurm log and progress only
# appears when the job ends.
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "python: $PY"
"$PY" -V
"$PY" -c "import torch, transformers, nnsight, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| nnsight', nnsight.__version__, '| accelerate', accelerate.__version__)"

nvidia-smi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('GPU:', torch.cuda.get_device_name(0))"

"$PY" experiments/reproduce.py \
    --model meta-llama/Llama-3.1-8B \
    --grid-size 4 \
    --num-steps 5000 \
    --target-layer 26 \
    --outdir outputs/reproduce \
    --cache outputs/reproduce/activations.npz
