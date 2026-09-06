#!/bin/bash
#SBATCH --job-name=obed-probe
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Re-draws the composition sweep from step-tagged checkpoints of a finished
# amortised run to test whether obedience was still improving at the end of
# training. Pure diagnostic: no training, and the archived
# eval/composition_sweep.json is left untouched (save=False).
#
# 8 checkpoints x 5 compositions x ~100 s ~= 70 min, so 4 h is ample.
#
# Run dir is a positional argument, never --export (this controller fails
# user-env retrieval and holds the job).
#   sbatch slurm/obedience_probe.sbatch results/02_constrained_soft/<run_dir>
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN_DIR=${1:?usage: sbatch obedience_probe.sbatch <run_dir>}

cd "$REPO"
echo "node=$(hostname) run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_soft_02.analysis.obedience_vs_training \
    --run-dir "$RUN_DIR"
