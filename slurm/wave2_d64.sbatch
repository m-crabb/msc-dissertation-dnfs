#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Wave-2 d64 house-table cell (s68): parametrised over (cfg, seed) so one
# script serves the whole 5-arm x 2-sigma x 3-seed matrix. Submit as
#   sbatch --job-name=w2-<arm>-<sigma>-s<seed> slurm/wave2_d64.sbatch <cfg> <seed>
# Fixed tag so a resubmission resumes into the same run dir. a30 partition:
# this is the d64 50k class (~0.7-6.8 h a100-equivalent by arm, ~2x on a30)
# and the a100s stay reserved for the d256 strategy per the
# priority-inversion rule; hold these while a100-gating jobs are PENDING.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260825-hard-w2-d64
CFG="$1"
SEED="$2"

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
