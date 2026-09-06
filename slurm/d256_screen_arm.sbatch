#!/bin/bash
#SBATCH --job-name=d256-screen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 16x16 arm runner (2026-08-15): $1=CFG, $2=SEED (42), $3=TAG (original
# screen default retained). Screen arms vary declared fields, pinned in
# test_screen_pins. Historical dispatch example with a readable job name:
#   sbatch -J scr5k-ma slurm/d256_screen_arm.sbatch H2_d256_scr5k_ma 42
# Config, seed and fixed tag reuse the run directory; raw metrics skip training.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=$1
SEED=${2:-42}
TAG=${3:-20260815-screen-s${SEED}}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
