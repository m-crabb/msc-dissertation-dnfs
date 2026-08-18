#!/bin/bash
#SBATCH --job-name=d256-screen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Generic 16x16 arm runner (2026-08-15): CFG is $1, SEED is $2 (default
# 42), TAG is $3 (defaults to the screen tag for compatibility with the
# original screen dispatches). One sbatch for the whole family because
# every arm is the same shape one declared variable apart (pinned in
# test_screen_pins); per-arm identity lives in the cell registry, not
# here. Submit with a job name so squeue stays readable:
#   sbatch -J scr5k-ma slurm/d256_screen_arm.sbatch H2_d256_scr5k_ma 42
# Fixed --tag (cfg+seed derived) so a resubmission after preemption
# reuses the run dir and a completed run is skipped rather than repeated.
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
