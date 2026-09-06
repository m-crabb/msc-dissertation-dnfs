#!/bin/bash
#SBATCH --job-name=d256-clip50
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# RETIRED BEFORE LAUNCH (2026-08-11) — DO NOT SUBMIT. The adversarial review
# refuted this arm by algebra: the diverged twin's minimum pre-clip norm was
# 186, so clip=50 saturates every step, and AdamW erases the constant factor
# between two always-saturated thresholds — this job would near-exactly
# retrace job 271892. Kept as the record of a rejected arm; replaced by
# slurm/d256_smoke12k_arm.sbatch (the five-arm smoke ladder).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260811-d256-clip50

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_50k_curr_clip50 seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_50k_curr_clip50 --seed 42 --tag "$TAG"
