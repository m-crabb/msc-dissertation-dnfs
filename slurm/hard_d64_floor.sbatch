#!/bin/bash
#SBATCH --job-name=d64-floor
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Mixing-probe floor cell: d64 sigma=0.10 direct training (no curriculum),
# 50k steps, seed 42. The probe's control operating point -- pairs with the
# sigma_c headline cell's converged ma_100k checkpoint. a30: d64 fits easily
# and the partition has the best availability. Fixed --tag so resubmission
# after preemption resumes into the same run dir.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260812-floor

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s010_letf_ma_50k seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s010_letf_ma_50k --seed 42 --tag "$TAG"
