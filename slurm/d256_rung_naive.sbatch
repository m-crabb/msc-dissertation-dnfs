#!/bin/bash
#SBATCH --job-name=d256-rung-naive
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# The 16x16 rescue 50k: the MA curriculum recipe with the naive_mc
# estimator (the CV inversion seen at d256 removed), following the five
# 20260811-smokewave cells. The (sigma_c, 16x16) reference chain is run
# alongside it. Fixed --tag so a resubmission resumes into the same run dir.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260813-d256-rescue

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_50k_curr_naive seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_50k_curr_naive --seed 42 --tag "$TAG"
