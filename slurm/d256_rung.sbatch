#!/bin/bash
#SBATCH --job-name=d256-rung
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 16x16 rung (hard.tex §5.6): masked attention, vertex-disjoint matching,
# 50k sigma ladder, seed 42 as at d64. Fixed tag reuses the run directory;
# train_swap resumes from checkpoints/resume.pt.
#
# Absolute pixi path avoids sbatch's missing ~/.bashrc. Offline W&B logs
# stay in repo wandb/ for later sync; key rotation was pending at launch.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260809-d256-launch

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_50k_curr --seed 42 --tag "$TAG"
