#!/bin/bash
#SBATCH --job-name=fac-d64-fab8
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Factorised-head d64 rung (after the 4x4 gate): seed 42
# twin of archived MA 50k curriculum; only head_kind and dual-eval EMA differ.
# EMA never touches training; eval/ raw remains primary, eval_ema/ alongside.
# Submitted WITHOUT nice to outrank nice'd walkback topups. Fixed tag keeps
# the resume location; resume.pt carries the EMA shadow.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260813-fac-d64

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s223_letf_fab8_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s223_letf_fab8_50k_curr --seed 42 --tag "$TAG"
