#!/bin/bash
#SBATCH --job-name=d256-rung
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# The 16x16 rung (hard.tex §5.6 plan of record): masked-attention head on the
# vertex-disjoint matching step, 50k sigma-ladder recipe, seed 42 (the d64
# ladder's seed). Fixed --tag so a resubmission lands in the SAME run dir and
# train_swap resumes from checkpoints/resume.pt instead of starting over.
#
# sbatch does not load ~/.bashrc, so pixi is addressed by absolute path.
# WANDB_MODE=offline: cluster wandb login is pending key rotation; logs land
# in wandb/ inside the repo for a later `wandb sync`.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260809-d256-launch

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_50k_curr --seed 42 --tag "$TAG"
