#!/bin/bash
#SBATCH --job-name=d256-clip50
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Clip-50 twin of the diverged 16x16 rung (job 271892; see the config's
# comment for the mechanism and the pre-stated health criterion). NOT
# auto-launched — the recipe change is outside the _100k horizon rule the
# first rung was launched under, so this waits on an explicit user decision.
# Fixed --tag so a resubmission resumes from checkpoints/resume.pt.
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
