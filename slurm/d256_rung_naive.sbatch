#!/bin/bash
#SBATCH --job-name=d256-rung-naive
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# The 16x16 rescue 50k (arm-B mechanism: naive_mc estimator, CV inversion
# removed). LAUNCH GATE: all five 20260811-smokewave verdicts judged against
# the pre-stated criteria AND the user's launch call (frozen amendment
# prereg-s7-amend01: launch by 14 Aug EOD, health-clean by 21 Aug EOD, else
# 16x16 degrades to smoke-level evidence). At launch, also start the
# (sigma_c, 16x16) reference chain on the Mac per the amendment's outcome C.
# Fixed --tag so a resubmission resumes into the same run dir.
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
