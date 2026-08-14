#!/bin/bash
#SBATCH --job-name=fac-d64-fmo2
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# A-prime at scale (2026-08-14, user GO — LOGGED AMENDMENT): fab8 d64 rung +
# the row+col causal streams, single-variable twin pinned in test_configs.
# The 4x4 no-regression band FIRED (0.883/0.938/0.879 vs 0.911); the
# amendment declares the 4x4 read inconclusive for size-dependent interior
# mechanisms (interiors <= 14 sites there; the motivating forensic was
# measured at d64) and prices the question where interiors are long enough.
# Bands frozen in the cell comment (configs.py) BEFORE launch: MEANINGFUL
# raw >= 0.60; STRONG raw >= 0.70; NEGATIVE raw <= 0.55 -> A-prime refuted
# at scale, R1 (fint) the only live interior repair. Fixed --tag so a
# resubmission resumes into the same run dir.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-fmo2-d64

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s223_letf_fmo2_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s223_letf_fmo2_50k_curr --seed 42 --tag "$TAG"
