#!/bin/bash
#SBATCH --job-name=d256-ma-s010-w3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 16x16 house-table floor row, masked attention (s71, 2026-08-26). Cell
# comment in configs.py; one-variable relationships pinned by
# test_d256_house_cells_are_declared_transforms_of_arm_b. Fixed --tag so a
# resubmission resumes into the same run dir from resume.pt. Walltime is
# generous by intent: MA is the expensive d256 head (87.8 ms/step forward vs
# the factorised head's 18.6 at the bench volume), halved on the forward by
# gather_triu_pairs, and no b512 wall-clock measurement exists for it.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s010_letf_ma_50k_b512_ne128_cv2_w3
TAG=20260826-d256-s010

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG"
