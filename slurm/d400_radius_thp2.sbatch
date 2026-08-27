#!/bin/bash
#SBATCH --job-name=d400-thp2-s010
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --array=42-44

# 20x20 radius probe, R=2, sigma = 0.10 floor (2026-08-27). Cell comment in
# configs.py; the one-variable relationships are pinned by
# test_d400_radius_cells_are_declared_transforms_of_arm_b and
# test_d400_radius_cells_isolate_the_radius.
#
# a100 (80 GB) is REQUIRED, not preferred. Profiled on a Modal A100-80GB --
# same card class as this partition -- one compiled train step over 512 rows
# at d=400 peaks 58.25 GB at R=2 and 63.41 GB at R=3, against a d256 R=2
# control reproducing its recorded 24.93 GB exactly. The a40 (48 GB) and a30
# (24 GB) partitions cannot hold either radius.
#
# Fixed --tag so a resubmission resumes into the same run dir from resume.pt.
# The array index IS the seed, so a single job id covers 42/43/44 and the
# per-user 3-GPU cap schedules them as slots free.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d400_c50_s010_letf_thp2_50k_b512_ne128_cv2_w4
TAG=20260827-d400-s010
SEED=$SLURM_ARRAY_TASK_ID

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
