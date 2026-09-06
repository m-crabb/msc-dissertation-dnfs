#!/bin/bash
#SBATCH --job-name=d400-thp2-bf16
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --array=42-43

# 20x20 radius x precision probe, R=2, bf16, sigma = 0.10 floor.
# Cell comments in configs.py; the one-variable relationships
# are pinned by test_d400_radius_cells_are_declared_transforms_of_arm_b,
# test_d400_radius_cells_isolate_the_radius and
# test_bf16_probe_cells_are_their_fp32_twins_plus_the_flag.
#
# Two crossed axes, two seeds each, eight jobs at the QOS submission cap.
# The radius is the head's one architectural knob (R=1 -> R=2 was worth
# +0.188 ESS fraction at d256 sigma_c for +4% step time). The precision is
# a wall-clock lever whose quality cost is unmeasured -- and unmeasurable
# at one seed, since this lineage has read 0.423 and 0.899 from the same
# config and seed under floating-point non-determinism alone.
#
# a100 (80 GB) is REQUIRED, not preferred. Profiled on a Modal A100-80GB --
# same card class as this partition -- one compiled train step over 512
# rows at d=400 peaks 41.6 GB (R=3) / 39.6 GB (R=2), against a d256 R=2 control that
# reproduced its recorded 24.93 GB exactly. a40 (48 GB) would hold the bf16
# arms but not the fp32 ones; a30 (24 GB) holds neither.
#
# Fixed --tag so a resubmission resumes into the same run dir from
# resume.pt. The array index IS the seed, and the per-user 3-GPU cap
# schedules the eight as slots free.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d400_c50_s010_letf_thp2_50k_b512_ne128_cv2_w4bf16
TAG=20260827-d400-s010
SEED=$SLURM_ARRAY_TASK_ID

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
