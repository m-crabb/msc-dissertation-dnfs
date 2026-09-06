#!/bin/bash
#SBATCH --job-name=m6-d64-replay2
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Replay-buffer staleness ablation: the d64 MA curriculum recipe at the
# 12k smoke horizon with buffer window 8 -> 2 the ONLY declared change
# (expected ranges in the configs.py cell comment; twin-ness pinned by
# test_m6_replay2_smoke_mirrors_ma_recipe_except_declared_fields). Read
# against the archived MA twin's first 12k steps: final train-ESS within
# +-10% relative at matched steps rules staleness out as a d256 confound;
# a sensitive read either way buys a full 50k run and a re-think of the
# buffer window in the d256 recipe. Fixed tag keeps resume location.
# Scheduling: ~25 min class; 12 h wall is deliberate headroom (a30 nodes
# run ~2x the a100 estimates). a30 because a100 stays reserved for the
# d256 cv2 job — to avoid priority inversion, submit only once that job
# is RUNNING (or scontrol hold this until then).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-m6-replay2

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_smoke12k_replay2 seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_smoke12k_replay2 --seed 42 --tag "$TAG"
