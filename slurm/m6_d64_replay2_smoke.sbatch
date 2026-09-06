#!/bin/bash
#SBATCH --job-name=m6-d64-replay2
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# M6 (M-campaign Task 6, plan 2026-08-14): replay-buffer staleness ablation.
# The d64 MA curriculum recipe at the 12k smoke horizon with buffer window
# 8 -> 2 the ONLY declared change (cell comment in configs.py carries the
# frozen bands; twin-ness pinned by
# test_m6_replay2_smoke_mirrors_ma_recipe_except_declared_fields). Read
# against the archived MA twin's first 12k steps: INSENSITIVE (final
# train-ESS within +-10% relative at matched steps) rules staleness out as
# a d256 confound; SENSITIVE either way buys a full 50k arm and a re-think
# of the buffer window in the d256 recipe. Fixed tag keeps resume location.
# Scheduling: ~25 min class on the plan's estimate; 12 h wall request is
# deliberate headroom (the a30 nodes run ~2x the a100 estimates that priced
# Task 6's neighbours). a30 because a100 stays reserved for cv2 — per the
# priority-inversion rule (imperial-doc-cluster.md §4) submit only once
# 274187 is RUNNING (or scontrol hold this until then).
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
