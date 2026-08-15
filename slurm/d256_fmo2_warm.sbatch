#!/bin/bash
#SBATCH --job-name=d256-fmo2-warm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# Cross-volume transfer at 16x16 (2026-08-15), ARM SELECTED BY $1:
#   coarse -> n_euler 128, the grid every archived 16x16 cell ran (0.5d)
#   fine   -> n_euler 512, the clip-safe 2d budget the builder states
# The two differ in that field ALONE (pinned in test_configs), so the
# resolution read is separable from the transfer they share.
#
# Why the pair exists. The 16x16 warm-start arm has never actually been run:
# the archived phase-2 cell continued 16x16 from a 16x16 checkpoint, which
# tests an estimator switch and not transfer. And a resolution sweep on a
# frozen 16x16 checkpoint found Var[log w] still falling monotonically at
# the largest grid tested (19.44 -> 15.57 over 128 -> 384), which is what an
# under-resolved grid looks like — but a frozen-checkpoint sweep can only
# ask how an ALREADY-TRAINED model behaves when re-rolled finer. It cannot
# separate "the grid is coarse" from "the model was fitted to a coarse
# grid", because a model trained under a biased discretisation learns to
# compensate that bias. Only the fine TRAINING arm tests the second reading.
#
# Source: the 8x8 fmo2 run's EMA parameters (per-site 0.00328, the best 8x8
# model in the archive), resampled onto the larger torus by the warm-start
# script — 109 keys copied shape-identical, 6 positional tables interpolated
# bicubically with circular padding, 0 missing / 0 unexpected on load.
#
# Reference points, measured, not bars: current 16x16 per-site 0.0707;
# ESS 0.10 at this volume would need 0.0090. No measured lever approaches
# that (best product of everything priced so far is ~1.44x against 7.9x),
# which is why the untested axes are what these arms carry. Read on
# Var[log w]/site — the ESS column at this volume is at its finite-N floor
# and inverted against Var on the archived arms.
#
# Fixed --tag per arm so a resubmission resumes into the same run dir;
# resume.pt takes precedence over --init-from, so a resumed job continues
# rather than restarting from the transfer.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
INIT=results/03_hard/warm_start_fmo2_d64ema_to_d256.pt

case "${1:?arm required: coarse|fine}" in
  coarse) CFG=H2_d256_c50_s223_letf_fmo2_20k_sc_warm;       TAG=20260815-d256-fmo2-warm-ne128 ;;
  fine)   CFG=H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512; TAG=20260815-d256-fmo2-warm-ne512 ;;
  *) echo "unknown arm '$1' (coarse|fine)" >&2; exit 2 ;;
esac

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG" --init-from "$INIT"
