#!/bin/bash
#SBATCH --job-name=d256-fmo2-warm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# Historical cross-volume transfer at 16x16, $1 selects:
#   coarse -> n_euler 128, the archived 16x16 grid (0.5d)
#   fine   -> n_euler 512, the builder's clip-safe 2d budget
# The fine arm never ran and is not to be relaunched: the ne512 recipe
# endpoint came back null and the eval-grid sweep answered the question.
# The two differ in that field ALONE (pinned in test_configs), so the
# resolution read is separable from the transfer they share.
#
# Rationale: the cv2 continuation continued 16x16 from 16x16, testing the
# estimator switch rather than transfer. Frozen-checkpoint Var[log w] fell
# 19.44 -> 15.57 over grids 128 -> 384. Re-rolling cannot separate coarse
# scoring from a model fitted to coarse-grid bias; the fine-training arm
# addressed the latter question.
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
# Fixed tags reuse each run directory; resume.pt overrides the loaded init.
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
