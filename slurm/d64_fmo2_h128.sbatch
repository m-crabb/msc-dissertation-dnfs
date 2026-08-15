#!/bin/bash
#SBATCH --job-name=d64-fmo2-h128
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Capacity, first variation ever (2026-08-15). hidden_dim=32 is set once in
# the shared cell builder and EVERY cell at EVERY volume has used it, so it
# has never appeared in a cell diff. This is the 8x8 fmo2 rung with
# hidden_dim 32 -> 128 as the only change (pinned in test_configs; n_heads
# and n_layers deliberately do not move, so head_dim rides 8 -> 32 as a
# consequence rather than as a second knob).
#
# Motivation is the rank argument, not the fit statistic. The head must
# produce the d x d pair field, and parameters per pair-field entry fall
# 663 -> 42.7 -> 3.4 across the 16 / 64 / 256-site rungs while total
# capacity grows 1.4x; the factorised-head forensics separately measured the
# REQUIRED effective rank of that field growing ~d/8 at sigma_c. The
# relative-fit statistic that was also cited for this does NOT survive
# recomputation — it tracks which c_t estimator is switched on (two 8x8 runs
# of near-identical quality read 0.446 with the control variate and 0.028
# without), and the 16x16 value quoted for it came from the run killed for
# divergence rather than the health-clean one. It is not used here.
#
# THIS ARM IS THE NULL AND IS EXPECTED TO BE NEUTRAL. Rank headroom at 8x8
# is already 32/(64/8) = 4x, so capacity should not bind at this volume; a
# GAIN here would mean the rank argument mis-locates the constraint, which
# is why the null is worth its GPU. It also serves as the matched-capacity
# transfer source for a larger rung, which is why it runs at 8x8 first — a
# cross-volume warm start resamples positional tables but cannot bridge a
# hidden_dim change, since every weight matrix changes shape.
#
# Compare against the 8x8 fmo2 rung: raw 0.00459 / EMA 0.00328 per site.
# a30 rather than a100: 8x8 at 128 Euler steps is small, and the a100 slots
# belong to the volume and transfer arms.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260815-d64-fmo2-h128

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s223_letf_fmo2_h128_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s223_letf_fmo2_h128_50k_curr --seed 42 --tag "$TAG"
