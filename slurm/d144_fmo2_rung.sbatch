#!/bin/bash
#SBATCH --job-name=d144-fmo2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 12x12 rung (2026-08-15), previously unrun: 8x8 Var[log w]/site 0.0040
# versus 16x16 0.0707, an unbracketed 18x per-site gap over 4x volume.
#
# Read on Var[log w]/site, with ESS reported alongside and expected to be
# HONEST at this size: the lognormal identity ESS/N = exp(-Var) reproduces
# every 8x8 arm to ~1%, and at 144 sites a Var/site anywhere under ~0.03
# keeps Var[log w] below log N, so the self-normalised estimator is not at
# its floor. That is not true at 16x16, where the same column is wrong by
# 2.3e5 and reports the largest weight in the draw rather than the sampler.
#
# Reference points for reading the result, all measured, none a pass/fail
# bar: 8x8 fmo2 raw 0.00459 / EMA 0.00328 per site; 8x8 masked-attention
# 0.00395; 16x16 masked-attention 0.0707. Extensivity alone predicts a flat
# per-site number if volume costs nothing, so the interesting quantity is
# the SHAPE of 0.0040 -> (144) -> 0.0707, not any single value.
#
# n_euler=288 = 2d, the clip-safe budget the config builder itself states
# and the 8x8 rung honours; the multi-event step because Lambda ~ d^2/2 has
# outgrown the one-event step's clip budget at this volume. Both are pinned
# in test_configs against the 8x8 twin. Fixed tag preserves resume location.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260815-d144-fmo2

cd "$REPO"
echo "node=$(hostname) cfg=H2_d144_c50_s223_letf_fmo2_50k_curr seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d144_c50_s223_letf_fmo2_50k_curr --seed 42 --tag "$TAG"
