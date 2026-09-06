#!/bin/bash
#SBATCH --job-name=d144-ma-bracket
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 12x12 volume bracket of the archived 16x16 failure (2026-08-15). The
# 16x16 naive-rescue recipe with the lattice side the ONLY mechanism change
# (pinned in test_screen_pins); supersedes the fmo2 12x12 cell as the
# bracketing rung, which sat three variables from the failure it was meant
# to bracket (head, estimator, Euler grid — rationale in the cell builder
# docstring). Read: a single-estimator single-head scaling curve
# 8x8 0.00472 -> 12x12 (this) -> 16x16 0.0707 Var[log w]/site, plus a free
# stage-1 screen read en route (the ladder's first 5k steps are flat
# sigma=0.10; stage-tail loss/var_dt_log_p_tilde lands day one against the
# 8x8/16x16 yardsticks 0.033 / 0.119).
#
# Seed $1 defaults to 42; headline scaling point uses separate seeds 42, 43.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
SEED=${1:-42}
TAG=20260815-d144-bracket-s${SEED}

cd "$REPO"
echo "node=$(hostname) cfg=H2_d144_c50_s223_letf_ma_50k_curr_naive seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d144_c50_s223_letf_ma_50k_curr_naive --seed "$SEED" --tag "$TAG"
