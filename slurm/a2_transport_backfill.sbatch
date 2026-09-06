#!/bin/bash
#SBATCH --job-name=a2-transport-backfill
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --nice=1000
#SBATCH --output=slurm/logs/%x_%j.out

# Transport backfill. One-event transport counters
# (jumps_per_site_proposed / _accepted / _state_changing, run.py:276-282)
# were added after the in-flight evals started. These three passes
# measure the per-site jump rates previously estimated for the d256 claim.
#
# Why --eval-ne at each cell's OWN native grid: the override's real job
# here is to divert the artefacts into a side dir (eval_ne<k>/) so the
# frozen eval/ the headline numbers were read from is never overwritten.
# The grid itself does not move -- ne512 for the recipe pair, ne128 for the
# rescue, each matching its cell -- so the draw is the frozen draw repeated
# with the counters switched on. That makes the pass a free reproduction
# check as a side effect: the eval_ne<k>/ metrics should land on top of
# eval/, and a wide gap would be an FP-nondeterminism finding rather than
# a transport one.
#
# Three sequential passes share one slot under the 8-job cap (six family
# jobs occupied slots). Budget: ~2h + ~2h + ~0.5h at the measured d256 rate
# (~29 min at ne128, linear in grid); 9h includes cold-import headroom.
#
# --nice=1000 for the same reason as rw_stage_best_eval.sbatch: while five
# a100 family jobs are PENDING the 3-GPU per-user cap goes to whichever job
# can start first, not to the highest priority.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
R=results/03_hard

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# name:native_grid -- the grid is the cell's own in every case.
PASSES=(
  "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed42_20260818-d256-recipe-s42:512"
  "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed43_20260818-d256-recipe-s43:512"
  "H2_d256_c50_s223_letf_ma_50k_curr_naive_seed42_20260813-d256-rescue:128"
)

for pass in "${PASSES[@]}"; do
    RUN_DIR="${pass%:*}"
    NE="${pass##*:}"
    echo "=== $RUN_DIR at ne$NE ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --eval-only "$R/$RUN_DIR" --eval-ne "$NE"
done
