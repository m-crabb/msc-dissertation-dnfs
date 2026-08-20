#!/bin/bash
#SBATCH --job-name=d256-gridprobe
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Train-grid vs eval-grid decoupling for the KEYSTONE (2026-08-20, s41).
#
# The keystone (b512 + ne128) judged GRID-HELPS: EMA Var[log w]/site 0.00485
# (CI 0.00463-0.00508) against the ne512 recipe parents' 0.00612 (s42) and
# 0.00735 (s43), CI-disjoint below both and surviving the family-wise
# correction. But `n_euler_steps` is ONE field shared by training and eval,
# so the arm moved both grids at once: part of the gain could be that the
# model is merely being SCORED on a coarser grid. A1 measured the eval grid
# flat at d256 (0.9% across 8x, all CIs overlapping) but did so on the
# rescue checkpoint at Var/site ~0.069 -- two orders from this arm's ~0.005,
# so that flatness is not automatically transferable.
#
# Three eval-only draws complete a 2x2 in {train grid} x {eval grid}; the
# other two cells already exist as the runs' own frozen eval_ema/.
#   (a) keystone   @ ne128  -- PIPELINE VALIDATION, not a result: the
#       keystone already trains and evals at 128, so this must reproduce
#       its frozen eval_ema/ Var/site 0.00485. If it does not, the probe
#       is not measuring what it claims and (b)/(c) are void.
#   (b) keystone   @ ne512  -- the test. If Var/site stays near 0.00485 the
#       advantage is the TRAINED MODEL; if it climbs toward the parent's
#       0.00612 the advantage was the eval grid and the recipe claim falls.
#   (c) parent s42 @ ne128  -- the converse. If the ne512-trained parent
#       improves to ~0.005 when merely scored at ne128, the whole effect is
#       eval-side. Together (b) and (c) separate the two readings; either
#       alone leaves the other open.
#
# EMA weights throughout because every d256 verdict is EMA-primary (raw
# eval ESS at sigma_c is top-weight dominated and does not resolve).
# Artefacts land in eval_ema_ne<k>/ and CANNOT overwrite the frozen
# eval_ema/ -- run.py refuses --eval-ema without --eval-ne for exactly that
# reason. These numbers are re-draws and must never be quoted as the frozen
# eval numbers.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
KEYSTONE=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive_seed42_20260819-mars-grid-depth-a
PARENT=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed42_20260818-d256-recipe-s42

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_probe() {  # run_probe <run-dir> <grid> <label>
    echo "=== $3 : $1 @ ne$2 (EMA)"
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --eval-only "$1" --eval-ne "$2" --eval-ema
}

run_probe "$KEYSTONE" 128 "(a) validation - must reproduce eval_ema/"
run_probe "$KEYSTONE" 512 "(b) keystone on the fine grid"
run_probe "$PARENT"   128 "(c) parent on the coarse grid"
