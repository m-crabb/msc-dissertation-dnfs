#!/bin/bash
#SBATCH --job-name=d256-naivecont
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# NAIVE twin of the CV continuation (2026-08-20, s41, user GO at judging).
# Continues the SAME seed-43 recipe final.pt with `estimator` naive_mc the
# only declared change, so the CV continuation's 13.4x (EMA eval ESS/N
# 0.2655 vs its parent's 0.0198, at sigma_c on 16x16) can be split between
# the control variate and the 20k additional fixed-sigma_c steps it also
# bought. Bands frozen in the cell comment (configs.py) BEFORE this launch:
# CONTINUATION-OWNS / ESTIMATOR-OWNS / SPLIT.
#
# Venue deliberately matched: the CV continuation ran on this same a100
# partition (A100-80GB), so the split is read across one variable and not
# across hardware as well. Same walltime envelope as its twin for the same
# reason.
#
# Fixed --tag so a resubmission resumes into the same run dir; resume.pt
# takes precedence over --init-from by design, which is the desired
# restart semantics.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s223_letf_fmo2_20k_sc_naive_b512_ne512
TAG=20260820-d256-naivecont
INIT=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed43_20260818-d256-recipe-s43/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=43 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 43 --tag "$TAG" \
    --init-from "$INIT"
