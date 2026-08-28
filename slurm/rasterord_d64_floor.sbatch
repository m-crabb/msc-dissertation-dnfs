#!/bin/bash
#SBATCH --job-name=rfloor-d64
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Sweep-ladder FLOOR rung, 8x8 at sigma = 0.10 (2026-08-28). One cell per
# job, THREE SEEDS SERIAL INSIDE IT:
#   sbatch -J rfloor-mamo2 slurm/rasterord_d64_floor.sbatch H2_d64_c50_s010_letf_mamo2_50k_w2
#
# WHY SEEDS LOOP RATHER THAN FAN OUT AS AN ARRAY. Six cells x three seeds is
# eighteen jobs against six free slots, so an array would leave two thirds
# PENDING behind a QOS cap while the ladder stayed unreadable -- the column
# is judged three-seeds-paired or not at all. Serialised, all six cells
# advance together and the column completes in one pass. A d64 50k run is
# ~1.3 h a100-class and ~2x that on a30, so three seeds is ~15 h against
# this 2-day limit.
#
# WHAT THE COLUMN IS FOR. tab:eval-hard-8x8 prints these cells as `--` and
# its caption calls them "queued rather than failed". Every head there
# already sits between 0.984 (`ma`) and 0.999 (`thp`), so 0.015 of headroom
# cannot hold the 0.150 the ladder spans at sigma_c: this is NOT a ranking.
# It is the control that licenses the sigma_c reading -- a head that were
# merely better trained, rather than better matched to critical structure,
# would separate at BOTH couplings. Every other arm carries both columns;
# the ladder's blank halves are the one place that argument cannot be made.
#
# THE MASKED-ATTENTION ARMS RUN SEPARABLE, the prefix-sum arms do not (they
# have no score tensor to factorise). `separable_band_scores` is an exact
# rewrite -- forward agreement 1.5e-7, gradients matched parameter by
# parameter -- so quality cannot move; it buys 1.44x fewer head-forward
# FLOPs at d64 and 3.11x the training step at d256. Its trajectory
# seed-check read 0/6 catastrophic seeds against the dense twin. The `masep`
# anchor is here for the same reason the arms are separable: with a dense
# `ma` anchor the step `ma -> mamo2` would price the orderings AND the
# contraction order at once, and the MA family's FLOP/es would FALL as
# sweeps were added.
#
# Runs COMPILED (compile_head=True, inherited from the wave-2 floor parent),
# matching the recipe of every cell it is read against and reproducing the
# stress the arm-C seed check cleared.
#
# Fixed --tag so a resubmission resumes: run.py skips a seed whose
# eval/metrics.json exists and continues a partial one from resume.pt.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=${1:?usage: sbatch rasterord_d64_floor.sbatch <cfg name>}
TAG=20260828-rasterfloor-d64

cd "$REPO"
echo "node=$(hostname) cfg=$CFG tag=$TAG seeds=42,43,44"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SEED in 42 43 44; do
    echo "=== $CFG seed $SEED starting $(date -Is) ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
done
echo "=== $CFG all seeds done $(date -Is) ==="
