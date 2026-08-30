#!/bin/bash
#SBATCH --job-name=rlad-d256
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Sweep-ladder FLOOR rung, 16x16 at sigma = 0.10 (2026-08-30). Comma-chained
# configs, THREE SEEDS SERIAL INSIDE EACH:
#   sbatch -J rlad-masep slurm/rasterord_d256_floor.sbatch <cfg[,cfg,...]>
#
# WHY THE FLOOR RUNS HERE AND sigma_c RUNS ON MARS. The sigma_c ladder is
# the science (does the ladder rescue the rung where bare `ma` fails at
# 0.015 +- 0.019?) and lands fastest on the borrowed B200s; the floor is
# the control column that licenses that reading, and it fits DoC's five
# free QOS slots exactly (six cells, the two cheapest prefix-band cells
# chained in one job). Each COLUMN stays same-venue, which is what the
# within-column comparisons need.
#
# WHAT THE FLOOR COLUMN IS FOR, as at 8x8: a head that were merely better
# trained, rather than better matched to critical structure, would separate
# at BOTH couplings. The archived floor row it extends: `ma` 0.862 (a
# two-seed mean with a degenerate seed excluded -- the `masep` cell here
# repairs that anchor), thp/thp2 0.998-1.000.
#
# THE MASKED-ATTENTION ARMS RUN SEPARABLE WITH THE GATHER OFF; the interval
# arms drop the gather too (their head reads it) and cannot read the
# separable flag. Both knobs are function-preserving (separable: forward
# agreement 1.5e-7, gradients matched parameter by parameter; gather:
# bit-class equivalent GEMM shape), so quality cannot move and the chain
# still prices one field per step against the archived anchors.
#
# Fixed --tag so a resubmission resumes: run.py skips a seed whose
# eval/metrics.json exists and continues a partial one from resume.pt.
# A wall kill at 2 days is therefore recoverable, not fatal.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFGS=${1:?usage: sbatch rasterord_d256_floor.sbatch <cfg[,cfg,...]>}
TAG=${2:-20260830-rasterord-d256}

cd "$REPO"
echo "node=$(hostname) cfgs=$CFGS tag=$TAG seeds=42,43,44"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

IFS=',' read -r -a CFG_LIST <<< "$CFGS"
for CFG in "${CFG_LIST[@]}"; do
    for SEED in 42 43 44; do
        echo "=== $CFG seed $SEED starting $(date -Is) ==="
        if ! "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
                --cfg "$CFG" --seed "$SEED" --tag "$TAG"; then
            echo "SEED $SEED FAILED for $CFG -- continuing the chain"
        fi
    done
done
echo "=== all configs done $(date -Is) ==="
