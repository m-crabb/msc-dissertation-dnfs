#!/bin/bash
#SBATCH --job-name=softcamort-d8
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Matched-base amortised soft cells at 8x8 (s101, plan
# 2026-08-31-soft-camort-matched-base): one conditioned model per
# coupling, draws uniform over every realisable composition in
# [0.25, 0.5] (17 values at 1/64), matched Bernoulli(c) base per cycle,
# house recipe, no curriculum. Seeds CHAINED inside the job (the 8x8
# floor-rung pattern) so one GPU slot serves the family serially:
#   sbatch -J softcamort-sc slurm/softcamort_d8.sbatch \
#       S2_d8_camort_l50_letf_ne128_house_sc
# Walltime: the walkback note prices a 50k ne128 d8 run at ~2.8 h on an
# A30; four chained seeds ~11-12 h, 2x headroom -> 24 h. Fixed --tag so
# a resubmission reuses run dirs and completed seeds are skipped, not
# retrained. Per-composition numbers come from the composition sweep at
# judging time, not here.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional arguments, not --export (this cluster's controller fails
# user-env retrieval and holds the job).
CFG=${1:?usage: sbatch softcamort_d8.sbatch <cfg> [seeds...]}
shift
SEEDS=${@:-"42 43 44 45"}
TAG=20260831-softcamort-d64

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seeds=$SEEDS tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Continue after failed seeds; exit nonzero at the end to record failure in sacct.
FAILED=""
for SEED in $SEEDS; do
    echo "=== seed $SEED start $(date -Is) ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG" \
        || FAILED="$FAILED $SEED"
    echo "=== seed $SEED end $(date -Is) ==="
done
if [ -n "$FAILED" ]; then
    echo "FAILED seeds:$FAILED"
    exit 1
fi
