#!/bin/bash
#SBATCH --job-name=amort-d4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/%x_%A_%a.out

# Amortised sampler, small-lattice validation: one composition-conditioned
# model per seed, trained over a widening window of target compositions, then
# swept at ten compositions. The cell is a controlled clone of the archived
# S2_d4_c05_l50_letf specialist (10k steps, ne50, fixed λ=50, hidden 64), so
# the c=0.30 and c=0.50 sweep rows compare directly against the four-seed
# specialist record on disk before any D=10 compute is committed.
#
# Array seeds 42-45 match the comparator; the per-user 3-GPU cap queues task four.
#
# Absolute pixi path: sbatch omits ~/.bashrc; --frozen uses the locked env.
export CONDA_OVERRIDE_CUDA=12.9
# Keep the package cache on gpudata (fast NFS), off the small /homes quota.
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
# Offline W&B works without credentials; upload later with `wandb sync`.
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RESULTS=results/02_constrained_soft
# Positional $1 avoids --export's user-env retrieval failure on this controller
# (whole array "requeued held").
#   sbatch slurm/amort_d4_validate.sbatch S2_d4_camort_w15_l50_letf
CFG=${1:-S2_d4_camort_l50_letf}

SEEDS=(42 43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Exit on training failure: recency lookup could otherwise sweep an older run.
if ! "$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
        --cfg "$CFG" --seed "$SEED"; then
    echo "training failed for seed $SEED; skipping sweep"
    exit 1
fi
RUN_DIR=$(ls -dt "$RESULTS/${CFG}_seed${SEED}"_* | head -1)
echo "sweeping $RUN_DIR"

"$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
    --sweep --run-dir "$RUN_DIR"
