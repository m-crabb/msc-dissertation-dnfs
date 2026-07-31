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
# One array task per seed, matching the comparator's seed set 42-45 (the
# per-user cap is 3 GPUs, so the fourth task queues behind the first three).
#
# sbatch does not load ~/.bashrc, so pixi is addressed by absolute path.
# --frozen: activate the already-installed locked env, never re-solve.
export CONDA_OVERRIDE_CUDA=12.9
# Keep pixi's package cache off the small /homes quota; on gpudata (fast NFS).
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
# Offline is the credential-free path that is known to work here: the job
# still exercises the real wandb code path, and `wandb sync` can upload later.
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=S2_d4_camort_l50_letf
RESULTS=results/02_constrained_soft

SEEDS=(42 43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The run dir carries a timestamp, so the sweep resolves it by recency —
# which means a failed training run must exit here rather than let the sweep
# silently pick up some older run dir and report it as this seed's result.
if ! "$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
        --cfg "$CFG" --seed "$SEED"; then
    echo "training failed for seed $SEED; skipping sweep"
    exit 1
fi
RUN_DIR=$(ls -dt "$RESULTS/${CFG}_seed${SEED}"_* | head -1)
echo "sweeping $RUN_DIR"

"$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
    --sweep --run-dir "$RUN_DIR"
