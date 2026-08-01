#!/bin/bash
#SBATCH --job-name=amort-d10
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Amortised sampler at D=10: one composition-conditioned model trained on the
# full 50k-step ne128 anneal recipe, then swept at ten compositions. Single
# seed per submission (the comparator specialists hold seeds 42-45; more seeds
# are a follow-up decision, not a default). The archived D=10 ne128 anneal
# specialist took ~12 h on an A100, so 48 h covers an a30 plus the sweep.
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
RESULTS=results/02_constrained_soft
# Cell and seed are positional arguments, not --export: passing values through
# the environment made this cluster's controller fail user-env retrieval and
# hold the job ("requeued held", 2026-08-01).
#   sbatch slurm/amort_d10.sbatch S2_d10_camort_l50_letf_ne128_anneal [seed]
CFG=${1:?usage: sbatch amort_d10.sbatch <cell> [seed]}
SEED=${2:-42}

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
