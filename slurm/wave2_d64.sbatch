#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Wave-2 d64 house-table cells (s68): runs SEQUENTIALLY through cfg:seed
# tokens (the wave-1 pattern) so one queue slot carries a whole bundle:
#   sbatch --job-name=w2-<bundle> slurm/wave2_d64.sbatch <cfg>:<seed> [...]
# Fixed tag: a resubmit after timeout or preemption reuses each run dir,
# skips finished seeds and resumes the unfinished one from resume.pt, so a
# bundle that overruns the walltime just gets resubmitted with the same
# tokens. a100 header (user, s68): the mo bundles run here (~6.8 h/run,
# 3-seed bundle ~20.4 h + headroom). Cheaper one-pass bundles may be
# routed to a30 at submit time via `sbatch --partition=a30` (~2x slower);
# honour the priority-inversion rule there — hold a30 submissions while
# a100-gating d256 jobs are PENDING.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional cfg:seed tokens, not --export: passing values through the
# environment made this cluster's controller fail user-env retrieval.
[ $# -ge 1 ] || { echo "usage: sbatch wave2_d64.sbatch <cfg>:<seed> [...]"; exit 1; }
TAG=20260825-hard-w2-d64

cd "$REPO"
echo "node=$(hostname) tag=$TAG specs=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SPEC in "$@"; do
    CFG=${SPEC%%:*}
    SEED=${SPEC##*:}
    echo "=== $(date -Is) start cfg=$CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
    echo "=== $(date -Is) done cfg=$CFG seed=$SEED rc=$? ==="
done
