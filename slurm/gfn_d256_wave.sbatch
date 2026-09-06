#!/bin/bash
#SBATCH --job-name=gfn-d256
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=42:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# GFN 16x16 wave: one config per job, seeds 42-44 run serially.
#   sbatch -J gfn-d256-<short> slurm/gfn_d256_wave.sbatch <cfg name>
#
# THE FOUR CELLS (house d256 budgets: 50k floor / 100k sigma_c):
#   GFN_d256_c50_s010_{tb,fldb}_50k_par
#   GFN_d256_c50_s220_{tb,fldb}_100k_par
# Projected from the measured d64 rate (60 min per 50k seed, A30,
# latency-bound rollout x4 sequential steps): ~4-6 h per 50k seed and
# ~8-12 h per 100k seed -> --time above covers a serial 100k trio; the
# armer submits the s010 jobs with --time=24:00:00. Read the launch
# bench's train-step projection before trusting these further.
#
# Require gfn_d256_launch_bench.sbatch to pass on this stack first:
# compile_policy=True needs size-specific certification; d64 does not cover it.
# slurm/retry_gfn_d256.sh sequences the gate and wave.
# Fixed tags resume partial seeds from resume.pt and skip completed raw+EMA evals.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=${1:?usage: sbatch gfn_d256_wave.sbatch <cfg name> [tag]}
TAG=${2:-20260831-gfn-d256}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG tag=$TAG seeds=42,43,44"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SEED in 42 43 44; do
    echo "=== $CFG seed $SEED starting $(date -Is) ==="
    "$PIXI" run --frozen -e cuda python -m \
        experiments.constrained_hard_03.run_gfn \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
done
echo "=== $CFG all seeds done $(date -Is) ==="
