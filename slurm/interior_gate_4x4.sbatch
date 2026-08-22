#!/bin/bash
#SBATCH --job-name=int-gate
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Exterior-vs-interior separation gate, 4x4 (2026-08-23): one job per arm,
# both sigma points x seeds 42-44 sequentially (six 10k runs). Arms are
# single-variable twins of the MA demo cells (tests/test_configs.py).
#   sbatch -J int-gate-fib slurm/interior_gate_4x4.sbatch fib
#   arms: fib | fatt | fimo2 | fmoatt (factorised + band) | iv (interval head)
# Fixed --tag so a resubmission reuses completed run dirs.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
ARM=${1:?usage: sbatch interior_gate_4x4.sbatch <fib|fatt|fimo2|fmoatt|iv>}
TAG=20260823-intgate

cd "$REPO"
echo "node=$(hostname) arm=$ARM tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SIGMA in s010 s223; do
  for SEED in 42 43 44; do
    CFG="H2_d16_c50_${SIGMA}_letf_${ARM}_10k"
    echo "=== $CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
  done
done
