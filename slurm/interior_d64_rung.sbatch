#!/bin/bash
#SBATCH --job-name=int-d64
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Exterior-vs-interior separation d64 rung: ONE cell, seed 42,
# twin of the fmo2 d64 rung (EMA instrument kept) with an interior band on
# the factorised head's per-pair path. Expected ranges in configs.py.
#   sbatch -J int-d64-fib slurm/interior_d64_rung.sbatch H2_d64_c50_s223_letf_fib_50k_curr
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=${1:?usage: sbatch interior_d64_rung.sbatch <cfg name>}
TAG=20260823-int-d64

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG"
