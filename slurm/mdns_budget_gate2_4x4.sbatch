#!/bin/bash
#SBATCH --job-name=mdns-gate2-4x4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Amendment-01 rerun of the budget-masked MDNS 4x4 gate (second dated
# pass; prereg in docs/plans/2026-08-13-mdns-budget-4x4-gate.md, cuts and
# protocol frozen BEFORE launch). Budget-only change from the first pass:
# 10k steps (5x), 3 seeds, both operating points, all three arms — 18
# runs. The 2k-step CPU pass took ~24 s/arm training + ~40 s eval, so the
# full slate is well under an hour even CPU-bound; 4 h is headroom.
# Primary purpose: the DNFS 4x4 side-by-side (comparison harness:
# experiments/constrained_hard_03/mdns_vs_dnfs_4x4.py, runs locally off
# the rsynced results).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=${1:-20260813-gate2-10k}

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4 \
    --steps 10000 --seeds 42,43,44 --sigmas 0.10,0.223 --arms a,b,c \
    --device cuda --tag "$TAG" \
    --results-dir results/03_hard/mdns_budget_gate2_4x4
