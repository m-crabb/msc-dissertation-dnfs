#!/bin/bash
#SBATCH --job-name=mdns-gate3-4x4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# PREPARED, NOT LAUNCHED — user call required (forensics slate,
# docs/reviews/2026-08-13-mdns-budget-4x4-gate-run.md forensics section).
# Two factorised interventions at the sigma_c point, 10k steps, so each
# compares against the existing Amendment-01 a/c rows at MATCHED budget:
#   1) arm A' (a2): V0 with learnable two-term gate — does the C-over-A
#      unlearning bill vanish when the prior's scale is a trainable
#      parameter instead of trunk cancellation?
#   2) arm A + near-boundary loss boost kappa=4 — minimiser-safe gradient
#      reallocation to the starved b in {1, m-1} contexts.
# ~15 min GPU total (0.34 of the 44-min 18-run slate).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4 \
    --steps 10000 --seeds 42,43,44 --sigmas 0.223 --arms a2 \
    --device cuda --tag 20260813-gate3-gated \
    --results-dir results/03_hard/mdns_budget_gate3_4x4

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4 \
    --steps 10000 --seeds 42,43,44 --sigmas 0.223 --arms a \
    --near-boundary-boost 4.0 \
    --device cuda --tag 20260813-gate3-nbboost \
    --results-dir results/03_hard/mdns_budget_gate3_4x4
