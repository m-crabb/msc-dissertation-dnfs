#!/bin/bash
#SBATCH --job-name=mdns-gate3-4x4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# PREPARED, NOT LAUNCHED — user call required. Amendment-02 slate
# (docs/plans/2026-08-13-mdns-budget-4x4-gate.md): four factorised
# sigma_c invocations at 10k steps, all carrying the paper-recipe bundle
# (EMA 0.9999 + AdamW + R=8) so single-variable diffs read against
# invocation 1, and invocation 1 reads against the Amendment-01 arm-a
# row for the recipe effect itself. ~30 min GPU total.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RESULTS=results/03_hard/mdns_budget_gate3_4x4
COMMON="--steps 10000 --seeds 42,43,44 --sigmas 0.223 --device cuda \
    --optimiser adamw --ema-decay 0.9999 --replicates 8 \
    --results-dir $RESULTS"

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN="$PIXI run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4"

$RUN $COMMON --arms a  --tag 20260813-gate3-recipe
$RUN $COMMON --arms a  --objective lv --tag 20260813-gate3-lv
$RUN $COMMON --arms a2 --tag 20260813-gate3-gated
$RUN $COMMON --arms a  --near-boundary-boost 4.0 \
    --tag 20260813-gate3-nbboost
