#!/bin/bash
#SBATCH --job-name=mdns-gate3-4x4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# PREPARED, NOT LAUNCHED — release is the user's call. Amendment-02 FINAL
# (dated amendment, frozen before results): five-arm gate-3 —
# arm 0 = the UNCONSTRAINED CONTROL reproducing MDNS's published 4x4
# critical cell on our implementation (their protocol: 2k steps, R=16,
# AdamW, EMA 0.9999, eval-on-EMA + recorded raw-param eval; 10k twins
# for our budget axis; both objectives), then the four factorised
# constrained invocations, all carrying the recipe bundle. ~60 min GPU.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RESULTS=results/03_hard/mdns_budget_gate3_4x4
RECIPE="--sigmas 0.223 --seeds 42,43,44 --device cuda \
    --optimiser adamw --ema-decay 0.9999 --results-dir $RESULTS"

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN="$PIXI run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4"

# arm 0: unconstrained control, their protocol + 10k twins, both objectives
$RUN $RECIPE --arms u --steps 2000  --replicates 16 \
    --tag 20260813-gate3-u0-wdce-2k
$RUN $RECIPE --arms u --steps 2000  --objective lv \
    --tag 20260813-gate3-u0-lv-2k
$RUN $RECIPE --arms u --steps 10000 --replicates 16 \
    --tag 20260813-gate3-u0-wdce-10k
$RUN $RECIPE --arms u --steps 10000 --objective lv \
    --tag 20260813-gate3-u0-lv-10k

# arms 1-4: factorised constrained invocations under the recipe bundle
$RUN $RECIPE --arms a  --steps 10000 --replicates 8 \
    --tag 20260813-gate3-recipe
$RUN $RECIPE --arms a  --steps 10000 --objective lv \
    --tag 20260813-gate3-lv
$RUN $RECIPE --arms a2 --steps 10000 --replicates 8 \
    --tag 20260813-gate3-gated
$RUN $RECIPE --arms a  --steps 10000 --replicates 8 \
    --near-boundary-boost 4.0 --tag 20260813-gate3-nbboost
