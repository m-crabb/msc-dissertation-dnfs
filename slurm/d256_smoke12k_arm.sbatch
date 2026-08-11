#!/bin/bash
#SBATCH --job-name=d256-arm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# One arm of the 16x16 rescue smoke ladder (configs.py "smoke ladder" block,
# adversarial review 2026-08-11). Usage:
#   sbatch -J d256-<arm> slurm/d256_smoke12k_arm.sbatch <CFG_NAME> [INIT_FROM]
# 12k steps crosses the sigma=0.14 (5k) and 0.17 (10k) boundaries; the 0.17
# transition is where the diverged twin re-ignited, so it is the smoke's
# decisive window. Fixed --tag so resubmission resumes.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG="$1"
INIT_FROM="$2"
TAG=20260811-smokewave

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG init_from=${INIT_FROM:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA=()
if [ -n "$INIT_FROM" ]; then
    EXTRA=(--init-from "$INIT_FROM")
fi
"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG" "${EXTRA[@]}"
