#!/bin/bash
#SBATCH --job-name=gfn-d576-sc
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --array=42-44

# One critical TB seed per job, with the size/venue compile gate in-job.
# Submit: sbatch slurm/gfn_d576_critical.sbatch
# Fixed tag resumes checkpoints and skips completed raw+EMA evaluations.
# Training: 100k updates, batch 128, hidden 76, original d400 sigma ladder.
set -euo pipefail
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=GFN_d576_c50_s220_tb_100k_par
TAG=20260906-gfn-d576-sc
SEED=${SLURM_ARRAY_TASK_ID:?submit as a seed array}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum experiments/constrained_hard_03/{gfn_configs,gfn_launch_bench,run_gfn}.py \
    src/discrete_flow_sampler/models/raster_gfn_policy.py

GATE_LOG=$(mktemp)
trap 'rm -f "$GATE_LOG"' EXIT
"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.probes.gfn_launch_bench --rung d576 | tee "$GATE_LOG"
grep -q 'GATE PASSED' "$GATE_LOG" || { echo 'GATE DID NOT PASS'; exit 1; }

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.run_gfn \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
echo "=== seed $SEED done $(date -Is) ==="
