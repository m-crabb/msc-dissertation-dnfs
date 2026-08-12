#!/bin/bash
#SBATCH --job-name=probe-reps
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Probe neural replicate draws (S7 amendment DECIDE-1): R=8 independent
# sampling runs, fresh eval seeds 101-108, off ONE converged checkpoint.
# Usage: sbatch slurm/hard_probe_replicates.sbatch <run_dir>
# One job per checkpoint spends one queue slot against the 8-job submission
# cap instead of eight. Already-drawn replicates are skipped (metrics.json
# present), so resubmission after preemption resumes where it stopped.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN_DIR=$1

cd "$REPO"
if [ ! -f "$RUN_DIR/checkpoints/final.pt" ]; then
    echo "no checkpoint at $RUN_DIR/checkpoints/final.pt" >&2
    exit 1
fi
echo "node=$(hostname) run_dir=$RUN_DIR replicate seeds 101-108"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for replicate_seed in 101 102 103 104 105 106 107 108; do
    if [ -f "$RUN_DIR/eval_replicate_s${replicate_seed}/metrics.json" ]; then
        echo "replicate s${replicate_seed} already drawn, skipping"
        continue
    fi
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --eval-only "$RUN_DIR" --eval-seed "$replicate_seed"
done
