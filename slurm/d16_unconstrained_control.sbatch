#!/bin/bash
#SBATCH --job-name=dnfs-d16-unc
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Unconstrained 16x16 control (2026-08-18): the hard chapter's 50k ladder
# frame on the unconstrained engine at d=256 — does the sigma_c wall
# appear without the constraint machinery? Bands frozen at the registry
# entry (experiments/dnfs_baseline_01/configs.py,
# stage_4_d16_critical_50k_ladder) before launch.
#   sbatch slurm/d16_unconstrained_control.sbatch \
#       stage_4_d16_critical_50k_ladder 42
# Walltime: the d8 walkback decomposed to ~9.5 h for 200k steps with 400
# full 5000-draw evals on an A30. This cell is 50k steps at ~5x the
# per-step cost (d 64 -> 256) with the in-training eval cut to 1000
# draws: inner ~2 h + ~100 small evals ~1.5 h + rollouts + one final
# 5000-draw eval => ~5-7 h estimated; 16 h is 2x headroom. Fixed --tag so
# a preemption resubmit reuses the run dir; NO mid-run resume in this
# runner — a preempted job restarts from step 0.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional arguments, not --export: passing values through the environment
# made this cluster's controller fail user-env retrieval and hold the job.
CFG=${1:?usage: sbatch d16_unconstrained_control.sbatch <cfg> [seed]}
SEED=${2:-42}
TAG=20260818-d16-unconstrained

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
