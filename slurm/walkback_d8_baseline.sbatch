#!/bin/bash
#SBATCH --job-name=wb8-base
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Walk-back-to-8x8 slate (2026-08-12), baseline family: the D=8 twin of the
# d10 critical paper-curriculum cell, one seed per submission, seeds 42-45
# to match the archived d10 four-seed record.
#   sbatch -J wb8-base-s42 slurm/walkback_d8_baseline.sbatch \
#       stage_4_d8_critical_paper_curriculum 42
# Walltime: the archived d10 twin decomposes to ~9.5 h end-to-end on an A30
# (200k inner updates ~1.6 h; the 400 in-training 5000-sample evals dominate,
# calibrated from the one cluster-timed 50k ne128 soft run at ~103 s/eval,
# halved for ne64). Scaled by 64/100 sites ~= 6 h; 2x headroom -> 12 h.
# Fixed --tag so a job resubmitted after preemption reuses the run dir and a
# completed run is skipped, not retrained. NO mid-run resume in this runner:
# a preempted job restarts from step 0.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional arguments, not --export: passing values through the environment
# made this cluster's controller fail user-env retrieval and hold the job.
CFG=${1:?usage: sbatch walkback_d8_baseline.sbatch <cfg> [seed]}
SEED=${2:-42}
TAG=20260812-walkback

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
