#!/bin/bash
#SBATCH --job-name=dnfs-grid-depth-mps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# mars-node (B200 183GB) variant of mars_d256_recipe_mps.sbatch: runs TWO
# DIFFERENT cells at ONE seed on the single allocated GPU under user-level
# MPS, where the parent script runs one cell at two seeds. Same rationale
# (each d256 b512 run peaks <=80GB, proven on the DoC A100-80GB, so two fit
# in 183GB; without MPS two CUDA contexts merely time-slice) and the same
# etiquette envelope: one GPU, dnfs-* name, bounded walltime. 30h covers a
# fully-serialised degraded co-run (B200 single-seed measured ~9-12h at 50k
# steps for the ne512 shape; the ne128 arm carries a quarter of that
# rollout). wall_clock_step_s from a co-run is contention-contaminated:
# never quote timings from these runs, only the verdict bands.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
CFG_A=${1:?cfg A required}
CFG_B=${2:?cfg B required}
SEED=${3:-42}
TAG=${4:-mars-grid-depth}

export CUDA_MPS_PIPE_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control' EXIT

cd "$REPO"
echo "node=$(hostname) cfgA=$CFG_A cfgB=$CFG_B seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG_A" --seed "$SEED" --tag "${TAG}-a" \
    > "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_a.out" 2>&1 &
PID_A=$!
"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG_B" --seed "$SEED" --tag "${TAG}-b" \
    > "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_b.out" 2>&1 &
PID_B=$!

wait "$PID_A"; EXIT_A=$?
wait "$PID_B"; EXIT_B=$?
echo "cfgA exit=$EXIT_A cfgB exit=$EXIT_B"
exit $(( EXIT_A || EXIT_B ))
