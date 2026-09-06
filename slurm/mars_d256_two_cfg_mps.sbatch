#!/bin/bash
#SBATCH --job-name=dnfs-grid-depth-mps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: mars permanently withdrawn 2026-08-31; do not submit.
#
# B200 183GB variant of mars_d256_recipe_mps.sbatch: two configs, one seed.
# Each d256 b512 run peaks <=80GB on DoC A100-80GB; MPS co-schedules kernels
# otherwise time-sliced by separate CUDA contexts. B200 measured ~9-12h
# per ne512 seed at 50k steps; ne128 has a quarter of that rollout and 30h
# allows a serialised pair. Borrowed-node rules: one GPU, dnfs-* name,
# bounded walltime. Co-run wall_clock_step_s includes contention: quote
# verdict bands only, never timings.
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
