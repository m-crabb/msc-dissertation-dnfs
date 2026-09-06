#!/bin/bash
#SBATCH --job-name=dnfs-recipe-mps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: mars permanently withdrawn 2026-08-31; do not submit.
#
# B200 183GB twin of d256_screen_arm.sbatch: two recipe seeds under MPS.
# Each peaks <=80GB on DoC A100-80GB. Recorded packing rationale: MPS
# co-schedules kernels (~1.2-1.5x single-run time); separate CUDA contexts
# time-slice (~serial). B200 measured ~9-12h/seed at 50k steps; 36h allows
# a serialised pair. Borrowed-node rules (docs/imperial-doc-cluster.md §8):
# one GPU, dnfs-* name, bounded walltime. Co-run wall_clock_step_s includes
# contention: quote ESS/FVU verdict bands only, never timings.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
CFG=${1:-H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive}
SEED_A=${2:-42}
SEED_B=${3:-43}
TAG_PREFIX=${4:-20260818-d256-recipe}

# Job-private MPS pipe/log directories; the EXIT trap requests shutdown.
export CUDA_MPS_PIPE_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control' EXIT

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seeds=$SEED_A,$SEED_B tag_prefix=$TAG_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED_A" --tag "${TAG_PREFIX}-s${SEED_A}" \
    > "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_s${SEED_A}.out" 2>&1 &
PID_A=$!
"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED_B" --tag "${TAG_PREFIX}-s${SEED_B}" \
    > "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_s${SEED_B}.out" 2>&1 &
PID_B=$!

wait "$PID_A"; EXIT_A=$?
wait "$PID_B"; EXIT_B=$?
echo "seed${SEED_A} exit=$EXIT_A seed${SEED_B} exit=$EXIT_B"
exit $(( EXIT_A || EXIT_B ))
