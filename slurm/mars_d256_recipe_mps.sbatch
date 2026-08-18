#!/bin/bash
#SBATCH --job-name=dnfs-recipe-mps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# mars-node (B200 183GB) twin of d256_screen_arm.sbatch: runs BOTH recipe
# seeds concurrently on the single allocated GPU under user-level MPS.
# Rationale: each seed peaks <=80GB (proven on the DoC A100-80GB), so two
# fit in 183GB; without MPS two CUDA contexts merely time-slice (~serial),
# with MPS their kernels co-schedule and the pair finishes in ~1.2-1.5x a
# single run. Etiquette (docs/imperial-doc-cluster.md section 8): one GPU,
# dnfs-* name, bounded walltime. 36h covers even a fully-serialised
# degraded co-run (B200 single-seed measured ~9-12h at 50k steps).
# wall_clock_step_s from a co-run is contention-contaminated: never quote
# timings from these runs, only the ESS/FVU verdict bands.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
CFG=${1:-H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive}
SEED_A=${2:-42}
SEED_B=${3:-43}
TAG_PREFIX=${4:-20260818-d256-recipe}

# User-level MPS scoped to this job: private pipe/log dirs keep the daemon
# invisible to the box's other users and let the EXIT trap tear it down on
# any end state (completion, failure, scancel).
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
