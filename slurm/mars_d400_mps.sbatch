#!/bin/bash
#SBATCH --job-name=dnfs-d400-sc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: mars permanently withdrawn 2026-08-31; do not submit.
#
# 20x20 at sigma_c on one borrowed B200 under MPS (2026-08-29):
#
#     sbatch slurm/mars_d400_mps.sbatch <TAG> <CFG:SEED> [<CFG:SEED> ...]
#
# Choose the R=2/R=3 pack in the submit-line specs to fit GPU memory.
#
# WHY MPS AND NOT SEQUENTIAL. Two CUDA contexts on one card merely
# time-slice (~serial); under MPS their kernels co-schedule and a pack
# finishes in ~1.2-1.5x a single run rather than Nx. The cost is that
# `wall_clock_step_s` from a co-run is contention-contaminated. That is
# ACCEPTABLE HERE and would not have been at the floor: the floor wave
# already settled the precision COST half end-to-end on 4/4 matched seeds
# (-21% to -27%), and this wave is spent on the QUALITY half, which is an
# ESS/weight-variance read that contention does not touch. NOTHING from
# these runs may be quoted as a timing.
#
# THE PACK IS A MEMORY BUDGET, not a preference. Profiled on an A100-80GB at
# d=400, batch 512, compiled, single-shot backward: R=2 peaks 58.25 GB fp32 /
# 39.62 GB bf16, R=3 peaks 63.41 GB fp32 / 41.63 GB bf16. Against 183 GB that
# gives, per card, at most two fp32 clients plus one bf16 (156.1 GB at R=2,
# 168.5 GB at R=3 -- the latter too tight, so the R=3 wave packs 2+3 and
# trails a seed). THREE fp32 clients is 174.75 GB at R=2 and does NOT fit.
# These are A100 numbers on a different allocator, so the first card of each
# wave is verified from nvidia-smi before the second is submitted; the
# watcher below writes that evidence into this log.
#
# Borrowed-node rules (docs/imperial-doc-cluster.md section 8): spillover
# only, dnfs-* name, bounded walltime. Before submission, require physically
# empty GPUs in nvidia-smi AND free Slurm TRES; check owner pending jobs for
# delays. Historical two-card exception: four of eight were at 0 MiB / 0%
# with zero pending jobs; the usual allowance is one card.
#
# Fixed --tag reuses the run directory and any resume.pt after a wall kill;
# the 2-day request relies on resume within the partition's 7-day maximum.
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache
# Blackwell fix: job 2344 failed every cell's step-0 diagnostic on 2026-08-28
# with `InductorError: Cannot find ptxas-blackwell`. Triton selects that
# sm_100 assembler (triton/knobs.py:492); the locked env has plain ptxas only.
# Patch heads here compile; decision (c) exempts only factorised arms.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi

TAG=${1:?usage: mars_d400_mps.sbatch <TAG> <CFG:SEED> [<CFG:SEED> ...]}
shift
SPECS=("$@")
[ ${#SPECS[@]} -gt 0 ] || { echo "no CFG:SEED specs given"; exit 2; }

# Job-private MPS pipe/log directories; the EXIT trap requests shutdown.
export CUDA_MPS_PIPE_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control' EXIT

cd "$REPO"
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG clients=${#SPECS[@]}"
printf '  spec: %s\n' "${SPECS[@]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Log GPU memory/utilisation samples for the next wave's packing decision.
( for WAIT in 300 600 1800 5400; do
    sleep "$WAIT"
    echo "=== mem @ $(date -Is) ==="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
  done ) &
WATCHER=$!

PIDS=(); LABELS=()
for SPEC in "${SPECS[@]}"; do
  CFG=${SPEC%:*}
  SEED=${SPEC##*:}
  LOG="slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${CFG}_s${SEED}.out"
  echo "=== launch $(date -Is) cfg=$CFG seed=$SEED -> $LOG ==="
  "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
      --cfg "$CFG" --seed "$SEED" --tag "$TAG" > "$LOG" 2>&1 &
  PIDS+=($!)
  LABELS+=("${CFG}:${SEED}")
done

RC=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"; EXIT=$?
  echo "${LABELS[$i]} exit=$EXIT"
  [ "$EXIT" -eq 0 ] || RC=1
done
kill "$WATCHER" 2>/dev/null
echo "=== PACK DONE $(date -Is) rc=$RC ==="
exit $RC
