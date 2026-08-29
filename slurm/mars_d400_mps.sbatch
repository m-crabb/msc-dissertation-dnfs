#!/bin/bash
#SBATCH --job-name=dnfs-d400-sc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 20x20 at sigma_c on ONE borrowed B200, several cells co-resident under
# user-level MPS (2026-08-29). SPEC-DRIVEN on purpose:
#
#     sbatch slurm/mars_d400_mps.sbatch <TAG> <CFG:SEED> [<CFG:SEED> ...]
#
# The packing decision is a MEMORY decision that differs between the R=2 and
# R=3 waves, so it lives in the submit line rather than in a per-wave copy of
# this file.
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
# ETIQUETTE (docs/imperial-doc-cluster.md section 8). Borrowed capacity,
# spillover only: dnfs-* name, bounded walltime, and BOTH halves of the idle
# check must agree before submitting -- GPUs physically empty in nvidia-smi
# AND free in Slurm's TRES -- plus the owner's pending jobs read for whether
# ours would delay them. Two cards rather than the usual one is a deliberate
# deviation, taken because four of eight sat at 0 MiB / 0% with ZERO pending
# jobs on the node.
#
# A WALL KILL IS RECOVERABLE, NOT FATAL. --tag is fixed, so a requeue lands
# in the same run dir and resumes from resume.pt. That is what lets this ask
# for 2 days instead of padding to the partition's 7-day maximum.
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache
# BLACKWELL COMPILE FIX (2026-08-28, job 2344 died on this at step 0 of every
# cell). Triton resolves a separate assembler for sm_100 -- `ptxas-blackwell`,
# triton/knobs.py:492 -- and the locked env ships only plain `ptxas`, so
# torch.compile raises `InductorError: Cannot find ptxas-blackwell` inside the
# step-0 rate diagnostic. Every cell here is compiled (patch heads; decision
# (c) exempts only factorised arms), so this is load-bearing.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi

TAG=${1:?usage: mars_d400_mps.sbatch <TAG> <CFG:SEED> [<CFG:SEED> ...]}
shift
SPECS=("$@")
[ ${#SPECS[@]} -gt 0 ] || { echo "no CFG:SEED specs given"; exit 2; }

# User-level MPS scoped to this job: private pipe/log dirs keep the daemon
# invisible to the box's other users, and the EXIT trap tears it down on ANY
# end state (completion, failure, scancel, wall kill).
export CUDA_MPS_PIPE_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=$REPO/slurm/logs/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control' EXIT

cd "$REPO"
echo "node=$(hostname) job=$SLURM_JOB_ID tag=$TAG clients=${#SPECS[@]}"
printf '  spec: %s\n' "${SPECS[@]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Memory evidence for the packing decision, written into this log at the
# points where a compiled d400 step has reached steady state. Cheap, and it
# is the artefact the next wave's pack is sized from.
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
