#!/bin/bash
#SBATCH --job-name=dnfs-mal-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: mars permanently withdrawn 2026-08-31; do not submit.
#
# 8x8 whole-lattice attention window (`mal`), six cells run SEQUENTIALLY on
# ONE borrowed B200 (2026-08-28).
#
# Serial runs keep wall_clock_step_s free of MPS contention. The window's
# +20%/step at the 4x4 gate may reflect launch-bound mask construction,
# not asymptotic cost. Estimated six-cell budget: ~3.25h at B200's ~2x A100
# rate, within the 6h request.
#
# Borrowed-node rules: spillover only, one GPU, dnfs-* name, bounded walltime.
# Historical idle check (2026-08-28): GPUs 4-7 at 0 MiB / 0%, owner's four
# at ~173 GB / 100%, Slurm gres/gpu:4, no pending jobs. Recheck physical
# occupancy, Slurm allocation and owner pending jobs before submission.
#
# sigma_c first preserves the discriminating half if walltime runs out;
# sigma=0.10 saturates near the sampling floor (all d256 heads 0.99-1.00).
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache
# Blackwell fix: job 2344 failed every cell's step-0 diagnostic on 2026-08-28
# with `InductorError: Cannot find ptxas-blackwell`. Triton's arch >= 100
# assembler is knobs.nvidia.ptxas_blackwell (triton/knobs.py:492; env naming
# at :194-201); the locked env has plain ptxas only. Mars CUDA 13.1 ptxas
# compiled a correct kernel on B200 capability (10, 0) before relaunch.
# Earlier sm_100 verification covered native kernels/eager matmul only,
# not Inductor; compile_head=True exposed the missing assembler.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
TAG=20260828-win-d64

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for CFG in H2_d64_c50_s220_letf_mal_50k_curr_win H2_d64_c50_s010_letf_mal_50k_win; do
  for SEED in 42 43 44; do
    echo "=== $(date -Is)  cfg=$CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG" || echo "FAILED cfg=$CFG seed=$SEED"
  done
done
echo "=== ALL SIX DONE $(date -Is) ==="
