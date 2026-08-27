#!/bin/bash
#SBATCH --job-name=dnfs-mal-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 8x8 whole-lattice attention window (`mal`), six cells run SEQUENTIALLY on
# ONE borrowed B200 (2026-08-28).
#
# WHY SEQUENTIAL AND NOT MPS. The co-run pattern exists and works, but it
# contaminates `wall_clock_step_s` beyond quoting, and the cost of this
# window is precisely an open question -- it measured +20% per step at the
# 4x4 gate, in a launch-bound regime where that is probably an artefact of
# mask construction rather than the asymptotic price. Running serially keeps
# those numbers usable. Six cells at a B200's ~2x A100 rate is ~3.25 h
# against a 6 h request.
#
# ETIQUETTE. Borrowed capacity, spillover only: one GPU, dnfs-* name,
# bounded walltime. Idle check passed before submission on BOTH required
# counts (2026-08-28): GPUs 4-7 physically empty at 0 MiB / 0% against the
# owner's four at ~173 GB / 100%, Slurm holding gres/gpu:4, and NO pending
# jobs at all -- so taking one of four free GPUs delays nobody.
#
# ORDER IS DELIBERATE: sigma_c first. That is the coupling that
# discriminates -- at every rung the sigma=0.10 row saturates near the
# sampling floor (every d256 head 0.99-1.00 there) -- so if the walltime is
# ever cut short, the informative half is already done.
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache

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
