#!/bin/bash
#SBATCH --job-name=dnfs-marope2-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 8x8 sigma_c: the PERIODIC-RoPE BACKBONE at PATCH SIZE 2 under the
# masked-attention head, three seeds SEQUENTIAL on one borrowed B200
# (2026-08-28). Companion to the p = 1 job; read as the triple
# `ma` -> rope1 -> rope2.
#
# WHAT p = 2 ADDS OVER p = 1. p = 1 is dense causal attention in two pieces,
# so it changes the position code and nothing else. p = 2 additionally splits
# the keys: `near` = individual sites in the query's own patch-row band,
# `far` = ONE POOLED KEY PER PATCH, admitted only when every member site
# precedes the query. So this cell is NOT single variable against `ma`; it is
# single variable against p = 1, and the pair separates the code from the
# pooling.
#
# WHY IT EARNS A SLOT -- COST, NOT A SECOND QUALITY READ. Keys per query are
# 1 + pL + d/p^2, which at d = 64 (L = 8) reads 65 / 33 / 37 for p = 1/2/4:
# the optimum is INTERIOR, p = 4 is worse than p = 2 on cost and coarser in
# the far field too, so p = 2 is the only larger p worth running at this
# rung. The lever matters at scale rather than here -- at d = 256 (L = 16)
# the same formula reads 257 / 97 / 81 -- and the masked-attention head's
# footprint is exactly what blocks it at that size. "RoPE is quality-neutral
# AND p = 2 cuts attention cost" is a materially different result from "RoPE
# is quality-neutral", which is the whole reason to spend the seeds.
#
# WHAT IT COSTS IN EXACTNESS. Pooling aliases shifts that are not multiples
# of p, so the bidirectional stack's exact torus equivariance holds on pZ^2
# only, and the far field is seen at patch resolution. Blindness is UNTOUCHED
# -- every far patch lies entirely before the query, which is what the
# admission rule enforces. The pooled patch key is the MEAN over members of
# the ROTATED site keys, so the patch logit is the mean of the member logits
# and the periodicity argument survives term by term.
#
# DEPENDENCY, NOT A SECOND CONCURRENT GPU. Submitted with
# --dependency=afterany on the p = 1 job so this node never carries more than
# one of our GPUs at a time, which is the standing etiquette on borrowed
# capacity. It costs latency and nothing else.
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache
# Mandatory on Blackwell: Triton resolves a separate `ptxas-blackwell` for
# sm_100 and the locked env ships only plain `ptxas`, so a compiled cell dies
# at step 0 without this. The cell inherits compile_head=True, and compiling
# the head compiles the backbone forward with it.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
TAG=20260828-rope-ma-d64

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CFG=H2_d64_c50_s220_rope2_ma_50k_curr_w2
for SEED in 42 43 44; do
  echo "=== $(date -Is)  cfg=$CFG seed=$SEED ==="
  "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
      --cfg "$CFG" --seed "$SEED" --tag "$TAG" || echo "FAILED cfg=$CFG seed=$SEED"
done
echo "=== ALL THREE DONE $(date -Is) ==="
