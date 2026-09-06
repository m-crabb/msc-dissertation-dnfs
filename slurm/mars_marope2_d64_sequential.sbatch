#!/bin/bash
#SBATCH --job-name=dnfs-marope2-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: the borrowed mars node is no longer available; do not submit.
#
# 8x8 sigma_c: the PERIODIC-RoPE BACKBONE at PATCH SIZE 2 under the
# masked-attention head, three seeds SEQUENTIAL on one borrowed B200.
# Companion to the p = 1 job; read as the triple
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
# WHY IT EARNED A SLOT -- COST, NOT A SECOND QUALITY READ. Keys per query are
# 1 + pL + d/p^2, which at d = 64 (L = 8) reads 65 / 33 / 37 for p = 1/2/4:
# the optimum is INTERIOR, p = 4 is worse than p = 2 on cost and coarser in
# the far field too, so p = 2 is the only larger p worth running at this rung.
#
# MEASURED OUTCOME. The reasoning above was WRONG on both halves, and the
# claim "the lever matters at scale rather than here" is FALSE.
#
# 1. NO SPEED. Measured on this node: 8.402 ms/step against p=1's 8.263, and
#    17m07s against 16m33s end to end per seed. p = 2 is marginally SLOWER.
#    Pooling shrinks the BACKBONE's causal attention, but the masked-attention
#    head's (B, d^2, n) band-score tensor is what dominates -- 5.00 GB at
#    B=32, d=256, back-solving to n ~ 3d. So the head is O(d^3) and the
#    backbone O(d^2), and at d=256 the head's element count is ~150x the
#    backbone's. Halving under 1% of the work is invisible.
# 2. AND THE ASYMPTOTICS RUN THE WRONG WAY. Because the head grows as d^3 and
#    the backbone as d^2, the backbone's SHARE SHRINKS with d -- so p = 2 is
#    LESS useful at 16x16 than here, not more.
# 3. WORSE, IT COSTS QUALITY. Seeds 42/43 read 0.648/0.697 raw and
#    0.709/0.750 EMA against `ma`'s 0.7814 +- 0.0193 / 0.8458 +- 0.0219 --
#    disjoint BELOW, and below `mar` too. The far field at patch resolution
#    loses information the band needs.
#
# So p = 2 loses on both axes. p = 1 is the keeper (level quality at
# -8.2% parameters). The lever that actually reaches the head's cost is the
# BAND FACTORISATION, which attacks the (B, d^2, n) tensor itself.
#
# WHAT IT COSTS IN EXACTNESS. Pooling aliases shifts that are not multiples
# of p, so the bidirectional stack's exact torus equivariance holds on pZ^2
# only, and the far field is seen at patch resolution. Blindness is UNTOUCHED
# -- every far patch lies entirely before the query, which is what the
# admission rule enforces. The pooled patch key is the MEAN over members of
# the ROTATED site keys, so the patch logit is the mean of the member logits
# and the periodicity argument survives term by term.
#
# Submitted with --dependency=afterany on the p = 1 job to respect the
# borrowed-node allowance of one concurrent GPU; dependency adds latency.
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
