#!/bin/bash
#SBATCH --job-name=dnfs-marope-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Historical launcher: the borrowed mars node is no longer available; do not submit.
#
# 8x8 sigma_c: the PERIODIC-RoPE BACKBONE under the masked-attention head,
# three seeds run SEQUENTIALLY on one borrowed B200.
#
# WHAT THIS ARM ASKS. `ma` builds its band's keys and values from a backbone
# carrying d FREE ABSOLUTE position vectors, so the network must learn from d
# independent embeddings that site 0 and site d-L are bonded across the wrap.
# The RoPE backbone has no position parameters at all: positions enter as
# rotary phases theta_s = 2 pi m (row_s, col_s) / L with m integer, so every
# backbone attention logit is a function of the signed torus offset alone,
#     q_s . k_s' -> q_s . R(theta_s' - theta_s) k_s'.
# Single variable against the `ma` w2 sigma_c twins: `model.kind` and nothing
# else, pinned by test.
#
# WHY IT IS NOT A FOURTH REPEAT OF THE SAME NULL. Three arms have now come
# back null or negative on this head -- the whole-lattice window (`mal` 0.831
# +- 0.030 EMA against `ma` 0.846 +- 0.022 at 8x8 sigma_c), the band query's
# relative pair code (`mar` 0.788 +- 0.025 EMA, disjoint BELOW on raw and EMA
# both, Var[log w] 46% worse), and RoPE itself. But RoPE was measured on the
# FACTORISED head (rope1/rope2 fimo2 twins 0.861-0.933 and 0.884-0.915 at
# sigma_c against the parent's 0.903-0.925), never on this one. `mar` changed
# the band QUERY's own pair-level embedding; RoPE changes the site-level code
# feeding the band's KEYS and VALUES -- disjoint tensors, different layers.
# The `mar` null says the pair code is not the lever; it says nothing about
# the terms being pooled. Masked attention is the head that reads d free
# position vectors through a softmax, so if the backbone code ever pays it
# pays here.
#
# p = 1, NOT p = 2. At p = 1 the causal stacks are dense attention in two
# pieces, so the only change is the position code. p = 2 additionally pools
# far keys into 2x2 patches, a second variable and a d-scaling cost lever,
# and aliases shifts that are not multiples of p. Both landed in the same
# place on the factorised head.
#
# WHY SEQUENTIAL AND NOT MPS. The co-run pattern works but contaminates
# `wall_clock_step_s` beyond quoting, and the cost of this backbone is a live
# question -- it benched FREE on an A100 at d=256 (24.0 against leTF's 24.8
# ms, identical 1.93 GB) but has never been compiled anywhere.
#
# SIGMA_C ONLY. The sigma = 0.10 floor saturates at every rung (`mar` read
# 0.9863 there against `ma`'s 0.9845) and cannot discriminate, so the three
# seeds all go to the coupling that does.
#
# Borrowed-node rules: spillover only, one GPU, dnfs-* name, bounded walltime.
# Idle check at the time: GPUs 4-7 at 0 MiB / 0%, owner's four
# at ~173 GB, Slurm gres/gpu:4 for job 2342, no pending jobs. Recheck physical
# occupancy, Slurm allocation and owner pending jobs before submission.
#
# Results written here are NOT synced back automatically; `results/` is
# excluded from the rsync in both directions.
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIXI_CACHE_DIR=$HOME/.pixi-cache
# Use mars CUDA 13.1 ptxas for Blackwell: Triton selects ptxas-blackwell for
# arch >= 100, absent from the locked env. Without it, torch.compile fails
# the step-0 diagnostic with `InductorError: Cannot find ptxas-blackwell`.
# Inherited compile_head=True also compiles the backbone forward; this was
# RoPEViTRateMatrix's first Inductor run.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
TAG=20260828-rope-ma-d64

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CFG=H2_d64_c50_s220_rope1_ma_50k_curr_w2
for SEED in 42 43 44; do
  echo "=== $(date -Is)  cfg=$CFG seed=$SEED ==="
  "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
      --cfg "$CFG" --seed "$SEED" --tag "$TAG" || echo "FAILED cfg=$CFG seed=$SEED"
done
echo "=== ALL THREE DONE $(date -Is) ==="
