#!/bin/bash
#SBATCH --job-name=dnfs-mar-d64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 8x8 RELATIVE torus position code (`mar`), six cells run SEQUENTIALLY on
# ONE borrowed B200 (2026-08-28).
#
# WHAT THIS ARM ASKS. `ma` indexes its band query's position embedding by
# ABSOLUTE site, so it must learn that sites 0 and d-1 are torus neighbours;
# `thp` -- which wins at every rung where both ran -- indexes by the SIGNED
# TORUS DISPLACEMENT of j from i, so translation-equivalent pairs share a
# code by construction. That is the one structural difference left between
# the families after the whole-lattice window came back NULL at 8x8 (`mal`
# 0.831 +- 0.030 EMA against `ma` 0.846 +- 0.022 at sigma_c, overlapping and
# on the wrong side). Single variable against the `ma` w2 twins.
#
# NOT what RoPE tested: that swapped the BACKBONE's position code, reaching
# only P_i and S_j, never this embedding. RoPE benched free on an A100
# (24.0 vs 24.8 ms at d=256, identical memory) and was a quality null.
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
# BLACKWELL COMPILE FIX (2026-08-28, job 2344 died on this at step 0 of every
# cell). Triton resolves a SEPARATE assembler for sm_100 -- `knobs.nvidia.
# ptxas_blackwell` at triton/knobs.py:492, i.e. `ptxas-blackwell`, chosen
# whenever arch >= 100 -- and the locked env ships only plain `ptxas`, so
# `torch.compile` raises `InductorError: Cannot find ptxas-blackwell` and the
# run dies inside the step-0 rate diagnostic. The env var is derived from the
# binary name (knobs.py:194-201), and mars carries a CUDA 13.1 ptxas that
# does support sm_100. Verified on a B200 before relaunch: capability (10, 0),
# a compiled kernel built and returned the right value.
#
# NOTE what the earlier "sm_100 verified" record does and does not cover: it
# established that the locked torch carries native Blackwell kernels and that
# a matmul fires EAGER. It did not exercise Inductor, which is why this only
# surfaced once the optimised recipe (compile_head=True) reached this node.
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas

REPO=$HOME/msc-dissertation-dnfs
PIXI=$HOME/.pixi/bin/pixi
TAG=20260828-relpos-d64

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for CFG in H2_d64_c50_s220_letf_mar_50k_curr_rel H2_d64_c50_s010_letf_mar_50k_rel; do
  for SEED in 42 43 44; do
    echo "=== $(date -Is)  cfg=$CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG" || echo "FAILED cfg=$CFG seed=$SEED"
  done
done
echo "=== ALL SIX DONE $(date -Is) ==="
