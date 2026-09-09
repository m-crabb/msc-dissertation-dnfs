#!/bin/bash
#SBATCH --job-name=fmo2-bench-d256
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# What does the fmo2 head ACTUALLY cost at d=256, and are we getting the
# speedup we should?
#
# Two gaps this closes.
#
# 1. fmo2 has never been benched. `--site-orderings row,col` runs TWO causal
#    streams, so its forward is roughly double the single-ordering fab8
#    arm's 20.5 ms — yet every fmo2 cost figure on record inherits fab8's
#    number and is optimistic by up to 2x. The decision this feeds:
#    converting to cost per effective sample at d=256, fmo2's variance
#    factor of 1.19 means it needs 1.54x more draws than masked attention,
#    against MA's 94.0 ms — so fmo2 must beat 61.0 ms/forward to be a net
#    win. Every other head already fails its own break-even (interval needs
#    < 16.3 ms and is 31.8; fab8 needs < 1.9 ms and is 20.5).
#
# 2. Batch saturation. Every d=256 cell uses eval_sample_chunk=64, tuned for
#    masked attention at 5.00 GB peak. The factorised family peaks 0.85 GB
#    at B=32, so it could run 1024-2048 — 16-32x fewer sequential launches
#    than it is getting — and at B=32 the rank-8 GEMMs badly underutilise an
#    A100. The batch ladder measures how much throughput that leaves on the
#    table, in ms per SAMPLE rather than ms per forward.
#
# The masked-attention control arm is deliberate: it must reproduce the
# archived 95.3 ms / 5.0 GB (Modal measured 94.0 / 5.00, a 1.4% match) or
# the protocol has drifted and the fmo2 numbers are not comparable.
#
# a100 for its 80 GB: the batch=2048 arm wants ~54 GB and OOMed a 40 GB card
# on Modal (exit 144) — Modal's bare gpu="A100" is the 40 GB variant, while
# this partition is nvidia_a100_80gb_pcie.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN="$PIXI run --frozen -e cuda python -m \
    experiments.constrained_hard_03.probes.profile_swap --mode head --d 256 --sdpa"

echo "### control: masked_attention must reproduce ~94 ms / 5.0 GB"
$RUN --head-kind masked_attention --batch 32

for BATCH in 32 128 512 2048; do
    echo "### fmo2 (row,col) batch=$BATCH"
    $RUN --head-kind factorised --site-orderings row,col --batch "$BATCH"
done

for BATCH in 512 2048; do
    echo "### fab8 (row) batch=$BATCH  [single-ordering reference]"
    $RUN --head-kind factorised --batch "$BATCH"
done
