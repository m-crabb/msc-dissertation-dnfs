#!/bin/bash
#SBATCH --job-name=gfn-d256-bench
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# d256 GFN launch gate + bench (s102): the gfn_d64_launch_bench sibling at
# --rung d256. Compile numerical parity (eager vs compiled scoring, both
# objectives, at the 16x16 size) and the cached-rollout wall clock. MUST
# PASS on this stack before any gfn_d256_wave job is submitted -- the wave
# ships compile_policy=True and inductor kernels are certified per venue
# AND size, never inherited from the d64 gate. The train-step timing
# replaces the extrapolated wall-time estimate: read the projected hours
# from this log before trusting the wave's --time.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.gfn_launch_bench --rung d256
