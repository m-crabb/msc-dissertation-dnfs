#!/bin/bash
#SBATCH --job-name=gfn-d64-bench
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out

# d64 GFN launch gate + bench (s94): compile numerical parity (eager vs
# compiled scoring, both objectives) and the cached-rollout wall clock with
# phase and batch named. MUST PASS on this stack before any gfn_d64_wave
# job is submitted -- the wave ships compile_policy=True, and inductor
# kernels are certified per venue, never inherited from the CPU tests.
# The train-step timing also replaces the design's wall-time guess: read
# the projected hours from the log before picking the wave's --time.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.gfn_launch_bench
