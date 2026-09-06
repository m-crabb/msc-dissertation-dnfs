#!/bin/bash
#SBATCH --job-name=gfn-d64-bench
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out

# d64 GFN gate: eager/compiled parity for both objectives and cached-rollout
# timing by phase and batch. Must pass on the launch stack before gfn_d64_wave:
# compile_policy=True requires venue certification; CPU tests do not certify it.
# Use the log's train-step projection to choose the wave's --time.
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
