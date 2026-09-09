#!/bin/bash
#SBATCH --job-name=gfn-d256-bench
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# d256 GFN gate: eager/compiled parity for both objectives and cached-rollout
# timing at 16x16. Must pass on the launch stack before gfn_d256_wave:
# compile_policy=True requires certification per venue AND size, not d64's gate.
# Check the log's train-step projection against the wave's estimated --time.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.probes.gfn_launch_bench --rung d256
