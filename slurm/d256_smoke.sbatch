#!/bin/bash
#SBATCH --job-name=d256-smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out

# Pre-launch A100 smoke: d256 masked attention, matching-step training and
# eval chunking; 4 steps / 8 Euler steps / 64 eval samples. Scratch output
# keeps smoke artefacts outside the results tree.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname) smoke of H2_d256_c50_s223_letf_ma_50k_curr"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_50k_curr --smoke --no-wandb \
    --output-dir /tmp/d256_smoke_$SLURM_JOB_ID

echo "smoke exit code: $?"
