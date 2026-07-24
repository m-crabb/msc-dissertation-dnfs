#!/bin/bash
#SBATCH --job-name=train-smoke-off
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
cd /vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
echo "node=$(hostname)"
/vol/bitbucket/mc625/.pixi/bin/pixi run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d16_c50_s223_letf_dh --smoke --tag slurm_smoke_offline
