#!/bin/bash
#SBATCH --job-name=m3-d256-ctb512
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# c_t_batch mechanism cell: the smoke12k naive recipe with c_t_batch
# 128 -> 512 the ONLY declared change (cell comment in configs.py;
# twin-ness pinned by
# test_m3_ctb512_smoke_mirrors_naive_arm_except_declared_fields), read
# against the ARCHIVED naive 50k run's first 12k at matched steps. c_t's
# per-slot standard error halves (variance /4) while the inner batch and
# replay buffer stay at 128. Expected: meaningful = train-ESS median on
# the sigma=0.17 rung (steps 10k-12k) >= 2x the archived naive run's at the
# same steps, OR the logged per-slot var_estimator_integrand down >= 2x.
# Never bundle with the c_t EMA lever — one lever per cell. Cost ~5 h a100 (trajectory
# phase +2.25x on ~75% of wall); the no-grad MA pass at B=512 peaks ~20 GB,
# in-cap on the 80 GB a100. Fixed --tag so a resubmission resumes.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-m3-ctb512

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_smoke12k_naive_ctb512 seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_smoke12k_naive_ctb512 --seed 42 --tag "$TAG"
