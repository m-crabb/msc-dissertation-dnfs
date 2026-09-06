#!/bin/bash
#SBATCH --job-name=d256-cv2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Phase-2 estimator switch at 16x16 (2026-08-14, user GO): continue the
# completed naive rescue (its final.pt, ~/results path below) with the Stein
# control variate re-enabled, flat sigma_c, 20k steps. Mechanism measured on
# the trained checkpoint before launch: CV integrand-variance ratio 0.140
# (reduces 7x; it ADDED variance on the diverged model). Bands frozen in the
# cell comment (configs.py): PASS = train-ESS > 15/256 sustained+rising over
# the final 5k AND eval ess_fraction >= 0.05.
# Fixed tag reuses the run directory; resume.pt overrides the loaded init.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-d256-cv2
INIT=results/03_hard/H2_d256_c50_s223_letf_ma_50k_curr_naive_seed42_20260813-d256-rescue/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_20k_sc_cv2 seed=42 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_20k_sc_cv2 --seed 42 --tag "$TAG" \
    --init-from "$INIT"
