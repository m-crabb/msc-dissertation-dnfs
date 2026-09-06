#!/bin/bash
#SBATCH --job-name=d256-A-cvcont-ne128
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# ne128 x CV composition family: the keystone continued under the control
# variate for 20k steps at fixed sigma_c.
# Expected ranges in the configs.py cell comment; one-variable relationships
# pinned by test_ne128_cv_family_arms_are_one_variable_twins_of_their_comparators.
# Fixed --tag reuses the run directory; resume.pt overrides --init-from.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne128
TAG=20260821-A-cvcont-ne128
INIT=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive_seed42_20260819-mars-grid-depth-a/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG" \
    --init-from "$INIT"
