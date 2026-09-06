#!/bin/bash
#SBATCH --job-name=d256-C-cvcont-h128L3-fromP03
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# h128L3 capacity continuation of the ne128 x CV family: a 20k CV
# continuation at fixed sigma_c, the capacity twin of the keystone's CV
# continuation (d256-A-cvcont-ne128). It is run from BOTH candidate h128L3
# parents (P and P_lr03) rather than from a parent chosen after seeing
# their endpoints; both endpoints are reported, and the comparison against
# the keystone continuation is made twice, so it carries a Bonferroni
# factor of 2 (alpha 0.05 -> 0.025 per arm). Expected ranges are in the
# configs.py cell comment. This job continues P_lr03.
#
# It cannot continue from the keystone directly: hidden_dim 32 -> 128
# changes the shape of every weight matrix, so it needs an h128L3 parent,
# which is exactly what P and P_lr03 are.
#
# Fixed --tag reuses the run directory; resume.pt overrides --init-from.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128
TAG=20260822-C-cvcont-h128L3-fromP03
INIT=results/03_hard/H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive_seed42_20260821-P03-h128L3-lr03/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG" \
    --init-from "$INIT"
