#!/bin/bash
#SBATCH --job-name=d256-cvcont
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# CV continuation of the b512+ne512 recipe: continue the seed-43 recipe
# run's final.pt with the control variate
# re-enabled, flat sigma_c, 20k steps, lr at the ladder's final 3e-4 —
# the cv2 pattern at the recipe shape. Tripwire ARMED in the cell
# (halt_on_cv_inversion_after=2000, window 10): a sustained
# controlled/naive integrand-variance inversion halts the run, which is
# the designed cost-capped negative outcome. Expected ranges are in the
# configs.py cell comment.
# Fixed tag reuses the run directory; resume.pt overrides the loaded init.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne512
TAG=20260819-d256-cvcont
INIT=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed43_20260818-d256-recipe-s43/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=43 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 43 --tag "$TAG" \
    --init-from "$INIT"
