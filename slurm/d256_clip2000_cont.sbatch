#!/bin/bash
#SBATCH --job-name=d256-clip2000-cont
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Clip-threshold continuation (2026-08-15): the completed naive rescue
# continued from its final checkpoint for 10k steps at flat sigma_c with
# grad_clip_max_norm 500 -> 2000 the only mechanism change (estimator stays
# naive — the archived cv2 continuation switched estimator at this same
# juncture, which is why it cannot serve as a clip control; pinned in
# test_screen_pins). The clip fires on ~25% of the rescue's final-plateau
# steps (p50 grad norm 384, p99 ~1050, ~9-10% of gradient magnitude
# removed); at threshold 2000 that tail runs essentially unclipped, so this
# isolates the brake's steady-state cost with none of the early-phase
# divergence risk a cold unclipped run takes (the clip=20000 smoke arm
# never escaped rung 0). Null band = the rescue's own flatness continuing
# (stage-tail FVU 0.137 +- 0.005, eval Var/site 0.0707 at N=5000).
# Fixed tag reuses the run directory; resume.pt overrides the loaded init.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260815-d256-clip2000-cont
INIT=results/03_hard/H2_d256_c50_s223_letf_ma_50k_curr_naive_seed42_20260813-d256-rescue/checkpoints/final.pt

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_c50_s223_letf_ma_10k_sc_clip2000 seed=42 tag=$TAG init_from=$INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_c50_s223_letf_ma_10k_sc_clip2000 --seed 42 --tag "$TAG" \
    --init-from "$INIT"
