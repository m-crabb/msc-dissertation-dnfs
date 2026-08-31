#!/bin/bash
#SBATCH --job-name=gfn-d64
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# GFN 8x8 wave (s94): one cell per job, THREE SEEDS SERIAL INSIDE IT (the
# rasterord_d64_floor pattern -- 12 cells x 3 seeds as an array would sit
# 2/3 PENDING behind the QOS cap while no cell completed its band):
#   sbatch -J gfn-d64-<short> slurm/gfn_d64_wave.sbatch <cfg name>
#
# THE TWELVE CELLS. Centres at both couplings (the printed rows):
#   GFN_d64_c50_s010_{tb,fldb}_50k_par
#   GFN_d64_c50_s220_{tb,fldb}_50k_par
# plus the discriminating star, sigma_c only (d64 house ESS spans
# 0.735-0.953 there; the 4x4 grid was a saturated filter):
#   GFN_d64_c50_s220_{tb,fldb}_50k_{l3e4_e005,l3e3_e005,l1e3_e000,l1e3_e010}_swp
# QOS admits 8 queued: submit the four centres + first star arms, feed the
# rest as slots free (or chain with --dependency=singleton per name).
#
# GATE FIRST: gfn_d64_launch_bench.sbatch must have PASSED on this stack
# before the first submission -- these cells ship compile_policy=True.
#
# Fixed --tag so a resubmission resumes: train_gfn short-circuits a seed
# whose eval set is complete (eval/ AND eval_ema/ -- the dual-eval
# instrument) and continues a partial one from resume.pt (EMA shadow and
# its warmup counter ride the checkpoint).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=${1:?usage: sbatch gfn_d64_wave.sbatch <cfg name> [tag]}
# Default = the s94 wave tag; later arms (e.g. the s100 flr pair) pass
# their own so a resubmission resumes against the right dirs.
TAG=${2:-20260830-gfn-d64}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG tag=$TAG seeds=42,43,44"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SEED in 42 43 44; do
    echo "=== $CFG seed $SEED starting $(date -Is) ==="
    "$PIXI" run --frozen -e cuda python -m \
        experiments.constrained_hard_03.run_gfn \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
done
echo "=== $CFG all seeds done $(date -Is) ==="
