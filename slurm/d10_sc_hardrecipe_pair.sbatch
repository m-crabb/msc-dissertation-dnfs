#!/bin/bash
#SBATCH --job-name=d10-hr-pair
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --nice=1000
#SBATCH --output=slurm/logs/%x_%j.out

# 10x10 unconstrained sigma_c on the 16x16 100k hard house recipe with the
# exact-field channel: two seeds CO-RUN on one card so the
# four-seed set fits the two free submission slots. At d=100 the let 128x3
# flip model leaves most of an A100 idle per process, so the pair overlaps
# at well under 2x the single-seed time; wall_clock_step_s from a packed
# card is contaminated and is never quoted. --nice=1000 keeps the pending
# 24x24 sigma_c array ahead of these on the a100 partition.
#   sbatch -J d10-hr-s42s43 slurm/d10_sc_hardrecipe_pair.sbatch \
#       stage_4_d10_sc_hardrecipe_efc 42 43
# Fixed --tag so a resubmission reuses the run dirs; the baseline runner has
# no mid-run resume, so a preempted job restarts both seeds from step 0.
# Moved to Modal 2026-09-07: the pairs sat pending on the 3-GPU cap for four
# days behind the 24x24 GFN array. Relaunched unpacked, one seed per
# A100-80GB, tag 20260907-d10-hardrecipe via
# dnfs_baseline_01.modal_app::batch_seeds; kept here as the recipe record.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=${1:?usage: sbatch d10_sc_hardrecipe_pair.sbatch <cfg> <seedA> <seedB>}
SEED_A=${2:?seedA}
SEED_B=${3:?seedB}
TAG=20260903-d10-hardrecipe

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seeds=$SEED_A,$SEED_B tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SEED in "$SEED_A" "$SEED_B"; do
    "$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG" \
        > "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_s${SEED}.out" 2>&1 &
done
wait
echo "pair finished: exit statuses above are per-seed logs"
