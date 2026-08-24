#!/bin/bash
#SBATCH --job-name=fc-c030-topup
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --array=0-3

# c=0.30 ne128 top-up (2026-08-24, s62): the launch prereg's thin-tail clause
# (top up if fewer than three seeds clear the 0.30 ESS floor) fired at 0/4 --
# seeds 42-45 evaled at ESS fraction 0.076/0.030/0.049/0.0005, against 3/6
# passing in the ne64 family (0.84/0.80/0.38 best three). A git-history
# provenance check cleared the code: the only behavioural deltas versus the
# recovering 2026-08-09 ne64 launches are the Euler grid (64 -> 128) and the
# seed draw; every newly serialized config field is inert on the flip route.
# Frozen judgement for this family: c=0.30 prints from ne128 only if >= 3 of
# the 8 total seeds clear the 0.30 floor; otherwise the printed point stays
# on the ne64 family corrected by its own measured eval-grid offset and is
# marked as trained on the coarser grid.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
SEEDS=(46 47 48 49)
CFG=S2_d10_c030_l50_letf_ne128_anneal
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
    --cfg "$CFG" --seed "$SEED"
