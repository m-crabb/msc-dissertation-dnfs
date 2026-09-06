#!/bin/bash
#SBATCH --job-name=efc-sweep
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Lambda-sweep exact-field-channel twins: rerun
# tab:soft-lambda-sweep with the closed-form flip channel, one declared
# change per cell (pinned by tests/test_exact_flip_channel.py). One job per
# (lattice, lambda) config with seeds 42-45 chained on the one card: the
# archived cells measure ~3 min (d4, 10k steps) and ~21 min (d10, 50k steps)
# per seed, so a full 4-seed chain sits far inside the wall even at a30
# speeds. A stuck seed is a RESULT here (the printed 10x10 lambda>=50 rows
# are all-or-nothing), so seeds continue on failure rather than aborting
# the chain; a crash still prints loudly.
#
#   sbatch slurm/efc_lambda_sweep.sbatch <cfg[,cfg,...]> [tag]
#   e.g. sbatch slurm/efc_lambda_sweep.sbatch S2_d10_c05_l50_letf_ne64_efc
# The comma form chains several configs on one card — used for the four d4
# cells (~12 min each, all four ~1 h) because the cluster's
# QOSMaxSubmitJobPerUserLimit counts 8 QUEUED jobs per user and one-job-per-
# config left no slots for the two heaviest d10 rows.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFGS=${1:?usage: sbatch efc_lambda_sweep.sbatch <cfg[,cfg,...]> [tag]}
TAG=${2:-20260829-efc-sweep}

cd "$REPO"
echo "node=$(hostname) cfgs=$CFGS tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Split the comma-separated configs explicitly.
IFS=',' read -r -a CFG_LIST <<< "$CFGS"
for CFG in "${CFG_LIST[@]}"; do
    for SEED in 42 43 44 45; do
        echo "=== $CFG seed $SEED"
        if ! "$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
                --cfg "$CFG" --seed "$SEED" --tag "$TAG"; then
            echo "SEED $SEED FAILED for $CFG — continuing the chain"
        fi
    done
done
