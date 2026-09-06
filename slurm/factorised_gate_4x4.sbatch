#!/bin/bash
#SBATCH --job-name=fac-gate
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Factorised-head 4x4 gate: one job per head variant, covering both
# sigma points (s010 floor, s223 critical) x seeds 42-44 sequentially --
# six 10k-step 4x4 runs. Cells are single-variable twins of the MA demo
# cells (test_factorised_gate_cells_mirror_ma_twin_except_declared_fields).
#   sbatch -J fac-gate-fab8 slurm/factorised_gate_4x4.sbatch fab8
#   arms: fab8 | fab16 | fbil | fglo | fmp40 (matched-param arm)
#       | fmo2 (multi-order row+col causal streams)
# Submitted WITHOUT nice, so these outrank the nice'd walkback topups and
# take freed GPU slots ahead of them.
# Walltime: archived demo 10k 4x4 cells are minutes-scale per run on an
# A30; 8 h is pure headroom. Fixed --tag so a resubmitted job reuses run
# dirs instead of retraining completed cells.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional argument, not --export: passing values through the environment
# made this cluster's controller fail user-env retrieval and hold the job.
ARM=${1:?usage: sbatch factorised_gate_4x4.sbatch <fab8|fab16|fbil|fglo|fmp40>}
TAG=20260813-facgate

cd "$REPO"
echo "node=$(hostname) arm=$ARM tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SIGMA in s010 s223; do
  for SEED in 42 43 44; do
    CFG="H2_d16_c50_${SIGMA}_letf_${ARM}_10k"
    echo "=== $CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
  done
done
