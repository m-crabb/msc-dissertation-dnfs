#!/bin/bash
#SBATCH --job-name=fc-d10-ne128
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --array=0-15

# F(c) campaign retrain at ne128: the four
# windows without an ne128 family x seeds 42-45, one array task each. Cells
# are one-variable twins of the c=0.50 ne128 anneal cell (test-pinned). The
# archived D=10 ne128 anneal specialist took ~12 h on an A100; 48 h covers an
# a30. sbatch does not load ~/.bashrc, so pixi is addressed by absolute path.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
WINDOWS=(c030 c055 c060 c065)
SEEDS=(42 43 44 45)
CFG=S2_d10_${WINDOWS[$((SLURM_ARRAY_TASK_ID / 4))]}_l50_letf_ne128_anneal
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 4))]}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
    --cfg "$CFG" --seed "$SEED"
