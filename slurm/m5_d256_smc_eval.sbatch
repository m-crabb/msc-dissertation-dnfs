#!/bin/bash
#SBATCH --job-name=m5-d256-smc
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/%x_%A_%a.out

# SMC-eval DIAGNOSTIC on the archived d256 naive-rescue checkpoint — one
# array task per ESS threshold
# tau in {0.5, 0.9}. Separates weight-concentration (SMC-fixable) from
# support-miss (isn't): at d256 the unique-count collapse is expected to be
# severe (top-10 rows already own 69.5% of mass), so read min(ESS,
# n_unique), not ESS alone. Artefacts land in eval_smc_tau<tau>/ beside the
# untouched plain-IS eval/, which stays the quoted baseline (SMC is
# reported ALONGSIDE plain IS, never instead).
# Expected: SMC ESS >= 0.2 with n_unique >= 500 would mean a caveated
# 16x16 sampler exists; otherwise the failure is support/coverage.
# Scheduling: eval-only, no wandb. a30 because a100 stays reserved for the
# d256 cv2 job — to avoid priority inversion, submit only once that job is
# RUNNING (or scontrol hold this array until then).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN_DIR=results/03_hard/H2_d256_c50_s223_letf_ma_50k_curr_naive_seed42_20260813-d256-rescue

ESS_THRESHOLDS=(0.5 0.9)
TAU=${ESS_THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "node=$(hostname) tau=$TAU run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --eval-only "$RUN_DIR" --smc-tau "$TAU"
