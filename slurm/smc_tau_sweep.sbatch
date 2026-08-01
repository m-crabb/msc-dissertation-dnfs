#!/bin/bash
#SBATCH --job-name=smc-tau-sweep
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%A_%a.out

# SMC-resampled eval-only pass over the frozen MA d64 100k run: one array
# task per ESS threshold tau. Writes eval_smc_tau<tau>/metrics.json beside
# the untouched plain-IS eval/, which stays the quoted baseline.
#
# sbatch does not load ~/.bashrc, so pixi is addressed by absolute path.
# --frozen: activate the already-installed locked env, never re-solve.
# CONDA_OVERRIDE_CUDA satisfies the env's __cuda system requirement even if
# pixi's driver detection misfires; the lock pins the cuda 12.9 stack anyway.
export CONDA_OVERRIDE_CUDA=12.9
# Keep pixi's package cache off the small /homes quota; on gpudata (fast NFS).
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN_DIR=results/03_hard/H2_d64_c50_s223_letf_ma_100k_curr_seed42_20260722-124315

ESS_THRESHOLDS=(0.3 0.5 0.7)
TAU=${ESS_THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "node=$(hostname) tau=$TAU run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --eval-only "$RUN_DIR" --smc-tau "$TAU"
