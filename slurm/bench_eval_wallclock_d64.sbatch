#!/bin/bash
#SBATCH --job-name=wallclock-d64
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out

# Eval-phase wall-clock bench (s100): both sampling paradigms' frozen-eval
# chunk timed on ONE GPU -- the latency counterpoint's measurement (s93:
# FLOP/es prices arithmetic, not the AR policy's d sequential one-token
# kernels). Same partition as the GFN launch gate (a30) so the numbers sit
# on the stack the d64 GFN wave actually ran on. Writes
# results/03_hard/eval_wallclock_bench_d64/wallclock.json -- pull that file
# down and re-run house_table_8x8.py to fill the Wall/es entries.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.bench_eval_wallclock_d64
