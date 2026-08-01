#!/bin/bash
#SBATCH --job-name=mchammer-base
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Timed mchammer MCMC baselines on a cluster CPU. `--gres=gpu:0` is required,
# not cosmetic: the a30 partition assigns a GPU by default even to a job that
# never touches CUDA, so without it this job queues behind — and idles — a
# scarce accelerator. These are the wall-clock-per-effective-sample numbers
# quoted against DNFS, so they come from the same node class as the DNFS runs.
# One job covers both legs; each (composition, seed) cell takes seconds to
# minutes, so a single 2 h allocation is generous.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"

# Soft leg: the D=10 lambda=50 campaign operating points.
"$PIXI" run --frozen -e cuda python scripts/mchammer_baselines.py vcsgc \
    --D 10 --sigma 0.1 --lam 50 \
    --compositions 0.30 0.50 0.55 0.60 0.65 0.80 \
    --n-steps 1000000

# Hard leg: size-matched to the d64 (D=8) sigma_c cells. Near-critical
# swap dynamics mix slowly, hence the larger step budget.
"$PIXI" run --frozen -e cuda python scripts/mchammer_baselines.py canonical \
    --D 8 --sigma 0.223 --compositions 0.5 --n-steps 5000000
