#!/bin/bash
#SBATCH --job-name=train-smoke
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Verify wandb logging from a DoC compute node: --smoke uses 4 training
# steps, 8 Euler steps and 64 eval samples.

export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

# Run `wandb login <KEY>` once on the cluster; appending ~/.netrc proved
# unreliable (docs/imperial-doc-cluster.md §5a). Keep keys out of this file.
# For key-free local logging, uncomment below; `wandb sync` uploads later:
# export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d16_c50_s223_letf_dh --smoke --tag slurm_smoke
