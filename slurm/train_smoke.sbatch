#!/bin/bash
#SBATCH --job-name=train-smoke
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Minutes-scale training smoke (--smoke: 4 steps, 8 Euler steps, 64 eval
# samples) whose real purpose is verifying wandb logging works from a DoC
# compute node.

export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

# wandb auth: run `wandb login <KEY>` ONCE on the cluster beforehand (the
# ~/.netrc append route proved unreliable — see docs/imperial-doc-cluster.md
# §5a). No key belongs in this file. To prove the pipeline with no key at
# all, uncomment the next line (logs locally; `wandb sync` later):
# export WANDB_MODE=offline

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d16_c50_s223_letf_dh --smoke --tag slurm_smoke
