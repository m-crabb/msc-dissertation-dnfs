#!/bin/bash
#SBATCH --job-name=d256-scr5k-mo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Never submitted: the mask_one head OOMs at the recipe batch at 16x16
# (75.9 GiB backward, measured).
# mask_one screen cell: the best 8x8 head on record (0.00129
# Var[log w]/site at 100k, 2.2x under the masked-attention twin) has never
# touched 16x16. It runs the stage-1 screen frame (flat sigma=0.10, 5k
# steps, naive c_t — expected ranges in the scr5k cell builder
# docstring). On the CLUSTER rather than metered hardware because its
# per-step cost at 256 sites is unbenchmarked — the head runs d separate
# anchor passes, so cost scales with d in a way no archived 16x16 timing
# covers — and the 24h cap turns a cost blowout into a bounded negative
# result whose training_log still prices the head (wall_clock_step_s
# column) instead of a lost detached job.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260815-d256-scr5k-mo

cd "$REPO"
echo "node=$(hostname) cfg=H2_d256_scr5k_mo seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d256_scr5k_mo --seed 42 --tag "$TAG"
