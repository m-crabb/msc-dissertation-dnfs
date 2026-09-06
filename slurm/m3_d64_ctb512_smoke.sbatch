#!/bin/bash
#SBATCH --job-name=m3-d64-ctb512
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# c_t_batch d64 plumbing smoke: the d64 MA curriculum recipe at the 12k
# smoke horizon with c_t_batch=512 the ONLY
# mechanism change (twin-ness pinned by
# test_m3_ctb512_d64_smoke_mirrors_ma_recipe_except_declared_fields). The
# ~25-min validation of the enlarged-rollout outer cycle (buffer prefix,
# c_t over the full 512-row set, resume carriage) on GPU BEFORE the d256
# mechanism cell if a100 scheduling blocks it — the unit tests
# (test_c_t_batch.py) already pin the semantics, so this carries NO quality
# band; it is an integration smoke. Fixed tag keeps resume location.
# 12 h wall is deliberate headroom over the ~25-min
# class estimate (a30 runs ~2x the a100 estimates).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-m3-ctb512-d64

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_smoke12k_ctb512 seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_smoke12k_ctb512 --seed 42 --tag "$TAG"
