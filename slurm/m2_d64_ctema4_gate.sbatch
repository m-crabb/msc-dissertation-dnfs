#!/bin/bash
#SBATCH --job-name=m2-d64-ctema4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# c_t EMA no-regression gate; comparator is the ARCHIVED MA twin.
# Archived d64 MA 50k curriculum, only c_t_ema_halflife_cycles=4 changed
# (cell comment in configs.py; twin-ness pinned by
# test_m2_ctema4_gate_mirrors_ma_twin_except_declared_fields). Expected:
# no regression = final eval ESS frac >= 0.755 (the archived MA twin's seed
# spread 0.755/0.769/0.781); below that the across-cycle smoothing is not
# free. Seed 42 is the d64 ladder's standing seed; fixed tag keeps resume.
# Scheduling: d64 50k is the 1.3 h A100 class (~2x wall on a30); 2-day
# request matches the fmo2/fab16 d64 rungs' headroom. a30 because a100
# stays reserved for the d256 cv2 job — to avoid priority inversion,
# submit only once that job is RUNNING.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-m2-ctema4

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s223_letf_ma_50k_curr_ctema4 seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s223_letf_ma_50k_curr_ctema4 --seed 42 --tag "$TAG"
