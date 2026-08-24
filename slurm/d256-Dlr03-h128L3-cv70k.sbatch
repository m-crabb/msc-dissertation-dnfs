#!/bin/bash
#SBATCH --job-name=d256-Dlr03-h128L3-cv70k
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Flat-lr twin of arm D (2026-08-24, user GO), owed by P_lr03's LR-ARTEFACT
# read. Bands frozen in the cell comment (configs.py) BEFORE this launch;
# one-variable relationship (lr only, vs D) pinned by
# test_ne128_cv_family_arms_are_one_variable_twins_of_their_comparators.
# Fixed --tag so a resubmission resumes into the same run dir; resume.pt
# takes precedence over --init-from by design, which is the desired restart
# semantics if this job hits its walltime.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_c50_s223_letf_fmo2_h128L3_lr03_70k_curr_b512_ne128_cv2
TAG=20260824-Dlr03-h128L3-cv70k

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed 42 --tag "$TAG"
