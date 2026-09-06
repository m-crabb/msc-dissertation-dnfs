#!/bin/bash
#SBATCH --job-name=dnfs-wave1-sc
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Wave-1 sigma_c retrains (s63, 2026-08-24): the baseline chapter's critical
# cells retrained at the exact SIGMA_C = 0.220343. Cells, twin discipline and
# the FROZEN judging bands live at the registry block in
# experiments/dnfs_baseline_01/configs.py (Wave-1 sigma_c retrain twins):
# >= 3 of 4 seeds over the family floor (d10 0.86 / d8 0.89 / d4 0.93) on
# the final fp32 5000-draw eval/ess_fraction, else c030-style regression
# investigation and the legacy print stays.
#
# Sequential cfg:seed cells share a queue slot (2-slot budget, 2026-08-24).
# Fixed tag reuses run dirs, skips finished seeds and resumes unfinished ones
# from checkpoints/resume.pt after timeout/preemption. Two-job launch plan:
#   sbatch slurm/wave1_unconstrained_sc.sbatch \
#       stage_4_d10_critical_paper_curriculum_sc:{42,43,44,45}
#   sbatch slurm/wave1_unconstrained_sc.sbatch \
#       stage_4_d8_critical_paper_curriculum_sc:{42,43,44,45} \
#       stage_4_d4_critical_sc:{42,43,44,45}
# Walltime: d8 legacy 200k = ~9.5 h eager on a30; compiled ~1.6x faster and
# d10 ~1.5x d8 per step => 4 seeds ~= 28 h (d10) / ~24 h + minutes-scale d4
# (job 2); 36 h is headroom, and the resume tag covers an overrun.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional cfg:seed tokens, not --export: passing values through the
# environment made this cluster's controller fail user-env retrieval.
[ $# -ge 1 ] || { echo "usage: sbatch wave1_unconstrained_sc.sbatch <cfg>:<seed> [...]"; exit 1; }
TAG=20260824-wave1-sc

cd "$REPO"
echo "node=$(hostname) tag=$TAG specs=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SPEC in "$@"; do
    CFG=${SPEC%%:*}
    SEED=${SPEC##*:}
    echo "=== $(date -Is) start cfg=$CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.dnfs_baseline_01.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
    echo "=== $(date -Is) done cfg=$CFG seed=$SEED rc=$? ==="
done
