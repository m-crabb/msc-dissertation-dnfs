#!/bin/bash
#SBATCH --job-name=ctv-d64-naive
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# c_t transfer-function cell. Every variance lever in flight (Stein CV, the
# c_t EMA, the enlarged c_t rollout) reduces c_t NOISE, but the quantity
# that has to fall ~7.9x for a usable d256 is per-site Var[log w]. The slope
# between the two has never been measured: all 21 archived d64 cells run
# control_variate, so there is no naive arm at any healthy size in either
# direction.
#
# This is the archived MA 50k curriculum twin with estimator
# control_variate -> naive_mc the ONLY declared change (twin-ness pinned by
# test_ctv_naive_twin_mirrors_ma_twin_except_the_estimator). It measures the
# transfer where the control variate is known to WORK (a healthy run),
# rather than where it inverted (the diverged d256) — the confound that
# makes the archived d256 naive-vs-CV comparison uninterpretable.
#
# PRE-REGISTERED OUTCOMES (twin ESS 0.7806; the naive integrand variance
# 26.0 is already logged as the CV runs' own Var[dt log p tilde] column, so
# these are arithmetic, not guesses):
#   ~0.78  -> c_t noise does not drive log-weight variance; the whole c_t
#             thrust has a ceiling and M4/warm-start become the mainline
#   ~0.25  -> transfer linear in c_t standard error
#   ~0.001 -> transfer linear in c_t variance
#   diverges -> the CV is a STABILITY crutch, not only a variance reducer,
#             which would mean the archived d256 naive rescue ran 50k steps
#             without one
#
# a30 not a100: this is the d64 50k class (~1.3 h a100, ~2x on a30) and the
# a100s are reserved for the d256 strategy per the priority-inversion rule.
# Fixed --tag so a resubmission resumes into the same run dir.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260814-ctv-naive

cd "$REPO"
echo "node=$(hostname) cfg=H2_d64_c50_s223_letf_ma_50k_curr_naive seed=42 tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg H2_d64_c50_s223_letf_ma_50k_curr_naive --seed 42 --tag "$TAG"
