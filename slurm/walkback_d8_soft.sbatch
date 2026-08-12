#!/bin/bash
#SBATCH --job-name=wb8-soft
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Walk-back-to-8x8 slate (2026-08-12), soft family: D=8 twins of the three
# load-bearing d10 specialists, one seed per submission, seeds 42-45 to
# match each archived d10 family. Fixed-composition specialists only, so no
# post-training composition sweep (that step is amortised-cells-only).
#   sbatch -J wb8-c05l50-s42 slurm/walkback_d8_soft.sbatch \
#       S2_d8_c05_l50_letf_ne64 42
# Cells: S2_d8_c05_l10_letf_ne64, S2_d8_c05_l50_letf_ne64,
#        S2_d8_c03_l50_letf_ne128.
# Walltime is the family max: the one cluster-timed 50k ne128 D=10 run took
# ~4.3 h on an A30 (in-training 5000-sample evals at ~103 s dominate);
# scaled by 64/100 sites that is ~2.8 h for the ne128 cell and ~1.5 h for
# the ne64 cells (eval + outer sampling halve with n_euler). 2x headroom on
# the ne128 estimate -> 6 h; the ne64 cells just carry extra slack.
# Fixed --tag so a job resubmitted after preemption reuses the run dir and a
# completed run is skipped, not retrained. NO mid-run resume in this runner:
# a preempted job restarts from step 0.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
# Positional arguments, not --export: passing values through the environment
# made this cluster's controller fail user-env retrieval and hold the job.
CFG=${1:?usage: sbatch walkback_d8_soft.sbatch <cfg> [seed]}
SEED=${2:-42}
TAG=20260812-walkback

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_soft_02.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
