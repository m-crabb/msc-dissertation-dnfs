#!/bin/bash
#SBATCH --job-name=camort-d256-ps
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=42-44
#SBATCH --output=slurm/logs/%x_%A_%a.out

# 16x16 composition-amortised cell RE-RUN under the per-slice c_t baseline
# (swap_ctmc.mean_per_slice, 2026-09-05): the archived seed-42 run under tag
# 20260831-camort-d256 trained with a c_t pooled across the five slices.
# Same config, seeds 42-44, new tag; the seed-42 pair is the matched
# comparison, seeds 43-44 give the spread the archived run lacked.
# Fixed --tag so a resubmission resumes from checkpoints/resume.pt.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d256_camort_s220_letf_thp2_100k_curr
TAG=20260905-camort-d256-perslice
SEED=${SLURM_ARRAY_TASK_ID:?seed comes from the Slurm array}

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
grep -q "mean_per_slice" src/discrete_flow_sampler/samplers/swap_ctmc.py \
    || { echo "per-slice c_t fix NOT in the synced tree -- refusing to train"; exit 1; }

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
