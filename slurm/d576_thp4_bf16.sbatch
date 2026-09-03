#!/bin/bash
#SBATCH --job-name=d576-thp4-bf16
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --array=42-44

# 24x24 at sigma_c, R=4, bf16, loss microbatch 128 (2026-09-03). Cell
# comments in configs.py; the one-variable relationships are pinned by
# test_d576_cells_are_the_d400_r3_bf16_cell_moved_to_the_lattice and
# test_d576_cells_isolate_the_radius.
#
# Two radii x three seeds = six jobs, filling the QOS submission cap beside
# the two GFN d256 jobs. R=3 is the measured d400 sigma_c winner (EMA ESS
# 0.79-0.81 vs R=2 0.63-0.73, disjoint) moved to the lattice; R=4 is the
# knob's one continuation. No fp32 twin: bf16-only is the rule at d400+.
#
# a100 (80 GB) is REQUIRED. The (B, d, d, f) slab grows as d^2, so bf16
# single-shot at R=3 projects from the measured 41.6 GB at d400 to ~86 GB;
# loss microbatching at 128 (gradient-exact) brings the backward to ~33 GB
# against a ~48 GB rollout. 3-day wall: the d400 bf16 step was 0.271 s, so
# d576 projects to ~0.6 s/step, ~17 h of training plus rollout and eval.
#
# Fixed --tag so a resubmission resumes into the same run dir from
# resume.pt. The array index IS the seed.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
CFG=H2_d576_c50_s220_letf_thp4_100k_curr_b512_ne128_cv2_w5bf16
TAG=20260903-d576-sc
SEED=$SLURM_ARRAY_TASK_ID

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
