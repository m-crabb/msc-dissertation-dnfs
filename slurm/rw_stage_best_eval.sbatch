#!/bin/bash
#SBATCH --job-name=rw-stage-best
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --nice=1000
#SBATCH --output=slurm/logs/%x_%j.out

# The checkpoint-SELECTION read both rw cells pre-registered: does the
# sigma_c stage's best-trailing-median-ESS checkpoint beat final.pt?
# Stage 6 peaked at step 45500 against a 50k run, so there are 4500 steps
# of training between the two -- a real gap, not a tie by construction.
#
# Drawn at the cell's NATIVE ne512 so it is paired with the frozen eval/,
# which was drawn at ne512 by the training run; artefacts land in
# eval_stage6/ and eval/ is never touched. Budget: d256 eval is ~29 min at
# ne128 and scales linearly with the grid, so ~2h here -- the s43 prompt's
# "~29 min" was the ne128 number and does not apply to an ne512 cell.
#
# --nice=1000 deliberately: five deadline-gating a100 family arms are
# PENDING, and the per-user 3-GPU cap is consumed by whichever job can
# start first, not by priority. Nice orders jobs competing for the SAME
# resource, and this job is a100-only like the arms, so it can only take a
# slot none of them wanted first.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN_DIR=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw_seed43_20260819-rw-s43

cd "$REPO"
echo "node=$(hostname) run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --eval-only "$RUN_DIR" --stage-best 6
