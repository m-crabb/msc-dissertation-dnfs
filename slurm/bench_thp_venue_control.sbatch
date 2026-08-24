#!/bin/bash
#SBATCH --job-name=dnfs-bench-venue
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out

# Venue control for the thp-vs-B step-time story (s59). Arm B recorded
# 0.499 s/step on THIS cluster; the thp arms recorded 0.545-0.591 on Modal
# A100-80GB, and a same-device Modal bench gives fmo2 4x0.161 = 0.64 s/step
# equivalent -- i.e. on Modal thp is ~16% FASTER, and the recorded "thp
# slower at 16x16" is suspected to be a cross-venue artefact. This job runs
# the identical profile_swap train_step matrix on the DoC a100 so both
# heads have same-device numbers at both venues.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for ARGS in \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind factorised --site-orderings row,col --sdpa" \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind two_hole_patch" \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind two_hole_patch --patch-radius 2" \
    "--mode train_step --d 64 --batch 128 --repeats 5 --head-kind factorised --site-orderings row,col --sdpa" \
    "--mode train_step --d 64 --batch 128 --repeats 5 --head-kind two_hole_patch"; do
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.profile_swap $ARGS
    echo
done

echo "bench exit code: $?"
