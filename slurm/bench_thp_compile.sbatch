#!/bin/bash
#SBATCH --job-name=dnfs-bench-compile
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out

# torch.compile measurement for the thp head (profiling pass). Runs the eager d256
# train_step matrix first (same-node baseline), then the compiled twins.
# On Modal this failed: inductor's cuda_utils.c needs cuda.h and the image's
# pixi cuda env ships no CUDA dev headers; the DoC nodes carry a system
# toolkit, so the measurement lives here.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for ARGS in \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind two_hole_patch" \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind two_hole_patch --compile" \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind two_hole_patch --patch-radius 2 --compile" \
    "--mode train_step --d 256 --batch 128 --repeats 5 --head-kind factorised --site-orderings row,col --sdpa --compile"; do
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.profile_swap $ARGS
    echo
done

echo "bench exit code: $?"
