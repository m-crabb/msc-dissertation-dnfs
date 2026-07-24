#!/bin/bash
#SBATCH --job-name=env-probe
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out

export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/bitbucket/mc625/.pixi-cache
REPO=/vol/bitbucket/mc625/msc-dissertation-dnfs
cd "$REPO"

echo "node=$(hostname)"
echo "--- NFS seq read 100MB:"
dd if=.pixi/envs/cuda/lib/libtorch_cuda.so of=/dev/null bs=1M count=100 2>&1 | tail -1
echo "--- timed torch import + cuda check + run.py import:"
time /vol/bitbucket/mc625/.pixi/bin/pixi run --frozen -e cuda python -c "
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0))
import experiments.constrained_hard_03.run
print('run.py imports OK')
"
