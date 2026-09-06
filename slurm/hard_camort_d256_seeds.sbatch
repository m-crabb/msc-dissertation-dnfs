#!/bin/bash
#SBATCH --job-name=hard-camort-d256
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=43-44
#SBATCH --output=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs/slurm/logs/%x_%A_%a.out

# Replicate the existing seed-42 composition-amortised 16x16 recipe.
# The only experimental change is the training seed. No widened mixture,
# symmetry augmentation, warm start or new evaluation protocol is introduced.
# Use an isolated source snapshot and the already installed CUDA environment;
# PYTHONPATH makes this snapshot take precedence over editable installations.
# Fixed tag preserves the runner's normal checkpoint/resume behaviour.
set -euo pipefail
SNAPSHOT=${1:?usage: sbatch hard_camort_d256_seeds.sbatch /absolute/source/snapshot}
SHARED_REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
TRAIN_PYTHON="$SHARED_REPO/.pixi/envs/cuda/bin/python"
CFG=H2_d256_camort_s220_letf_thp2_100k_curr
TAG=20260905-camort-d256
SEED=${SLURM_ARRAY_TASK_ID:?seed must come from the declared Slurm array}
case "$SEED" in 43|44) ;; *) exit 2 ;; esac

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$SNAPSHOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Runs on the allocated compute node, not the shared login node.
"$TRAIN_PYTHON" - <<'PY'
import json
from dataclasses import asdict
from pathlib import Path
import discrete_flow_sampler
from experiments.constrained_hard_03.configs import CONFIGS

manifest = json.loads(Path('launch_manifest.json').read_text())
actual = json.loads(json.dumps(asdict(CONFIGS[manifest['config_name']])))
assert actual == manifest['expected_config'], 'Training recipe differs from the audited snapshot'
assert Path(discrete_flow_sampler.__file__).resolve().is_relative_to(Path.cwd()), 'Wrong source installation'
print('Audited recipe:', manifest['config_name'], 'source:', manifest['source_commit'])
PY

"$TRAIN_PYTHON" -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG" \
    --output-dir "$SHARED_REPO/results/03_hard"
