#!/bin/bash
#SBATCH --job-name=dh-oracle
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Doubly-hollow ORACLE row of tab:eval-hard-4x4 (2026-08-29). One job, both
# couplings x seeds 42/43/44 sequentially -- six 10k runs on the w2 recipe,
# the only deviation from the mask-one row being head_kind (asserted by
# tests/test_house_table_4x4.py).
#
# WHY IT IS RUN AT ALL, given mask_one agrees with doubly_hollow numerically:
# the fidelity columns are EXPECTED to reproduce the mask-one row within seed
# noise, and the row exists for the FLOP/es column -- the price of the
# O(d^2) architecture-agnostic gate. A fidelity disagreement beyond seed
# noise would instead be a correctness finding about the O(d) head.
#
# 4x4 ONLY. There is no d64 oracle config and there must not be: 7,036 ms per
# forward at d=64 against the masked-attention head's 6.0 ms (~1,170x). At
# d=16 the same bench reads 417 ms.
#
# Seeds loop INSIDE the job so one 3-GPU-capped slot yields a COMPLETE cell
# rather than a third of three (the s84 floor-column lesson). run.py skips a
# seed whose eval/metrics.json exists and resumes a partial one from
# resume.pt, so resubmission is idempotent.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260829-dh-oracle-d16

cd "$REPO"
echo "node=$(hostname) tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SIGMA in s010 s220; do
  for SEED in 42 43 44; do
    CFG="H2_d16_c50_${SIGMA}_letf_dh_10k_w2"
    echo "=== $CFG seed=$SEED ==="
    "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
  done
done
echo "ORACLE GATE DONE"
