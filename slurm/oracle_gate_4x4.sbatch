#!/bin/bash
#SBATCH --job-name=dh-oracle
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Doubly-hollow ORACLE row of tab:eval-hard-4x4.
#   sbatch slurm/oracle_gate_4x4.sbatch <s010|s220> <seed>
#
# ONE RUN PER JOB: measured 0.55 steps/s (1.82 s/step), so 10k steps ~5.1 h.
# Six sequential runs ~30 h exceed the 12 h wall by two and a half times;
# separate ~5.1 h jobs give 2.4x margin. Under the 3-GPU cap, use two waves:
# SUBMIT THE THREE s010 SEEDS FIRST so the first wave yields a COMPLETE cell.
#
# EAGER: a first attempt with the house recipe's compile_head=True spent
# 26 minutes without reaching step 1 -- pinned at ~95% of ONE core,
# 373 MiB on the GPU, i.e. inductor still compiling a graph that unrolls
# to d^2 = 256 masked pair forwards. The configs carry compile_head=False
# (c_t_from_rollout kept).
#
# mask_one agrees with doubly_hollow NUMERICALLY, so the
# fidelity columns are EXPECTED to reproduce the mask-one row within seed
# noise. The new information is FLOP/es -- the price of the O(d^2)
# architecture-agnostic gate. A fidelity disagreement beyond seed noise would
# be a correctness finding about the O(d) head instead.
#
# 4x4 ONLY. There is no d64 oracle config and there must not be: 7,036 ms per
# forward at d=64 against the masked-attention head's 6.0 ms (~1,170x). At
# d=16 the same bench reads 417 ms.
#
# Fixed tag reuses the run dir: run.py skips existing eval/metrics.json
# and resumes partial runs from resume.pt.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
SIGMA=${1:?usage: sbatch oracle_gate_4x4.sbatch <s010|s220> <seed>}
SEED=${2:?usage: sbatch oracle_gate_4x4.sbatch <s010|s220> <seed>}
TAG=20260829-dh-oracle-d16
CFG="H2_d16_c50_${SIGMA}_letf_dh_10k_w2"

cd "$REPO"
echo "node=$(hostname) cfg=$CFG seed=$SEED tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
    --cfg "$CFG" --seed "$SEED" --tag "$TAG"
echo "ORACLE CELL DONE: $CFG seed=$SEED"
