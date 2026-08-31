#!/bin/bash
#SBATCH --job-name=rlad-resume
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-06:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# 16x16 ladder RESCUE on the DoC a100s (2026-08-31): mars was withdrawn
# mid-wave, so the 17 partial cells of tag 20260830-rasterord-d256 finish
# here from their staged resume.pt checkpoints. Spec-driven, one cell at a
# time on the card:
#
#     sbatch -J <name> [--time=...] slurm/rasterord_d256_a100_resume.sbatch \
#         <CFG:SEED> [<CFG:SEED> ...]
#
# WHY CFG:SEED SPECS AND NOT THE FLOOR SCRIPT'S SEED LOOP. Only the partial
# run dirs are staged on this cluster (quota discipline); the 16 finished
# cells live on the Mac. run.py's short-circuit reads eval/metrics.json in
# the LOCAL run dir, so looping seeds 42/43/44 here would find no dir for a
# finished seed and silently retrain it from scratch. Naming each (cfg,
# seed) pair explicitly makes that failure unbuildable.
#
# SAME-VENUE CAVEAT FOR JUDGING: these cells trained partly on a B200 under
# MPS and finish on an A100. Quality reads (ESS, weight variance) carry;
# nothing timing-shaped from these runs is quotable, same rule as the MPS
# packs they resume from. Cross-hardware frozen-eval drift is ~1%
# (A30-vs-A100 measurement, 2026-07-24) — quote-as-drift if it ever shows.
#
# Fixed --tag: a wall kill is recoverable, a requeue resumes from resume.pt.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=20260830-rasterord-d256

SPECS=("$@")
[ ${#SPECS[@]} -gt 0 ] || { echo "usage: sbatch $0 <CFG:SEED> [<CFG:SEED> ...]"; exit 2; }

cd "$REPO"
echo "node=$(hostname) tag=$TAG clients=${#SPECS[@]}"
printf '  spec: %s\n' "${SPECS[@]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RC=0
for SPEC in "${SPECS[@]}"; do
    CFG=${SPEC%:*}
    SEED=${SPEC##*:}
    echo "=== $CFG seed $SEED starting $(date -Is) ==="
    if ! "$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.run \
            --cfg "$CFG" --seed "$SEED" --tag "$TAG"; then
        echo "SPEC $SPEC FAILED -- continuing the chain"
        RC=1
    fi
done
echo "=== all specs done $(date -Is) rc=$RC ==="
exit $RC
