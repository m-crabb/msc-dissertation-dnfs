#!/bin/bash
#SBATCH --job-name=gfn-d400
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# GFN 20x20 wave: a job is an ORDERED LIST of cfg:seed cells, run
# serially, so the six TB cells pack into two jobs under the 3-day
# walltime cap instead of one-coupling-per-card (sigma_c trio 60-78 h
# projected = at the cap):
#   sbatch -J gfn-d400-sc   slurm/gfn_d400_wave.sbatch <tag> \
#       GFN_d400_c50_s220_tb_100k_par:42 GFN_d400_c50_s220_tb_100k_par:43
#   sbatch -J gfn-d400-s010 slurm/gfn_d400_wave.sbatch <tag> \
#       GFN_d400_c50_s010_tb_50k_par:42 GFN_d400_c50_s010_tb_50k_par:43 \
#       GFN_d400_c50_s010_tb_50k_par:44 GFN_d400_c50_s220_tb_100k_par:44
# Projected from the measured d256 rate (0.32 s/step A30, ~13 h per 100k
# seed with evals) x 1.6-1.9 for 400 vs 256 sequential rollout steps:
# 20-26 h per 100k seed, 10-13 h per 50k -> 40-52 h and 50-65 h.
#
# The in-job gate certifies compile_policy=True per venue AND size; d256's
# gate does not cover d400. It takes ~15 min on an A30 without another slot
# under the 8-job QOS cap. Check its train-step projection against --time.
# Fixed tags resume partial cells from resume.pt and skip completed raw+EMA
# evals; after a timeout, resubmit the same command.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
TAG=${1:?usage: sbatch gfn_d400_wave.sbatch <tag> <cfg:seed> [<cfg:seed> ...]}
shift
[ $# -ge 1 ] || { echo "no cfg:seed cells given"; exit 2; }

cd "$REPO"
echo "node=$(hostname) tag=$TAG cells=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== compile parity gate + bench at d400 $(date -Is) ==="
GATE_LOG=$(mktemp)
"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.probes.gfn_launch_bench --rung d400 | tee "$GATE_LOG"
grep -q "GATE PASSED" "$GATE_LOG" || { echo "GATE DID NOT PASS -- wave not run"; exit 1; }

for CELL in "$@"; do
    CFG=${CELL%%:*}
    SEED=${CELL##*:}
    echo "=== $CFG seed $SEED starting $(date -Is) ==="
    "$PIXI" run --frozen -e cuda python -m \
        experiments.constrained_hard_03.run_gfn \
        --cfg "$CFG" --seed "$SEED" --tag "$TAG"
done
echo "=== all cells done $(date -Is) ==="
