#!/bin/bash
#SBATCH --job-name=neuler-sweep-d256
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Does the IS weight variance depend on the time discretisation at d=256?
#
# The time grid never scaled with the lattice: 100 steps at 4x4, 128 at 8x8,
# and still 128 at 16x16 — 4x the sites at the same resolution. Beskos,
# Crisan & Jasra (Ann. Appl. Probab. 24(4), 2014) prove a single
# importance-sampling step needs sample size exponential in dimension while
# a SEQUENCE of intermediate targets keeps ESS non-degenerate at fixed N —
# but only if the number of intermediate targets grows with the dimension.
#
# The earlier grid study cannot answer this: it reanalysed already-recorded
# 128-step rollouts (128 vs subsampled 64/32 vs warped-64), so its negative
# result is about ALLOCATION at a fixed budget. RESOLUTION was never
# testable from those artefacts.
#
# Eval-only off the archived d256 naive checkpoint — no retraining. Reports
# Var[log w], not ESS: at Var 18.1 a 5000-draw self-normalised estimator is
# floor-bound and reads 0.0031 regardless, whereas Var[log w] still
# estimates cleanly (log w near-Gaussian, skew 0.15).
#
# READ IT ONE-SIDED. n_euler drives the trajectory law AND the quadrature,
# and the model was TRAINED at 128, so a finer grid is off-distribution for
# it. Improvement => resolution matters, conclusive. No improvement =>
# INCONCLUSIVE, because a model trained at 384 could still be better. The
# d64 arm of this sweep came back at +16% then flat (Euler converged by 256
# steps), which is weakly negative and exactly what a size that already
# works should show.
#
# Harness check at d64: at the as-run resolution it reproduces
# the archived Var[log w] 0.659 to 1.2 sigma. Re-check the 128 arm here
# against the archived d256 Var[log w] = 18.1 before trusting the finer arms.
#
# a100 because a d=256 masked-attention forward is ~94 ms there and the
# finest arm multiplies that by the resolution; the dev Mac's MPS is the
# wrong home entirely (one arm runs about an hour and starves the machine).
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RUN=results/03_hard/H2_d256_c50_s223_letf_ma_50k_curr_naive_seed42_20260813-d256-rescue
OUT=results/03_hard/n_euler_sweep_d256.json

cd "$REPO"
echo "node=$(hostname) run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m scripts.n_euler_resolution_sweep \
    --run-dir "$RUN" --n-euler 128,256,384 --n-draws 512 --chunk 32 \
    --out "$OUT"
