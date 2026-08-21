#!/bin/bash
#SBATCH --job-name=d256-N5-transport
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# N5 (2026-08-21, user GO): the d256 twin of the B(a)+B(b) transport probes.
#
# WHY. The warm-base review's size claim -- "per-swap SELECTION capture falls
# 68.6% -> 53.8% from d64 to d256" -- rests on ONE unmeasured number. Capture
# is (achieved - random)/(oracle - random); at d256 the oracle ceiling (14.44)
# and the random-pair mean (-9.39) are MEASURED on reference draws at t=1,
# sigma_c, but ACHIEVED = 3.44 is INHERITED from the d64 audit (0.93/0.27) on
# an assumption of scale-invariance. The d64 side was measured at 3.907. This
# pass measures the d256 side so the size claim stops being half-estimated.
#
# FROZEN BAND, before launch. Achieved = net_edge_units_per_site /
# swaps_per_site_state_changing, both from transport_decomposition.json:
#   SIZE-EFFECT-CONFIRMED iff achieved < 6.96 (capture < d64's measured
#                         68.6%) -- per-swap selection really does degrade
#                         with lattice size, and the review's headline holds
#                         with both legs measured.
#   SIZE-EFFECT-REFUTED   iff achieved >= 6.96 -- the inherited 3.44 was an
#                         artefact of assuming scale-invariance, and the
#                         "selection degrades with size" claim is WITHDRAWN.
#   Report the measured capture explicitly either way, beside 53.8%, rather
#   than rounding to the nearer story. `cancellation_fraction` rides
#   alongside: d64 measured 0.184, and a d256 value near 1 would move the
#   diagnosis from TARGETING to CANCELLATION, which would in turn remove the
#   measured justification for the periodic-architecture thread (N6).
#
# TWO RUN DIRS, one allocation. PRIMARY is the keystone: it is the standing
# d256 recipe AND it is ne128, matching the d64 rung's grid, so the size
# comparison is not confounded by the Euler grid. SECONDARY is the ne512
# recipe seed 42, declared as a grid control -- if the decomposition moves
# between them, the d64->d256 comparison needs the grid stated, and finding
# that out here costs ~30 min on an already-allocated GPU rather than a
# whole extra queue slot (queue wait, not compute, is the binding cost).
# RECIPE CAVEAT ON THE RECORD: the d64 rung is b128 + control_variate; both
# d256 legs are b512 + naive_mc. These are each size's own standing recipe,
# not a matched twin, and the comparison must be reported that way.
#
# n-samples 512 = 8x the d64 draw (64). `net_std_over_draws` in the output is
# the precision check. NOTE the script's own guard: this draw uses
# return_all_states=True, which is mutually exclusive with log-weights, so it
# produces NO importance weights and its samples must NEVER be quoted as an
# eval. It writes transport_decomposition.json and never touches eval/.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
KEYSTONE=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive_seed42_20260819-mars-grid-depth-a
NE512=results/03_hard/H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_seed42_20260818-d256-recipe-s42

cd "$REPO"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for RUN in "$KEYSTONE" "$NE512"; do
    echo "=== transport decomposition: $RUN"
    "$PIXI" run --frozen -e cuda python -m \
        experiments.constrained_hard_03.analysis_transport_decomposition \
        "$RUN" --n-samples 512
done
