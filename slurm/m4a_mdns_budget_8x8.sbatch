#!/bin/bash
#SBATCH --job-name=m4a-mdns-8x8
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x_%j.out

# Preparation status: unlaunched; all 8 submit slots were occupied.
# Release required the user's approval.
#
# M4a (M-campaign Task 4a): the budget-masked MDNS smoke at 8x8, asking
# the one question no swap-CTMC cell can answer — is the d256 wall
# CTMC-specific or target-intrinsic? The constructive family has no
# pair-rate field, no c_t, no Euler error, and structural feasibility, so
# the entire W3 complex that starves d256 simply does not exist here.
#
# WHY THIS RECIPE. Gate-3 settled the ingredients at the enumerable size
# and this cell changes ONE thing, the lattice:
#   --objective lv    F_LV beat WDCE on every column at every seed at 4x4
#                     (ESS 0.856/0.853/0.857 vs 0.826/0.829/0.827,
#                     non-overlapping). --replicates is a WDCE-only knob
#                     and is deliberately absent.
#   --arms a          the V0 budget-tilted preconditioner.
#   --optimiser adamw --ema-decay 0.9999 --ema-warmup
#                     the recipe-parity bundle. NOTE --ema-warmup is
#                     passed EXPLICITLY: gate-3's own $RECIPE omitted it,
#                     so six of its eight cells ran the plain shadow,
#                     which at 2k steps still holds ~82% of init. Only
#                     the two amendment-03 re-runs had the corrected
#                     schedule. Do not drop this flag.
#   --steps 10000     the gate-3 constrained-arm horizon.
#   --seeds 42,43,44  three seeds, as gate-3.
#
# REFERENCE BLOCK (the M4a build). C(64,32) ~ 1.8e18 rules enumeration
# out, so --probe-root switches the energy reference to the certified
# 8-chain Kawasaki reference and --free-energy-ref supplies the slice-TI
# constant. The enumeration-only instruments (conditional KL,
# late-generation error, within-level uniformity) are DROPPED, not
# approximated, and report as NaN/absent — see the driver's build_space
# docstring. The two energy conventions are pinned bit-equal in
# tests/test_mdns_gate_driver.py.
#
# FROZEN BANDS (8x8 sigma_c, vs the certified references; frozen before
# any M4a result exists):
#   reference F/d      = -1.90410 +/- 0.00005  (slice-TI, run8x8)
#   MA swap energy TV  =  0.012476, own 95% noise floor 0.010129
#   MEANINGFUL: ESS >= 0.5 AND energy TV <= 0.025 (2x the MA swap head's)
#   STRONG:     ESS >= 0.8 AND energy TV <= 0.0125 (parity with it, which
#               at this TV's noise floor means indistinguishable from it)
#   Decision rule: MDNS >= MA-swap at 8x8 -> reprioritise the workshop
#   scaling story to the constructive family. The swap chapter's claims
#   are unaffected either way; they are frozen-band results.
#
# OPEN CHOICES (retained defaults):
#   1. hidden_width stays at the module default 128. The trunk width is
#      therefore UNCHANGED from 4x4; only the size-forced input (3d) and
#      output (d) layers grow, ~25k -> ~49k params. The 4x4 width was
#      chosen to parameter-match their 26k ViT, a constraint that does
#      not carry to 8x8, so holding it fixed is the clean single-variable
#      move rather than a tuned one.
#   2. --eval-rollouts stays at the frozen 65536. Each rollout is now 64
#      sequential trunk calls rather than 16, so eval is ~4x the 4x4
#      cost per rollout; the 8h walltime carries that with headroom, but
#      if the first seed's wall clock says otherwise this is the knob.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
RESULTS=results/03_hard/mdns_budget_8x8

cd "$REPO"
echo "node=$(hostname) cell=m4a-mdns-8x8 seeds=42,43,44"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m \
    experiments.constrained_hard_03.mdns_budget_gate_4x4 \
    --lattice-side 8 \
    --probe-root results/kawasaki_probe --probe-point sc \
    --free-energy-ref -1.90410 \
    --sigmas 0.223 --seeds 42,43,44 --arms a --steps 10000 \
    --objective lv --optimiser adamw --ema-decay 0.9999 --ema-warmup \
    --device cuda --results-dir $RESULTS \
    --tag 20260814-m4a-8x8
