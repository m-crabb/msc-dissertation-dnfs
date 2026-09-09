#!/bin/bash
#SBATCH --job-name=d256-zeroshot
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=42-44
#SBATCH --output=slurm/logs/%x_%A_%a.out

# Zero-shot transfer probe on the landed 16x16 thp2 sigma_c cells: how much
# amortisation the EXISTING checkpoints already carry, before any amortised
# training is designed. Sampling only -- no optimiser, no gradients, no new
# weights. Rationale and both derivations live in the probe's module docstring.
#
# Two axes off one set of sampler passes:
#   coupling      -- on the fixed-composition slice base_log_eta is constant, so
#                    the geometric path reduces to log p~_t ∝ t*sigma*xAx: the
#                    path IS a coupling anneal and stopping at t* lands on the
#                    target at t*sigma. Weights are a running integral, so every
#                    stop time is read off ONE pass.
#   composition   -- the head sees c only through x, so it transfers unchanged;
#                    amending the target's slice re-targets the IS weights by
#                    construction. Needs its own pass (different base draw).
#
# Both axes are EXACT whatever the model does: the learned rates are only a
# proposal and xi_t is evaluated against whichever target is handed in. Bad
# transfer shows up as low ESS, never as a biased number.
#
# Grid choices, all deliberate:
#   --n-euler-steps 128 is the cells' own production grid, so the t*=1, c=0.5
#     row is an ANCHOR: it must land near the published EMA eval (seed 42/43/44
#     read 0.805 / 0.838 / 0.836, mean 0.826 = the printed table cell). Draws
#     are re-seeded rather than replayed, so agreement is statistical (within
#     the seed spread), not bit-for-bit. A miss there means the probe is wrong,
#     not that transfer failed.
#   Stop times are EXACT grid points k/127 (k = 16, 32, 58, 76, 95, 111, 127),
#     so nothing snaps and each row's coupling is exact. k=58 gives sigma =
#     0.100629, the closest the production grid comes to the certified sigma=0.1
#     reference; comparing correlations there needs a reference regenerated at
#     0.100629, not the 0.1 one, which is a 0.6% coupling offset.
#   Compositions are k/256 so n_plus is integral: 128 (trained), 120, 112, 96,
#     80, 64 -- plus 160, the exact Z2 mirror of 96. The TARGET family is exactly
#     symmetric under the global spin flip (x^T A x is), so a Z2-equivariant
#     model would give matching rows. thp is NOT equivariant by construction --
#     the two token embeddings are independent learned vectors and a global flip
#     merely swaps them -- so a mismatch is a finding about the MODEL, not a
#     probe bug. Measured 2026-08-27: they match to 0.005 in ESS fraction on the
#     3-seed mean at sigma_c (0.276 vs 0.281), against a per-seed spread of
#     0.208-0.325, so the trained head is Z2-symmetric within seed noise.
#   final_ema.pt, because the EMA eval is what the published table reports.
export CONDA_OVERRIDE_CUDA=12.9
export PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
PIXI=/vol/bitbucket/mc625/.pixi/bin/pixi
SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR="results/03_hard/H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3_seed${SEED}_20260826-d256-sc"

STOP_TIMES=0.125984252,0.251968504,0.456692913,0.598425197,0.748031496,0.874015748,1.0
COMPOSITIONS=0.5,0.46875,0.4375,0.375,0.3125,0.25,0.625

cd "$REPO"
echo "node=$(hostname) seed=$SEED run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PIXI" run --frozen -e cuda python -m experiments.constrained_hard_03.probes.probe_zero_shot_transfer \
    --run-dir "$RUN_DIR" \
    --checkpoint final_ema.pt \
    --compositions "$COMPOSITIONS" \
    --stop-times "$STOP_TIMES" \
    --n-samples 5000 \
    --n-euler-steps 128 \
    --sample-chunk 500 \
    --out "$RUN_DIR/zero_shot_transfer.json"
