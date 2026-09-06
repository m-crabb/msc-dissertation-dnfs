#!/bin/bash
# Refill the walk-back queue up to the DoC per-user submission cap
# (QOSMaxSubmitJobPerUserLimit = 8 queued+running).
# Safe to re-run any time: skips jobs already queued (by name) and cells whose
# run dir is already complete (eval/metrics.json exists — the fixed-tag runner
# also short-circuits, this just avoids burning a queue slot to find out).
# Hitting the cap is the expected stop condition, not an error.
# Run on the cluster from the repo root: bash slurm/topup_walkback.sh
TAG=20260812-walkback
declare -a RUNS=(
  "wb8-base-s42   walkback_d8_baseline stage_4_d8_critical_paper_curriculum 42 results/01_baseline"
  "wb8-base-s43   walkback_d8_baseline stage_4_d8_critical_paper_curriculum 43 results/01_baseline"
  "wb8-base-s44   walkback_d8_baseline stage_4_d8_critical_paper_curriculum 44 results/01_baseline"
  "wb8-base-s45   walkback_d8_baseline stage_4_d8_critical_paper_curriculum 45 results/01_baseline"
  "wb8-c05l50-s42 walkback_d8_soft S2_d8_c05_l50_letf_ne64 42 results/02_constrained_soft"
  "wb8-c05l50-s43 walkback_d8_soft S2_d8_c05_l50_letf_ne64 43 results/02_constrained_soft"
  "wb8-c05l50-s44 walkback_d8_soft S2_d8_c05_l50_letf_ne64 44 results/02_constrained_soft"
  "wb8-c05l50-s45 walkback_d8_soft S2_d8_c05_l50_letf_ne64 45 results/02_constrained_soft"
  "wb8-c05l10-s42 walkback_d8_soft S2_d8_c05_l10_letf_ne64 42 results/02_constrained_soft"
  "wb8-c05l10-s43 walkback_d8_soft S2_d8_c05_l10_letf_ne64 43 results/02_constrained_soft"
  "wb8-c05l10-s44 walkback_d8_soft S2_d8_c05_l10_letf_ne64 44 results/02_constrained_soft"
  "wb8-c05l10-s45 walkback_d8_soft S2_d8_c05_l10_letf_ne64 45 results/02_constrained_soft"
  "wb8-c03l50-s42 walkback_d8_soft S2_d8_c03_l50_letf_ne128 42 results/02_constrained_soft"
  "wb8-c03l50-s43 walkback_d8_soft S2_d8_c03_l50_letf_ne128 43 results/02_constrained_soft"
  "wb8-c03l50-s44 walkback_d8_soft S2_d8_c03_l50_letf_ne128 44 results/02_constrained_soft"
  "wb8-c03l50-s45 walkback_d8_soft S2_d8_c03_l50_letf_ne128 45 results/02_constrained_soft"
)
submitted=0
for spec in "${RUNS[@]}"; do
  set -- $spec
  name=$1; script=$2; cfg=$3; seed=$4; results_root=$5
  squeue --me -h -n "$name" -o '%i' | grep -q . && continue
  [ -f "$results_root/${cfg}_seed${seed}_${TAG}/eval/metrics.json" ] && continue
  if out=$(sbatch --nice=1000 -J "$name" "slurm/${script}.sbatch" "$cfg" "$seed" 2>&1); then
    echo "$out ($name)"; submitted=$((submitted + 1))
  else
    echo "stopped at cap after $submitted new submission(s)"; break
  fi
done
echo "queue now: $(squeue --me -h | wc -l)/8"
