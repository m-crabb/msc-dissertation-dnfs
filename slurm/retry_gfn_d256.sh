#!/bin/bash
# GFN d256 wave armer (s102): gate-then-wave, fed in as QOS slots free.
# The retry_submit_softcamort_s010 pattern, extended to a sequence:
#   1. submit gfn_d256_launch_bench and WAIT for "GATE PASSED" in its log
#      (the wave ships compile_policy=True; inductor kernels are certified
#      per venue AND size, so the d64 gate carries nothing);
#   2. submit the four wave jobs in value order -- sigma_c first (the
#      discriminating coupling; the 256-step fairness claim lives there),
#      floor second -- each retried every 10 min while the QOS submit cap
#      is the refusal.
# Survives the laptop session (launch with nohup+setsid on the submission
# host). Log + pid in slurm/logs. Kill: kill $(cat slurm/logs/retry_gfn_d256.pid)
# ONE ARMER AT A TIME: a second submitter against the same fixed tag
# double-submits into shared run dirs.
W=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
cd "$W"
DEADLINE=$(( $(date +%s) + 96*3600 ))

deadline_ok() { [ "$(date +%s)" -lt "$DEADLINE" ]; }

# Retry one sbatch submission while the QOS cap is the refusal.
# Echoes the job id on success; returns 1 on deadline or non-QOS failure.
submit_when_free() {
  local label=$1; shift
  while deadline_ok; do
    result=$(sbatch "$@" 2>&1)
    if echo "$result" | grep -q "Submitted batch job"; then
      echo "$(date -Is) SUBMITTED $label: $result" >&2
      echo "$result" | awk '{print $4}'
      return 0
    fi
    if ! echo "$result" | grep -q "QOSMax"; then
      echo "$(date -Is) NON-QOS FAILURE on $label, stopping: $result" >&2
      return 1
    fi
    sleep 600
  done
  echo "$(date -Is) deadline reached before $label found a slot" >&2
  return 1
}

# --- Stage 1: the compile/parity gate at the new size ---------------------
BENCH_ID=$(submit_when_free "gate" slurm/gfn_d256_launch_bench.sbatch) || exit 1
BENCH_LOG="slurm/logs/gfn-d256-bench_${BENCH_ID}.out"
while deadline_ok && squeue -h -j "$BENCH_ID" 2>/dev/null | grep -q .; do
  sleep 120
done
if ! grep -q "GATE PASSED" "$BENCH_LOG" 2>/dev/null; then
  echo "$(date -Is) GATE DID NOT PASS (see $BENCH_LOG) -- wave NOT submitted"
  exit 1
fi
echo "$(date -Is) gate passed ($BENCH_LOG); feeding the wave"

# --- Stage 2: the four cells, sigma_c first ------------------------------
submit_when_free "tb-sc" -J gfn-d256-tb-sc \
  slurm/gfn_d256_wave.sbatch GFN_d256_c50_s220_tb_100k_par || exit 1
submit_when_free "fldb-sc" -J gfn-d256-fldb-sc \
  slurm/gfn_d256_wave.sbatch GFN_d256_c50_s220_fldb_100k_par || exit 1
submit_when_free "tb-floor" -J gfn-d256-tb-s010 --time=24:00:00 \
  slurm/gfn_d256_wave.sbatch GFN_d256_c50_s010_tb_50k_par || exit 1
submit_when_free "fldb-floor" -J gfn-d256-fldb-s010 --time=24:00:00 \
  slurm/gfn_d256_wave.sbatch GFN_d256_c50_s010_fldb_50k_par || exit 1
echo "$(date -Is) all four wave jobs submitted"
