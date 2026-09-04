#!/bin/zsh
# File-by-file sequential pull of landed run dirs matching a pattern from the Modal volume
# (a directory `volume get` clobbers; parallel gets collide). Skips dirs without a final eval.
# Usage: zsh pull_modal_runs.sh '<grep pattern>'      e.g. 'cuau64-grid'
cd /Users/mitchcrabb/Documents/Imperial/Modules/Dissertation/msc-dissertation-dnfs
MODAL=.pixi/envs/dev/bin/modal
FILES=(config.json metadata.json training_log.csv eval/metrics.json eval/log_weights.pt eval/samples.pt eval_ema/metrics.json eval_ema/log_weights.pt eval_ema/samples.pt)
$MODAL volume ls dnfs-results / 2>/dev/null | grep -E "$1" | sort | while IFS= read -r d; do
  case "$d" in H2_*|GFN_*) dest=results/03_hard;; stage_4_*) dest=results/01_baseline;; *) dest=results/02_constrained_soft;; esac
  $MODAL volume ls dnfs-results "$d/eval" 2>/dev/null | grep -q metrics.json || { echo "not landed $d"; continue; }
  [ -s "$dest/$d/eval/metrics.json" ] && [ -s "$dest/$d/eval_ema/metrics.json" ] && { echo "have $d"; continue; }
  mkdir -p "$dest/$d/eval" "$dest/$d/eval_ema"
  for f in $FILES; do
    [ -s "$dest/$d/$f" ] && continue
    $MODAL volume get --force dnfs-results "/$d/$f" "$dest/$d/$f" >/dev/null 2>&1 || echo "MISSING $d/$f"
  done
  echo "pulled $d"
done
