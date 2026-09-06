#!/bin/zsh
# Pull every cuau16-desk run dir from the Modal volume, sequentially (parallel gets collide).
cd "$(dirname "$0")/../../.."
MODAL=.pixi/envs/dev/bin/modal
$MODAL volume ls dnfs-results / 2>/dev/null | grep -o "[A-Z0-9_a-z.-]*cuau16-[a-z0-9]*\|stage_4_d[0-9]*_sc_hardrecipe[A-Za-z0-9_-]*" | sort -u | while IFS= read -r d; do
  case "$d" in H2_*) dest=results/03_hard;; stage_4_*) dest=results/01_baseline;; *) dest=results/02_constrained_soft;; esac
  [ -f "$dest/$d/eval/metrics.json" ] && { echo "have $d"; continue; }
  $MODAL volume get dnfs-results "/$d" "$dest/" >/dev/null 2>&1 && echo "pulled $d" || echo "FAILED $d"
done
