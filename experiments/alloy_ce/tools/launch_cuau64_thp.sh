#!/bin/zsh
# 64-site Cu-Au two-hole patch cells (s122, 2026-09-03), Modal A100-80GB, tag 20260903-cuau64-thp.
# Replaces mask-one killed at step 9k (3.76 s/step); patch benchmark: 0.048 s/step one-shell,
# registered with two shells. Both compositions, seeds 42-44, no channel, 256 in-training
# eval draws. Gate: logged ESS vs mask-one's first 9k steps (20260903-cuau64-house, still on volume).
cd /Users/mitchcrabb/Documents/Imperial/Modules/Dissertation/msc-dissertation-dnfs
M=.pixi/envs/dev/bin/modal
T=20260903-cuau64-thp
H=experiments/constrained_hard_03/modal_app.py
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c25_T500_thp_50k_curr --seeds 42,43,44 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_thp_50k_curr --seeds 42,43,44 --tag $T
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep -o '[A-Za-z0-9_]*cuau64-thp[A-Za-z0-9_-]*' | sort -u
