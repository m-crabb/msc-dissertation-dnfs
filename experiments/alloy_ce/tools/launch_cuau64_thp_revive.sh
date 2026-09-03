#!/bin/zsh
# 64-site Cu-Au revival arms on the two-hole patch head (s122 night, 2026-09-04), Modal A100-80GB,
# tag 20260904-cuau64-revive. First thp wave: c25 ordered at ESS 0.17-0.20, c50 gave up gradually
# from 818 K down (see the configs.py comment). Arms: ladder x2 (l14 at 50k / 100k), trajectory
# (ne256), both (100k_l14_ne256); c25 gets ne256 alone. c50 seeds 42,43; c25 seed 42.
cd /Users/mitchcrabb/Documents/Imperial/Modules/Dissertation/msc-dissertation-dnfs
M=.pixi/envs/dev/bin/modal
T=20260904-cuau64-revive
H=experiments/constrained_hard_03/modal_app.py
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_thp_50k_l14 --seeds 42,43 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_thp_100k_l14 --seeds 42,43 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_thp_50k_ne256 --seeds 42,43 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_thp_100k_l14_ne256 --seeds 42,43 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c25_T500_thp_50k_ne256 --seeds 42 --tag $T
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep 'cuau64-revive' | sort
