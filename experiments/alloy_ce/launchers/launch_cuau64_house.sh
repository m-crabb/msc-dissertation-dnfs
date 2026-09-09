#!/bin/zsh
# The 64-site Cu-Au house cells on Modal A100-80GB, tag 20260903-cuau64-house.
# Hard c=0.25/0.5 mask-one house recipe (no channel: the swap channel regressed on the alloy at 16 sites);
# free with and without the flip channel; soft lambda=50 with the channel (channel-free soft died at 16 sites).
cd "$(dirname "$0")/../../.."
M=.pixi/envs/dev/bin/modal
T=20260903-cuau64-house
H=experiments/constrained_hard_03/modal_app.py
S=experiments/constrained_soft_02/modal_app.py
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c25_T500_mask_one_50k_curr --seeds 42,43,44 --tag $T
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_c50_T500_mask_one_50k_curr --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau64_T500_letf_50k_curr --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau64_T500_letf_50k_curr_efc --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name S2_cuau64_c25_l50_T500_letf_50k_curr_efc --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name S2_cuau64_c50_l50_T500_letf_50k_curr_efc --seeds 42,43,44 --tag $T
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep -o '[A-Za-z0-9_]*cuau64[A-Za-z0-9_-]*' | sort -u
