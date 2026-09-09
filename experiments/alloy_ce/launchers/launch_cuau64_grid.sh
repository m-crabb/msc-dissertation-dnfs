#!/bin/zsh
# 64-site Cu-Au MetaDNS grid, Modal A100-80GB, tag 20260904-cuau64-grid.
# Revival left c=0.5 dead at 500 K on every ladder x trajectory arm; report 1200 / 680 K
# with the ladder stopped there (configs.py), 500 K as the limit. Hard c25/c50 use the
# two-hole patch head, free A1 the flip head; three seeds each (18 cells, ~$1 each).
cd "$(dirname "$0")/../../.."
M=.pixi/envs/dev/bin/modal
T=20260904-cuau64-grid
H=experiments/constrained_hard_03/modal_app.py
S=experiments/constrained_soft_02/modal_app.py
for c in c25 c50; do
  $M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_${c}_T1200_thp_10k --seeds 42,43,44 --tag $T
  $M run --detach "$H::batch_seeds" --cfg-name H2_cuau64_${c}_T680_thp_30k_l4 --seeds 42,43,44 --tag $T
done
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau64_T1200_letf_10k --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau64_T680_letf_30k_l4 --seeds 42,43,44 --tag $T
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep 'cuau64-grid' | sort
