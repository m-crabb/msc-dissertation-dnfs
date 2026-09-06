#!/bin/zsh
# 16-site Cu-Au F(c) at 500 K, Modal tag 20260904-cuau16-fc:
# house c=0.5 recipe at n_Au = 4, 5, 6, 7 of 16, three seeds each (the c=0.5 cell already exists).
# Exact-enumeration comparison: judge_16site_cells.py.
cd "$(dirname "$0")/../../.."
M=.pixi/envs/dev/bin/modal; T=20260904-cuau16-fc; H=experiments/constrained_hard_03/modal_app.py
for c in c25 c31 c38 c44; do
  $M run --detach "$H::batch_seeds" --cfg-name H2_cuau16_${c}_T500_mask_one_50k_house --seeds 42,43,44 --tag $T
done
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep 'cuau16-fc' | sort
# Same-tag composition-amortised twin (one checkpoint, five slices) and free cells
# at MetaDNS's 1200 / 680 K; the 500 K free house cell already exists. One pull collects the set.
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau16_camort_T500_mask_one_50k_house --seeds 42,43,44 --tag $T
S=experiments/constrained_soft_02/modal_app.py
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau16_T1200_letf_10k --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau16_T680_letf_30k_l4 --seeds 42,43,44 --tag $T
