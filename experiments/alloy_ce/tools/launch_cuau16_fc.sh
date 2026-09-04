#!/bin/zsh
# 16-site Cu-Au composition sweep for the canonical F(c) at 500 K (s123, 2026-09-04), Modal, tag
# 20260904-cuau16-fc: the house c=0.5 recipe at n_Au = 4, 5, 6, 7 of 16 (c50 house cells landed s118),
# three seeds each; judged against exact enumeration by judge_16site_cells.py.
cd /Users/mitchcrabb/Documents/Imperial/Modules/Dissertation/msc-dissertation-dnfs
M=.pixi/envs/dev/bin/modal; T=20260904-cuau16-fc; H=experiments/constrained_hard_03/modal_app.py
for c in c25 c31 c38 c44; do
  $M run --detach "$H::batch_seeds" --cfg-name H2_cuau16_${c}_T500_mask_one_50k_house --seeds 42,43,44 --tag $T
done
echo "=== volume after:"; $M volume ls dnfs-results / 2>/dev/null | grep 'cuau16-fc' | sort
# s123 addendum: the composition-amortised twin (one checkpoint, five slices) and the free cell on
# MetaDNS's 1200 / 680 K rows (500 K free house cell landed s119), same tag so one pull collects the set.
$M run --detach "$H::batch_seeds" --cfg-name H2_cuau16_camort_T500_mask_one_50k_house --seeds 42,43,44 --tag $T
S=experiments/constrained_soft_02/modal_app.py
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau16_T1200_letf_10k --seeds 42,43,44 --tag $T
$M run --detach "$S::batch_seeds" --cfg-name A1_cuau16_T680_letf_30k_l4 --seeds 42,43,44 --tag $T
