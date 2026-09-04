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
