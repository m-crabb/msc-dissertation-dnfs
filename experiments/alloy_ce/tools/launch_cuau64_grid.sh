#!/bin/zsh
# 64-site Cu-Au cells on MetaDNS's temperature grid (s123, 2026-09-04), Modal A100-80GB,
# tag 20260904-cuau64-grid. The revival wave left 500 K at c=0.5 dead on every ladder x
# trajectory arm, so the 64-site rows are reported at MetaDNS's 1200 K and 680 K with the
# ladder stopped there (configs.py comments), 500 K printed as the limit. Hard c25/c50 on the
# two-hole patch head, free A1 on the flip head; three seeds each (18 cells, ~$1 each).
cd /Users/mitchcrabb/Documents/Imperial/Modules/Dissertation/msc-dissertation-dnfs
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
