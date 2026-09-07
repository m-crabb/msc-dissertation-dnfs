# Cu–Au cluster-expansion experiments

[All experiments](../README.md) · [Cu–Au tools](tools/README.md)

The materials application evaluates free, softly constrained and exactly
constrained sampling on a binary Cu–Au cluster expansion. Exported coefficients
for the 16-site and 64-site FCC cells are committed under [data/ce/](../../data/ce/).
[export_binary_expansion.py](export_binary_expansion.py) prepares coefficients;
[reference_chain.py](reference_chain.py) supplies classical reference sampling.
The earlier fitting workflow is in [icet-ce/](../../icet-ce/README.md).

Training reuses the existing experiment families:

| Cells | Configuration registry | Trainer |
| --- | --- | --- |
| `A1_cuau*` free and `S2_cuau*` soft | [Soft configs](../constrained_soft_02/configs.py) | [Soft entrypoint](../constrained_soft_02/run.py) |
| `H2_cuau*` specialist and composition-amortised | [Hard configs](../constrained_hard_03/configs.py) | [Hard entrypoint](../constrained_hard_03/run.py) |

## Indicative training and sampling

For a 16-site fixed-composition Cu–Au run at 500 K, the following recipe trains
for 50,000 steps and performs end-of-run evaluation. It uses the committed
cluster-expansion coefficients and is intended for a CUDA machine:

```bash
pixi run -e cuda python -m experiments.constrained_hard_03.run \
  --cfg H2_cuau16_c50_T500_mask_one_50k_house --seed 42 --no-wandb \
  --tag readme --output-dir results/demo-alloy-training
pixi run -e cuda python -m experiments.constrained_hard_03.run \
  --eval-only results/demo-alloy-training/H2_cuau16_c50_T500_mask_one_50k_house_seed42_readme \
  --eval-seed 0
```

The second command draws a fresh batch into `eval_replicate_s0/`, with the saved
evaluation budget. It writes proposal states and log importance weights.
Cu–Au checkpoints are not bundled; the [pretrained Ising examples](../../checkpoints/README.md)
provide a smaller starting point without training. This indicative recipe does
not replace the selected historical runs discussed below.

[judge_16site_cells.py](judge_16site_cells.py) uses exact enumeration;
[judge_64site_cells.py](judge_64site_cells.py) compares larger-cell outputs.
The 16-site judge performs enumeration and judging at module level. The 64-site
judge takes a positional glob rather than an argparse help flag. Read their
sources before execution.

For the retained 16-site composition-amortised figure, the selected tag ends in
`camort*perslice`; the older pooled `camort*fc` selection is a distinct result.
Local corrected-tagged evaluations do not establish the exact training revision
or checkpoint identity. Some original 64-site 500 K hard producers are missing
locally; later revival runs have different budgets/settings and cannot replace
them in a reproduction. The [tools guide](tools/README.md) identifies the retained
campaign and transfer wrappers.
