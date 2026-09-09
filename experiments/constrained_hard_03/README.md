# Exact composition through swap dynamics

[All experiments](../README.md) · [Launcher index](../../slurm/README.md)

Swap moves preserve species counts throughout the path. The experiments compare
swap-head architectures, Ising lattices through 24×24, composition conditioning,
Cu–Au cluster expansions, classical Kawasaki sampling, GFlowNets and budget-masked
diffusion. [configs.py](configs.py) defines swap cells and head construction;
[gfn_configs.py](gfn_configs.py) holds the GFlowNet comparisons.

```bash
pixi run -e dev python -m experiments.constrained_hard_03.run --help
```

[run.py](run.py) trains a selected `--cfg` and writes to `results/03_hard/` by
default. It also provides checkpoint evaluation, grid changes, EMA, selected-stage
and SMC modes. **`--eval-only RUN_DIR` generates samples.** It validates the saved
configuration against the current registry and refuses collisions with existing
evaluation artifacts. A rejected historical configuration requires provenance
review; changing the saved config to bypass the guard changes its meaning.

## Train and sample

The small 4×4 example trains for 2,000 steps at coupling 0.1. Add `--smoke`
and choose a different tag for a four-step pipeline check:

```bash
pixi run -e dev python -m experiments.constrained_hard_03.run \
  --cfg H2_d16_c50_s010_letf_dh --seed 42 --no-wandb \
  --tag readme --output-dir results/demo-hard-training
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-hard-4x4
```

The README's 24×24 radius-3 model uses a 100,000-step coupling curriculum and
bf16 training. The full recipe is intended for a CUDA machine:

```bash
pixi run -e cuda python -m experiments.constrained_hard_03.run \
  --cfg H2_d576_c50_s220_letf_thp3_100k_curr_b512_ne128_cv2_w5bf16 \
  --seed 42 --no-wandb --tag readme --output-dir results/demo-hard-training
```

You can sample its bundled EMA weights on CPU without training:

```bash
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_24x24 \
  --n-samples 8 --batch-size 1 --seed 0 --out results/demo-hard-24x24
```

For a fresh evaluation of your own trained 4×4 run, using its saved evaluation
budget and writing to `eval_replicate_s0/`:

```bash
pixi run -e dev python -m experiments.constrained_hard_03.run \
  --eval-only results/demo-hard-training/H2_d16_c50_s010_letf_dh_seed42_readme \
  --eval-seed 0
```

Both bundled examples preserve exact composition. The
[checkpoint guide](../../checkpoints/README.md) documents raw versus EMA selection,
importance weights and trajectory recording. Full training recipes illustrate
current code; they do not establish bit-for-bit historical training reproduction.

| Workflow | Entrypoints |
| --- | --- |
| Main comparison tables | [4×4](analysis/house_table_4x4.py), [8×8](analysis/house_table_8x8.py), [16×16](analysis/house_table_16x16.py), [20×20](analysis/house_table_20x20.py), [24×24](analysis/house_table_24x24.py) |
| Training and sample exhibits | [training_curves_hard.py](analysis/training_curves_hard.py), [hard_results_cell.py](analysis/hard_results_cell.py), [sample montages](analysis/sample_montages.py) |
| Learned swap rates | [rate_field_rows.py](analysis/rate_field_rows.py), [rate_field_strip.py](analysis/rate_field_strip.py), [retained manifest](../../assets/hard_rate_field_runs.json) |
| Composition transfer and free energy | [composition_transfer_figure.py](analysis/composition_transfer_figure.py), [fc_surface.py](analysis/fc_surface.py), [zero_shot_tables.py](analysis/zero_shot_tables.py), [slice_ti.py](probes/slice_ti.py) |
| Exact small-system checks | [gate_4x4.py](probes/gate_4x4.py), [gate_camort_4x4.py](probes/gate_camort_4x4.py), [demo_4x4.py](analysis/demo_4x4.py) |
| Kawasaki comparison and reference | [probe_kawasaki_8x8.py](probes/probe_kawasaki_8x8.py), [probe_analysis_8x8.py](analysis/probe_analysis_8x8.py), [plot_probe_8x8.py](analysis/plot_probe_8x8.py), [generate_kawasaki_reference_d256.py](probes/generate_kawasaki_reference_d256.py) |
| Transport and local-field diagnostics | [analysis_transport_decomposition.py](probes/transport_decomposition.py), [analysis_local_field_regression.py](analysis/local_field_regression.py) |
| Compute measurements | [measure_training_flops.py](probes/measure_training_flops.py), [compile_gate.py](probes/compile_gate.py) |
| Warm-base experiments | [warm_base_reference.py](probes/warm_base_reference.py), [warm_base_offline_table.py](analysis/warm_base_offline_table.py), [warm_base_t_grid.py](probes/warm_base_t_grid.py) |
| Neural comparators | [run_gfn.py](run_gfn.py), [mdns_vs_dnfs_4x4.py](analysis/mdns_vs_dnfs_4x4.py), [mdns_budget_gate_4x4.py](probes/mdns_budget_gate_4x4.py) |

House tables have their own frozen seed/tag/reference selections and raw/EMA
defaults. The README now uses the bundled EMA model's
[24×24 recording](../../assets/hard_rate_field_strip_24x24.npz); the report's
`fig:rate-field-strip` is the [16×16 recording](../../assets/hard_rate_field_strip_16x16.npz)
(bundle `checkpoints/ising_hard_16x16`, seed 20260906, chosen for visual clarity among
ten seeds 20260905–20260914, rendered with `--recorded-stride 32`); the earlier
[20×20 recording](../../assets/hard_rate_field_strip_20x20.npz) remains available.
Their `--recorded` rendering mode needs no checkpoint. The static renderer's live
mode uses a different draw contract. See [visual provenance](../../assets/readme/README.md).

Composition-amortised artifacts require particular care: corrected-tagged d64
outputs have retained evaluation evidence but no verified local trainer revision
or checkpoints. The historical d256 seed-42 result used a pooled training
baseline, and the protected additional-seed launcher snapshot predates the
per-slice correction. Raw, EMA and SMC results are separate evidence channels.
Keep these qualifications when interpreting or presenting results.
