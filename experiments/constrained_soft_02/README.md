# Soft composition constraints

[All experiments](../README.md)

Adds a finite composition penalty to the target while retaining single-spin
flip dynamics. A finite penalty biases the composition distribution; it does
not enforce an exact species count. [configs.py](configs.py) includes specialist,
composition-amortised, curriculum and Cu–Au cells. [run.py](run.py) passes these
configurations to the [shared flip trainer](../dnfs_baseline_01/run.py).

```bash
pixi run -e dev python -m experiments.constrained_soft_02.run --help
```

Training uses `--cfg`, with optional `--seed`, `--tag`, `--output-dir` and
`--no-wandb`; outputs default to `results/02_constrained_soft/`. Composition
sweeps use the baseline entrypoint's `--sweep --run-dir` interface. A single
end-of-training evaluation of an amortised cell covers its window centre;
per-composition results require the separate sweep.

## Train and sample

Train a 10,000-step critical 4×4 specialist at composition 0.5 and penalty 50,
or sample from the bundled checkpoint:

```bash
pixi run -e dev python -m experiments.constrained_soft_02.run \
  --cfg S2_d4_c0500_10k_l50_letf_house_sc --seed 42 --no-wandb \
  --tag readme --output-dir results/demo-soft-training
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_soft_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-soft-4x4
```

The shared baseline entrypoint redraws your own trained soft run (archives its
previous `eval/` first), using the sample count in its saved config:

```bash
pixi run -e dev python -m experiments.dnfs_baseline_01.run --eval-only --redraw \
  --run-dir results/demo-soft-training/S2_d4_c0500_10k_l50_letf_house_sc_seed42_readme \
  --redraw-seed 0
```

These draws can include off-composition states. See the
[checkpoint guide](../../checkpoints/README.md) for importance-weighted estimates.

| Workflow | Entrypoints |
| --- | --- |
| Main comparison tables | [4×4](analysis/house_table_soft_4x4.py), [8×8](analysis/house_table_soft_8x8.py), [10×10](analysis/house_table_soft_10x10.py) |
| Training and penalty dynamics | [training_curves_soft.py](analysis/training_curves_soft.py), [penalty_variance_traces.py](analysis/penalty_variance_traces.py), [shock_discharge_overlay.py](analysis/shock_discharge_overlay.py) |
| Composition and fidelity exhibits | [soft_results_cell.py](analysis/soft_results_cell.py), [composition_marginal_overlay.py](analysis/composition_marginal_overlay.py), [8×8 overlay](analysis/composition_marginal_overlay_8x8.py) |
| Free energy versus composition | [fc_curve.py](analysis/fc_curve.py), [fc_weighted_thermo.py](analysis/fc_weighted_thermo.py), [soft/hard comparison](analysis/fc_compare.py) |
| Classical free-energy reference | [fc_mchammer_reference.py](analysis/fc_mchammer_reference.py) |
| Amortised versus specialist sampling | [amortised_vs_specialist.py](analysis/amortised_vs_specialist.py), [amort_table_4x4.py](analysis/amort_table_4x4.py) |
| Acceptance and computational cost | [reject_off_soft.py](analysis/reject_off_soft.py), [cost_vs_quality.py](analysis/cost_vs_quality.py), [machinery_cost_repricing.py](analysis/machinery_cost_repricing.py) |
| Local-field diagnostic | [analysis_local_field_regression.py](analysis_local_field_regression.py) |
| Archived log-probability scatter | [plot_logp_scatter_4x4.py](analysis/plot_logp_scatter_4x4.py) |

Additional diagnostic entrypoints live in [analysis/](analysis/). Free-energy
figures distinguish the soft target, slice-mass correction and canonical hard
target. Reflected compositions reuse the original draws and are not independent
seeds. Historical 10×10 VC-SGC reference names also differ from later naming;
use the consumer's explicit selection rather than renaming the archive.
