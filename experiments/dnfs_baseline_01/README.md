# Unconstrained Ising baseline

[All experiments](../README.md)

Paper-derived DNFS replication with single-spin flips, progressing from a
generic MLP to locally equivariant models. [configs.py](configs.py) defines the
stages and later comparison cells; [run.py](run.py) owns the flip trainer shared
with the soft-constraint experiments. [modal_app.py](modal_app.py) contains the
remote interfaces.

Inspect the training interface from the repository root:

```bash
pixi run -e dev python -m experiments.dnfs_baseline_01.run --help
```

Training requires `--cfg`, with optional `--seed`, `--tag`, `--output-dir` and
`--no-wandb`. The default output root is `results/01_baseline/`. Fixed tags
support completion checks and resume from retained outer-cycle state.

## Train and sample

Train the 10,000-step critical 4×4 recipe, or sample from its bundled seed-42
checkpoint without training:

```bash
pixi run -e dev python -m experiments.dnfs_baseline_01.run \
  --cfg stage_4_d4_critical_sc --seed 42 --no-wandb \
  --tag readme --output-dir results/demo-baseline-training
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_baseline_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-baseline-4x4
```

To draw again from your own trained run (archives its previous `eval/` first):

```bash
pixi run -e dev python -m experiments.dnfs_baseline_01.run --eval-only --redraw \
  --run-dir results/demo-baseline-training/stage_4_d4_critical_sc_seed42_readme \
  --redraw-seed 0
```

This uses the run's configured evaluation sample count. The
[checkpoint guide](../../checkpoints/README.md) explains outputs and importance weights.

| Workflow | Entrypoint |
| --- | --- |
| 10×10 comparison table | [house_table_unconstrained_10x10.py](analysis/house_table_unconstrained_10x10.py) |
| Training curves | [training_curves_unconstrained.py](analysis/training_curves_unconstrained.py) |
| Configuration and energy demonstration | [unconstrained_clean_demo.py](analysis/unconstrained_clean_demo.py) |
| Classical reference pool and cross-check | [wolff_reference_pool.py](analysis/wolff_reference_pool.py), [reference_crosscheck.py](analysis/reference_crosscheck.py) |
| Composition acceptance | [composition_acceptance_10x10.py](analysis/composition_acceptance_10x10.py) |
| Archived log-probability scatter | [plot_logp_scatter_4x4.py](analysis/plot_logp_scatter_4x4.py) |

The shared trainer also provides saved-run evaluation and composition sweeps for
soft runs. Those operations write evaluation artifacts; see the
[run/exhibit guide](../README.md#reproduce-an-exhibit). Saved curriculum endpoints
determine the evaluation target. Historical `s223` and `s22305` couplings are
distinct from the project's exact critical `SIGMA_C`; preserve that distinction
when selecting reference data.
