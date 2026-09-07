# Test guide

[Project overview](../README.md) · [Source guide](../src/README.md)

Run from the repository root in the locked development environment:

```bash
pixi run -e dev test
pixi run -e dev test-serial
pixi run -e dev python -m pytest tests/test_swap_ctmc.py -q
```

The default task uses pytest-xdist with work stealing. The serial task is useful
for debugging. [conftest.py](conftest.py) limits PyTorch thread counts so parallel
workers do not oversubscribe CPU cores. Accelerator-specific checks may skip
when the required hardware is unavailable.

Tests are kept flat and named after the component or experiment behaviour they
verify. Useful starting points:

| Area | Representative checks |
| --- | --- |
| Targets and exact references | [Ising](test_ising_target.py), [exact torus](test_ising_exact.py), [Potts](test_potts.py), [cluster expansion](test_cluster_expansion_target.py) |
| Rate models and invariances | [leTF](test_letf.py), [periodic backbone](test_rope_vit.py), [swap readout](test_swap_readout.py), [patch head](test_two_hole_patch_swap_head.py) |
| Dynamics and residuals | [flip CTMC](test_ctmc.py), [flip residual](test_kolmogorov.py), [swap CTMC](test_swap_ctmc.py), [swap residual](test_swap_kolmogorov.py) |
| Estimators and composition | [log-normaliser](test_log_z_estimators.py), [resampling](test_resampling.py), [fixed composition](test_fixed_composition_target.py), [per-slice estimates](test_c_t_per_slice.py) |
| Training and checkpoint continuity | [flip training](test_training.py), [flip resume](test_training_resume.py), [swap training](test_swap_training.py), [swap resume](test_swap_training_resume.py), [EMA](test_ema.py) |
| Classical and neural comparators | [Gibbs](test_oracles_gibbs.py), [Wolff](test_wolff.py), [Kawasaki](test_kawasaki.py), [masked diffusion](test_budget_masked_sampler.py), [GFlowNet](test_gfn_comparator.py) |
| Experiment integration and reports | [configuration wiring](test_configs.py), [checkpoint selection](test_eval_checkpoint_selection.py), [figure entrypoints](test_figure_entry_points.py), [hard result tables](test_hard_results_cell.py) |

Passing tests checks the implementation and small fixtures. Reproducing report
figures also requires the selected runs and references described in the
[experiment guide](../experiments/README.md).
