# Shared scripts

[Project overview](../README.md) · [Experiment guide](../experiments/README.md)

These entrypoints serve classical baselines, comparisons across experiment
families and shared figures. Family-specific tables and diagnostics are linked
from the experiment guides.

| Task | Entrypoints |
| --- | --- |
| Recorded swap animation | [animate_recorded_swap.py](animate_recorded_swap.py) — stored arrays only |
| Ising phase illustration | [plot_ising_phases.py](plot_ising_phases.py) |
| Configuration calibration | [configuration_calibration_4x4.py](configuration_calibration_4x4.py), [plot_configuration_calibration_4x4.py](plot_configuration_calibration_4x4.py), [Modal producer](modal_configuration_calibration_4x4.py) |
| Historical scatter comparison | [plot_logp_scatters_combined.py](plot_logp_scatters_combined.py) |
| Kawasaki baselines | [kawasaki_mcmc.py](kawasaki_mcmc.py), [kawasaki_sweep.py](kawasaki_sweep.py), [kawasaki_annealed_check.py](kawasaki_annealed_check.py) |
| DNFS versus classical sampling | [compare_dnfs_vs_mcmc.py](compare_dnfs_vs_mcmc.py) |
| Mchammer and VC-SGC validation | [mchammer_baselines.py](mchammer_baselines.py), [vcsgc_mcmc_validation.py](vcsgc_mcmc_validation.py) |
| Seed aggregation | [aggregate_seeds.py](aggregate_seeds.py) |
| Euler-grid resolution | [n_euler_resolution_sweep.py](n_euler_resolution_sweep.py) |
| Swap-head transfer | [warm_start_swap_head.py](warm_start_swap_head.py) |
| Training-tail inspection | [find_hanging_tails.py](find_hanging_tails.py) |

Read each script's inputs and output defaults before running it. A figure script
may generate its own samples; a baseline script may run a long chain. The
recorded-animation example is self-contained:

```bash
pixi run -e dev python -m scripts.animate_recorded_swap --out /tmp/recorded_swap.gif
```

The five saved times are displayed without interpolation. For a static version
and the original draw selection, see [visual provenance](../assets/readme/README.md).
