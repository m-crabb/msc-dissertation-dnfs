# Cu–Au analysis and campaign tools

[Alloy experiment](../README.md) · [All experiments](../../README.md)

## Figures and diagnostics

| Task | Tool |
| --- | --- |
| Exact 16-site composition/free-energy exhibit | [cuau16_figure.py](cuau16_figure.py) |
| FCC configurations and ordered structures | [fcc_render.py](fcc_render.py) |
| Composition histograms | [composition_histograms.py](composition_histograms.py) |
| Flow versus uniform-slice statistics | [static_flow_report.py](static_flow_report.py), [samples_vs_static.py](samples_vs_static.py) |
| Training history and curriculum comparisons | [train_ess_trajectory.py](train_ess_trajectory.py), [compare_training_logs.py](compare_training_logs.py) |
| Exact curriculum slice statistics | [ladder_desk_stats.py](ladder_desk_stats.py) |
| Two-hole patch reach regression | [patch_reach_probe.py](patch_reach_probe.py) |

The cell-scoring scripts live one directory up: [16-site](../judge_16site_cells.py) and
[64-site](../judge_64site_cells.py). Several tools enumerate states, fit models,
or inspect run files on execution or import; read the source before using them.
The retained composition-amortised exhibit selects corrected-tagged
`*camort*perslice` outputs. Older `*camort*fc` examples select pooled predecessors.

## Historical launch and transfer wrappers

All seven wrappers use zsh and resolve the repo root from their own location. Launchers call
the soft/hard Modal apps with fixed tags and return after submission; they do
not establish completion of the remote jobs.

| Wrapper | Campaign or transfer selection |
| --- | --- |
| [launch_cuau16_fc.sh](launch_cuau16_fc.sh) | `20260904-cuau16-fc`: 16-site specialist/amortised hard cells and free high-temperature controls; predates the per-slice correction |
| [launch_cuau64_grid.sh](launch_cuau64_grid.sh) | `20260904-cuau64-grid`: 1200 K and 680 K hard/free grid |
| [launch_cuau64_house.sh](launch_cuau64_house.sh) | `20260903-cuau64-house`: 500 K hard, free and soft comparisons |
| [launch_cuau64_thp.sh](launch_cuau64_thp.sh) | `20260903-cuau64-thp`: two-hole patch hard cells |
| [launch_cuau64_thp_revive.sh](launch_cuau64_thp_revive.sh) | `20260904-cuau64-revive`: later ladder, budget and Euler-grid variants; separate from original house results |
| [pull_cuau16_runs.sh](pull_cuau16_runs.sh) | Sequential whole-directory pulls of Cu–Au 16-site tags and hard-recipe controls; skips on existing raw metrics alone |
| [pull_modal_runs.sh](pull_modal_runs.sh) | Regex-selected config, metadata, log and raw/EMA evaluation transfers; excludes checkpoints and auxiliary evaluations |

Pass an explicit selection when adapting a pull wrapper: the generic wrapper's
empty pattern can match all names. Its skip checks do not establish a complete
archive. Preserve original tags and partial-data distinctions; the
[launcher guide](../../../slurm/README.md) describes the shared execution context.
