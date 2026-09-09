# Cu–Au launch and transfer wrappers

[Alloy experiment](../README.md) · [All experiments](../../README.md)

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
