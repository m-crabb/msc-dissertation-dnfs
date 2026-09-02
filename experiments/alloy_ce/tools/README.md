# s117 (2026-09-02) Cu-Au 16-site judging tools

- `pull_cuau16_runs.sh` -- pull every `*cuau16-<tag>*` and `stage_4_d*_sc_hardrecipe*` run dir from
  the Modal volume, sequentially (parallel `volume get` collides), into the right results/ subdir.
  Skips dirs already holding eval/metrics.json. Run with `zsh`.
- `judge_16site_cells.py` (parent dir) -- exact 2^16 enumeration judge: ESS, F_lb / F_is vs exact,
  composition marginal. Globs `results/*/*cuau16*`, so every tag lands in one table.
- `static_flow_report.py "<glob>"` -- per hard run: train-ESS trajectory, eval ESS, samples' mean
  beta*E vs the uniform slice (identity flow) and the target, Var(log w) vs the static variance.
  The identity-flow collapse signature: samples' mean beta*E ~ uniform, Var(log w) ~ static.
- `train_ess_trajectory.py` -- train ESS every 500 steps for every cuau16 run (divide by 5000).
- `ladder_desk_stats.py` -- exact slice statistics per curriculum temperature (effN, static ESS,
  Var[beta E], KL between stages, swap stiffness).
- `samples_vs_static.py` -- the s116 cells' samples against uniform-slice and target energies.
- `compare_training_logs.py` -- windowed column means around the curriculum steps, dead vs live.
