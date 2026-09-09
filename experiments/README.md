# Experiment guide

[Project overview](../README.md) · [Launcher index](../slurm/README.md) ·
[Shared scripts](../scripts/README.md) · [Source guide](../src/README.md)

The dissertation follows three sampling regimes and then applies them to a
Cu–Au cluster expansion. The numbered directories record this progression;
the alloy experiments reuse the soft and hard trainers.

| Study | Start here | Training entrypoint | Default output root |
| --- | --- | --- | --- |
| Unconstrained Ising replication | [Baseline guide](dnfs_baseline_01/README.md) | `python -m experiments.dnfs_baseline_01.run` | `results/01_baseline/` |
| Soft composition penalties | [Soft guide](constrained_soft_02/README.md) | `python -m experiments.constrained_soft_02.run` | `results/02_constrained_soft/` |
| Exact composition via swaps | [Hard guide](constrained_hard_03/README.md) | `python -m experiments.constrained_hard_03.run` | `results/03_hard/` |
| Cu–Au materials application | [Alloy guide](alloy_ce/README.md) | Soft trainer for `A1_`/`S2_`; hard trainer for `H2_` cells | Soft or hard output root |

Run modules from the repository root in the Pixi environment. Append `--help`
to the three training entrypoints above to inspect their arguments without
training. Configurations live in each experiment's `configs.py`; their IDs
are also stored in historical run names. In names, baseline/soft `d8` usually
means an 8×8 lattice, whereas hard `d64` means 64 sites. Read the saved config
for the actual geometry, coupling, composition, curriculum and head.

## Start with training or sampling

The [checkpoint guide](../checkpoints/README.md) provides five bundled pretrained
models and copy-paste sampling commands, including the 24×24 README model.
For example:

```bash
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-hard-4x4
pixi run -e dev python -m experiments.constrained_hard_03.run \
  --cfg H2_d16_c50_s010_letf_dh --smoke --seed 42 --no-wandb \
  --tag readme-smoke --output-dir results/demo-training
```

The second command is a four-step pipeline check. Remove `--smoke` and use a new
tag for the configured 2,000-step run. Each family guide above has full-budget
training and fresh-sampling examples. Training includes end-of-run evaluation;
the bundled-checkpoint helper allows smaller standalone draws into a new directory.

## Read a run

A typical learned run is named `<cfg>_seed<seed>_<tag>` and contains:

| File or directory | Meaning |
| --- | --- |
| `config.json`, `metadata.json` | Saved configuration and available run provenance |
| `training_log.json` | Training and diagnostic history |
| `checkpoints/final.pt` | Final raw parameters |
| `checkpoints/final_ema.pt` | Exponential moving average parameters, when enabled |
| `checkpoints/resume.pt` | Training state at an outer-cycle boundary, when retained |
| `eval/` | Raw-checkpoint metrics, samples and log importance weights |
| `eval_ema/` | EMA-checkpoint evaluation |
| `eval_ne*`, `eval_ema_ne*` | Separate evaluations on an overridden Euler grid |
| `eval_smc_tau*`, `eval_replicate_s*`, `eval_stage*` | SMC, replicate-seed or selected-stage evaluations |

Archive contents vary by trainer and transfer history. A metrics file does not
establish that samples, weights, checkpoints or the exact training revision are
available. Production archives under `results/` are not distributed in Git;
the selected inference bundles under `checkpoints/`, small inputs under `data/`
and recorded README trajectory under `assets/` are.

## Reproduce an exhibit

Start with the analysis entrypoint in the family guide, then identify its
explicit run/tag/seed selection and required reference data. Keep raw, EMA,
SMC and grid evaluations separate. Some historical scripts select fixed tags
or newest matching directories; defaults alone do not specify a report result.
Write any new outputs to a separate directory and retain the selected inputs.

An evaluation command may sample a checkpoint or rewrite metrics. In particular,
hard `--eval-only RUN_DIR` draws samples; baseline `--eval-only --run-dir RUN_DIR`
recomputes stored evaluation statistics, and `--redraw` draws anew.
Several older analysis scripts have no argument parser or execute work at import,
so `--help` is not a universal inspection command. Read those sources first.

For a self-contained rendering example using the committed trajectory:

```bash
pixi run -e dev python -m scripts.animate_recorded_swap --out /tmp/recorded_swap.gif
pixi run -e dev python -m experiments.constrained_hard_03.analysis.rate_field_strip \
  --recorded assets/hard_rate_field_strip_24x24.npz --recorded-stride 32 \
  --out /tmp/recorded_swap.pdf
```

Both commands read stored frames. The [visual provenance](../assets/readme/README.md)
records their trajectory selection and the separate free-energy figure inputs.

## Organisation and compatibility

Each numbered experiment directory keeps its trainer, configuration registry
and Modal app at the top level (the hard experiment also keeps its GFlowNet
trainer and registry there); `alloy_ce/` has no trainer of its own and reuses
the soft and hard ones. `analysis/` holds the scripts that read archived
runs and emit report tables and figures; `probes/` holds everything that spends
compute without being a training run: exact-enumeration gates, reference
chains, benches, profilers and compile gates. The Cu–Au launch and pull
wrappers live in `alloy_ce/launchers/`. Historical filenames, config IDs and
checkpoint paths remain part of the reproduction record; the launchers under
`slurm/` were updated to the current module paths when the layout changed.

Retained experimental variants include negative controls, unsuccessful campaigns
and earlier architectures. Their presence does not imply a recommendation to run
them. The [launcher index](../slurm/README.md) records withdrawn campaigns and
other restrictions. Training provenance and missing archive inputs remain limits
on exact reproduction, including corrected-tagged composition-amortised runs
without local trainer revisions/checkpoints and missing original Cu–Au 64-site
500 K hard runs.
