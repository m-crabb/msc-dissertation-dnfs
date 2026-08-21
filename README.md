# Discrete Neural Samplers with Constraints

Code for an MSc dissertation at Imperial College London on **constrained discrete neural
samplers for materials configurations**.

## Overview

Many materials problems (high-entropy alloys, battery cathodes, disordered solids) require
drawing lattice configurations from a finite-temperature Boltzmann distribution

```
pi(x) ~ exp(-beta * E(x)),    x in {0, ..., S-1}^d,    subject to a constraint such as fixed composition c(x) = c_target.
```

The partition function is intractable, the target is multimodal at low temperature, and the
constraint reshapes the support. Classical Markov chain Monte Carlo struggles here (critical
slowing-down and broken ergodicity), and existing discrete neural samplers are unconstrained.

This project builds on Discrete Neural Flow Samplers (DNFS), which learn the rate matrix of a
continuous-time Markov chain (CTMC) that transports an easy base distribution to the target by
minimising a squared Kolmogorov-forward residual, using a discrete Stein control variate for the
intractable log-partition derivative and locally equivariant architectures for an O(1) residual.
The dissertation extends this to constrained sampling for materials, with the constraint handled
at the level of the CTMC move set.

## Status

Work in progress (MSc dissertation). Completed to date:

- a faithful, staged reproduction of the unconstrained DNFS baseline (MLP -> locally equivariant
  MLP -> Stein control variate -> locally equivariant CNN -> locally equivariant Transformer);
- motivating failure-mode experiments for classical MCMC in the constrained, low-temperature
  regime (critical slowing-down and mode collapse under composition-preserving Kawasaki dynamics);
- a soft (penalty / VCSGC-style) composition-constraint extension and an analysis of its limits;
- native hard composition constraints via a composition-preserving swap CTMC with learned rates
  (masked-attention pair head computing all swap rates in one pass), exact on the constraint
  manifold by construction: validated against exact enumeration at 4x4, scaled to an 8x8
  record, and benchmarked in a pre-registered mixing probe against tuned Kawasaki baselines
  (icet/mchammer and a compiled local-swap kernel) with certified reference chains;
- thermodynamic-integration free-energy references on the fixed-composition slice, validated
  at 4x4 against enumeration;
- a second constrained sampler family: budget-masked masked diffusion (MDNS-style) with the
  constrained reference process, weight algebra, preconditioner, and denoising loss derived
  and gate-tested at 4x4.

In progress: the 16x16 scaling run, cross-size classical baseline slates, and the dissertation
writeup.

## Setup

Environment and dependencies are managed with [pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # macOS / Linux; see pixi docs for other platforms
pixi install                                 # resolve and install the default environment
pixi run -e dev test                         # run the test suite
```

The lockfile (`pixi.lock`) is committed, so installs resolve against the same dependency set
used for the GPU runs. Supported platforms are `linux-64` and `osx-arm64`.

### Tasks

- `pixi run -e dev test` - run the pytest suite
- `pixi run -e dev lint` - ruff lint `src/`, `experiments/`, `tests/`
- `pixi run -e dev format` - ruff format the same directories
- `pixi run jupyter` - launch JupyterLab

## Repository layout

```
src/discrete_flow_sampler/   library code
  targets/                   target distributions (Ising, soft composition penalty)
  samplers/                  CTMC simulation, Kolmogorov residual, Stein log-Z estimator, training loop,
                             budget-masked masked diffusion (budget_masked.py)
  models/                    rate-matrix parameterisations (MLP, leMLP, leConv, leTF)
  mcmc/                      classical baselines (Gibbs, Kawasaki) for validation and failure-mode demos
  diagnostics/               sampler-quality metrics (ESS, etc.)
  constraints/               constraint handling
experiments/                 per-experiment configs, runners, and analysis
  dnfs_baseline_01/          paper-faithful unconstrained Ising replication (stages 0-4)
  constrained_soft_02/       soft composition-constraint extension
  constrained_hard_03/       hard-constraint work (in progress)
scripts/                     standalone baselines and figure scripts (Kawasaki MCMC, DNFS-vs-MCMC, plots)
slurm/                       batch scripts for the Imperial DoC GPU cluster (primary compute)
tests/                       pytest suite (correctness checks, see below)
notebooks/                   familiarisation and cross-check notebooks
data/                        small input assets
results/                     per-run artefacts (gitignored)
```

## Running experiments

Each directory under `experiments/` follows the same pattern:

- `configs.py` - named run configurations (lattice size, coupling, model stage, training budget);
- `run.py` - local entry point for a single run;
- `modal_app.py` - launcher for GPU runs on [Modal](https://modal.com) (single GPU per run;
  GPU runs now go primarily to a Slurm cluster via the scripts in `slurm/`, same locked env);
- `analysis/` - post-hoc analysis and figure generation.

Classical baselines and motivating experiments are standalone scripts under `scripts/` (for
example `kawasaki_mcmc.py`, `compare_dnfs_vs_mcmc.py`, `vcsgc_mcmc_validation.py`,
`plot_ising_phases.py`). Per-run artefacts (config, training log, checkpoints, evaluation files)
are written under `results/` and are not tracked.

## Acknowledgements

This work builds on Discrete Neural Flow Samplers by Zijing Ou and collaborators; the original
code is at https://github.com/J-zin/DNFS. Implementation here is derived from the papers'
equations and algorithmic descriptions.

## License

To be released under an open-source licence (MIT recommended) at final submission.

## Citation

Mitchell Crabb, *Discrete Neural Flow Samplers with Constraints for Materials Discovery*,
MSc dissertation, Imperial College London, 2026 (working title).
