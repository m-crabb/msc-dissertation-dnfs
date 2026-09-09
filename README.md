# Discrete Neural Samplers with Constraints

Code accompanying my MSc dissertation at Imperial College London on constrained
sampling of discrete materials configurations, supervised by Yingzhen Li,
Zijing Ou and Alex Ganose.

The project extends Discrete Neural Flow Samplers (DNFS) from unconstrained
Ising systems to composition penalties, exact composition through swap dynamics,
and Cu–Au cluster-expansion targets. A learned continuous-time Markov chain
transports a simple base distribution towards a Boltzmann target; importance
weights support target expectations and free-energy estimation.
Training minimises a squared Kolmogorov-forward residual, using a discrete Stein
control variate to estimate the log-normaliser derivative.

![A recorded 24×24 neural swap trajectory, alongside learned swap rates from a marked site.](assets/readme/recorded_swap.gif)

**Composition is fixed throughout the path:** all 129 recorded grid states contain
288 sites of each species. This is one raw proposal trajectory from the bundled
24×24 EMA checkpoint, recorded over 128 matching steps without interpolation.
The rate panel uses a common scale across frames.
[Static figure and provenance](assets/readme/README.md).

## Explore the project

| Study | What it investigates |
| --- | --- |
| [Unconstrained baseline](experiments/dnfs_baseline_01/README.md) | Paper-derived Ising replication with locally equivariant rate models |
| [Soft constraints](experiments/constrained_soft_02/README.md) | Composition penalties, acceptance, amortisation and free-energy corrections |
| [Hard constraints](experiments/constrained_hard_03/README.md) | Count-preserving swap heads, Ising lattices through 24×24, and neural/classical comparisons |
| [Cu–Au application](experiments/alloy_ce/README.md) | Free, soft and fixed-composition sampling on 16-site and 64-site FCC cluster expansions |

The [experiment guide](experiments/README.md) explains entrypoints, saved runs
and figure reproduction. The [launcher index](slurm/README.md) groups the retained
cluster campaigns; [shared scripts](scripts/README.md) cover classical baselines
and comparisons across experiments.
The [source guide](src/README.md) maps every core module and follows the flip and
swap training paths; the [test guide](tests/README.md) groups the main checks.

## A result: free energy across compositions

![Free energy per site and residuals for soft and hard samplers against thermodynamic integration on the critical 8×8 Ising lattice.](assets/readme/fc_hard_direct_8x8_sc.png)

At the critical 8×8 Ising cell, the retained figure compares soft estimates,
their slice-mass correction, and hard mean-log estimates against thermodynamic
integration. It uses Euler-grid extrapolation, three hard seeds and four soft
seeds. The hard branch uses EMA evaluations; the soft branch uses raw-checkpoint
evaluations. The image is copied unchanged from the dissertation.
[Input selections and reproduction limits](assets/readme/README.md#free-energy-comparison).

## Setup

The environment is managed with [Pixi](https://pixi.sh). From the repository root:

```bash
pixi install --locked -e dev
pixi run -e dev test
```

The committed `pixi.lock` covers `linux-64` and `osx-arm64`. CUDA runs use the
separate Linux `cuda` environment. Other project tasks are:

```bash
pixi run -e dev lint       # Ruff: src/, experiments/, tests/
pixi run -e dev format     # Format those directories (modifies files)
pixi run jupyter
```

To inspect the hard-training interface or render the committed snapshots:

```bash
pixi run -e dev python -m experiments.constrained_hard_03.run --help
pixi run -e dev python -m scripts.animate_recorded_swap --out /tmp/recorded_swap.gif
```

Training and evaluation examples are described in the [experiment guide](experiments/README.md).
The test suite covers analytic residuals, estimator checks, equivariance,
locality, exact composition and checkpoint/resume behaviour.

## Sample from pretrained checkpoints

Five small [checkpoint bundles](checkpoints/README.md) are included in Git:
unconstrained, soft and hard 4×4 Ising models, a hard 16×16 model, plus the
hard 24×24 model above.
Each includes its saved configuration, original weights and SHA-256 manifest.
From the repository root, after setup:

```bash
# Fast CPU example: 64 fixed-composition proposals and their importance weights.
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-hard-4x4

# The README model; keep batches small on CPU.
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_24x24 \
  --n-samples 8 --batch-size 1 --seed 0 --out results/demo-hard-24x24
```

Each command creates `samples.pt`, `log_weights.pt` and `metadata.json` in a new
output directory. These are raw proposal draws; use the importance weights for
target expectations. The [checkpoint guide](checkpoints/README.md) includes a
weighted-estimate example, a sampling command per bundle and animation reproduction.

## Train a sampler

For a short end-to-end check, then a 2,000-step hard 4×4 training example:

```bash
pixi run -e dev python -m experiments.constrained_hard_03.run \
  --cfg H2_d16_c50_s010_letf_dh --smoke --seed 42 --no-wandb \
  --tag readme-smoke --output-dir results/demo-training
pixi run -e dev python -m experiments.constrained_hard_03.run \
  --cfg H2_d16_c50_s010_letf_dh --seed 42 --no-wandb \
  --tag readme --output-dir results/demo-training
```

The family READMEs provide [baseline](experiments/dnfs_baseline_01/README.md),
[soft](experiments/constrained_soft_02/README.md),
[24×24 hard](experiments/constrained_hard_03/README.md) and
[Cu–Au](experiments/alloy_ce/README.md) training examples. Full training budgets
can be substantial; these commands illustrate the current recipes, rather than
promise bit-for-bit recovery of historical runs.

## Repository layout

See the [source guide](src/README.md) for the module-by-module map, and the
[data](data/README.md) and [notebook](notebooks/README.md) guides for retained inputs
and exploratory work.

```text
src/discrete_flow_sampler/
  targets/       Ising, Potts and cluster-expansion targets
  models/        Rate-model backbones and conditioning
  constraints/   Swap-readout heads and exact-field channels
  samplers/      Flip/swap CTMCs, residuals, training, SMC and neural baselines
  mcmc/          Gibbs, Wolff, Kawasaki and mchammer baselines
  diagnostics/   Metrics, FLOP accounting and figure style
experiments/     Configurations, trainers, analysis and experiment guides
scripts/         Shared baselines, comparisons and figure tools
slurm/           Historical cluster launchers and campaign index
assets/          Small retained figure inputs and README visuals
checkpoints/     Four pretrained examples, saved configs and checksum manifests
data/           Exported cluster expansions and reference database
icet-ce/         Cluster-expansion fitting and energy-prediction workflow
notebooks/       Familiarisation and cross-check notebooks
tests/           Correctness and regression checks
```

Production runs are stored locally under `results/` and are not tracked. A Git
checkout therefore supports code inspection, tests, pretrained sampling and the animation;
most report analyses also require the original run archives and references.
Some historical checkpoints, samples and exact training revisions remain
unavailable. The guides preserve these distinctions and the withdrawn campaign
restrictions.

## Acknowledgements

This work builds on Discrete Neural Flow Samplers by Zijing Ou and collaborators.
The original code is at [J-zin/DNFS](https://github.com/J-zin/DNFS).
Implementation here is derived from the papers' equations and algorithmic
descriptions.

## License

Original code and documentation are released under the [MIT License](LICENSE),
copyright © 2026 Mitchell Crabb. Third-party inputs, adapted tutorial material
and derived tutorial outputs retain their upstream notices and terms; see
[third-party notices](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/).

## Citation

If you find this work useful, please cite both of the below:

```bibtex
@thesis{
  crabb2026discrete,
  title={Discrete Neural Flow Samplers with Constraints for Materials Discovery},
  author={Mitchell Crabb},
  institution={Imperial College London},
  type={mathesis},
  year={2026},
  pubstate={inpreparation},
  editora={Yingzhen Li and Zijing Ou and Alex Ganose},
  editoratype={collaborator}
}

@inproceedings{
  ou2025discrete,
  title={Discrete Neural Flow Samplers with Locally Equivariant Transformer},
  author={Zijing Ou and Ruixiang ZHANG and Yingzhen Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=Wk65okms3T}
}
```
