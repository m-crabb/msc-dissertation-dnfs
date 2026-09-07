# Source guide

[Project overview](../README.md) · [Experiment guide](../experiments/README.md) ·
[Test guide](../tests/README.md)

`discrete_flow_sampler` contains the shared numerical implementation. Experiment
configuration, command-line entrypoints, saved-run selection and report analyses
live in [experiments/](../experiments/README.md). Install the locked Pixi
environment from the repository root; imports use `discrete_flow_sampler`, without
the `src` prefix. Import classes and functions from their defining modules.

## Follow a training path

| Regime | Target | Rates | Training and sampling |
| --- | --- | --- | --- |
| Free or softly constrained | `IsingTarget`, `PottsTarget` or `ClusterExpansionTarget` | Single-site rate model in `models/` | `training.train` → `ctmc.sample_ctmc`; residuals in `kolmogorov.py` |
| Exact composition | `FixedCompositionIsingTarget`, `FixedCompositionPottsTarget` or `FixedCompositionClusterExpansionTarget` | Pair-swap head in `constraints/` | `swap_training.train_swap` → `swap_ctmc.sample_swap_ctmc`; residuals in `swap_kolmogorov.py` |
| Composition-amortised | Composition-conditioned soft target, or mixture of fixed-composition Ising/cluster-expansion slices | `CompositionConditioned` adapter around a rate model/head | The corresponding flip or swap trainer, with composition bound for each batch |

The outer training step samples trajectories and estimates the log-normaliser
derivative on a time grid. Inner steps optimise the squared Kolmogorov-forward
residual. Sampling accumulates importance weights; optional resampling adds SMC.
The experiment runners construct the target and model, call these shared loops,
and save configurations, checkpoints and evaluations.

## Targets and state conventions

| Module | Responsibility |
| --- | --- |
| [targets/ising.py](discrete_flow_sampler/targets/ising.py) | Ising energy, composition penalties, fixed-composition and mixture targets |
| [targets/potts.py](discrete_flow_sampler/targets/potts.py) | Multi-species Potts targets, including fixed counts |
| [targets/cluster_expansion.py](discrete_flow_sampler/targets/cluster_expansion.py) | Batched binary spin-product energies and flip/swap changes from exported JSON |
| [targets/ising_exact.py](discrete_flow_sampler/targets/ising_exact.py) | Exact finite-torus Ising thermodynamics |

States have a leading batch dimension and flattened site dimension. Binary Ising
and alloy states use spins in `{-1, +1}`; Potts states use integer species labels.
The [data guide](../data/README.md) records alloy species and energy conventions.
Exact composition is preserved by the move set, whereas a soft penalty modifies
the target energy. Check each target's docstring for its coupling and normalisation.

## Models and swap heads

| Module | Responsibility |
| --- | --- |
| [models/mlp.py](discrete_flow_sampler/models/mlp.py) | General MLP rate model |
| [models/lemlp.py](discrete_flow_sampler/models/lemlp.py) | Locally equivariant MLP and shared time embedding |
| [models/letf.py](discrete_flow_sampler/models/letf.py) | Locally equivariant transformer body and attention readout |
| [models/leconv_deep.py](discrete_flow_sampler/models/leconv_deep.py) | Deep locally equivariant convolutional model |
| [models/rope_vit.py](discrete_flow_sampler/models/rope_vit.py) | Periodic positional encoding and lattice-attention backbone |
| [models/composition_conditioned.py](discrete_flow_sampler/models/composition_conditioned.py) | Exposes a composition-conditioned model through the shared `(x, t)` interface |
| [constraints/swap_readout.py](discrete_flow_sampler/constraints/swap_readout.py) | Antisymmetric pair readout, doubly hollow and mask-one heads |
| [constraints/interval_swap_head.py](discrete_flow_sampler/constraints/interval_swap_head.py) | Leave-two-out interval head and pair-index utilities |
| [constraints/masked_attention_swap_head.py](discrete_flow_sampler/constraints/masked_attention_swap_head.py) | Exclusion-mask band-attention head |
| [constraints/factorised_swap_head.py](discrete_flow_sampler/constraints/factorised_swap_head.py) | Low-rank factorised pair scores and site orderings |
| [constraints/grouped_anchor_swap_head.py](discrete_flow_sampler/constraints/grouped_anchor_swap_head.py) | Group-masked anchor contexts |
| [constraints/two_hole_patch_swap_head.py](discrete_flow_sampler/constraints/two_hole_patch_swap_head.py) | Local patch contexts and square/Bravais geometry |
| [constraints/exact_field_channel.py](discrete_flow_sampler/constraints/exact_field_channel.py) | Exact energy-change channels for swap and flip models |

These include retained ablations and earlier architectures. Use the experiment
guides and saved configuration to identify the head associated with a result.
Module docstrings describe the hollow-context assumptions and numerical caveats.

## Samplers and training support

| Module | Responsibility |
| --- | --- |
| [samplers/ctmc.py](discrete_flow_sampler/samplers/ctmc.py) | Single-site Euler CTMC rollout and importance-weight integrand |
| [samplers/kolmogorov.py](discrete_flow_sampler/samplers/kolmogorov.py) | General and locally equivariant flip residuals/loss |
| [samplers/log_z_estimators.py](discrete_flow_sampler/samplers/log_z_estimators.py) | Flip-path log-normaliser derivative estimator |
| [samplers/training.py](discrete_flow_sampler/samplers/training.py) | Flip outer/inner training loop |
| [samplers/swap_ctmc.py](discrete_flow_sampler/samplers/swap_ctmc.py) | Swap rollout, weight integrand and per-slice derivative estimates |
| [samplers/swap_kolmogorov.py](discrete_flow_sampler/samplers/swap_kolmogorov.py) | Swap residuals, loss and microbatched backward pass |
| [samplers/swap_training.py](discrete_flow_sampler/samplers/swap_training.py) | Swap outer/inner training loop |
| [samplers/_neighbours.py](discrete_flow_sampler/samplers/_neighbours.py), [samplers/_swap_neighbours.py](discrete_flow_sampler/samplers/_swap_neighbours.py) | Internal neighbour evaluation and pair indexing |
| [samplers/resampling.py](discrete_flow_sampler/samplers/resampling.py) | Adaptive systematic resampling and SMC normaliser estimates |
| [samplers/optim.py](discrete_flow_sampler/samplers/optim.py) | Shared stable AdamW implementation |
| [samplers/resume.py](discrete_flow_sampler/samplers/resume.py) | Checkpoint, RNG-state and training-log resume support |
| [ema.py](discrete_flow_sampler/ema.py) | Parameter EMA and time-grid estimator EMA |
| [composition.py](discrete_flow_sampler/composition.py) | Composition draws and contiguous batch expansion |
| [seeding.py](discrete_flow_sampler/seeding.py) | Shared random seeding |

## Comparators and diagnostics

| Module | Responsibility |
| --- | --- |
| [samplers/budget_masked.py](discrete_flow_sampler/samplers/budget_masked.py) | Fixed-composition masked-diffusion comparator and objectives |
| [models/raster_gfn_policy.py](discrete_flow_sampler/models/raster_gfn_policy.py), [samplers/gfn_objectives.py](discrete_flow_sampler/samplers/gfn_objectives.py) | Raster-order GFlowNet policy and training objectives |
| [mcmc/gibbs.py](discrete_flow_sampler/mcmc/gibbs.py) | Single-site heat-bath reference sampler |
| [mcmc/wolff.py](discrete_flow_sampler/mcmc/wolff.py) | Unconstrained Ising cluster reference sampler |
| [mcmc/kawasaki.py](discrete_flow_sampler/mcmc/kawasaki.py) | Local and nonlocal count-preserving swap chains |
| [mcmc/mchammer_ising.py](discrete_flow_sampler/mcmc/mchammer_ising.py) | icet/mchammer canonical, SGC and VCSGC baselines |
| [diagnostics/metrics.py](discrete_flow_sampler/diagnostics/metrics.py) | Importance-weight, observable, exact-enumeration and chain diagnostics |
| [diagnostics/flops.py](discrete_flow_sampler/diagnostics/flops.py) | Measured model FLOPs and analytical sampler/training accounting |
| [diagnostics/figure_style.py](discrete_flow_sampler/diagnostics/figure_style.py) | Shared figure palette, axes and uncertainty styling |

## Organisation conventions

Keep reusable target, model and sampler logic here; put campaign-specific
configuration and analysis beside the relevant experiment. Cross-experiment
drivers belong in [scripts/](../scripts/README.md), cluster launchers in
[slurm/](../slurm/README.md), and correctness checks in [tests/](../tests/README.md).
Historical module paths and checkpoint keys are part of the reproduction record.
