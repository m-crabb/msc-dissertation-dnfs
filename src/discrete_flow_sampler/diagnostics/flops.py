"""FLOP accounting for the house table's FLOP/es column.

The column prices what a practitioner pays at sampling time for one
independent-equivalent sample; training cost is amortised and lives in the
appendix recipe table. Both sampler families reduce to
(cost per raw draw) / (independence yield per draw):

  neural:  n_euler forwards + IS-weight target evals, divided by the
           ESS fraction of the frozen evaluation;
  chain:   total sweep/cluster work including burn-in, divided by
           N / tau_int effective records.

The network term is measured: one forward under
``torch.utils.flop_counter.FlopCounterMode`` at the run's shapes, so no
analytic formula has to track every head variant. Convention: 1
multiply-accumulate = 2 FLOPs (the counter's own), so all counts are even.
The chain terms are analytic per-work-unit constants (below), since a chain's
inner loop is scalar ops no tensor-level counter sees; the printed cells carry
2 significant figures and their uncertainty is dominated by tau_int.

The FLOP currency is architecture-blind (a dense-matmul FLOP and a
pointer-chasing chain FLOP buy different wall-clock); the thesis states this
once where the column is defined.
"""

import math

from torch.utils.flop_counter import FlopCounterMode

# One Ising target log-prob evaluation, per site: the neighbour sum is two
# torch.roll'd elementwise products (2 mult + 2 add per site under the
# double-counted x^T J x convention), the coupling scale and reduction add
# ~2 more; rounded up to 10. It sits four-plus orders below a network
# forward, so the rounding is invisible at 2 significant figures.
ISING_ENERGY_FLOPS_PER_SITE = 10

# One Gibbs heat-bath single-site update: 3 adds for the 4-neighbour local
# field, 1 multiply by the coupling, the conditional-flip probability's exp
# charged at 4, one compare against the uniform draw -- ~9, rounded up to
# 12 to cover the bookkeeping. Per sweep this multiplies by d.
GIBBS_FLOPS_PER_SITE_UPDATE = 12

# One site added to a Wolff cluster: its 4 torus bonds are examined
# (4 lookups + 4 alignment compares + 4 membership tests), aligned
# non-members draw a uniform against p_add (~2 each), plus frontier
# bookkeeping and the flip -- ~20, rounded up to 30. Total cluster work is
# proportional to the summed cluster sizes, which
# ``wolff_sample(cluster_size_log=...)`` records exactly; with the build
# seeds the certified pools are recounted bit-identically, never rebuilt.
WOLFF_FLOPS_PER_CLUSTER_SITE = 30


# One VC-SGC trial: a single-site flip proposal. The Ising part is the
# Gibbs local-field work (the 12 above); the penalty part is the difference
# kappa*d*[(c'-c_t)^2 - (c-c_t)^2] off a cached running composition --
# c' = c +- 1/d is one add, the two squared deviations and their difference
# ~3 more -- charged at 4. The column prices this cached form, not an
# O(d) re-sum of the composition per trial, consistent with pricing Gibbs
# off the local field.
VCSGC_FLOPS_PER_TRIAL = GIBBS_FLOPS_PER_SITE_UPDATE + 4


def measured_forward_flops(model, example_inputs: tuple) -> int:
    """FLOPs of one forward at the given shapes, measured by FlopCounterMode.

    Call on the eager model (a compiled wrapper can hide ops from the
    dispatch-level counter); compilation changes scheduling, not the
    mathematics, so the eager count prices the compiled run too.
    """
    counter = FlopCounterMode(display=False)
    with counter:
        model(*example_inputs)
    return counter.get_total_flops()


def ising_energy_eval_flops(n_sites: int) -> int:
    """One target log-prob evaluation (see ISING_ENERGY_FLOPS_PER_SITE)."""
    return ISING_ENERGY_FLOPS_PER_SITE * n_sites


def neural_sampling_flops_per_sample(
    per_forward_flops: int, n_euler_steps: int, n_sites: int
) -> int:
    """Price of one raw weighted sample from the trained sampler.

    Each Euler step is one rate-matrix forward (all d flip rates) plus one
    target evaluation for the IS-weight integrand. Base draw and final
    softmax are O(d) and O(1) per sample, below the energy term, and are not
    itemised.
    """
    return n_euler_steps * (per_forward_flops + ising_energy_eval_flops(n_sites))


def per_effective_sample(flops_per_raw_sample: float, ess_fraction: float) -> float:
    """Convert a raw-sample price to the independent-equivalent price."""
    if not 0.0 < ess_fraction <= 1.0:
        raise ValueError(f"ESS fraction must be in (0, 1], got {ess_fraction}")
    return flops_per_raw_sample / ess_fraction


def gibbs_run_flops(n_sites: int, n_sweeps: int) -> int:
    """Total price of a Gibbs run of n_sweeps full-lattice sweeps."""
    return GIBBS_FLOPS_PER_SITE_UPDATE * n_sites * n_sweeps


def wolff_run_flops(total_cluster_sites: int) -> int:
    """Total price of a Wolff run whose clusters summed to this many sites."""
    return WOLFF_FLOPS_PER_CLUSTER_SITE * total_cluster_sites


def vcsgc_run_flops(n_trials: int) -> int:
    """Total price of a VC-SGC chain of n_trials single-flip proposals.

    Burn-in trials belong in n_trials (paid before the first usable frame),
    mirroring gibbs_run_flops / wolff_run_flops.
    """
    return VCSGC_FLOPS_PER_TRIAL * n_trials


def sgc_run_flops(n_trials: int) -> int:
    """Total price of a semi-grand-canonical chain of n_trials flip proposals.

    At Delta-mu = 0 an SGC trial is a Gibbs single-site update: propose a
    species change at one site, evaluate the local field, accept. The
    chemical-potential term contributes mu_Au - mu_Ag = 0 to the acceptance
    exponent, so there is no composition rider: ``VCSGC_FLOPS_PER_TRIAL``
    minus its 4-FLOP cached-composition update. Burn-in trials belong in
    n_trials. Not ``gibbs_run_flops``, which is per lattice sweep; mchammer
    counts trial moves.
    """
    return GIBBS_FLOPS_PER_SITE_UPDATE * n_trials


def kawasaki_run_flops(n_trials: int) -> int:
    """Total price of a Kawasaki (canonical swap) chain of n_trials trials.

    One trial is a non-local unlike-pair swap proposal (mchammer's
    CanonicalEnsemble move, the swap CTMC's move set): the local field at
    both swapped sites, i.e. two Gibbs site-update budgets of 12, whose exp
    and accept-compare cover the O(1) riders (unlike-pair pick off cached
    up/down index lists, adjacent-pair Delta-E correction, swap bookkeeping).
    Priced in the cached-lists form, not a per-trial lattice rescan. Burn-in
    trials belong in n_trials.
    """
    return 2 * GIBBS_FLOPS_PER_SITE_UPDATE * n_trials


def chain_per_effective_sample(
    total_flops: float, n_records: int, tau_int: float
) -> float:
    """Chain price per independent-equivalent record.

    Burn-in belongs in total_flops (paid before the first usable record);
    tau_int is measured on the slowest tabled observable, so the divisor is
    the effective count N / tau_int.
    """
    return total_flops / (n_records / tau_int)


# --- Training-run accounting --------------------------------------------------
#
# Training cost is the other half of the amortisation argument ("one training
# run serves N targets") and gets two instruments that cover each other's
# assumption:
#
#   derived   `training_run_flops`: measured per-forward count times the
#             forward count the loop structure implies. Assumes backward is a
#             fixed multiple of forward and the recipe names every forward.
#   measured  the real loop at several horizons under FlopCounterMode, fitted
#             with `fit_flop_scaling`. Assumes linearity; cannot reach the
#             full horizon.
#
# They are compared, not reconciled: a gap is a finding about the loop (an
# unaccounted forward, a backward that is not 2x).


def training_forward_counts(
    n_steps: int,
    inner_steps_per_outer: int,
    n_euler_steps: int,
    *,
    c_t_from_rollout: bool = True,
) -> dict:
    """Head-forward counts for one training run, from the loop structure.

    Per outer cycle `train_swap` does one rollout of `n_euler_steps` head
    forwards at `outer_batch`, then `inner_steps_per_outer` loss updates at
    `batch_size`. `n_outer = n_steps / inner_steps_per_outer`.

    With `c_t_from_rollout` (control-variate mode, as the d256 cells run) the
    rollout hands back its own per-slot xi_t and the c_t grid pass is skipped;
    charging it anyway would double the rollout term.

    Whole cycles only, mirroring train_swap's validator: a partial cycle would
    price a rollout that never ran.
    """
    if n_steps % inner_steps_per_outer != 0:
        raise ValueError(
            f"n_steps={n_steps} is not whole outer cycles at "
            f"inner_steps_per_outer={inner_steps_per_outer}"
        )
    n_outer = n_steps // inner_steps_per_outer
    return {
        "n_outer": n_outer,
        "rollout_forwards": n_outer * n_euler_steps,
        "update_forwards": n_steps,
        "c_t_grid_forwards": 0 if c_t_from_rollout else n_outer * n_euler_steps,
    }


def training_run_flops(
    rollout_forward_flops: int,
    update_forward_flops: int,
    *,
    n_steps: int,
    inner_steps_per_outer: int,
    n_euler_steps: int,
    backward_multiplier: float = 2.0,
    c_t_from_rollout: bool = True,
) -> float:
    """Derived training cost: forward counts times measured per-forward FLOPs.

    Two per-forward numbers because the loops run at different batch sizes
    (`outer_batch` for the rollout, `batch_size` for the update) and the
    counter's reading is not linear in batch.

    `backward_multiplier` is the one soft assumption, default 2x; the
    measured leg tests it. Rollouts are under no_grad (the paper's
    stop-gradient R_t^{theta_sg}), so only the update term carries a backward.
    """
    counts = training_forward_counts(
        n_steps,
        inner_steps_per_outer,
        n_euler_steps,
        c_t_from_rollout=c_t_from_rollout,
    )
    rollout = counts["rollout_forwards"] + counts["c_t_grid_forwards"]
    return rollout * rollout_forward_flops + counts[
        "update_forwards"
    ] * update_forward_flops * (1.0 + backward_multiplier)


def diagnostic_eval_flops(
    update_forward_flops: int,
    *,
    update_batch_size: int,
    n_euler_steps: int,
    n_steps: int,
    eval_every: int | None,
    n_eval_draws: int,
) -> float:
    """Cost of the periodic in-training frozen-ESS eval, as its own term.

    Instrumentation, not the algorithm: frozen weights, no gradients,
    severable via eval_every, so training-proper never includes it and the
    appendix prints it beside the algorithmic bill. Calibration: the d64 thp
    measured-vs-derived gap reconciled to this term within 0.8%, which also
    confirmed backward = 2x forward on the training-proper leg.

    No backward is charged; the draw count scales the update-batch forward
    linearly. The count uses n_steps // eval_every: a step-0 firing is a
    fixed-prefix effect the measurement's fit absorbs.
    """
    if not eval_every:
        return 0.0
    n_evals = n_steps // eval_every
    per_eval = n_euler_steps * update_forward_flops * (n_eval_draws / update_batch_size)
    return n_evals * per_eval


def valid_measurement_horizons(
    inner_steps_per_outer: int, eval_every: int | None, n_horizons: int
) -> list[int]:
    """Step counts a multi-horizon FLOP measurement may legitimately use.

    Total FLOPs are a step function of n_steps: they jump once per outer
    cycle (a rollout lands) and whenever the periodic eval fires. A horizon
    that cuts a cycle or straddles an eval sits off the line, so horizons are
    multiples of lcm(inner_steps_per_outer, eval_every) -- the lcm, not the
    larger: a multiple of either alone can still cut the other.
    """
    period = (
        inner_steps_per_outer
        if eval_every is None
        else math.lcm(inner_steps_per_outer, eval_every)
    )
    return [period * (i + 1) for i in range(n_horizons)]


def fit_flop_scaling(outer_cycles, total_flops) -> dict:
    """Least-squares fit of total FLOPs against outer cycles (not steps).

    Returns `fixed_flops` (intercept: startup, the replay buffer's first fill),
    `flops_per_outer_cycle` (slope, the quantity that extrapolates),
    `max_relative_residual` (per point, so one bad horizon cannot hide behind
    a large total) and an `extrapolate` callable.

    A fit rather than measure-once-and-divide because the fixed prefix (the
    first `replay_buffer_cycles` cycles run with a partly-filled buffer) must
    cancel. Three horizons minimum: two points fit any line exactly and
    certify nothing.
    """
    if len(outer_cycles) < 3:
        raise ValueError(
            f"need at least three horizons to test linearity, got {len(outer_cycles)}"
        )
    n = len(outer_cycles)
    mean_x = sum(outer_cycles) / n
    mean_y = sum(total_flops) / n
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(outer_cycles, total_flops)
    )
    variance = sum((x - mean_x) ** 2 for x in outer_cycles)
    if variance == 0:
        raise ValueError("horizons must differ; a single horizon has no slope")
    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    residual = max(
        abs(y - (intercept + slope * x)) / abs(y) if y else 0.0
        for x, y in zip(outer_cycles, total_flops)
    )
    return {
        "fixed_flops": intercept,
        "flops_per_outer_cycle": slope,
        "max_relative_residual": residual,
        "extrapolate": lambda n_outer: intercept + slope * n_outer,
    }
