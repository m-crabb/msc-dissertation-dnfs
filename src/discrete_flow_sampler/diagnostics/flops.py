"""FLOP accounting for the house table's FLOP/es column (agreed s50/s63).

The column prices what a user realistically pays AT SAMPLING TIME to obtain
one independent-equivalent sample; training cost is amortised and lives in
the appendix recipe table. Both sampler families reduce to the same
two-factor shape -- (cost per raw draw) / (independence yield per draw):

  neural:  n_euler forwards + IS-weight target evals, divided by the
           ESS fraction of the frozen evaluation;
  chain:   total sweep/cluster work including burn-in, divided by
           N / tau_int effective records.

Instruments, and why each side gets a different one:
- The NETWORK term is MEASURED, not derived: one real forward under
  ``torch.utils.flop_counter.FlopCounterMode`` at the run's shapes. An
  analytic formula would need re-deriving for every head variant (leT,
  leTF, patch heads, ...) and silently rot; the counter reads the actual
  matmuls executed. Convention: 1 multiply-accumulate = 2 FLOPs (the
  counter's own), so all counts are even.
- The CHAIN terms are ANALYTIC per-work-unit constants (below), because a
  chain's inner loop is a handful of scalar ops that no tensor-level
  counter sees. The constants are documented derivations, not
  measurements; the printed cells carry 2 significant figures and the
  chain cells' uncertainty is dominated by tau_int, not by these
  constants.

The FLOP currency is architecture-blind (a dense-matmul FLOP and a
pointer-chasing chain FLOP buy different wall-clock on real hardware);
the thesis states this once where the column is defined, and the column
supports order-of-magnitude cross-family reads.
"""

import math

from torch.utils.flop_counter import FlopCounterMode

# One Ising target log-prob evaluation, per site: the neighbour sum is two
# torch.roll'd elementwise products (2 mult + 2 add per site under the
# double-counted x^T J x convention), the coupling scale and reduction add
# ~2 more; rounded UP to 10 so the weight-eval term is if anything
# over-charged. It sits four-plus orders below a network forward, so the
# rounding is invisible at the table's 2 significant figures.
ISING_ENERGY_FLOPS_PER_SITE = 10

# One Gibbs heat-bath single-site update: 3 adds for the 4-neighbour local
# field, 1 multiply by the coupling, the conditional-flip probability's exp
# charged at 4, one compare against the uniform draw -- ~9, rounded up to
# 12 to cover the bookkeeping. Per sweep this multiplies by d.
GIBBS_FLOPS_PER_SITE_UPDATE = 12

# One site ADDED to a Wolff cluster: its 4 torus bonds are examined
# (4 lookups + 4 alignment compares + 4 membership tests), aligned
# non-members draw a uniform against p_add (~2 each), plus frontier
# bookkeeping and the flip -- ~20, rounded up to 30. Total cluster work is
# then proportional to the summed cluster sizes, which
# ``wolff_sample(cluster_size_log=...)`` records exactly; with the build
# seeds the certified pools are RECOUNTED bit-identically, never rebuilt.
WOLFF_FLOPS_PER_CLUSTER_SITE = 30


# One VC-SGC trial: a single-site flip proposal. The Ising part is the
# Gibbs local-field work (charged inside the 12 above); the penalty part is
# the difference kappa*d*[(c'-c_t)^2 - (c-c_t)^2] off a CACHED running
# composition -- c' = c +- 1/d is one add, the two squared deviations and
# their difference ~3 more -- charged at 4. An implementation that re-sums
# the composition each trial would pay O(d) instead; the column prices the
# algorithmic (cached) form, consistent with pricing Gibbs off the local
# field rather than a full energy re-evaluation.
VCSGC_FLOPS_PER_TRIAL = GIBBS_FLOPS_PER_SITE_UPDATE + 4


def measured_forward_flops(model, example_inputs: tuple) -> int:
    """FLOPs of one forward at the given shapes, measured by FlopCounterMode.

    Call on the EAGER model (a compiled wrapper can hide ops from the
    dispatch-level counter) -- compilation changes scheduling, not the
    mathematics, so the eager count prices the compiled run too.
    """
    counter = FlopCounterMode(display=False)
    with counter:
        model(*example_inputs)
    return counter.get_total_flops()


def ising_energy_eval_flops(n_sites: int) -> int:
    """One target log-prob evaluation (see ISING_ENERGY_FLOPS_PER_SITE)."""
    return ISING_ENERGY_FLOPS_PER_SITE * n_sites


def neural_sampling_flops_per_sample(per_forward_flops: int,
                                     n_euler_steps: int,
                                     n_sites: int) -> int:
    """Price of ONE raw weighted sample from the trained sampler.

    Each Euler step is one rate-matrix forward (the flip route emits all d
    flip rates per forward) plus one target evaluation for the IS-weight
    integrand. Base draw and final softmax are O(d) and O(1) per sample
    and are not itemised -- both are below the energy term already counted.
    """
    return n_euler_steps * (per_forward_flops + ising_energy_eval_flops(n_sites))


def per_effective_sample(flops_per_raw_sample: float,
                         ess_fraction: float) -> float:
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

    At Delta-mu = 0 an SGC trial IS a Gibbs single-site update: propose a
    species change at one site, evaluate the local field, accept. The
    chemical-potential term contributes mu_Au - mu_Ag = 0 to the acceptance
    exponent, so unlike VC-SGC there is no composition rider to charge --
    this is exactly ``VCSGC_FLOPS_PER_TRIAL`` minus its 4-FLOP cached-
    composition update. Burn-in trials belong in n_trials, mirroring the
    other bills.

    Not expressed through ``gibbs_run_flops``: that one takes (n_sites,
    n_sweeps) because a Gibbs sweep is defined over the lattice, whereas
    mchammer counts individual trial moves. Same constant, different unit.
    """
    return GIBBS_FLOPS_PER_SITE_UPDATE * n_trials


def kawasaki_run_flops(n_trials: int) -> int:
    """Total price of a Kawasaki (canonical swap) chain of n_trials trials.

    One trial is a non-local unlike-pair swap proposal (mchammer's
    CanonicalEnsemble move, the same move set as the swap CTMC): the local
    field at BOTH swapped sites (two Gibbs site-update budgets, whose 12
    each already cover the exp and accept-compare once over), with the
    unlike-pair pick off cached up/down index lists, the adjacent-pair
    Delta-E correction and the swap bookkeeping all O(1) riders inside the
    double charge. As with VC-SGC the column prices the algorithmic
    (cached-lists) form, not an implementation that rescans the lattice per
    trial. Burn-in trials belong in n_trials, mirroring the other bills.
    """
    return 2 * GIBBS_FLOPS_PER_SITE_UPDATE * n_trials


def chain_per_effective_sample(total_flops: float, n_records: int,
                               tau_int: float) -> float:
    """Chain price per independent-equivalent record.

    Burn-in belongs in total_flops (it is paid before the first usable
    record); tau_int is measured on the SLOWEST tabled observable, so the
    divisor is the honest effective count N / tau_int.
    """
    return total_flops / (n_records / tau_int)


# --- Training-run accounting --------------------------------------------------
#
# The column above prices SAMPLING. Training cost is the other half of the
# amortisation argument -- "one training run serves N targets" cannot be stated
# without it -- and gets two independent instruments, because each covers the
# other's assumption:
#
#   derived   `training_run_flops`: a measured per-forward count times the
#             forward count the loop structure implies. Assumes backward is a
#             fixed multiple of forward, and that nothing does forwards the
#             recipe does not mention.
#   measured  run the real loop at several horizons under FlopCounterMode and
#             fit a line with `fit_flop_scaling`. Assumes linearity, and cannot
#             be run to the full horizon.
#
# They are COMPARED, not reconciled: a gap is a finding about the loop (an
# unaccounted forward, a backward that is not 2x) rather than a number to split
# the difference on.


def training_forward_counts(
    n_steps: int,
    inner_steps_per_outer: int,
    n_euler_steps: int,
    *,
    c_t_from_rollout: bool = True,
) -> dict:
    """Head-forward counts for one training run, from the loop structure.

    Per outer cycle `train_swap` does ONE rollout of `n_euler_steps` head
    forwards at `outer_batch`, then `inner_steps_per_outer` loss updates at
    `batch_size`. `n_outer = n_steps / inner_steps_per_outer`.

    `c_t_from_rollout` is the lever worth naming rather than burying: in
    control-variate mode with it on -- the mode the d256 cells run -- the
    rollout hands back its own per-slot xi_t and the c_t grid pass is skipped
    entirely. Charging that pass anyway would add a second full rollout's worth
    of forwards per cycle, i.e. double the rollout term.

    Whole cycles only, mirroring train_swap's own validator: a partial cycle
    would price a rollout that never ran.
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

    Two per-forward numbers because the two loops run at different batch sizes
    (`outer_batch` for the rollout, `batch_size` for the update), and the
    counter's reading is not linear in batch -- fixed per-call work does not
    scale.

    `backward_multiplier` is this figure's one soft assumption and is therefore
    a named parameter, defaulting to the usual 2x. The measured leg exists to
    test it: FlopCounterMode counts the backward's actual matmuls.

    Rollouts are under no_grad (the paper's stop-gradient R_t^{theta_sg}), so
    only the update term carries a backward.
    """
    counts = training_forward_counts(
        n_steps, inner_steps_per_outer, n_euler_steps,
        c_t_from_rollout=c_t_from_rollout,
    )
    rollout = counts["rollout_forwards"] + counts["c_t_grid_forwards"]
    return (
        rollout * rollout_forward_flops
        + counts["update_forwards"] * update_forward_flops
        * (1.0 + backward_multiplier)
    )


def valid_measurement_horizons(
    inner_steps_per_outer: int, eval_every: int | None, n_horizons: int
) -> list[int]:
    """Step counts a multi-horizon FLOP measurement may legitimately use.

    Total FLOPs are a STEP function of n_steps, not a smooth one: they jump once
    per outer cycle (a rollout lands) and again whenever the periodic
    in-training eval fires. A horizon that cuts a cycle in half, or that
    straddles an eval, sits off the line for reasons that have nothing to do
    with the per-cycle cost -- and a line fitted through such points is a
    plausible-looking wrong answer.

    So horizons are multiples of lcm(inner_steps_per_outer, eval_every). The lcm
    and not the larger of the two: a multiple of eval_every alone can still cut
    a cycle, and a multiple of inner_steps_per_outer alone can still straddle an
    eval.
    """
    period = (
        inner_steps_per_outer
        if eval_every is None
        else math.lcm(inner_steps_per_outer, eval_every)
    )
    return [period * (i + 1) for i in range(n_horizons)]


def fit_flop_scaling(outer_cycles, total_flops) -> dict:
    """Least-squares fit of total FLOPs against OUTER CYCLES (not steps).

    Returns `fixed_flops` (the intercept: process startup, the replay buffer's
    first fill, any one-off allocation), `flops_per_outer_cycle` (the slope, the
    quantity that extrapolates), `max_relative_residual`, and an `extrapolate`
    callable.

    Fitting rather than measuring-once-and-dividing is the point: differencing
    across horizons cancels the fixed prefix exactly, which matters because the
    first `replay_buffer_cycles` cycles run with a partly-filled buffer and are
    not representative of the steady state being extrapolated.

    Three horizons minimum. Two points fit any line exactly, so a residual from
    two points is identically zero and certifies nothing; the third is what
    makes the linearity claim falsifiable. The residual is per-point relative,
    so one badly-placed horizon cannot hide behind a large total.
    """
    if len(outer_cycles) < 3:
        raise ValueError(
            f"need at least three horizons to test linearity, got "
            f"{len(outer_cycles)}"
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
