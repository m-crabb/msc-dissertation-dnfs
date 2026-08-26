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
