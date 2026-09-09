"""Price the "why not just filter?" objection, exactly.

The objection: training a composition-conditioned sampler is elaborate, when
you could draw from the unconstrained Ising sampler and correct afterwards.
Two corrections, and they are not the same thing:

  * Soft (reweighting) — the like-for-like comparator, because it targets the
    same distribution DNFS is trained on,
        p_soft(x) ∝ p_unc(x) · exp(-λ d (c(x) - c_t)²).
    Draw x ~ p_unc, attach w(x) = exp(-λ d (c(x) - c_t)²), and self-normalise.
    The price is the ESS fraction of those weights.

  * Hard (rejection) — rejection sampling against the constraint indicator
    1[c(x) = c_t]. The price is the acceptance probability. This is a stricter
    target (the fixed-composition ensemble), so it prices the hard-constraint
    leg rather than this one, and it is the λ → ∞ limit of the soft row,
    which is what lets one crossover composition be quoted for both.

Reweighting rather than rejection for the soft row: rejecting against p_soft
with envelope constant max_x w(x) = 1 has acceptance rate E_unc[w], strictly
worse than the self-normalised ESS fraction.

Both corrections depend on x only through c(x), so w is constant on a
composition slice and everything collapses onto the unconstrained composition
marginal π(n) = P_unc(N₊ = n), a vector of length d + 1:

    ESS_frac(c_t) = (Σ_n π(n) w_n)² / (Σ_n π(n) w_n²),  w_n = exp(-λ d (n/d - c_t)²)
    accept(c_t)   = π(c_t · d)                          (0 unless c_t·d ∈ ℤ)

At D = 4 the marginal itself is an exact enumeration of all 2^16 states, so
these are exact numbers, not sampled estimates; the reduction is checked
against the brute-force state-level ESS in the tests.

Cost follows by composition: one effective constrained sample needs 1/fraction
effective unconstrained ones, so

    filter s/eff = (unconstrained s/eff) / fraction

with the reference cost taken from the DNFS sampler's own measured draw time.

Example:
    python -m experiments.constrained_soft_02.analysis.filter_row_comparator \\
        --D 4 --lam 50 --reference-s-per-eff 0.00100
"""

import argparse

import pandas as pd
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.targets.ising import IsingTarget

# The compositions the amortised model is swept at, so the filter row lands
# alongside the DNFS row for the same request rather than on its own grid.
SWEPT_COMPOSITIONS = (0.30, 0.35, 0.45, 0.50, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80)


def unconstrained_composition_marginal(*, D: int, sigma: float) -> torch.Tensor:
    """π(n) = P(N₊ = n) under the unconstrained Ising target, exactly.

    No composition penalty: the filter's premise is a sampler for the
    unpenalised model. Built from `IsingTarget` rather than a binomial because
    the coupling makes this marginal strongly non-binomial, and bimodal at σ
    below critical — which is what makes an off-centre request expensive.

    Returns a (d + 1,) tensor indexed by N₊, the number of up-spins.
    """
    d = D * D
    states = enumerate_states(d).float()
    log_p = IsingTarget(D=D, sigma=sigma).log_prob(states)
    probabilities = torch.softmax(log_p, dim=0)
    n_plus = (states > 0).sum(dim=1)
    return torch.zeros(d + 1).index_add_(0, n_plus, probabilities)


def soft_filter_ess_fraction(
    marginal: torch.Tensor, *, c_target: float, lam: float, d: int
) -> float:
    """ESS fraction of reweighting unconstrained draws onto the soft target.

    Self-normalised IS, so the figure of merit is (Σ π w)² / (Σ π w²) — the
    standard ESS, directly comparable to the `ess_fraction` DNFS reports for
    its own draws.
    """
    n_plus = torch.arange(d + 1, dtype=marginal.dtype)
    log_weights = -lam * d * (n_plus / d - c_target) ** 2
    weights = torch.exp(log_weights)
    first = float((marginal * weights).sum())
    second = float((marginal * weights**2).sum())
    return first**2 / second


def hard_filter_acceptance(marginal: torch.Tensor, *, c_target: float, d: int) -> float:
    """Acceptance rate of rejecting every draw off the requested composition.

    Zero unless c_target·d is an integer: at D = 4 the composition quantum is
    1/16, so a request like c = 0.575 is unachievable and rejection cannot
    service it at any cost.
    """
    n_requested = c_target * d
    if abs(n_requested - round(n_requested)) > 1e-9:
        return 0.0
    return float(marginal[round(n_requested)])


def soft_target_slice_acceptance(
    *, D: int, sigma: float, c_target: float, lam: float
) -> float:
    """P(c(x) = c_t) under the soft target — rejection off the trained sampler.

    Keep only those draws from the composition-constrained sampler whose
    composition is exactly the requested one. Exact rather than approximate:
    on the slice the penalty factor exp(-λ d (c(x) - c_t)²) equals 1 and
    cancels from the conditional,

        p_soft(x | c(x) = c_t) = p_unc(x | c(x) = c_t)   for every λ

    so λ buys acceptance rate and costs no bias. (Pinned by
    `test_rejecting_off_the_soft_target_is_exactly_the_constrained_ensemble`.)

    DNFS returns importance weights rather than exact draws, so this composes:
    restrict to the slice and renormalise, and the end-to-end efficiency is
    this acceptance times the sampler's own ESS fraction. Enumerated under the
    penalised target, so it reflects how much λ has already done.
    """
    d = D * D
    states = enumerate_states(d).float()
    target = IsingTarget(
        D=D,
        sigma=sigma,
        target_composition=c_target,
        composition_penalty_strength=lam,
    )
    probabilities = torch.softmax(target.log_prob(states), dim=0)
    n_requested = c_target * d
    if abs(n_requested - round(n_requested)) > 1e-9:
        return 0.0
    on_slice = (states > 0).sum(dim=1) == round(n_requested)
    return float(probabilities[on_slice].sum())


def build_filter_rows(
    *,
    D: int,
    sigma: float,
    lam: float,
    reference_s_per_eff: float,
    compositions=SWEPT_COMPOSITIONS,
) -> pd.DataFrame:
    d = D * D
    marginal = unconstrained_composition_marginal(D=D, sigma=sigma)
    rows = []
    for c_target in compositions:
        soft = soft_filter_ess_fraction(marginal, c_target=c_target, lam=lam, d=d)
        hard = hard_filter_acceptance(marginal, c_target=c_target, d=d)
        from_soft = soft_target_slice_acceptance(
            D=D, sigma=sigma, c_target=c_target, lam=lam
        )
        rows.append(
            {
                "composition": c_target,
                "soft_ess_fraction": soft,
                "soft_cost_multiplier": 1.0 / soft,
                "soft_s_per_eff": reference_s_per_eff / soft,
                "hard_acceptance": hard,
                "hard_cost_multiplier": 1.0 / hard if hard else float("inf"),
                "from_soft_acceptance": from_soft,
                "from_soft_cost_multiplier": (
                    1.0 / from_soft if from_soft else float("inf")
                ),
                "hard_over_from_soft": hard / from_soft if from_soft else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=50.0)
    parser.add_argument(
        "--reference-s-per-eff",
        type=float,
        default=0.00100,
        help="Seconds per effective sample of the sampler being filtered. "
        "Defaults to the measured DNFS specialist cost at D=4, which is "
        "charitable to filtering.",
    )
    parser.add_argument(
        "--achievable",
        action="store_true",
        help="Sweep the compositions the lattice can actually realise (n/d) "
        "instead of the DNFS sweep grid. At D=4 the quantum is 1/16, so "
        "9 of the 10 swept values are unreachable and every rejection "
        "route reports an honest zero there.",
    )
    parser.add_argument(
        "--scaling-sides",
        nargs="+",
        type=int,
        help="Also report how the cost of each route grows with lattice size, "
        "over these lattice sides (enumeration caps d at 20 sites).",
    )
    parser.add_argument("--out", help="Optional CSV path")
    args = parser.parse_args()

    d = args.D * args.D
    marginal = unconstrained_composition_marginal(D=args.D, sigma=args.sigma)
    mean_composition = float(
        (marginal * torch.arange(d + 1, dtype=marginal.dtype) / d).sum()
    )
    print(
        f"Unconstrained composition marginal at D={args.D} (d={d} sites), "
        f"sigma={args.sigma}: mean c = {mean_composition:.4f}, "
        f"P(c=0.5) = {float(marginal[d // 2]):.4g}"
    )
    print(f"lambda = {args.lam}, reference {args.reference_s_per_eff:.5f} s/eff\n")

    compositions = (
        tuple(n / d for n in range(1, d)) if args.achievable else SWEPT_COMPOSITIONS
    )
    table = build_filter_rows(
        D=args.D,
        sigma=args.sigma,
        lam=args.lam,
        reference_s_per_eff=args.reference_s_per_eff,
        compositions=compositions,
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    if args.scaling_sides:
        print(
            "\nScaling with lattice size, at the most-off-centre composition "
            "each lattice can realise within the c=0.30 request:"
        )
        for side in args.scaling_sides:
            sites = side * side
            marginal_at_side = unconstrained_composition_marginal(
                D=side, sigma=args.sigma
            )
            n_requested = round(0.30 * sites)
            c_at_side = n_requested / sites
            soft_filter_ess = soft_filter_ess_fraction(
                marginal_at_side, c_target=c_at_side, lam=args.lam, d=sites
            )
            unconstrained_rejection = hard_filter_acceptance(
                marginal_at_side, c_target=c_at_side, d=sites
            )
            reject_off_soft = soft_target_slice_acceptance(
                D=side, sigma=args.sigma, c_target=c_at_side, lam=args.lam
            )
            print(
                f"  d={sites:3d}  c={c_at_side:.4f}  "
                f"soft-filter ESS {soft_filter_ess:.4g}"
                f"  unconstrained-rejection {unconstrained_rejection:.4g}"
                f"  reject-off-soft {reject_off_soft:.4g}"
            )

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
