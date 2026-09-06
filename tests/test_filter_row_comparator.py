"""The "why not just filter?" comparator, computed exactly.

The objection this answers is the obvious one: rather than training a sampler
that respects a composition constraint, draw from the *unconstrained* Ising
sampler and correct afterwards — reweight to the soft target, or reject
everything off the requested composition. The comparator prices that.

Both corrections depend on x only through c(x), so the whole calculation
collapses onto the unconstrained composition marginal π(n) = P_unc(N₊ = n).
That reduction is what makes the row exact at D = 4 rather than another
sampled estimate, and it is the first thing these tests pin — if it were
wrong, the row would be cheap and meaningless.

The other three pin the boundary behaviours that make the numbers readable:
no reweighting when λ = 0, a marginal that is a probability distribution, and
the soft row collapsing onto the hard row as λ → ∞.
"""

import importlib
import math

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.targets.ising import IsingTarget

comparator = importlib.import_module(
    "experiments.constrained_soft_02.analysis.filter_row_comparator"
)

D = 3  # 3x3 = 9 sites; enumerable in milliseconds
SIGMA = 0.1


@pytest.fixture(scope="module")
def marginal():
    return comparator.unconstrained_composition_marginal(D=D, sigma=SIGMA)


def test_marginal_is_a_distribution_over_achievable_compositions(marginal):
    """One entry per achievable N₊, summing to one.

    A composition on a d-site lattice can only be a multiple of 1/d, and the
    filter rows divide by entries of this vector — a marginal that silently
    dropped or double-counted a slice would rescale every cost in the table.
    """
    assert len(marginal) == D * D + 1
    assert marginal.sum() == pytest.approx(1.0)
    assert (marginal > 0).all()


def test_soft_filter_is_free_when_there_is_no_penalty(marginal):
    """λ = 0 means every weight is 1, so the ESS fraction is exactly 1.

    This is the sanity anchor for the whole row: whatever the marginal looks
    like, an absent constraint must cost nothing. A formula that returned
    anything else here would be measuring the marginal's shape rather than the
    price of the constraint.
    """
    for c_target in (0.30, 0.50, 0.80):
        assert comparator.soft_filter_ess_fraction(
            marginal, c_target=c_target, lam=0.0, d=D * D
        ) == pytest.approx(1.0)


def test_soft_filter_matches_brute_force_over_all_states(marginal):
    """The c(x)-only reduction reproduces the state-level ESS exactly.

    This is the load-bearing claim. The row is computed from a length-(d+1)
    marginal instead of 2^d states because the reweighting factor
    exp(-λ d (c(x) - c_t)²) is constant on a composition slice. If that were
    not exactly true the cheap calculation would drift from the real one, so
    it is checked against the honest enumeration rather than assumed.
    """
    d = D * D
    lam, c_target = 50.0, 0.55
    states = enumerate_states(d).float()
    unconstrained = IsingTarget(D=D, sigma=SIGMA)
    log_p = unconstrained.log_prob(states)
    log_p = log_p - torch.logsumexp(log_p, dim=0)

    composition = (states > 0).float().mean(dim=1)
    weights = torch.exp(-lam * d * (composition - c_target) ** 2)
    # The IS ESS fraction by definition, (E_p[w])^2 / E_p[w^2], summed over
    # all 2^d states with each state carrying its own probability. The module
    # sums the same thing over d+1 slices; agreeing is the reduction.
    probabilities = log_p.exp()
    brute_force = float((probabilities * weights).sum()) ** 2 / float(
        (probabilities * weights**2).sum()
    )

    reduced = comparator.soft_filter_ess_fraction(
        marginal, c_target=c_target, lam=lam, d=d
    )
    assert reduced == pytest.approx(brute_force, rel=1e-5)


def test_hard_filter_is_the_limit_the_soft_filter_approaches(marginal):
    """As λ → ∞ the reweighting becomes rejection onto one slice.

    The two rows are the same calculation at two penalty strengths, and saying
    so is what lets the writeup quote one crossover composition rather than
    two unrelated ones. At finite λ the soft row must be the *cheaper* of the
    two, because it keeps neighbouring slices at reduced weight instead of
    discarding them.
    """
    d, c_target = D * D, 5 / (D * D)

    hard = comparator.hard_filter_acceptance(marginal, c_target=c_target, d=d)
    assert hard == pytest.approx(float(marginal[5]))

    huge_lambda = comparator.soft_filter_ess_fraction(
        marginal, c_target=c_target, lam=1e6, d=d
    )
    assert huge_lambda == pytest.approx(hard, rel=1e-6)

    operating_point = comparator.soft_filter_ess_fraction(
        marginal, c_target=c_target, lam=50.0, d=d
    )
    assert operating_point > hard


def test_rejecting_off_the_soft_target_is_exactly_the_constrained_ensemble():
    """Conditioning the SOFT target on a slice gives the unconstrained one.

    This is why rejecting off the trained soft sampler is a legitimate route
    to a hard constraint rather than an approximation: on the slice
    c(x) = c_t the penalty factor exp(-λ d (c(x) - c_t)²) is identically 1, so
    it cancels out of the conditional and

        p_soft(x | c(x) = c_t) = p_unc(x | c(x) = c_t)

    exactly, for every λ. λ therefore buys efficiency and costs no bias — the
    accepted samples are the fixed-composition ensemble however the sampler
    was tuned. If this failed, every hard-constraint number obtained this way
    would be λ-dependent and none of them would be the target.
    """
    d, n_requested = D * D, 5
    c_target = n_requested / d
    states = enumerate_states(d).float()
    on_slice = (states > 0).sum(dim=1) == n_requested

    def conditional(lam):
        target = IsingTarget(
            D=D,
            sigma=SIGMA,
            target_composition=c_target,
            composition_penalty_strength=lam,
        )
        log_p = target.log_prob(states)[on_slice]
        return torch.softmax(log_p, dim=0)

    unconstrained = IsingTarget(D=D, sigma=SIGMA)
    reference = torch.softmax(unconstrained.log_prob(states)[on_slice], dim=0)

    for lam in (10.0, 50.0, 500.0):
        assert torch.allclose(conditional(lam), reference, atol=1e-6)


def test_soft_sampler_acceptance_beats_unconstrained_rejection(marginal):
    """The premise of the third row: λ concentrates mass onto the slice.

    Both routes reject onto the same slice and return the same distribution
    (above), so the only difference is how often they accept. Rejecting off
    the soft target must accept more often than rejecting off the
    unconstrained one, or there would be no reason to train the sampler at
    all — and the gap is the quantitative answer to "why not just filter".
    """
    d, c_target = D * D, 5 / (D * D)

    soft = comparator.soft_target_slice_acceptance(
        D=D, sigma=SIGMA, c_target=c_target, lam=50.0
    )
    unconstrained = comparator.hard_filter_acceptance(marginal, c_target=c_target, d=d)

    assert soft > unconstrained
    assert 0.0 < soft <= 1.0


def test_off_lattice_composition_cannot_be_hard_filtered(marginal):
    """c·d must be an integer or the acceptance is exactly zero.

    At D = 4 the quantum is 1/16 = 0.0625, so a swept composition like 0.575
    is not achievable at all. Returning a small-but-positive number there
    would understate rejection sampling's real failure mode, which is that it
    cannot service the request at any cost.
    """
    assert comparator.hard_filter_acceptance(marginal, c_target=0.575, d=D * D) == 0.0
    assert not math.isnan(
        comparator.soft_filter_ess_fraction(marginal, c_target=0.575, lam=50.0, d=D * D)
    )
