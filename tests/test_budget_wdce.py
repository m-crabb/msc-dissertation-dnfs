"""Constrained WDCE on the fixed-composition fibre: exhaustive verification
of the three claims that carry the masked-diffusion sampler's WDCE loss
(their Eq. (16), built on the DCE corruption kernel of their Eq. (4)) onto
the budget-masked reference.

Claim 1 (corruption law): under the budget-masked reference, the bridge
conditional given the terminal fibre state is EXACTLY the unconstrained
corruption kernel mu_lambda -- mask each site independently, no budget-aware
correction. Why: a reference trajectory factorises as P(order) times the
assignment-conditional product, and the product is the same constant on
every feasible trajectory (the trajectory-constant lemma), so the revelation
order is INDEPENDENT of the terminal state; the masked set is a function of
the order and the exchangeable clocks alone. The correction WOULD appear if
the site-selection clock saw the budget (only the species draw does) -- that
is the same boundary-feasible-mass pathology the mask-and-renormalise
counterexample pins.

Claim 2 (minimiser): with exact importance weights the WDCE population
minimiser at every masked context is the CONSTRAINED target's masked
conditional -- the same object the tilted-generator lemma names as the
optimal control, and the budget-tilted preconditioner approximates. So the
loss trains the network towards exactly the conditional the theory wants,
with any positive corruption-size weighting w(lambda): the posterior over
completions factorises out of the lambda mixture.

Claim 3 (weights): the reference's contribution to the trajectory log-weight
(site-choice product times assignment product, plus the uniform-on-fibre
terminal base) is the same constant for every trajectory and terminal, so it
cancels under the batch-softmax self-normalisation and the implementable
weight needs only the target energy and the model's own rollout terms --
identical in shape to the unconstrained loss.
"""
from itertools import combinations, permutations
from math import comb, exp, factorial, isclose, log

import pytest

from tests.test_budget_masked_reference import assignment_conditional_product
from tests.test_budget_preconditioner import (
    exact_masked_conditional_plus,
    feasible_masked_states,
    masked_and_budget,
    ring_energy,
    ring_neighbour_pairs,
)


# ---------------------------------------------------------------------------
# Derivation artefacts (the implementation under test)
# ---------------------------------------------------------------------------


def fibre_states(n_sites, n_plus_target):
    """All fully revealed states on the composition fibre, as +/-1 tuples."""
    states = []
    for plus_sites in combinations(range(n_sites), n_plus_target):
        states.append(tuple(
            +1 if site in plus_sites else -1 for site in range(n_sites)
        ))
    return states


def revealed_set_law_given_terminal(n_sites, terminal, n_revealed):
    """Brute-force P(revealed set after k reveals | X_1 = terminal) under the
    budget-masked reference: sum honest trajectory probabilities (uniform
    site choice times the per-step urn conditionals, NOT assuming the
    trajectory-constant lemma) over all d! revelation orders, then condition
    on the terminal and aggregate by the first-k set."""
    a_sites = {site for site, spin in enumerate(terminal) if spin == +1}
    law = {}
    total = 0.0
    for order in permutations(range(n_sites)):
        probability = (
            assignment_conditional_product(order, a_sites, n_sites)
            / factorial(n_sites)
        )
        total += probability
        revealed = frozenset(order[:n_revealed])
        law[revealed] = law.get(revealed, 0.0) + probability
    return {revealed: p / total for revealed, p in law.items()}


def corrupt_by_masking(terminal, mask_set):
    """mu_lambda's action on a terminal state: None at the masked sites."""
    return tuple(
        None if site in mask_set else spin
        for site, spin in enumerate(terminal)
    )


def wdce_population_minimiser(sigma, n_sites, n_plus_target, neighbours,
                              size_weight):
    """Per-(context, site) minimiser of the population constrained WDCE.

    Assembles the loss's cross-entropy coefficients explicitly: for each
    fibre terminal x_1 (weighted by the exact target mass, standing in for
    converged self-normalised importance weights) and each nonempty mask set
    S (weighted by any positive size_weight(|S|)), the corrupted context
    x_tilde accumulates weight towards species x_1^d at each masked site d.
    The cross-entropy minimiser at fixed context is the normalised
    coefficient vector (the standard fact sum_i c_i (-log s_i) is minimised
    over the simplex at s = c / sum c), so no numerical optimisation is
    needed and the test is exact. Returns {(context, site): p_plus}.
    """
    coefficients = {}
    for terminal in fibre_states(n_sites, n_plus_target):
        target_mass = exp(sigma * ring_energy(terminal, neighbours))
        for set_size in range(1, n_sites + 1):
            for mask_set in combinations(range(n_sites), set_size):
                weight = target_mass * size_weight(set_size)
                context = corrupt_by_masking(terminal, set(mask_set))
                for site in mask_set:
                    key = (context, site)
                    plus_mass, minus_mass = coefficients.get(key, (0.0, 0.0))
                    if terminal[site] == +1:
                        plus_mass += weight
                    else:
                        minus_mass += weight
                    coefficients[key] = (plus_mass, minus_mass)
    return {
        key: plus_mass / (plus_mass + minus_mass)
        for key, (plus_mass, minus_mass) in coefficients.items()
    }


# ---------------------------------------------------------------------------
# Claim 1: the bridge is the unconstrained corruption kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites,n_plus", [(4, 2), (5, 2), (6, 3)])
def test_revealed_set_is_uniform_given_any_terminal(n_sites, n_plus):
    """Given X_1, every size-k revealed set is equally likely (1/C(d,k)) --
    the masked set carries no information about the terminal, which is the
    discrete-order statement of 'corruption = unconstrained mu_lambda'."""
    for terminal in fibre_states(n_sites, n_plus):
        for n_revealed in range(n_sites + 1):
            law = revealed_set_law_given_terminal(
                n_sites, terminal, n_revealed
            )
            expected = 1.0 / comb(n_sites, n_revealed)
            assert all(
                isclose(p, expected, rel_tol=1e-9) for p in law.values()
            )
            assert len(law) == comb(n_sites, n_revealed)


def test_every_masking_of_a_fibre_state_is_budget_feasible():
    """The budget bookkeeping is implied by the unmasked remainder: masking
    any subset of any fibre state lands at 0 <= b <= m, so the corruption
    step never needs (and never has) a feasibility correction."""
    n_sites, n_plus = 6, 3
    for terminal in fibre_states(n_sites, n_plus):
        for set_size in range(1, n_sites + 1):
            for mask_set in combinations(range(n_sites), set_size):
                context = corrupt_by_masking(terminal, set(mask_set))
                masked, budget = masked_and_budget(context, n_plus)
                assert masked == set_size
                assert 0 <= budget <= masked


# ---------------------------------------------------------------------------
# Claim 2: the population minimiser is the constrained masked conditional
# ---------------------------------------------------------------------------


def test_wdce_minimiser_is_the_exact_constrained_conditional():
    n_sites, n_plus, sigma = 6, 3, 0.4
    neighbours = ring_neighbour_pairs(n_sites)
    minimiser = wdce_population_minimiser(
        sigma, n_sites, n_plus, neighbours, size_weight=lambda size: 1.0
    )
    contexts_seen = {context for context, _ in minimiser}
    assert contexts_seen == set(feasible_masked_states(n_sites, n_plus))
    for (context, site), p_plus in minimiser.items():
        exact = exact_masked_conditional_plus(
            context, site, sigma, n_plus, neighbours
        )
        assert isclose(p_plus, exact, rel_tol=1e-9, abs_tol=1e-12)


def test_minimiser_is_invariant_to_the_corruption_size_weight():
    """w(lambda) reweights WHICH contexts are visited, never the posterior
    over completions at a fixed context, so any positive size weighting
    (uniform, or the any-order-autoregressive 1/k) trains towards the same
    conditional -- the freedom the paper's w(lambda) grants survives the
    constraint untouched."""
    n_sites, n_plus, sigma = 5, 2, 0.4
    neighbours = ring_neighbour_pairs(n_sites)
    uniform = wdce_population_minimiser(
        sigma, n_sites, n_plus, neighbours, size_weight=lambda size: 1.0
    )
    any_order = wdce_population_minimiser(
        sigma, n_sites, n_plus, neighbours, size_weight=lambda size: 1.0 / size
    )
    assert uniform.keys() == any_order.keys()
    assert all(
        isclose(uniform[key], any_order[key], rel_tol=1e-12)
        for key in uniform
    )


def test_boundary_contexts_minimise_to_deltas():
    """Exhausted budgets give deterministic targets (the states where the
    unconstrained preconditioner fails structurally); the loss's own
    minimiser respects them with no special handling."""
    n_sites, n_plus = 6, 3
    neighbours = ring_neighbour_pairs(n_sites)
    minimiser = wdce_population_minimiser(
        0.4, n_sites, n_plus, neighbours, size_weight=lambda size: 1.0
    )
    boundary_seen = 0
    for (context, site), p_plus in minimiser.items():
        masked, budget = masked_and_budget(context, n_plus)
        if budget == 0:
            assert p_plus == 0.0
            boundary_seen += 1
        elif budget == masked:
            assert p_plus == 1.0
            boundary_seen += 1
    assert boundary_seen > 0


# ---------------------------------------------------------------------------
# Claim 3: the reference's weight contribution cancels under softmax
# ---------------------------------------------------------------------------


def test_reference_log_weight_terms_shift_all_trajectories_equally():
    """The reference trajectory log-probability (uniform site choice plus urn
    assignments) and the uniform-on-fibre terminal base are the same for
    every (order, terminal) pair, so batch-softmax weights computed with and
    without them agree exactly: the implementable constrained weight keeps
    only the target energy and the model's rollout terms, as in the
    unconstrained loss."""
    n_sites, n_plus = 5, 2
    neighbours = ring_neighbour_pairs(n_sites)
    reference_log_probs = {
        round(log(assignment_conditional_product(order, set(a_sites),
                                                 n_sites))
              - log(factorial(n_sites)), 12)
        for a_sites in combinations(range(n_sites), n_plus)
        for order in permutations(range(n_sites))
    }
    assert len(reference_log_probs) == 1

    terminals = fibre_states(n_sites, n_plus)
    sigma = 0.4
    reference_constant = reference_log_probs.pop()
    fibre_base = -log(comb(n_sites, n_plus))
    energies = [sigma * ring_energy(t, neighbours) for t in terminals]

    def softmax(values):
        peak = max(values)
        masses = [exp(v - peak) for v in values]
        total = sum(masses)
        return [mass / total for mass in masses]

    with_constants = softmax([
        energy - fibre_base - reference_constant for energy in energies
    ])
    without = softmax(energies)
    assert all(
        isclose(a, b, rel_tol=1e-12) for a, b in zip(with_constants, without)
    )


def test_minimiser_is_invariant_to_context_dependent_loss_weights():
    """Lead-1 licence (near-boundary exposure boost): a per-context loss
    weight eta(m, b) — any positive function of the CONTEXT, here the
    budget-class boost 1 + kappa*1[b in {1, m-1}] — scales every
    completion coefficient at a fixed context equally, so the per-context
    minimiser (the exact fibre conditional) is untouched. Same argument as
    the size-weight invariance: eta is context-measurable, and b, m are
    functions of the context alone. A weight depending on the TERMINAL
    (not just the context) would NOT enjoy this — it would re-tilt the
    posterior over completions."""
    n_sites, n_plus, sigma = 5, 2, 0.4
    neighbours = ring_neighbour_pairs(n_sites)

    def near_boundary_boosted(sigma_, n_sites_, n_plus_, neighbours_, boost):
        coefficients = {}
        for terminal in fibre_states(n_sites_, n_plus_):
            target_mass = exp(sigma_ * ring_energy(terminal, neighbours_))
            for set_size in range(1, n_sites_ + 1):
                for mask_set in combinations(range(n_sites_), set_size):
                    context = corrupt_by_masking(terminal, set(mask_set))
                    masked, budget = masked_and_budget(context, n_plus_)
                    eta = 1.0 + boost * (budget in (1, masked - 1))
                    weight = target_mass * eta
                    for site in mask_set:
                        key = (context, site)
                        plus_mass, minus_mass = coefficients.get(
                            key, (0.0, 0.0))
                        if terminal[site] == +1:
                            plus_mass += weight
                        else:
                            minus_mass += weight
                        coefficients[key] = (plus_mass, minus_mass)
        return {key: p / (p + q) for key, (p, q) in coefficients.items()}

    unboosted = near_boundary_boosted(sigma, n_sites, n_plus, neighbours, 0.0)
    boosted = near_boundary_boosted(sigma, n_sites, n_plus, neighbours, 4.0)
    assert unboosted.keys() == boosted.keys()
    assert all(
        isclose(unboosted[key], boosted[key], rel_tol=1e-12)
        for key in unboosted
    )
