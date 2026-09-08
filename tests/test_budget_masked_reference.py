"""Exhaustive verification of the budget-masked reference-process derivation
(the thesis's budget-masked-MDNS note): revealing a uniformly chosen masked
site and assigning species A with probability b/m (b = remaining A-budget,
m = masked sites) gives an assignment-conditional product that is the same
constant N_A! N_B! / d! on every feasible trajectory, hence a terminal law
uniform on the fixed-composition fibre and importance weights whose reference
contribution cancels under self-normalisation. Also pins the failure mode the
derivation exists to prevent: masking the *unconstrained* conditionals and
renormalising at sampling time leaves a trajectory-dependent correction, so
keeping the unconstrained conditionals in the weights silently biases every
estimate."""

from itertools import combinations, permutations
from math import factorial, isclose

import pytest


def assignment_conditional_product(reveal_order, a_sites, n_sites):
    """Product of the hypergeometric assignment conditionals along one
    complete revelation trajectory: b/m when the revealed site is species A,
    (m - b)/m when species B."""
    remaining_budget = len(a_sites)
    product = 1.0
    for step, site in enumerate(reveal_order):
        masked_count = n_sites - step
        if site in a_sites:
            product *= remaining_budget / masked_count
            remaining_budget -= 1
        else:
            product *= (masked_count - remaining_budget) / masked_count
    return product


@pytest.mark.parametrize("n_sites,n_a", [(4, 2), (5, 2), (6, 3)])
def test_assignment_product_is_the_same_constant_on_every_trajectory(n_sites, n_a):
    expected = factorial(n_a) * factorial(n_sites - n_a) / factorial(n_sites)
    for a_sites in combinations(range(n_sites), n_a):
        for reveal_order in permutations(range(n_sites)):
            product = assignment_conditional_product(
                reveal_order, set(a_sites), n_sites
            )
            assert isclose(product, expected, rel_tol=1e-12)


@pytest.mark.parametrize("n_sites,n_a", [(4, 2), (5, 2)])
def test_terminal_law_is_uniform_on_the_composition_fibre(n_sites, n_a):
    """Site choice contributes 1/m per step (1/d! per trajectory); with the
    constant assignment product, summing the d! orders per terminal state
    gives exactly 1/C(d, N_A) per fibre state, and the fibre sums to one."""
    fibre = list(combinations(range(n_sites), n_a))
    order_probability = 1.0 / factorial(n_sites)
    terminal_probabilities = []
    for a_sites in fibre:
        total = sum(
            order_probability
            * assignment_conditional_product(order, set(a_sites), n_sites)
            for order in permutations(range(n_sites))
        )
        terminal_probabilities.append(total)
    assert all(
        isclose(p, 1.0 / len(fibre), rel_tol=1e-9) for p in terminal_probabilities
    )
    assert isclose(sum(terminal_probabilities), 1.0, rel_tol=1e-9)


def test_mask_and_renormalise_correction_is_trajectory_dependent():
    """The broken alternative: keep a state-dependent unconstrained
    conditional q(A) and impose the budget only by refusing infeasible
    assignments (renormalising at the boundary states b=0 / b=m). The
    per-step feasible mass Z multiplies into a per-trajectory correction
    that differs across trajectories, so weights computed with the
    unconstrained conditionals are trajectory-wise wrong by a non-constant
    factor -- feasibility without the conditional measure."""
    n_sites, n_a = 4, 2

    def unconstrained_q_a(revealed_a_count):
        # Any state-dependent conditional exhibits the effect; this one
        # leans towards balance, as a learned conditional would.
        return 0.5 - 0.2 * (revealed_a_count - 1)

    def feasible_mass_product(reveal_order, a_sites):
        remaining_budget = n_a
        revealed_a = 0
        product = 1.0
        for step, site in enumerate(reveal_order):
            masked_count = n_sites - step
            q_a = unconstrained_q_a(revealed_a)
            if remaining_budget == 0:
                feasible_mass = 1.0 - q_a  # A refused, B forced
            elif remaining_budget == masked_count:
                feasible_mass = q_a  # B refused, A forced
            else:
                feasible_mass = 1.0
            product *= feasible_mass
            if site in a_sites:
                remaining_budget -= 1
                revealed_a += 1
        return product

    corrections = {
        round(feasible_mass_product(order, set(a_sites)), 12)
        for a_sites in combinations(range(n_sites), n_a)
        for order in permutations(range(n_sites))
    }
    assert len(corrections) > 1
