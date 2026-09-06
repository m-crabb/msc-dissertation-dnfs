"""Exhaustive verification of the budget-tilted preconditioner derivation
(the constrained analogue of the masked-diffusion sampler's exact-conditional
preconditioning, its App. D.4: logits softmax(network + P(x)) where P is the
target's closed-form single-site conditional with masked neighbours imputed).

Two results are pinned here, both by brute-force enumeration on small rings
where the fixed-composition fibre is enumerable:

1. The constrained Lemma-3 transfer: with the budget-masked reference
   (reveal a uniform masked site, assign species A = spin +1 w.p. b/m), the
   optimally controlled generator's unmask rate, computed as
   reference-rate x exp(V(child) - V(state)) with the value function summed
   over fibre completions, equals EXACTLY the constrained target's masked
   conditional Pr_pi(X^i = s | unmasked pattern). The hinge is the exact
   cancellation (b/m) x C(m,b)/C(m-1,b-1) = 1 -- the urn factor in the
   reference against the completion count in the value ratio. This is what
   licenses the MDNS parameterisation (learn the masked conditional
   directly) on the constraint fibre.

2. The budget-tilted preconditioner itself: for the fibre target
   pi(x) proportional to exp(sigma x^T A x) restricted to sum-constrained
   spins, the analytic conditional approximation
       logit(+1) - logit(-1) = log(b/(m-b)) + 4 sigma f_i,
   with f_i the local field where unmasked neighbours contribute their spin
   and masked neighbours their urn mean mu = (2b - m)/m, is
   (a) EXACT at sigma = 0 (the hypergeometric urn law b/m),
   (b) EXACT when one site remains masked (budget-forced delta),
   (c) a delta whenever b = 0 or b = m at any sigma,
   (d) Z2-covariant on the symmetric fibre, and
   (e) strictly better than the unconstrained zero-imputation
       preconditioner (which cannot see the budget at all and assigns
       positive probability to species the budget forbids).

The failure mode this construction guards: initialising a constrained
masked sampler with the UNCONSTRAINED preconditioner starts it off the
fibre-conditional law precisely at the boundary states (b = 0 or b = m)
where the budget binds hardest -- the states every trajectory must pass
through late in generation."""
from itertools import combinations, product as cartesian_product
from math import comb, exp, isclose, log

RING_SIGMA = 0.4          # strong enough that the energy tilt matters
SMALL_RING = 6            # 3^6 = 729 masked patterns: fully exhaustive
SMALL_RING_PLUSSES = 3
LARGE_RING = 8            # sigma = 0 checks only (no completion sums)
LARGE_RING_PLUSSES = 4


def ring_neighbour_pairs(n_sites):
    """Periodic 1D ring: site i touches i-1 and i+1 (mod n)."""
    return [((i - 1) % n_sites, (i + 1) % n_sites) for i in range(n_sites)]


def masked_and_budget(state, n_plus_target):
    """(m, b): masked-site count and the remaining +1 budget."""
    masked = sum(1 for spin in state if spin is None)
    budget = n_plus_target - sum(1 for spin in state if spin == +1)
    return masked, budget


def feasible_masked_states(n_sites, n_plus_target):
    """Every partially masked state whose unmasked part can still complete
    to the fibre: at least one masked site and 0 <= b <= m."""
    for state in cartesian_product((+1, -1, None), repeat=n_sites):
        masked, budget = masked_and_budget(state, n_plus_target)
        if masked >= 1 and 0 <= budget <= masked:
            yield state


def ring_energy(assignment, neighbours):
    """x^T A x on the ring (ordered pairs, so each edge counts twice)."""
    return sum(
        assignment[site] * (assignment[left] + assignment[right])
        for site, (left, right) in enumerate(neighbours)
    )


def fibre_completion_mass(state, sigma, n_plus_target, neighbours):
    """Sum of exp(sigma E) over all fibre completions of the masked state."""
    masked_sites = [i for i, spin in enumerate(state) if spin is None]
    _, budget = masked_and_budget(state, n_plus_target)
    total = 0.0
    for plus_sites in combinations(masked_sites, budget):
        completion = [
            spin if spin is not None else (+1 if site in plus_sites else -1)
            for site, spin in enumerate(state)
        ]
        total += exp(sigma * ring_energy(completion, neighbours))
    return total


def exact_masked_conditional_plus(state, site, sigma, n_plus_target,
                                  neighbours):
    """Brute-force Pr_pi(X^site = +1 | unmasked pattern) over the fibre."""
    masked_sites = [i for i, spin in enumerate(state) if spin is None]
    _, budget = masked_and_budget(state, n_plus_target)
    total = plus_mass = 0.0
    for plus_sites in combinations(masked_sites, budget):
        completion = [
            spin if spin is not None else (+1 if s in plus_sites else -1)
            for s, spin in enumerate(state)
        ]
        weight = exp(sigma * ring_energy(completion, neighbours))
        total += weight
        if site in plus_sites:
            plus_mass += weight
    return plus_mass / total


def stable_sigmoid(logit_difference):
    if logit_difference >= 0:
        return 1.0 / (1.0 + exp(-logit_difference))
    e = exp(logit_difference)
    return e / (1.0 + e)


def budget_tilted_conditional_plus(state, site, sigma, n_plus_target,
                                   neighbours):
    """The budget-tilted preconditioner's conditional (the derived object).

    logit(+1) - logit(-1) = log(b/(m-b)) + 4 sigma f_i, with the field f_i
    imputing masked neighbours at the urn mean mu = (2b - m)/m. The budget
    term is the sigma = 0 exact conditional in logit form; the energy term
    is the single-site conditional tilt (E(+1) - E(-1) = 4 f_i on the
    ordered-pair energy) with the only unbiased-by-symmetry imputation the
    urn provides. Boundary budgets short-circuit to exact deltas, which is
    where log(b/(m-b)) diverges -- the closed form and the special case
    agree in the limit; the branch exists for float safety, not semantics.
    """
    masked, budget = masked_and_budget(state, n_plus_target)
    if budget == 0:
        return 0.0
    if budget == masked:
        return 1.0
    urn_mean = (2.0 * budget - masked) / masked
    left, right = neighbours[site]
    field = sum(
        state[j] if state[j] is not None else urn_mean
        for j in (left, right)
    )
    return stable_sigmoid(log(budget / (masked - budget)) + 4.0 * sigma * field)


def unconstrained_preconditioner_plus(state, site, sigma, neighbours):
    """The masked-diffusion paper's App. D.4 preconditioner verbatim
    (closed-form full conditional, masked neighbours imputed at zero,
    h = 0): blind to the budget by construction."""
    left, right = neighbours[site]
    field = sum(
        state[j] for j in (left, right) if state[j] is not None
    )
    return stable_sigmoid(4.0 * sigma * field)


def optimal_unmask_probability_via_value_function(state, site, spin, sigma,
                                                  n_plus_target, neighbours):
    """The controlled generator's unmask tilt computed the LONG way:
    reference rate x exp(V(child) - V(state)), with exp(V) equal to the
    completion mass divided by the completion count C(m, b) (the reference's
    conditional terminal law is uniform over the C(m, b) completions). The
    constrained Lemma-3 claim is that this equals the target's masked
    conditional -- the combinatorial factors must cancel exactly."""
    masked, budget = masked_and_budget(state, n_plus_target)
    reference_probability = (
        budget / masked if spin == +1 else (masked - budget) / masked
    )
    if reference_probability == 0.0:
        return 0.0
    child = list(state)
    child[site] = spin
    child_masked, child_budget = masked_and_budget(child, n_plus_target)
    state_value = (
        fibre_completion_mass(state, sigma, n_plus_target, neighbours)
        / comb(masked, budget)
    )
    child_value = (
        fibre_completion_mass(child, sigma, n_plus_target, neighbours)
        / comb(child_masked, child_budget)
    )
    return reference_probability * child_value / state_value


# ---------------------------------------------------------------------------
# 1. The constrained Lemma-3 transfer
# ---------------------------------------------------------------------------


def test_value_function_tilt_equals_exact_masked_conditional():
    """(b/m) C(m,b)/C(m-1,b-1) = 1: the reference's urn factor cancels the
    value ratio's completion counting, so the optimal generator IS the
    constrained target's masked conditional -- the licence for the MDNS
    parameterisation on the fibre."""
    neighbours = ring_neighbour_pairs(SMALL_RING)
    for state in feasible_masked_states(SMALL_RING, SMALL_RING_PLUSSES):
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            via_value = optimal_unmask_probability_via_value_function(
                state, site, +1, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
            )
            exact = exact_masked_conditional_plus(
                state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
            )
            assert isclose(via_value, exact, rel_tol=1e-10, abs_tol=1e-12)


def test_value_function_tilts_sum_to_one_per_site():
    """Per masked site the two species tilts sum to 1, so the controlled
    process keeps the reference's per-site unmask clock: control re-routes
    WHICH species is revealed, never how fast sites reveal."""
    neighbours = ring_neighbour_pairs(SMALL_RING)
    for state in feasible_masked_states(SMALL_RING, SMALL_RING_PLUSSES):
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            total = sum(
                optimal_unmask_probability_via_value_function(
                    state, site, spin, RING_SIGMA, SMALL_RING_PLUSSES,
                    neighbours,
                )
                for spin in (+1, -1)
            )
            assert isclose(total, 1.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 2. The budget-tilted preconditioner
# ---------------------------------------------------------------------------


def test_sigma_zero_is_exactly_the_urn_law():
    """At sigma = 0 the fibre conditional is hypergeometric: b/m exactly.
    The energy term vanishes and the budget logit must reproduce it to
    machine precision on every feasible masked state."""
    neighbours = ring_neighbour_pairs(LARGE_RING)
    for state in feasible_masked_states(LARGE_RING, LARGE_RING_PLUSSES):
        masked, budget = masked_and_budget(state, LARGE_RING_PLUSSES)
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            approx = budget_tilted_conditional_plus(
                state, site, 0.0, LARGE_RING_PLUSSES, neighbours
            )
            assert isclose(approx, budget / masked, rel_tol=1e-12,
                           abs_tol=1e-15)


def test_boundary_budgets_are_deltas_at_any_sigma():
    """b = 0 forbids +1 and b = m forces it, whatever the energy says."""
    neighbours = ring_neighbour_pairs(LARGE_RING)
    for state in feasible_masked_states(LARGE_RING, LARGE_RING_PLUSSES):
        masked, budget = masked_and_budget(state, LARGE_RING_PLUSSES)
        if budget not in (0, masked):
            continue
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            approx = budget_tilted_conditional_plus(
                state, site, RING_SIGMA, LARGE_RING_PLUSSES, neighbours
            )
            assert approx == (0.0 if budget == 0 else 1.0)


def test_last_masked_site_matches_the_exact_delta():
    """m = 1 is the fully-revealed limit where the exact conditional is a
    budget-forced delta; the preconditioner must agree exactly (the
    analogue of the unconstrained preconditioner being exact at full
    context is this single-masked-site exactness)."""
    neighbours = ring_neighbour_pairs(SMALL_RING)
    for state in feasible_masked_states(SMALL_RING, SMALL_RING_PLUSSES):
        masked, _ = masked_and_budget(state, SMALL_RING_PLUSSES)
        if masked != 1:
            continue
        site = next(i for i, s in enumerate(state) if s is None)
        exact = exact_masked_conditional_plus(
            state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
        )
        approx = budget_tilted_conditional_plus(
            state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
        )
        assert approx == exact
        assert exact in (0.0, 1.0)


def test_z2_mirror_covariance_on_the_symmetric_fibre():
    """On the half-filled fibre, negating every unmasked spin maps b to
    m - b and must swap the species probabilities, for the exact
    conditional and the preconditioner alike."""
    neighbours = ring_neighbour_pairs(SMALL_RING)
    for state in feasible_masked_states(SMALL_RING, SMALL_RING_PLUSSES):
        mirrored = tuple(
            None if spin is None else -spin for spin in state
        )
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            for conditional in (
                lambda s, i: exact_masked_conditional_plus(
                    s, i, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
                ),
                lambda s, i: budget_tilted_conditional_plus(
                    s, i, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
                ),
            ):
                assert isclose(
                    conditional(state, site),
                    1.0 - conditional(mirrored, site),
                    rel_tol=1e-9, abs_tol=1e-12,
                )


def test_budget_tilt_strictly_dominates_the_unconstrained_preconditioner():
    """The comparison the derivation exists for: against the brute-forced
    fibre conditional, the budget-tilted preconditioner must beat the
    unconstrained zero-imputation one in BOTH mean and worst-case absolute
    error over every (feasible state, masked site) pair. The unconstrained
    form's worst case is structural: at b = 0 with an aligned unmasked
    field it confidently proposes the species the budget forbids."""
    neighbours = ring_neighbour_pairs(SMALL_RING)
    tilted_errors, untilted_errors = [], []
    for state in feasible_masked_states(SMALL_RING, SMALL_RING_PLUSSES):
        for site, spin_value in enumerate(state):
            if spin_value is not None:
                continue
            exact = exact_masked_conditional_plus(
                state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
            )
            tilted_errors.append(abs(exact - budget_tilted_conditional_plus(
                state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
            )))
            untilted_errors.append(
                abs(exact - unconstrained_preconditioner_plus(
                    state, site, RING_SIGMA, neighbours
                ))
            )
    mean_tilted = sum(tilted_errors) / len(tilted_errors)
    mean_untilted = sum(untilted_errors) / len(untilted_errors)
    assert mean_tilted < mean_untilted
    assert max(tilted_errors) < max(untilted_errors)
    # the structural failure: somewhere the untilted form gives the
    # forbidden species probability > 0.5 while the truth is a hard 0/1
    assert max(untilted_errors) > 0.5


def test_untilted_form_violates_a_boundary_delta():
    """Pin one concrete instance of the failure mode: budget exhausted,
    but both unmasked neighbours are +1, so the unconstrained
    preconditioner leans +1 while the fibre forbids it."""
    state = (+1, +1, None, +1, -1, None)     # b = 0, m = 2 at N_+ = 3
    neighbours = ring_neighbour_pairs(SMALL_RING)
    masked, budget = masked_and_budget(state, SMALL_RING_PLUSSES)
    assert (masked, budget) == (2, 0)
    site = 2                                  # neighbours 1 and 3, both +1
    assert unconstrained_preconditioner_plus(
        state, site, RING_SIGMA, neighbours
    ) > 0.9
    assert budget_tilted_conditional_plus(
        state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
    ) == 0.0
    assert exact_masked_conditional_plus(
        state, site, RING_SIGMA, SMALL_RING_PLUSSES, neighbours
    ) == 0.0
