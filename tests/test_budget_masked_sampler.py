"""Tests for the vectorised budget-masked masked-diffusion sampler
(`samplers/budget_masked.py`), written BEFORE the implementation.

The pure-python derivation artefacts are already exhaustively verified
(`test_budget_preconditioner.py`, `test_budget_wdce.py`); what needs testing
here is the TORCH implementation that the 4x4 gate actually runs: that the
vectorised preconditioners reproduce the verified reference functions
exactly, that generation cannot leave the composition fibre (the gate's G0
is structural, so a violation here is a bug in the revelation loop, never a
training result), that the rollout law matches an independent
dynamic-programming enumeration of the same conditionals, that the
importance weights make the weighted estimator agree with the exactly
enumerated conditional, and that the WDCE cross-entropy is oriented (the
exact conditional scores better than a perturbation of it).

Conventions under test: masked states are float tensors with +1/-1 spins
and 0.0 at masked sites; adjacency is the symmetric 0/1 matrix of
`IsingTarget` (each undirected edge stored in both directions), so the
single-site energy tilt is 4*sigma*(A x)_i, matching the ring reference
where the field is the plain sum of the two neighbour spins.
"""

from math import comb, isclose

import pytest
import torch

from discrete_flow_sampler.samplers.budget_masked import (
    MaskedConditionalNet,
    feasibility_clamped_p_plus,
    preconditioner_logit_diff,
    rollout_budget_masked,
    wdce_cross_entropy,
)
from tests.test_budget_preconditioner import (
    budget_tilted_conditional_plus,
    exact_masked_conditional_plus,
    feasible_masked_states,
    masked_and_budget,
    ring_neighbour_pairs,
    unconstrained_preconditioner_plus,
)

RING_SIGMA = 0.4


def ring_adjacency(n_sites: int) -> torch.Tensor:
    """Symmetric 0/1 adjacency of the periodic ring (matches the pure-python
    reference's neighbour convention and IsingTarget's A convention)."""
    adjacency = torch.zeros(n_sites, n_sites)
    for site in range(n_sites):
        adjacency[site, (site + 1) % n_sites] = 1.0
        adjacency[site, (site - 1) % n_sites] = 1.0
    return adjacency


def as_masked_tensor(state) -> torch.Tensor:
    """Pure-python masked state (None = masked) -> the module's convention
    (0.0 = masked)."""
    return torch.tensor([[0.0 if spin is None else float(spin) for spin in state]])


def ring_energy_torch(states: torch.Tensor, adjacency: torch.Tensor):
    return torch.einsum("bi,ij,bj->b", states, adjacency, states)


# ---------------------------------------------------------------------------
# Preconditioners: vectorised == verified pure-python reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,reference",
    [
        (
            "budget_tilted",
            lambda state, site, neighbours: budget_tilted_conditional_plus(
                state, site, RING_SIGMA, 3, neighbours
            ),
        ),
        (
            "unconstrained",
            lambda state, site, neighbours: unconstrained_preconditioner_plus(
                state, site, RING_SIGMA, neighbours
            ),
        ),
    ],
)
def test_preconditioner_matches_reference_on_every_masked_state(mode, reference):
    """The torch preconditioner, pushed through sigmoid, must equal the
    exhaustively verified reference at every (feasible state, masked site)
    of the d=6 ring — same numbers, vectorised."""
    n_sites, n_plus = 6, 3
    adjacency = ring_adjacency(n_sites)
    neighbours = ring_neighbour_pairs(n_sites)
    for state in feasible_masked_states(n_sites, n_plus):
        logit = preconditioner_logit_diff(
            as_masked_tensor(state), adjacency, RING_SIGMA, n_plus, mode
        )
        p_plus = torch.sigmoid(logit)[0]
        for site, spin in enumerate(state):
            if spin is not None:
                continue
            assert isclose(
                p_plus[site].item(),
                reference(state, site, neighbours),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )


def test_none_preconditioner_is_zero():
    state = as_masked_tensor((+1, None, -1, None, None, +1))
    logit = preconditioner_logit_diff(state, ring_adjacency(6), RING_SIGMA, 3, "none")
    assert torch.all(logit == 0.0)


def test_budget_tilted_boundary_is_a_saturating_delta():
    """b = 0 must force p(+1) ~ 0 and b = m must force ~ 1 even after a
    finite trunk perturbation is added (the pseudo-infinite branch has to
    dominate any realistic network logit)."""
    exhausted = as_masked_tensor((+1, +1, +1, None, None, -1))  # b=0, m=2
    forced = as_masked_tensor((-1, -1, -1, None, None, +1))  # b=m=2
    adjacency = ring_adjacency(6)
    for state, expected in ((exhausted, 0.0), (forced, 1.0)):
        logit = preconditioner_logit_diff(
            state, adjacency, RING_SIGMA, 3, "budget_tilted"
        )
        perturbed = (
            torch.sigmoid(logit + 10.0)
            if expected == 0.0
            else (torch.sigmoid(logit - 10.0))
        )
        masked_sites = state[0] == 0.0
        assert torch.all(torch.abs(perturbed[0][masked_sites] - expected) < 1e-6)


# ---------------------------------------------------------------------------
# Generation: G0 is structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["budget_tilted", "none", "unconstrained"])
def test_rollout_never_leaves_the_fibre(mode):
    """The gate's G0: with an ADVERSARIAL (randomly initialised, not
    zero-init) trunk and any preconditioner — including the unconstrained
    one that leans towards forbidden species — every terminal has exactly
    N_+ up spins, because the species draw is feasibility-clamped. This is
    the design decision that makes G0 structural rather than learned."""
    n_sites, n_plus, sigma = 16, 8, 0.223
    adjacency = ring_adjacency(n_sites)
    torch.manual_seed(0)
    net = MaskedConditionalNet(n_sites, hidden_width=32)
    for parameter in net.parameters():  # adversarial: break zero-init
        parameter.data.normal_(0.0, 0.5)

    def logit_fn(x):
        return net(x) + preconditioner_logit_diff(x, adjacency, sigma, n_plus, mode)

    generator = torch.Generator().manual_seed(7)
    terminals, _ = rollout_budget_masked(logit_fn, 4096, n_sites, n_plus, generator)
    assert torch.all(terminals.abs() == 1.0)  # fully revealed
    assert torch.all(((terminals + 1) / 2).sum(dim=1) == n_plus)


def test_feasibility_clamp_forces_boundary_draws():
    p_plus = torch.tensor([0.9, 0.4, 0.5])
    budget = torch.tensor([0, 3, 1])
    masked_count = torch.tensor([2, 3, 4])
    clamped = feasibility_clamped_p_plus(p_plus, budget, masked_count)
    assert clamped[0] == 0.0  # exhausted budget: +1 forbidden
    assert clamped[1] == 1.0  # budget == masked: +1 forced
    assert clamped[2] == 0.5  # interior: untouched


# ---------------------------------------------------------------------------
# Rollout law: sampled == independent dynamic-programming enumeration
# ---------------------------------------------------------------------------


def rollout_law_by_dynamic_programming(logit_fn, n_sites, n_plus):
    """Exact terminal law of the revelation chain: recurse over partially
    masked states (uniform site clock x feasibility-clamped species draw),
    accumulating probability into fully revealed states. Independent of the
    sampling code path — it only shares the conditional itself."""
    law = {}

    def recurse(state, probability):
        masked_sites = [i for i, spin in enumerate(state) if spin == 0.0]
        if not masked_sites:
            law[state] = law.get(state, 0.0) + probability
            return
        x = torch.tensor([state])
        _, budget = masked_and_budget(
            tuple(None if s == 0.0 else int(s) for s in state), n_plus
        )
        p_plus_row = torch.sigmoid(logit_fn(x))[0]
        for site in masked_sites:
            p_plus = feasibility_clamped_p_plus(
                p_plus_row[site : site + 1],
                torch.tensor([budget]),
                torch.tensor([len(masked_sites)]),
            ).item()
            for spin, spin_probability in ((+1.0, p_plus), (-1.0, 1 - p_plus)):
                if spin_probability == 0.0:
                    continue
                child = list(state)
                child[site] = spin
                recurse(
                    tuple(child),
                    probability * spin_probability / len(masked_sites),
                )

    recurse((0.0,) * n_sites, 1.0)
    return law


def test_zero_trunk_with_budget_tilt_at_sigma_zero_is_uniform_on_fibre():
    """At sigma = 0 the budget-tilted preconditioner IS the urn law, and the
    urn law's terminal distribution is uniform on the fibre (the reference-
    process theorem). With a zero trunk the DP law must therefore put
    exactly 1/C(4,2) on each of the 6 fibre states."""
    n_sites, n_plus = 4, 2
    adjacency = ring_adjacency(n_sites)

    def logit_fn(x):
        return preconditioner_logit_diff(x, adjacency, 0.0, n_plus, "budget_tilted")

    law = rollout_law_by_dynamic_programming(logit_fn, n_sites, n_plus)
    assert len(law) == comb(n_sites, n_plus)
    for probability in law.values():
        assert isclose(probability, 1.0 / comb(n_sites, n_plus), rel_tol=1e-9)


def test_sampled_rollout_matches_the_dynamic_programming_law():
    """Bind the SAMPLING path (site clock, species draw, bookkeeping) to the
    DP law with a nonzero trunk and nonzero sigma: each terminal's sampled
    frequency must sit within 5 binomial standard errors of the DP
    probability. Deterministic seed, so a failure is a bug, not noise."""
    n_sites, n_plus, sigma = 4, 2, 0.4
    adjacency = ring_adjacency(n_sites)
    torch.manual_seed(1)
    net = MaskedConditionalNet(n_sites, hidden_width=16)
    for parameter in net.parameters():
        parameter.data.normal_(0.0, 0.3)

    def logit_fn(x):
        return net(x) + preconditioner_logit_diff(
            x, adjacency, sigma, n_plus, "budget_tilted"
        )

    with torch.no_grad():
        law = rollout_law_by_dynamic_programming(logit_fn, n_sites, n_plus)
        generator = torch.Generator().manual_seed(11)
        n_rollouts = 120_000
        terminals, _ = rollout_budget_masked(
            logit_fn, n_rollouts, n_sites, n_plus, generator
        )
    for state, probability in law.items():
        matches = (
            torch.all(terminals == torch.tensor(state), dim=1).float().sum().item()
        )
        standard_error = (probability * (1 - probability) / n_rollouts) ** 0.5
        assert abs(matches / n_rollouts - probability) < 5 * standard_error


# ---------------------------------------------------------------------------
# Importance weights: the weighted estimator hits the exact conditional
# ---------------------------------------------------------------------------


def test_weighted_estimator_recovers_the_exact_fibre_conditional():
    """End-to-end weight correctness: with a perturbed trunk (so the rollout
    law is NOT the target) the self-normalised importance weights must
    reweight the empirical terminal distribution back to the exactly
    enumerated fibre conditional. A wrong rollout log-probability or a
    missing weight term shows up here as irreducible TV."""
    n_sites, n_plus, sigma = 6, 3, RING_SIGMA
    adjacency = ring_adjacency(n_sites)
    torch.manual_seed(2)
    net = MaskedConditionalNet(n_sites, hidden_width=16)
    for parameter in net.parameters():
        parameter.data.normal_(0.0, 0.2)  # mild, keeps ESS healthy

    def logit_fn(x):
        return net(x) + preconditioner_logit_diff(
            x, adjacency, sigma, n_plus, "budget_tilted"
        )

    with torch.no_grad():
        generator = torch.Generator().manual_seed(23)
        terminals, rollout_log_prob = rollout_budget_masked(
            logit_fn, 200_000, n_sites, n_plus, generator
        )
    log_weights = sigma * ring_energy_torch(terminals, adjacency) - rollout_log_prob
    weights = torch.softmax(log_weights, dim=0)

    # exact conditional over the 20 fibre states
    from itertools import combinations

    fibre, exact_masses = [], []
    for plus_sites in combinations(range(n_sites), n_plus):
        state = torch.tensor(
            [[+1.0 if i in plus_sites else -1.0 for i in range(n_sites)]]
        )
        fibre.append(state)
        exact_masses.append((sigma * ring_energy_torch(state, adjacency)).exp().item())
    exact = torch.tensor(exact_masses)
    exact = exact / exact.sum()

    weighted_pmf = torch.zeros(len(fibre))
    for index, state in enumerate(fibre):
        weighted_pmf[index] = weights[torch.all(terminals == state[0], dim=1)].sum()
    total_variation = 0.5 * (weighted_pmf - exact).abs().sum().item()
    assert total_variation < 0.02


# ---------------------------------------------------------------------------
# WDCE loss: oriented towards the exact conditional
# ---------------------------------------------------------------------------


def test_wdce_loss_prefers_the_exact_conditional():
    """The population minimiser theorem says the exact fibre conditional
    minimises the constrained WDCE. On a large weighted batch the loss
    evaluated with an exact-conditional oracle must beat the same oracle
    perturbed in logit space — the orientation the training loop relies on."""
    n_sites, n_plus, sigma = 6, 3, RING_SIGMA
    adjacency = ring_adjacency(n_sites)
    neighbours = ring_neighbour_pairs(n_sites)

    def exact_logit_fn(x_batch):
        logits = torch.zeros(x_batch.shape[0], n_sites)
        for row, x in enumerate(x_batch):
            state = tuple(None if spin == 0.0 else int(spin) for spin in x.tolist())
            for site, spin in enumerate(state):
                if spin is not None:
                    continue
                p_plus = exact_masked_conditional_plus(
                    state, site, sigma, n_plus, neighbours
                )
                p_plus = min(max(p_plus, 1e-9), 1 - 1e-9)
                logits[row, site] = torch.tensor(p_plus).logit()
        return logits

    with torch.no_grad():
        generator = torch.Generator().manual_seed(31)
        terminals, rollout_log_prob = rollout_budget_masked(
            exact_logit_fn, 4096, n_sites, n_plus, generator
        )
        log_weights = sigma * ring_energy_torch(terminals, adjacency) - rollout_log_prob
        weights = torch.softmax(log_weights, dim=0)

        torch.manual_seed(5)
        perturbation = torch.randn(1, n_sites)

        losses = {}
        for name, logit_fn in (
            ("exact", exact_logit_fn),
            ("perturbed", lambda x: exact_logit_fn(x) + perturbation),
        ):
            loss_generator = torch.Generator().manual_seed(41)
            losses[name] = wdce_cross_entropy(
                logit_fn,
                terminals,
                weights,
                n_replicates=8,
                generator=loss_generator,
            ).item()
    assert losses["exact"] < losses["perturbed"]


# ---------------------------------------------------------------------------
# Forensics interventions
# ---------------------------------------------------------------------------


def test_gated_offset_at_init_is_exactly_v0():
    """GatedBudgetTiltOffset with gates at their 1.0 init must reproduce
    preconditioner_logit_diff('budget_tilted') bit-for-bit at masked sites
    — the whole point of the gate is keeping V0's start."""
    from discrete_flow_sampler.samplers.budget_masked import (
        GatedBudgetTiltOffset,
    )

    n_sites, n_plus = 6, 3
    adjacency = ring_adjacency(n_sites)
    gated = GatedBudgetTiltOffset(adjacency, RING_SIGMA, n_plus)
    for state in feasible_masked_states(n_sites, n_plus):
        x = as_masked_tensor(state)
        with torch.no_grad():
            v0 = preconditioner_logit_diff(
                x, adjacency, RING_SIGMA, n_plus, "budget_tilted"
            )
            gated_logit = gated(x)
        masked_sites = x[0] == 0.0
        assert torch.equal(gated_logit[0][masked_sites], v0[0][masked_sites])


def test_gated_rollout_stays_on_fibre():
    """G0 is generation-side (the feasibility clamp), so it must survive
    arbitrary gate values — including adversarial negative ones."""
    from discrete_flow_sampler.samplers.budget_masked import (
        GatedBudgetTiltOffset,
    )

    n_sites, n_plus = 16, 8
    adjacency = ring_adjacency(n_sites)
    gated = GatedBudgetTiltOffset(adjacency, 0.223, n_plus)
    with torch.no_grad():
        gated.gate_budget.fill_(-2.0)
        gated.gate_field.fill_(3.0)
    generator = torch.Generator().manual_seed(3)
    terminals, _ = rollout_budget_masked(
        lambda x: gated(x), 2048, n_sites, n_plus, generator
    )
    assert torch.all(((terminals + 1) / 2).sum(dim=1) == n_plus)


def test_context_loss_weight_none_is_the_frozen_protocol():
    """Regression pin: the default (no context weight) computes the same
    loss as before the argument existed, and a CONSTANT weight matches it
    too (the batch-mean normalisation makes constant weights a no-op)."""
    n_sites, n_plus, sigma = 6, 3, RING_SIGMA
    adjacency = ring_adjacency(n_sites)
    torch.manual_seed(9)
    net = MaskedConditionalNet(n_sites, hidden_width=16)
    for parameter in net.parameters():
        parameter.data.normal_(0.0, 0.2)

    def logit_fn(x):
        return net(x) + preconditioner_logit_diff(
            x, adjacency, sigma, n_plus, "budget_tilted"
        )

    with torch.no_grad():
        generator = torch.Generator().manual_seed(13)
        terminals, rollout_log_prob = rollout_budget_masked(
            logit_fn, 512, n_sites, n_plus, generator
        )
        weights = torch.softmax(
            sigma * ring_energy_torch(terminals, adjacency) - rollout_log_prob,
            dim=0,
        )
        losses = []
        for context_weight in (
            None,
            lambda contexts: torch.full((contexts.shape[0],), 7.0),
        ):
            loss_generator = torch.Generator().manual_seed(17)
            losses.append(
                wdce_cross_entropy(
                    logit_fn,
                    terminals,
                    weights,
                    n_replicates=4,
                    generator=loss_generator,
                    context_loss_weight=context_weight,
                ).item()
            )
    assert isclose(losses[0], losses[1], rel_tol=1e-6)


def test_log_variance_loss_is_near_zero_at_the_exact_conditional():
    """F_LV's optimality signature: at the exact conditional the trajectory
    log-RN-derivative is CONSTANT across trajectories (Var = 0 up to float
    noise); a perturbed conditional must score strictly higher. Also pins
    that gradients flow to the model through the rollout (the property
    WDCE never exercises)."""
    from discrete_flow_sampler.samplers.budget_masked import (
        log_variance_loss,
    )

    n_sites, n_plus, sigma = 6, 3, RING_SIGMA
    adjacency = ring_adjacency(n_sites)
    neighbours = ring_neighbour_pairs(n_sites)

    def exact_logit_fn(x_batch):
        logits = torch.zeros(x_batch.shape[0], n_sites)
        for row, x in enumerate(x_batch):
            state = tuple(None if spin == 0.0 else int(spin) for spin in x.tolist())
            for site, spin in enumerate(state):
                if spin is not None:
                    continue
                p_plus = exact_masked_conditional_plus(
                    state, site, sigma, n_plus, neighbours
                )
                p_plus = min(max(p_plus, 1e-9), 1 - 1e-9)
                logits[row, site] = torch.tensor(p_plus).logit()
        return logits

    def target_log_prob(states):
        return sigma * ring_energy_torch(states, adjacency)

    with torch.no_grad():
        exact_loss, _, _ = log_variance_loss(
            exact_logit_fn,
            target_log_prob,
            1024,
            n_sites,
            n_plus,
            torch.Generator().manual_seed(51),
        )
        torch.manual_seed(6)
        perturbation = torch.randn(1, n_sites)
        perturbed_loss, _, _ = log_variance_loss(
            lambda x: exact_logit_fn(x) + perturbation,
            target_log_prob,
            1024,
            n_sites,
            n_plus,
            torch.Generator().manual_seed(51),
        )
    assert exact_loss.item() < 1e-6
    assert perturbed_loss.item() > 100 * max(exact_loss.item(), 1e-12)

    net = MaskedConditionalNet(n_sites, hidden_width=16)
    for parameter in net.parameters():
        parameter.data.normal_(0.0, 0.2)
    loss, terminals, log_rn = log_variance_loss(
        lambda x: net(x),
        target_log_prob,
        64,
        n_sites,
        n_plus,
        torch.Generator().manual_seed(53),
    )
    loss.backward()
    gradient_norms = [p.grad.abs().sum().item() for p in net.parameters()]
    assert sum(gradient_norms) > 0
    assert not terminals.requires_grad and not log_rn.requires_grad


def test_ema_shadow_tracks_and_swaps():
    from experiments.constrained_hard_03.mdns_budget_gate_4x4 import (
        ExponentialMovingAverage,
    )

    parameter = torch.nn.Parameter(torch.ones(3))
    ema = ExponentialMovingAverage([parameter], decay=0.9)
    with torch.no_grad():
        parameter.mul_(2.0)  # parameter now 2, shadow 1
    ema.update()  # shadow = 0.9*1 + 0.1*2 = 1.1
    assert torch.allclose(ema.shadow[0], torch.full((3,), 1.1))
    ema.swap_in()
    assert torch.allclose(parameter.detach(), torch.full((3,), 1.1))
    ema.swap_out()
    assert torch.allclose(parameter.detach(), torch.full((3,), 2.0))


def test_ema_warmup_forgets_init_where_plain_shadow_cannot():
    """The plain-shadow failure mode, pinned. A plain 0.9999 shadow after k
    updates is decay^k init + (1-decay^k) recent-params: at k=200 that is
    98% init, at the paper's k=2000 still 82% init, so eval-on-EMA reads a
    nearly-untrained model no matter how good training was. The warmup
    schedule effective_decay = min(decay, (1+t)/(10+t)) makes the init
    weight prod_{t<=k}(1+t)/(10+t) = 10!(k+1)!/(10+k)! — already ~1e-17 at
    k=200 — while still capping at the requested decay for t >= 9e4."""
    from experiments.constrained_hard_03.mdns_budget_gate_4x4 import (
        ExponentialMovingAverage,
    )

    init_value, trained_value = 1.0, 3.0
    plain_parameter = torch.nn.Parameter(torch.full((3,), init_value))
    warm_parameter = torch.nn.Parameter(torch.full((3,), init_value))
    plain = ExponentialMovingAverage([plain_parameter], decay=0.9999)
    warm = ExponentialMovingAverage([warm_parameter], decay=0.9999, warmup=True)
    with torch.no_grad():
        plain_parameter.fill_(trained_value)
        warm_parameter.fill_(trained_value)
    for _ in range(200):
        plain.update()
        warm.update()
    # Plain shadow: 0.9999^200 = 0.980 of init survives.
    plain_expected = 0.9999**200 * init_value + (1 - 0.9999**200) * trained_value
    assert torch.allclose(plain.shadow[0], torch.full((3,), plain_expected), atol=1e-4)
    # Warmup shadow: init contribution is ~1e-17 — indistinguishable from
    # the trained value.
    assert torch.allclose(warm.shadow[0], torch.full((3,), trained_value), atol=1e-6)
    # Late time: the schedule caps at the requested decay, so warmup and
    # plain agree asymptotically.
    assert warm.effective_decay(step=10**6) == 0.9999
    assert warm.effective_decay(step=1) == 2.0 / 11.0


# ---------------------------------------------------------------------------
# Unconstrained control (gate-3 arm 0): n_plus_target=None switches the
# budget machinery off and the reference becomes the paper's own masked
# diffusion on the free space. The algebra to pin: the trajectory constant
# becomes (1/2)^d — exactly the uniform base on {-1,+1}^d — so
# log w = log p_tilde - rollout_log_prob still holds with no bookkeeping.
# ---------------------------------------------------------------------------


def test_unconstrained_rollout_log_prob_is_exact_species_product():
    """Zero logits = the unconstrained REFERENCE process: every species
    draw is Bernoulli(1/2) and every draw counts (no clamp, no forced
    steps), so rollout_log_prob must be exactly d*log(1/2) on every
    trajectory — the trajectory 'constant' of the free space. The
    terminals must also actually leave the fibre (compositions vary),
    otherwise the budget machinery was not off."""
    n_sites = 16
    generator = torch.Generator().manual_seed(0)
    terminals, rollout_log_prob = rollout_budget_masked(
        lambda x: torch.zeros_like(x), 512, n_sites, None, generator
    )
    expected = n_sites * torch.log(torch.tensor(0.5))
    assert torch.allclose(rollout_log_prob, expected.expand(512), atol=1e-6)
    compositions = ((terminals + 1) / 2).sum(dim=1)
    assert len(compositions.unique()) > 1


def test_unconstrained_oracle_conditionals_give_constant_weights():
    """The sharp end-to-end check of the unconstrained algebra: rolling
    out with the EXACT free-space conditionals must give importance
    weights that are constant across trajectories (log w = log Z for
    every rollout) — this exercises the rollout law, the log-prob
    accumulation, and the uniform-base cancellation at once. Uses the
    2x2 torus (d = 4, 16 states) so the oracle is an exact enumeration
    with no sampling floor."""
    from discrete_flow_sampler.targets.ising import IsingTarget

    target = IsingTarget(D=2, sigma=0.3)
    states = torch.cartesian_prod(*([torch.tensor([-1.0, 1.0])] * 4))
    log_p_tilde = target.log_prob(states)
    log_z = torch.logsumexp(log_p_tilde, dim=0)

    def oracle_logit_fn(x_masked):
        logits = torch.zeros_like(x_masked)
        for row, context in enumerate(x_masked):
            unmasked = context != 0.0
            consistent = (states[:, unmasked] == context[unmasked]).all(dim=1)
            posterior = torch.softmax(log_p_tilde[consistent], dim=0)
            p_plus = posterior @ (states[consistent] == 1.0).float()
            p_plus = p_plus.clamp(1e-9, 1 - 1e-9)
            logits[row] = torch.log(p_plus) - torch.log1p(-p_plus)
        return logits

    generator = torch.Generator().manual_seed(1)
    terminals, rollout_log_prob = rollout_budget_masked(
        oracle_logit_fn, 256, 4, None, generator
    )
    log_w = target.log_prob(terminals) - rollout_log_prob
    assert torch.allclose(log_w, log_z.expand(256), atol=1e-4)
