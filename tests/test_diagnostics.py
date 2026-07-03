"""Tests for paper-faithful diagnostics (paper Appendix D.1, Table 2).

Tests for the IS estimators (`free_energy_lb_estimate`,
`internal_energy_estimate`, `entropy_estimate`) and their
enumeration-based exact references encode "what correct looks like"
before the user fills in the bodies. They start failing with
NotImplementedError; they pass once each body lands.

Tests for the off-paper utilities removed in the 2026-05 metric refactor
(TVD, KL, 1-D Wasserstein, log_prob_w1) have been deleted alongside the
function definitions.
"""
import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up,
    composition_observables,
    conditional_pmf_at_composition,
    diagonal_correlation,
    entropy_estimate,
    enumerate_states,
    ess_from_log_weights,
    exact_free_energy,
    exact_internal_energy,
    exact_log_probs,
    free_energy_lb_estimate,
    internal_energy_estimate,
    magnetisation,
    nn_correlation,
    z2_asymmetry_from_samples,
)
from discrete_flow_sampler.targets.ising import IsingTarget

# ---------------------------------------------------------------------------
# ESS (already-implemented; pinning behaviour during the refactor)
# ---------------------------------------------------------------------------


def test_ess_equals_n_for_uniform_weights():
    log_w = torch.zeros(100)
    assert ess_from_log_weights(log_w).item() == pytest.approx(100.0)


def test_ess_one_for_dominant_weight():
    log_w = torch.full((100,), -100.0)
    log_w[0] = 0.0
    # one weight dominates → ESS → 1
    assert ess_from_log_weights(log_w).item() == pytest.approx(1.0, abs=1e-3)


def test_composition_and_magnetisation_observables():
    x = torch.tensor([
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
    ])

    torch.testing.assert_close(composition_fraction_up(x), torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(magnetisation(x), torch.tensor([0.0, -0.5]))

    metrics = composition_observables(
        x,
        target_composition=0.25,
        composition_penalty_strength=50.0,
    )
    assert metrics["composition_mean"] == pytest.approx(0.375)
    assert metrics["magnetisation_mean"] == pytest.approx(-0.25)
    assert metrics["target_composition"] == pytest.approx(0.25)
    assert metrics["composition_penalty_strength"] == pytest.approx(50.0)
    assert metrics["composition_abs_error_mean"] == pytest.approx(0.125)
    assert metrics["composition_sq_violation_mean"] == pytest.approx(0.03125)


# ---------------------------------------------------------------------------
# Enumeration helpers (still used by exact-reference D ≤ 20 path)
# ---------------------------------------------------------------------------


def test_enumerate_states_shape_and_count():
    D = 3  # 2^3 = 8 states for binary spins
    states = enumerate_states(D)
    assert states.shape == (8, D)
    # all entries ∈ {-1, +1}
    assert torch.all((states == 1) | (states == -1))
    # all states unique
    assert len({tuple(s.tolist()) for s in states}) == 8


def test_exact_log_probs_normalises_to_one():
    # tiny target: log_prob(x) = sum(x). enumerate over D=2 → 4 states.
    class TinyTarget:
        def log_prob(self, x):
            return x.sum(dim=-1).float()

    states = enumerate_states(D=2)
    log_p = exact_log_probs(TinyTarget(), states)
    probs = log_p.exp()
    assert probs.sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# free_energy_lb_estimate (paper Eq. 37)
# ---------------------------------------------------------------------------


def test_free_energy_lb_constant_log_weights():
    """If every w_k = c, then log Ẑ_lb = c and F/D = -c / (2σD)."""
    K, sigma, D = 16, 0.1, 4
    c = -3.0
    log_weights = torch.full((K,), c)
    F_per_site = free_energy_lb_estimate(log_weights, sigma=sigma, D=D)
    expected = -c / (2 * sigma * D)
    assert F_per_site.item() == pytest.approx(expected, abs=1e-6)


def test_free_energy_lb_uses_arithmetic_mean_of_log_weights():
    """log Ẑ_lb = (1/K) Σ w_k. Numerical pin so a `logsumexp - log K`
    mistake (the geometric-vs-arithmetic-mean trap) is caught."""
    sigma, D = 0.1, 2
    log_weights = torch.tensor([0.0, 2.0, 4.0])
    F_per_site = free_energy_lb_estimate(log_weights, sigma=sigma, D=D)
    expected = -(log_weights.mean().item()) / (2 * sigma * D)
    assert F_per_site.item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# internal_energy_estimate (paper Eq. 38)
# ---------------------------------------------------------------------------


def test_internal_energy_uniform_weights_equals_mean_neg_log_p_tilde():
    """Uniform log-weights → softmax(w) is uniform → IS reduces to a plain
    arithmetic mean of -log p̃ / (2σD). The 2σ in the denominator is the
    per-β conversion explained in `internal_energy_estimate` (E/D = u,
    physics per-spin, not E_paper/D)."""
    K, sigma, D = 8, 0.1, 4
    log_weights = torch.zeros(K)
    log_p_tilde = torch.tensor([0.5, -1.0, 0.0, 2.0, -0.3, 0.7, 1.1, -1.4])
    E_per_site = internal_energy_estimate(
        log_weights, log_p_tilde, sigma=sigma, D=D
    )
    expected = -log_p_tilde.mean().item() / (2 * sigma * D)
    assert E_per_site.item() == pytest.approx(expected, abs=1e-6)


def test_internal_energy_dominant_weight_picks_single_sample():
    """If log_weights[0] dominates, softmax(w) is essentially a one-hot at 0,
    so E/D ≈ -log_p_tilde[0] / (2σD)."""
    sigma, D = 0.1, 4
    log_weights = torch.tensor([0.0, -100.0, -100.0, -100.0])
    log_p_tilde = torch.tensor([0.7, 0.0, 0.0, 0.0])
    E_per_site = internal_energy_estimate(
        log_weights, log_p_tilde, sigma=sigma, D=D
    )
    expected = -log_p_tilde[0].item() / (2 * sigma * D)
    assert E_per_site.item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# entropy_estimate (paper Table 2 caption: S = 2σ(E - F))
# ---------------------------------------------------------------------------


def test_entropy_is_2sigma_times_energy_minus_free_energy():
    sigma = 0.1
    F_per_site = torch.tensor(-3.6727)
    E_per_site = torch.tensor(-0.4282)
    S_per_site = entropy_estimate(F_per_site, E_per_site, sigma=sigma)
    expected = 2 * sigma * (E_per_site.item() - F_per_site.item())
    assert S_per_site.item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# exact_free_energy / exact_internal_energy (D ≤ 20 enumeration)
# ---------------------------------------------------------------------------


def test_exact_free_energy_matches_logsumexp_definition():
    """Sanity: given a target whose log_p̃ enumerates trivially, the
    exact F/D should equal -logsumexp(log_p̃(states)) / (2σD)."""
    sigma, D = 0.1, 2

    class TinyTarget:
        device = "cpu"
        def log_prob(self, x):
            # log p̃(x) = σ · sum(x) (linear; not Ising, but enumerates fine)
            return sigma * x.sum(dim=-1).float()

    target = TinyTarget()
    F_per_site = exact_free_energy(target, sigma=sigma, D=D)

    states = enumerate_states(D).float()
    log_p_unnorm = target.log_prob(states)
    log_Z = torch.logsumexp(log_p_unnorm, dim=0)
    expected = -log_Z.item() / (2 * sigma * D)
    assert F_per_site.item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# conditional_pmf_at_composition / z2_asymmetry_from_samples
# ---------------------------------------------------------------------------


def test_conditional_pmf_slice_normalises_and_picks_right_count():
    """At D=4, the c=0.5 slice (n_plus=2) should contain C(4,2) = 6 states
    and the conditional log-pmf should logsumexp to 0."""
    D = 4
    states = enumerate_states(D)

    class TinyTarget:
        def log_prob(self, x):
            return torch.zeros(x.shape[0])  # uniform → π is uniform over 2^D

    log_pi = exact_log_probs(TinyTarget(), states)
    slice_states, log_pi_cond = conditional_pmf_at_composition(
        states, log_pi, n_plus_target=2
    )
    assert slice_states.shape == (6, D)
    assert torch.logsumexp(log_pi_cond, dim=0).item() == pytest.approx(0.0, abs=1e-6)
    # uniform target → uniform conditional → all log-probs equal log(1/6)
    torch.testing.assert_close(
        log_pi_cond,
        torch.full((6,), -torch.log(torch.tensor(6.0)).item()),
        atol=1e-6, rtol=0,
    )


def test_conditional_pmf_z2_symmetric_at_c_half():
    """At c_target = 0.5 with bias = 0, the conditional p(·|c=0.5) should
    be Z_2-symmetric: π(x|c=0.5) == π(-x|c=0.5) for every x on the slice."""
    from discrete_flow_sampler.targets.ising import IsingTarget

    target = IsingTarget(
        D=2, sigma=0.1, bias=0.0,
        target_composition=0.5, composition_penalty_strength=50.0,
    )
    D_total = 4  # 2x2 = 4 sites
    states = enumerate_states(D_total)
    log_pi = exact_log_probs(target, states.float())
    _, log_pi_cond = conditional_pmf_at_composition(states, log_pi, n_plus_target=2)
    # Z_2 symmetry: every state x on the slice has -x also on the slice
    # (because n_plus(-x) = D - n_plus(x), which equals 2 iff n_plus(x) = 2).
    # Sorting the conditional probabilities should give the same multiset
    # as the reverse-ordered version (since the slice is closed under x → -x
    # and each pair has equal probability).
    sorted_probs = torch.sort(log_pi_cond.exp())[0]
    # paired structure: the conditional pmf at c=0.5 should split into
    # (x, -x) pairs with equal prob, so the sorted multiset equals itself
    # under reversal trivially; the strong claim is that pairing x with -x
    # gives matching probs.
    n_plus = ((states + 1) // 2).sum(dim=-1)
    slice_idx = torch.where(n_plus == 2)[0]
    slice_states = states[slice_idx]
    probs = log_pi_cond.exp()
    for i, s in enumerate(slice_states):
        neg_s = -s
        # find index of -s in slice_states
        j = ((slice_states == neg_s).all(dim=-1).nonzero(as_tuple=True))[0].item()
        assert probs[i].item() == pytest.approx(probs[j].item(), abs=1e-6)


def test_z2_asymmetry_zero_for_symmetric_weights():
    """Pair each x with -x and give equal IS weight → e_m_is = 0 and
    mass_pos == mass_neg, so asymmetry = 0."""
    samples = torch.tensor([
        [1.0, -1.0, 1.0, -1.0],   # m = 0
        [1.0, 1.0, -1.0, -1.0],   # m = 0
        [1.0, 1.0, 1.0, -1.0],    # m = +0.5
        [-1.0, -1.0, -1.0, 1.0],  # m = -0.5  (negation of previous)
    ])
    log_w = torch.zeros(4)  # uniform IS weights
    result = z2_asymmetry_from_samples(samples, log_w)
    assert result["e_m_is"] == pytest.approx(0.0, abs=1e-6)
    assert result["mass_pos"] == pytest.approx(result["mass_neg"], abs=1e-6)
    assert result["asymmetry"] == pytest.approx(0.0, abs=1e-6)


def test_z2_asymmetry_flags_imbalanced_samples():
    """All samples have m > 0 → mass_pos = 1, mass_neg = 0, asymmetry = 1."""
    samples = torch.tensor([
        [1.0, 1.0, 1.0, -1.0],    # m = +0.5
        [1.0, 1.0, -1.0, 1.0],    # m = +0.5
    ])
    log_w = torch.zeros(2)
    result = z2_asymmetry_from_samples(samples, log_w)
    assert result["mass_pos"] == pytest.approx(1.0, abs=1e-6)
    assert result["mass_neg"] == pytest.approx(0.0, abs=1e-6)
    assert result["asymmetry"] == pytest.approx(1.0, abs=1e-6)


def test_exact_internal_energy_matches_pi_weighted_neg_log_p_tilde():
    sigma, D = 0.1, 2

    class TinyTarget:
        device = "cpu"
        def log_prob(self, x):
            return sigma * x.sum(dim=-1).float()

    target = TinyTarget()
    E_per_site = exact_internal_energy(target, sigma=sigma, D=D)

    states = enumerate_states(D).float()
    log_p_unnorm = target.log_prob(states)
    log_pi = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=0)
    expected = -(log_pi.exp() * log_p_unnorm).sum().item() / (2 * sigma * D)
    assert E_per_site.item() == pytest.approx(expected, abs=1e-6)


# --- MCMC-trace diagnostics (integrated autocorrelation time, R-hat) ---------
import numpy as np

from discrete_flow_sampler.diagnostics.metrics import (
    integrated_autocorr,
    gelman_rubin,
)


def test_integrated_autocorr_white_noise_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    tau = integrated_autocorr(x)
    assert 0.8 < tau < 1.3  # white noise → τ_int ≈ 1


def test_integrated_autocorr_ar1_matches_theory():
    # AR(1) x_t = phi x_{t-1} + eps has τ_int = (1+phi)/(1-phi).
    phi = 0.8
    rng = np.random.default_rng(1)
    n = 500_000
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    tau_theory = (1 + phi) / (1 - phi)  # = 9.0
    tau = integrated_autocorr(x)
    assert abs(tau - tau_theory) / tau_theory < 0.15


def test_gelman_rubin_agreeing_chains_near_one():
    rng = np.random.default_rng(2)
    chains = rng.standard_normal((4, 5000))  # iid, same dist
    assert gelman_rubin(chains) < 1.05


def test_gelman_rubin_separated_chains_large():
    rng = np.random.default_rng(3)
    offsets = np.array([0.0, 10.0, 20.0, 30.0])[:, None]
    chains = rng.standard_normal((4, 5000)) + offsets  # trapped in diff modes
    assert gelman_rubin(chains) > 2.0


# --- Short-range-order diagnostic (nearest-neighbour spin correlation) ------


def test_nn_correlation_all_up_is_one():
    tgt = IsingTarget(D=4, sigma=0.1)
    x = torch.ones(3, 16)
    assert torch.allclose(nn_correlation(x, tgt.A), torch.ones(3), atol=1e-6)


def test_nn_correlation_checkerboard_is_minus_one():
    tgt = IsingTarget(D=4, sigma=0.1)
    # bipartite checkerboard on the 4×4 torus: sign = (-1)^(row+col)
    coords = torch.arange(16)
    row, col = coords // 4, coords % 4
    x = ((-1.0) ** (row + col)).unsqueeze(0)     # (1, 16)
    assert torch.allclose(nn_correlation(x, tgt.A), -torch.ones(1), atol=1e-6)


def test_diagonal_correlation_all_up_is_one():
    x = torch.ones(3, 16)
    assert torch.allclose(diagonal_correlation(x, 4), torch.ones(3), atol=1e-6)


def test_diagonal_correlation_checkerboard_is_plus_one():
    # checkerboard sign = (-1)^(row+col); diagonal neighbours (r±1,c±1) share
    # colour, so every diagonal product is +1 — the OPPOSITE of nn_correlation.
    coords = torch.arange(16)
    row, col = coords // 4, coords % 4
    x = ((-1.0) ** (row + col)).unsqueeze(0)     # (1, 16)
    assert torch.allclose(diagonal_correlation(x, 4), torch.ones(1), atol=1e-6)
