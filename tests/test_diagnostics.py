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
import math

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    entropy_estimate,
    ess_from_log_weights,
    exact_free_energy,
    exact_internal_energy,
    exact_log_probs,
    free_energy_lb_estimate,
    internal_energy_estimate,
)


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
    arithmetic mean of -log p̃ / D."""
    K, sigma, D = 8, 0.1, 4
    log_weights = torch.zeros(K)
    log_p_tilde = torch.tensor([0.5, -1.0, 0.0, 2.0, -0.3, 0.7, 1.1, -1.4])
    E_per_site = internal_energy_estimate(
        log_weights, log_p_tilde, sigma=sigma, D=D
    )
    expected = -log_p_tilde.mean().item() / D
    assert E_per_site.item() == pytest.approx(expected, abs=1e-6)


def test_internal_energy_dominant_weight_picks_single_sample():
    """If log_weights[0] dominates, softmax(w) is essentially a one-hot at 0,
    so E/D ≈ -log_p_tilde[0] / D."""
    sigma, D = 0.1, 4
    log_weights = torch.tensor([0.0, -100.0, -100.0, -100.0])
    log_p_tilde = torch.tensor([0.7, 0.0, 0.0, 0.0])
    E_per_site = internal_energy_estimate(
        log_weights, log_p_tilde, sigma=sigma, D=D
    )
    expected = -log_p_tilde[0].item() / D
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
    expected = -(log_pi.exp() * log_p_unnorm).sum().item() / D
    assert E_per_site.item() == pytest.approx(expected, abs=1e-6)
