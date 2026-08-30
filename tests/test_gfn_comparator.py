"""Correctness gates for the GFlowNet comparator (hard chapter).

Written BEFORE the implementation (tests-first convention). The comparator is
a fixed-raster-order GFlowNet over the fixed-composition Ising slice:

  * Construction chain: sites assigned in raster order, so each state (prefix)
    has exactly one parent and P_B == 1. The exactly-N_A constraint is enforced
    by count masking: with n_up up-spins placed and r sites remaining, the
    policy is forced to -1 once n_up == N_A and forced to +1 once
    N_A - n_up == r. Feasibility is by construction, not by penalty.
  * TB arm (Malkin et al. 2022): on the chain the trajectory-balance loss
    collapses to (log Z_theta + log q_theta(x) - log p_tilde(x))^2 with
    q_theta the autoregressive product of masked per-site conditionals.
  * FL-DB arm (Pan et al. 2023): per-step detailed balance with the state flow
    reparameterised by the prefix partial energy R_tilde(s); with P_B == 1 the
    balance is  logFres(s_i) + log P_F(x_i|s_i) - delta_i = logFres(s_{i+1}),
    terminal residual logFres(s_d) == 0 because the complete prefix's partial
    energy IS the full energy.

The gates below are enumeration-exact (2x2 and 4x4 lattices), which is why the
losses are hand-rolled rather than imported from torchgfn: the library's
state-map abstraction re-encodes every prefix (O(d^3) attention for a
transformer policy at d=256), and an enumeration oracle is a stronger
correctness authority than a reference implementation.
"""

import math

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.raster_gfn import RasterGFNPolicy
from discrete_flow_sampler.samplers.gfn_raster import (
    forward_looking_db_loss,
    raster_prefix_log_reward_increments,
    trajectory_balance_loss,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _target(D=2, sigma=0.3, c=0.5, bias=0.0):
    return FixedCompositionIsingTarget(
        D=D, sigma=sigma, target_composition=c, bias=bias
    )


def _policy(target, hidden_dim=32, n_layers=2, n_heads=2, with_flow_head=False):
    torch.manual_seed(0)
    return RasterGFNPolicy(
        D=target.D,
        n_plus_target=target.n_plus_target,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        with_flow_head=with_flow_head,
    )


def _slice_states(target):
    """All configurations on the fixed-composition slice, (N_slice, d) float."""
    states = enumerate_states(target.d).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    return states[n_plus == target.n_plus_target]


def _exact_slice_log_probs(target, slice_states):
    """Normalised log-probs of the target restricted to the slice."""
    log_p_unnorm = target.log_prob(slice_states)
    return log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=0)


# ---------------------------------------------------------------------------
# Gate 1: masking is exact — every sample lands on the manifold, untrained.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.5, 0.25])
def test_samples_land_exactly_on_manifold(c):
    target = _target(D=4, c=c)
    policy = _policy(target)
    x, _ = policy.sample(256)
    target.assert_on_manifold(x)  # raises on any off-slice row
    assert set(x.unique().tolist()) <= {-1.0, 1.0}


def test_epsilon_exploration_stays_on_manifold():
    # Off-policy exploration must respect the forced moves: the mask binds
    # the behaviour policy too, not just the greedy one.
    target = _target(D=4, c=0.5)
    policy = _policy(target)
    x, _ = policy.sample(256, epsilon=0.5)
    target.assert_on_manifold(x)


def test_untrained_sampler_covers_many_slice_configs():
    # Uniform-ish init must not collapse to a handful of arrangements —
    # this is the cheap canary for a mask bug that over-forces.
    target = _target(D=4, c=0.5)
    policy = _policy(target)
    x, _ = policy.sample(512)
    assert x.unique(dim=0).shape[0] > 50


# ---------------------------------------------------------------------------
# Gate 2: the AR likelihood is coherent — sampling, scoring, normalisation.
# ---------------------------------------------------------------------------


def test_sample_log_prob_matches_parallel_scoring():
    # The sequential sampler and the one-pass causal scorer must agree; this
    # is the classic off-by-one (AR shift) bug detector.
    target = _target(D=4, c=0.5)
    policy = _policy(target)
    x, log_q_sequential = policy.sample(64)
    log_q_scored = policy.log_prob(x)
    assert torch.allclose(log_q_sequential, log_q_scored, atol=1e-5)


def test_slice_probabilities_normalise_exactly():
    # A masked AR factorisation is a proper distribution ON THE SLICE for any
    # logits: sum over the enumerated slice must be exactly 1, untrained.
    target = _target(D=2, c=0.5)
    policy = _policy(target)
    slice_states = _slice_states(target)
    assert slice_states.shape[0] == 6  # C(4, 2)
    total = torch.logsumexp(policy.log_prob(slice_states), dim=0)
    assert torch.allclose(total, torch.zeros(()), atol=1e-5)


def test_off_slice_state_scores_minus_infinity():
    target = _target(D=4, c=0.5)
    policy = _policy(target)
    x, _ = policy.sample(4)
    x[:, 0] = -x[:, 0]  # break composition: n_plus != N_A
    log_q = policy.log_prob(x)
    assert torch.all(torch.isinf(log_q) & (log_q < 0))


# ---------------------------------------------------------------------------
# Gate 3: prefix partial energies (the FL-DB intermediate reward).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [0.0, 0.3])
def test_prefix_increments_sum_to_target_log_prob(bias):
    # Telescoping identity: assigning each raster site's bonds to its
    # larger-index endpoint partitions the edge set, so the increments must
    # sum to log p_tilde exactly (wrap bonds included).
    target = _target(D=4, c=0.5, bias=bias)
    x = target.sample_base(32, device="cpu")
    increments = raster_prefix_log_reward_increments(target, x)
    assert increments.shape == (32, target.d)
    assert torch.allclose(increments.sum(dim=-1), target.log_prob(x), atol=1e-4)


def test_prefix_increments_match_bruteforce_partial_energies():
    # Prefix energy computed by zeroing unassigned sites (which kills every
    # bond touching them) must equal the running sum of increments.
    target = _target(D=4, c=0.5, bias=0.2)
    x = target.sample_base(8, device="cpu")
    running = raster_prefix_log_reward_increments(target, x).cumsum(dim=-1)
    for prefix_len in range(target.d + 1):
        masked = x.clone()
        masked[:, prefix_len:] = 0.0
        brute = (
            torch.einsum("bi,ij,bj->b", masked, target.J, masked)
            + target.bias * masked.sum(dim=-1)
        )
        expected = (
            torch.zeros(8) if prefix_len == 0 else running[:, prefix_len - 1]
        )
        assert torch.allclose(expected, brute, atol=1e-4)


# ---------------------------------------------------------------------------
# Gate 4: loss algebra vanishes at the enumeration-exact optimum.
# ---------------------------------------------------------------------------


def test_trajectory_balance_loss_zero_at_exact_conditionals():
    # TB residual = log Z + log q(x) - log p_tilde(x); with q the exact
    # normalised slice distribution and log Z the exact slice log-partition,
    # the residual is identically zero for every state.
    target = _target(D=2, c=0.5)
    slice_states = _slice_states(target)
    log_p_unnorm = target.log_prob(slice_states)
    log_z_slice = torch.logsumexp(log_p_unnorm, dim=0)
    exact_log_q = log_p_unnorm - log_z_slice
    loss = trajectory_balance_loss(log_z_slice, exact_log_q, log_p_unnorm)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_forward_looking_db_loss_zero_at_exact_flows():
    # Exact chain flows: F(s_i) = sum of exp(log p_tilde) over slice states
    # sharing the prefix. With logpf_i = log F(s_{i+1}) - log F(s_i) and
    # residuals logFres = log F - partial energy, every per-step FL-DB
    # residual is zero — including the terminal convention logFres(s_d) = 0.
    target = _target(D=2, c=0.5)
    slice_states = _slice_states(target)
    log_p_unnorm = target.log_prob(slice_states)
    increments = raster_prefix_log_reward_increments(target, slice_states)
    partial = torch.cat(
        [torch.zeros(len(slice_states), 1), increments.cumsum(dim=-1)], dim=1
    )  # (N, d+1): partial energy of s_0 .. s_d

    d = target.d
    log_flow = torch.empty(len(slice_states), d + 1)
    for i in range(d + 1):
        for row, x in enumerate(slice_states):
            shares_prefix = (slice_states[:, :i] == x[:i]).all(dim=-1)
            log_flow[row, i] = torch.logsumexp(log_p_unnorm[shares_prefix], dim=0)

    site_log_probs = log_flow[:, 1:] - log_flow[:, :-1]
    flow_residuals = (log_flow - partial)[:, :-1]  # model outputs s_0 .. s_{d-1}
    # Terminal check baked into the convention: logFres(s_d) must already be 0.
    assert torch.allclose(log_flow[:, -1], partial[:, -1], atol=1e-5)

    loss = forward_looking_db_loss(site_log_probs, increments, flow_residuals)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_flow_residual_head_shapes():
    target = _target(D=4, c=0.5)
    policy = _policy(target, with_flow_head=True)
    x, _ = policy.sample(8)
    site_log_probs, flow_residuals = policy.site_log_probs_and_flow_residuals(x)
    assert site_log_probs.shape == (8, target.d)
    assert flow_residuals.shape == (8, target.d)
    assert torch.isfinite(flow_residuals).all()


# ---------------------------------------------------------------------------
# Gate 5: short-training recovery of the exact slice distribution (2x2).
# ---------------------------------------------------------------------------


def _total_variation_to_exact(policy, target):
    slice_states = _slice_states(target)
    q = policy.log_prob(slice_states).exp()
    p = _exact_slice_log_probs(target, slice_states).exp()
    return 0.5 * (q - p).abs().sum().item()


def test_tb_training_recovers_exact_distribution_and_log_z():
    torch.manual_seed(42)
    target = _target(D=2, sigma=0.3, c=0.5)
    policy = _policy(target)
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(800):
        with torch.no_grad():
            x, _ = policy.sample(128, epsilon=0.05)
        loss = trajectory_balance_loss(
            policy.log_z, policy.log_prob(x), target.log_prob(x)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert _total_variation_to_exact(policy, target) < 0.05
    log_z_slice = torch.logsumexp(target.log_prob(_slice_states(target)), dim=0)
    assert abs(policy.log_z.item() - log_z_slice.item()) < 0.05


def test_fldb_training_recovers_exact_distribution():
    torch.manual_seed(42)
    target = _target(D=2, sigma=0.3, c=0.5)
    policy = _policy(target, with_flow_head=True)
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(800):
        with torch.no_grad():
            x, _ = policy.sample(128, epsilon=0.05)
        site_log_probs, flow_residuals = policy.site_log_probs_and_flow_residuals(x)
        increments = raster_prefix_log_reward_increments(target, x)
        loss = forward_looking_db_loss(site_log_probs, increments, flow_residuals)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert _total_variation_to_exact(policy, target) < 0.05
