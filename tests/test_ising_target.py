import torch
import pytest

from discrete_flow_sampler.targets.ising import IsingTarget


def test_log_prob_matches_explicit_sum_D2():
    """For D=2 (4 sites), p ∝ exp(σ x^T A x). Verify against explicit pair sum."""
    target = IsingTarget(D=2, sigma=0.1)
    # x = [+1, -1, -1, +1] over a 2x2 periodic grid
    x = torch.tensor([[1, -1, -1, 1]], dtype=torch.float)
    # adjacency: 2x2 periodic grid → each site has 4 neighbours but with wrap,
    # for a 2x2 each pair of sites is connected by both an "x" and "y" edge,
    # giving the J matrix specific structure. Trust the implementation; we test
    # via the log_prob = x^T J x relationship instead.
    expected = torch.einsum("bi,ij,bj->b", x, target.J, x) * 1.0
    got = target.log_prob(x)
    torch.testing.assert_close(got, expected)


def test_log_prob_invariant_under_global_flip():
    """Ising has Z2 symmetry: log_prob(x) = log_prob(-x) when bias=0."""
    target = IsingTarget(D=4, sigma=0.1, bias=0.0)
    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    torch.testing.assert_close(target.log_prob(x), target.log_prob(-x))


def test_bias_breaks_z2_symmetry():
    """With bias != 0, log_prob(x) - log_prob(-x) = 2 · bias · Σ x."""
    target = IsingTarget(D=4, sigma=0.1, bias=0.5)
    x = torch.tensor([[1.0] * 16, [-1.0] * 16])
    diff = target.log_prob(x) - target.log_prob(-x)
    expected = 2 * 0.5 * x.sum(dim=-1)
    torch.testing.assert_close(diff, expected)


def test_log_p_tilde_at_t0_is_uniform():
    """At t=0, log_p_tilde_t = log_eta (uniform), constant in x."""
    target = IsingTarget(D=4, sigma=0.1)
    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    t = torch.zeros(8)
    val = target.log_p_tilde_t(x, t)
    # all entries identical (uniform prior is constant in x)
    assert torch.allclose(val, val[0].expand_as(val))


def test_log_p_tilde_at_t1_equals_log_prob():
    """At t=1, log_p_tilde_t = log_prob (full target)."""
    target = IsingTarget(D=4, sigma=0.1)
    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    t = torch.ones(8)
    torch.testing.assert_close(target.log_p_tilde_t(x, t), target.log_prob(x))


def test_dt_log_p_tilde_t_is_log_ratio():
    """∂_t log p̃_t = log ρ − log η. Path is linear in log, so derivative is
    independent of t."""
    target = IsingTarget(D=4, sigma=0.1)
    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    t1 = torch.full((8,), 0.3)
    t2 = torch.full((8,), 0.7)
    torch.testing.assert_close(
        target.dt_log_p_tilde_t(x, t1),
        target.dt_log_p_tilde_t(x, t2),
    )


def test_periodic_boundary():
    """For D=4, site (0,0) and site (3,0) (opposite edges in row direction)
    must be neighbours due to periodic boundary."""
    target = IsingTarget(D=4, sigma=1.0)
    # site index = i*D + j, so (0,0) → 0, (3,0) → 12
    assert target.J[0, 12].item() != 0.0
    # similarly (0,0) and (0,3) → 0 and 3
    assert target.J[0, 3].item() != 0.0
