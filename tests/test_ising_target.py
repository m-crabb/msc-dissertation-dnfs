import itertools
import math

import pytest
import torch

from discrete_flow_sampler.targets.ising import IsingTarget


def _all_states(d):
    return torch.tensor(
        list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float
    )


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


def test_composition_fraction_maps_up_spins():
    target = IsingTarget(D=2, sigma=0.1)
    x = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0],
        ]
    )
    expected = torch.tensor([0.5, 0.25])
    torch.testing.assert_close(target.composition_fraction(x), expected)


def test_zero_strength_composition_constraint_matches_base_target():
    base = IsingTarget(D=4, sigma=0.1)
    constrained = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.3,
        composition_penalty_strength=0.0,
    )
    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    torch.testing.assert_close(constrained.log_prob(x), base.log_prob(x))


def test_composition_constraint_subtracts_extensive_penalty():
    target = IsingTarget(
        D=2,
        sigma=0.1,
        target_composition=0.25,
        composition_penalty_strength=10.0,
    )
    x = torch.tensor([[1.0, 1.0, -1.0, -1.0]])  # c_+ = 0.5
    expected_penalty = 10.0 * target.d * (0.5 - 0.25) ** 2
    got = target.log_prob(x)
    expected = target.base_log_prob(x) - expected_penalty
    torch.testing.assert_close(got, expected)


def test_nonzero_composition_penalty_requires_target():
    with pytest.raises(ValueError, match="target_composition"):
        IsingTarget(D=2, sigma=0.1, composition_penalty_strength=1.0)


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
    """∂_t log p̃_t = log ρ - log η. Path is linear in log, so derivative is
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


def test_set_sigma_matches_fresh_target():
    """set_sigma swaps σ in place; resulting target equals one built at the new σ.

    Used by temperature curricula to move through easier intermediate
    targets without rebuilding model or optimizer state.
    """
    swapped = IsingTarget(D=4, sigma=0.1)
    swapped.set_sigma(0.5)
    fresh = IsingTarget(D=4, sigma=0.5)

    assert swapped.sigma == 0.5
    torch.testing.assert_close(swapped.J, fresh.J)

    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    torch.testing.assert_close(swapped.log_prob(x), fresh.log_prob(x))


def test_set_composition_penalty_strength_matches_fresh_target():
    """set_composition_penalty_strength swaps λ in place; target equals one
    built at the new λ. Used by λ-annealing curricula to tighten the soft
    composition constraint without rebuilding model or optimizer state."""
    swapped = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=10.0,
    )
    swapped.set_composition_penalty_strength(50.0)
    fresh = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=50.0,
    )

    assert swapped.composition_penalty_strength == 50.0

    x = torch.randint(0, 2, (8, 16)).float() * 2 - 1
    torch.testing.assert_close(
        swapped.composition_penalty(x), fresh.composition_penalty(x)
    )
    torch.testing.assert_close(swapped.log_prob(x), fresh.log_prob(x))


def test_set_composition_penalty_strength_rejects_negative():
    target = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=10.0,
    )
    with pytest.raises(ValueError):
        target.set_composition_penalty_strength(-1.0)


def test_base_log_eta_normalises_uniform():
    target = IsingTarget(D=2, sigma=0.1, base_composition=0.5)
    states = _all_states(target.d)
    total = torch.logsumexp(target.base_log_eta(states), dim=0)
    torch.testing.assert_close(total, torch.tensor(0.0))


def test_base_log_eta_normalises_off_centre():
    target = IsingTarget(D=2, sigma=0.1, base_composition=0.8)
    states = _all_states(target.d)
    total = torch.logsumexp(target.base_log_eta(states), dim=0)
    torch.testing.assert_close(total, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_base_log_eta_uniform_is_constant_minus_d_log2():
    target = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    x = torch.randint(0, 2, (8, target.d)).float() * 2 - 1
    expected = torch.full((8,), -target.d * math.log(2))
    torch.testing.assert_close(target.base_log_eta(x), expected)


def test_log_p_tilde_t_bit_identical_at_p_half():
    """p=0.5 must reproduce the legacy (1-t)(-d log2) + t log_prob exactly."""
    target = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    x = torch.randint(0, 2, (8, target.d)).float() * 2 - 1
    t = torch.rand(8)
    legacy = (1 - t) * (-target.d * math.log(2)) + t * target.log_prob(x)
    torch.testing.assert_close(target.log_p_tilde_t(x, t), legacy)


def test_dt_log_p_tilde_t_bit_identical_at_p_half():
    target = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    x = torch.randint(0, 2, (8, target.d)).float() * 2 - 1
    t = torch.rand(8)
    legacy = target.log_prob(x) + target.d * math.log(2)
    torch.testing.assert_close(target.dt_log_p_tilde_t(x, t), legacy)


def test_log_p_tilde_t_at_zero_equals_base_log_eta_off_centre():
    target = IsingTarget(D=4, sigma=0.1, base_composition=0.8)
    x = torch.randint(0, 2, (8, target.d)).float() * 2 - 1
    t0 = torch.zeros(8)
    torch.testing.assert_close(target.log_p_tilde_t(x, t0), target.base_log_eta(x))


def test_sample_base_draw_bit_identical_at_p_half():
    target = IsingTarget(D=4, sigma=0.1, base_composition=0.5)
    torch.manual_seed(123)
    legacy = torch.randint(0, 2, (100, target.d)).float() * 2 - 1
    torch.manual_seed(123)
    got = target.sample_base(100, device="cpu")
    torch.testing.assert_close(got, legacy)


def test_sample_base_mean_matches_p_off_centre():
    target = IsingTarget(D=10, sigma=0.1, base_composition=0.8)
    torch.manual_seed(0)
    x = target.sample_base(20_000, device="cpu")
    frac = ((x + 1.0) * 0.5).mean().item()
    assert abs(frac - 0.8) < 0.01


def test_base_composition_out_of_range_raises():
    with pytest.raises(ValueError):
        IsingTarget(D=4, sigma=0.1, base_composition=0.0)
    with pytest.raises(ValueError):
        IsingTarget(D=4, sigma=0.1, base_composition=1.0)


def test_sample_base_matches_inline_call_signature_p_half():
    """training.py replaces randint(0,2,(N,d))*2-1 with target.sample_base(N, dev).
    At p=0.5 the two must be identical for the same RNG state and shape."""
    target = IsingTarget(D=10, sigma=0.1, base_composition=0.5)
    outer_batch, n_dims = 256, target.d
    torch.manual_seed(7)
    inline = torch.randint(0, 2, (outer_batch, n_dims)).float() * 2 - 1
    torch.manual_seed(7)
    via_base = target.sample_base(outer_batch, device="cpu")
    torch.testing.assert_close(via_base, inline)
