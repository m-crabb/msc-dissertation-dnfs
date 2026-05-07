"""Tests for the locally equivariant MLP rate-matrix (stages 1-2).

These pin "what correct looks like" for LeMLPRateMatrix:
    1) shape contract (B, D) -> (B, D, S);
    2) output non-trivial — the network depends on inputs (rules out a
       degenerate constant network that would pass every other check
       trivially);
    3) self-slot-zero (the τ = x_i slot must be exactly 0);
    4) local equivariance (Eq. 20):
           G(τ, i | x) = -G(x_i, i | Swap(x, i, τ));
    5) t-dependence;
    6) other-site-x-dependence (rules out a degenerate hollow MLP with
       all-zero off-diagonal weights);
    7) is_locally_equivariant flag.
"""
import pytest
import torch

from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix


@pytest.fixture
def model():
    torch.manual_seed(0)
    return LeMLPRateMatrix(d=8, vocab_size=2, hidden_dim=32, n_summands=3)


def _sample_state(batch: int, d: int) -> torch.Tensor:
    """Random {-1, +1}^d batch — codebase spin convention."""
    return torch.randint(0, 2, (batch, d)).float() * 2 - 1


def test_forward_shape(model):
    """Output must be (B, D, S)."""
    x = _sample_state(4, 8)
    t = torch.rand(4)
    G = model(x, t)
    assert G.shape == (4, 8, 2)


def test_output_nontrivial(model):
    """Output must not be identically zero — guards every other test
    from a stub-only pass."""
    x = _sample_state(4, 8)
    t = torch.rand(4)
    G = model(x, t)
    assert G.abs().max() > 1e-4, "leMLP output is identically (near-)zero"


def test_self_slot_is_zero(model):
    """G(τ=x_i, i | x) must be exactly 0 for every i (scatter check)."""
    x = _sample_state(4, 8)
    t = torch.rand(4)
    x_idx = ((x + 1) / 2).long()
    G = model(x, t)
    self_slot = G.gather(-1, x_idx.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        self_slot, torch.zeros_like(self_slot), atol=0, rtol=0
    )


def test_local_equivariance(model):
    """Eq. (20): G(τ, i | x) = -G(x_i, i | Swap(x, i, τ)) for every i.

    Load-bearing test — this is the property that makes Eq. (10) valid."""
    x = _sample_state(1, 8)
    t = torch.rand(1)
    x_idx = ((x + 1) / 2).long()
    G = model(x, t)
    for i in range(8):
        original_value = x_idx[0, i].item()
        tau = 1 - original_value
        x_swapped = x.clone()
        x_swapped[0, i] = -x[0, i]
        G_swapped = model(x_swapped, t)
        diff = (G[0, i, tau] + G_swapped[0, i, original_value]).abs().item()
        assert diff < 1e-5, (
            f"LE violated at site {i}: "
            f"G(τ={tau}, i={i} | x) + G(x_i={original_value}, i | Swap) = {diff:.3e}"
        )


def test_t_dependence(model):
    """Output must depend on t (sinusoidal-MLP cond_t actually wired in)."""
    x = _sample_state(1, 8)
    t1 = torch.zeros(1)
    t2 = torch.ones(1)
    G1 = model(x, t1)
    G2 = model(x, t2)
    assert not torch.equal(G1, G2)


def test_x_dependence_at_other_sites(model):
    """For sites j ≠ i, G(·, i | x) must depend on x_j — rules out
    a degenerate hollow MLP where all off-diagonal weights are zero."""
    torch.manual_seed(1)
    x_a = _sample_state(1, 8)
    t = torch.rand(1)
    G_a = model(x_a, t)
    for j in range(8):
        x_b = x_a.clone()
        x_b[0, j] = -x_b[0, j]
        G_b = model(x_b, t)
        # Aggregate: across all sites i ≠ j, at least one entry must differ
        # by a meaningful margin. Asserting per-(i, j) is fragile under
        # different seeds; the property being tested is only that the
        # hollow MLP isn't degenerate (all-zero off-diagonal weights).
        other_sites = [i for i in range(8) if i != j]
        max_diff = max(
            (G_a[0, i] - G_b[0, i]).abs().max().item() for i in other_sites
        )
        assert max_diff > 1e-6, (
            f"Flipping x_{j} did not change the output at any other site — "
            "hollow MLP appears to ignore other-site inputs."
        )


def test_is_locally_equivariant_flag(model):
    """Dispatch flag must be True (kolmogorov.loss reads this attribute
    in Task 4)."""
    assert model.is_locally_equivariant is True
