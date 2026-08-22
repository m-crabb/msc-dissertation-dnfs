"""Tests for the MLP rate-matrix parameterisation (Stage 1).

These encode "what correct looks like" for the MLPRateMatrix module:
- shape contract (B, d) -> (B, d),
- non-negativity of rates (CTMC requirement),
- non-trivial dependence on both inputs (t and x).

We do NOT test learned behaviour here; that's the job of the training-loop
integration tests later. The point of these tests is to catch architecture
mistakes (wrong output shape, missing softplus, t/x ignored) before training
silently fails.
"""
import pytest
import torch

from discrete_flow_sampler.models.mlp import MLPRateMatrix


@pytest.fixture
def model():
    # d = 16 corresponds to D = 4 lattice (D**2 sites). Small enough to be fast,
    # large enough that bugs that only surface for d > 1 will show up.
    return MLPRateMatrix(d=16, hidden_dim=32, n_layers=2)


def _sample_state(batch: int, d: int) -> torch.Tensor:
    """Draw a random {-1, +1}^d batch."""
    return torch.randint(0, 2, (batch, d)).float() * 2 - 1


def test_forward_shape(model):
    """Output must be (B, d): one rate per site, per batch element."""
    x = _sample_state(8, 16)
    t = torch.rand(8)
    rates = model(x, t)
    assert rates.shape == (8, 16)


def test_forward_nonnegative(model):
    """CTMC rates are non-negative by definition. Softplus on raw scores
    enforces this; if a model forgets to apply it, this catches it."""
    x = _sample_state(8, 16)
    t = torch.rand(8)
    rates = model(x, t)
    assert (rates >= 0).all()


def test_t_dependence(model):
    """Rates should depend on t. If the network ignores t (e.g. the model forgot
    to concatenate t into the input), the two outputs will be bit-equal."""
    x = _sample_state(1, 16)
    t1 = torch.zeros(1)
    t2 = torch.ones(1)
    r1 = model(x, t1)
    r2 = model(x, t2)
    assert not torch.equal(r1, r2)


def test_x_dependence(model):
    """Rates should depend on x. A model that only conditions on t (or that
    feeds zeros for x) would produce identical outputs for opposite states."""
    x1 = torch.ones((1, 16))
    x2 = -torch.ones((1, 16))
    t = torch.tensor([0.5])
    r1 = model(x1, t)
    r2 = model(x2, t)
    assert not torch.equal(r1, r2)


def test_batch_independence(model):
    """Row b of the output should depend only on row b of the input.
    This catches accidental cross-batch leakage (e.g. a BatchNorm with a
    bug, or a reshape that mixes batch into features)."""
    x = _sample_state(4, 16)
    t = torch.rand(4)
    rates_full = model(x, t)
    # Re-run row 0 alone; should match rates_full[0] up to fp tolerance.
    rates_row0 = model(x[:1], t[:1])
    assert torch.allclose(rates_full[0], rates_row0[0], atol=1e-6)
