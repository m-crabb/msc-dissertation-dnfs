"""Tests for the paired-swap antisymmetric readout (P1.1, design note 2026-06-29 §5)."""
import torch

from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.constraints.swap_readout import swap2, _masked_body

ATOL = 1e-5


def _backbone(d=9, seed=42, hidden_dim=8, n_heads=2, n_layers=2):
    torch.manual_seed(seed)
    m = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads
    )
    m.eval()
    return m


def _state(d=9, seed=1):
    """A (1, d) ±1 state guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (1, d)) * 2 - 1).float()
    if bool((x > 0).all()) or bool((x < 0).all()):
        x[0, 0] *= -1
    return x


def test_swap2_is_involution_and_exchanges():
    x = _state(d=9)
    i, j = 2, 5
    y = swap2(x, i, j)
    assert y[0, i].item() == x[0, j].item()
    assert y[0, j].item() == x[0, i].item()
    # off {i,j} unchanged
    mask = [k for k in range(9) if k not in (i, j)]
    assert torch.equal(y[0, mask], x[0, mask])
    # involution
    assert torch.equal(swap2(y, i, j), x)


def test_masked_body_double_hollow():
    """H from masking anchor i: H[:,j,:] is blind to x_i (masked) and x_j (hollow)."""
    m = _backbone(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = 2, 5
    H = _masked_body(m, x, t, (i,))
    base = H[:, j, :].clone()

    # flip x_i (masked) -> H[:,j,:] unchanged
    x_fi = x.clone(); x_fi[0, i] *= -1
    d_i = (_masked_body(m, x_fi, t, (i,))[:, j, :] - base).abs().max().item()
    # flip x_j (hollow at j) -> H[:,j,:] unchanged
    x_fj = x.clone(); x_fj[0, j] *= -1
    d_j = (_masked_body(m, x_fj, t, (i,))[:, j, :] - base).abs().max().item()

    assert d_i < ATOL, f"H_ij not blind to x_i: {d_i:.2e}"
    assert d_j < ATOL, f"H_ij not hollow in x_j: {d_j:.2e}"
