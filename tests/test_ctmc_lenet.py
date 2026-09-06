"""Tests for the LE branch of compute_xi_t.

The LE form must agree with the non-LE explicit-rate computation when both
are evaluated on the same underlying G — i.e. the LE shortcut is just an
algebraic rearrangement of the general Eq. 8 form, not a separate
approximation.
"""

import torch

from discrete_flow_sampler.samplers.ctmc import compute_xi_t
from discrete_flow_sampler.targets.ising import IsingTarget


class FixedGModel:
    """Locally equivariant 'rate matrix' that ignores its inputs and
    returns a fixed pre-defined G tensor. Used to make compute_xi_t
    deterministic in tests without training a leMLP."""

    is_locally_equivariant: bool = True
    vocab_size: int = 2

    def __init__(self, G: torch.Tensor):
        # G: (B, D, S). Caller is responsible for zeroing the τ=x_i slot
        # — we don't re-zero here so the test can spot-check.
        self.G = G

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.G


def test_compute_xi_t_le_branch_shape():
    """LE branch returns (B,) — same shape as the non-LE branch."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    B, D, S = 3, 4, 2
    G = torch.randn(B, D, S) * 0.1
    x = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]
    )
    x_idx = ((x + 1) / 2).long()
    G.scatter_(-1, x_idx.unsqueeze(-1), 0.0)
    model = FixedGModel(G)
    t = torch.tensor([0.3, 0.5, 0.7])

    xi = compute_xi_t(x, t, model, target)
    assert xi.shape == (B,)


def test_compute_xi_t_le_matches_explicit_rate_form():
    """LE shortcut must equal the explicit-rate computation on the same G.

    Build R_forward[i] = [G(y_i, i | x)]_+ and R_reverse[i] =
    [-G(y_i, i | x)]_+ by hand for the binary case, then verify that ξ_t
    computed by the LE-branch dispatch agrees with the manual computation.
    """
    torch.manual_seed(1)
    target = IsingTarget(D=2, sigma=0.1)
    B, D, S = 2, 4, 2
    G = torch.randn(B, D, S) * 0.5
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    x_idx = ((x + 1) / 2).long()
    G.scatter_(-1, x_idx.unsqueeze(-1), 0.0)
    model = FixedGModel(G)
    t = torch.tensor([0.4, 0.6])

    xi_le = compute_xi_t(x, t, model, target)

    # Hand-compute. For binary, the only non-self τ at site i is 1 - x_i_idx.
    flip_idx = (1 - x_idx).unsqueeze(-1)  # (B, D, 1)
    G_at_flip = G.gather(-1, flip_idx).squeeze(-1)  # (B, D)
    G_plus = torch.relu(G_at_flip)
    neg_G_plus = torch.relu(-G_at_flip)

    flip_signs = 1.0 - 2.0 * torch.eye(D, dtype=x.dtype)
    flip_neighbours = x.unsqueeze(1) * flip_signs.unsqueeze(0)  # (B, D, D)
    flat = flip_neighbours.reshape(B * D, D)
    t_per = t.repeat_interleave(D)
    log_p_flip = target.log_p_tilde_t(flat, t_per).reshape(B, D)
    log_p_x = target.log_p_tilde_t(x, t)
    neighbour_ratio = (log_p_flip - log_p_x.unsqueeze(-1)).exp()

    outflow_sum = G_plus.sum(-1)
    inflow_sum = (neg_G_plus * neighbour_ratio).sum(-1)
    xi_explicit = target.dt_log_p_tilde_t(x, t) + outflow_sum - inflow_sum

    torch.testing.assert_close(xi_le, xi_explicit, atol=1e-6, rtol=1e-5)
