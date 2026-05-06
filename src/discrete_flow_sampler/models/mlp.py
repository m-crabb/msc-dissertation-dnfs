"""MLP rate-matrix parameterisation for Stage 1 of DNFS replication.

Concept (paper Sec. 3, Eq. (4)):
    DNFS learns a CTMC generator R_theta(x, t) so that, when integrated from
    t = 0 (base p_0) to t = 1 (target p_1), the marginals follow the
    annealing path. For binary spins (S = 2) we only need one rate per site
    (the flip rate); the rate matrix collapses to shape (B, D**2).

Architecture (Stage 1 -- intentionally vanilla, will be replaced by LeT in
Stage 3):

    inputs  : concat(x, t) of shape (B, d + 1)
    hidden  : `n_layers` blocks of (Linear -> ReLU), width = `hidden_dim`
    output  : Linear -> (B, d) raw scores
    rates   : softplus(scores), guaranteed >= 0

Why softplus rather than exp:
    Both produce non-negative outputs. Softplus has a well-conditioned
    gradient near zero (slope 1/2) whereas exp can either saturate (large
    negative scores -> 0 with vanishing gradient) or blow up early. Softplus
    is the paper's choice; we follow it.

Why a single MLP over the whole state (not site-wise):
    Stage 1's whole point is to be the simple, naive baseline. We want the
    failure mode (high estimator variance, low ESS at D=10) to be observable
    so Stage 3's LeT improvements have something to compare against.

This is research-bearing -- the user implements the body. The interface,
docstrings, and tests are scaffolding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPRateMatrix(nn.Module):
    """Plain MLP that maps (state, time) -> flip rates per site.

    Args:
        d: number of sites (D**2 for a D x D Ising lattice).
        hidden_dim: width of every hidden layer.
        n_layers: number of (Linear, ReLU) hidden blocks. Must be >= 1.
    """

    def __init__(self, d: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.d = d
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input is the spin state (d entries) concatenated with a scalar time:
        # one extra feature, hence d + 1.
        state_plus_time_dim = d + 1

        # Build the layer stack as: one input projection, (n_layers - 1)
        # hidden blocks, one readout projection. Counting `n_layers` as the
        # number of hidden (Linear, ReLU) blocks matches the paper's wording.
        layers: list[nn.Module] = [
            nn.Linear(state_plus_time_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Readout: raw flip-rate scores. Non-negativity is enforced later by
        # softplus in forward(), not by an activation here -- stacking ReLU on
        # top of softplus would clip the expressive range of the rates.
        layers.append(nn.Linear(hidden_dim, d))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute non-negative flip rates.

        Args:
            x: (B, d) state tensor in {-1, +1}.
            t: (B,) time tensor in [0, 1].

        Returns:
            rates: (B, d) tensor with rates[b, i] = flip rate for site i in
                   batch element b. All entries >= 0.
        """
        # Glue the scalar time onto the end of each state vector, so every
        # batch element gets a (d + 1)-vector input. Unsqueezing turns
        # t : (B,) into (B, 1) so it concatenates along the feature axis.
        state_and_time = torch.cat([x, t.unsqueeze(-1)], dim=-1)  # (B, d + 1)

        # MLP outputs raw real-valued scores per site -- can be negative.
        flip_logits = self.net(state_and_time)  # (B, d)

        # Softplus maps R -> R_{>= 0}, smoothly. Required because the output
        # is interpreted as a CTMC rate (non-negative by definition).
        flip_rates = F.softplus(flip_logits)  # (B, d), >= 0
        return flip_rates
