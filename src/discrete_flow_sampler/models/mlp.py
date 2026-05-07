"""MLP rate-matrix parameterisation — Stage 0 broken-baseline only.

Concept (paper Sec. 3, Eq. (4)):
    DNFS learns a CTMC generator R_theta(x, t) so that, when integrated from
    t = 0 (base p_0) to t = 1 (target p_1), the marginals follow the
    annealing path. For binary spins (S = 2) the original Stage 1 design
    naïvely emitted one rate per site (the flip rate); the rate matrix
    collapsed to shape (B, D**2).

Status (post-2026-05-07 redo):
    Stages 1 and 2 now use `LeMLPRateMatrix` (`models/lemlp.py`) — the
    paper's Eq. (10) loss only makes sense over the locally equivariant /
    one-way family that leMLP parameterises. This module is retained for
    `stage_0_*` configs only, where it serves as a *broken baseline* whose
    purpose is to motivate LE empirically by direct comparison against the
    stage_1 leMLP run.

    `is_locally_equivariant = False` is what routes `stage_0_*` runs to
    the Eq. (7) `residual_general` path in `samplers/kolmogorov.py`.

Architecture (intentionally vanilla):

    inputs  : concat(x, t) of shape (B, d + 1)
    hidden  : `n_layers` blocks of (Linear -> ReLU), width = `hidden_dim`
    output  : Linear -> (B, d) raw scores
    rates   : softplus(scores), guaranteed >= 0

Why softplus rather than exp:
    Both produce non-negative outputs. Softplus has a well-conditioned
    gradient near zero (slope 1/2) whereas exp can either saturate (large
    negative scores -> 0 with vanishing gradient) or blow up early. Softplus
    was the paper's choice for the original family.
"""
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

    is_locally_equivariant: bool = False

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
