"""MLP rate-matrix parameterisation — Stage 0 high-variance baseline.

Concept (paper Sec. 3, Eq. (4)):
    DNFS learns a CTMC generator R_theta(x, t) so that, when integrated from
    t = 0 (base p_0) to t = 1 (target p_1), the marginals follow the
    annealing path. For binary spins (S = 2) the original Stage 1 design
    naïvely emitted one rate per site (the flip rate); the rate matrix
    collapsed to shape (B, D**2).

Baseline role:
    Stages 1 and 2 use `LeMLPRateMatrix` (`models/lemlp.py`). This
    module is retained for `stage_0_*` configs as a *high-variance*
    baseline.

    Eq. (7) (`residual_general` in `samplers/kolmogorov.py`) is a valid
    loss for any single-site-flip parameterisation — it does NOT require
    local equivariance. Per Zijing, the empirical variance of Eq. (7) is
    intractable at lattice scale even under the paper's control-variate
    estimator (Eq. 8); the pre-redo stage_2 d=10 R≡0 collapse corroborates
    this. Eq. (10) (`residual_lenet`) is the LE specialisation that uses
    Prop. 1 + Eq. (20) to fold the reverse rate into the same forward
    tensor — *that* is the path requiring `is_locally_equivariant = True`.

    `MLPRateMatrix.is_locally_equivariant = False` routes stage_0 runs to
    Eq. (7). The stage_0 → stage_1 → stage_2 ladder thus demonstrates two
    stacking variance-reduction strategies: architectural (Eq. 10 via
    leMLP) and estimator (control variates).

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

        state_plus_time_dim = d + 1

        layers: list[nn.Module] = [
            nn.Linear(state_plus_time_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Keep scores unrestricted before softplus so rates can approach zero.
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
        state_and_time = torch.cat([x, t.unsqueeze(-1)], dim=-1)  # (B, d + 1)
        flip_logits = self.net(state_and_time)  # (B, d)
        flip_rates = F.softplus(flip_logits)  # (B, d), >= 0
        return flip_rates
