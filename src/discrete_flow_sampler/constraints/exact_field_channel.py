"""Exact-field channel: the closed-form Kawasaki energy change as a fixed
additive score, in front of any learned swap head (s54, 2026-08-23).

    G(i, j | x) = G_head(i, j | x) + gain(t) * sigma * Delta_ij(x),
    sigma * Delta_ij = log p(swap2(x, i, j)) - log p(x)          (t = 1)
                     = sigma * [2 (x_j - x_i)(h_i - h_j) - 2 (x_j - x_i)^2 A_ij],
    h = x A  (neighbour sums),  gain(t) = gain_constant + gain_slope * t.

Why this channel. Under binary swap antisymmetry every score is
(x_i - x_j) S_ij(x_{-ij}), and the exact equilibrium log-ratio is the
rank-one, linear, hole-excluded field difference sigma (x_i - x_j)(h~_i -
h~_j) -- h~ = the neighbour field at each hole with its partner excluded,
which is what the -2 diff^2 A_ij term does for adjacent pairs (the i-j bond
is swap-invariant and must not be counted). The 4x4 regression of trained
heads (2026-08-22) put ~half of Var S on exactly this field; MDNS's
preconditioning is the same move for the flip process (exact local
conditional as a fixed logit, learned residual). Here the head only has to
learn the residual.

Why not the log-ratio t * sigma * Delta. With one-way rates relu(+-G) a
nonzero G is TRANSPORT, not equilibrium dynamics: along p_t ~ eta^{1-t} p^t
mass must flow toward lower energy from t = 0 onward (the KFE source
-sigma E + const is nonzero at t = 0), so the channel is the source
direction sigma * Delta with a learned time-dependent gain, not the
log-ratio that vanishes at t = 0. The magnitude of the optimal transport
field is the non-local Poisson solution the residual carries.

Why gain starts at ZERO. Bit-identity with the base head at initialisation
(tests assert it), so every archived cell's init telemetry -- rate-clip
fraction, per-pair rate scale -- is unchanged, and the channel is something
the optimiser switches on rather than something that rescales the rates
before the first step (at sigma_c, |sigma Delta| reaches ~1.8 on a lattice
whose trained mean per-pair rate is ~0.004).

Two antisymmetries, and which one the channel has for free. sigma * Delta
is STATE-antisymmetric (its sign flips at the swapped state, the property
the reverse rate needs) but SYMMETRIC in the pair labels -- swapping i and j
is one physical move whichever site is called i. The heads' matrices carry
INDEX antisymmetry G[j,i] = -G[i,j] as a convention imposed by the mirror
(keep i < j, subtract the transpose), and the loss reads i < j only; the
channel is given the same convention by the same mirror, so the sum is
exactly index-antisymmetric with a zero diagonal. sigma and A are
read LIVE from the target so the sigma-curriculum propagates.
"""
import torch
import torch.nn as nn
from torch import Tensor


class ExactFieldSwapHead(nn.Module):
    """Wrap a swap head with the exact-field channel. Exposes `backbone` and
    `compile` so trainers, profilers and row counters see the inner head."""

    def __init__(self, head: nn.Module, target):
        super().__init__()
        self.head = head
        self._target = [target]  # list, not attribute: the target is not a Module
        self.gain_constant = nn.Parameter(torch.zeros(()))
        self.gain_slope = nn.Parameter(torch.zeros(()))

    @property
    def backbone(self):
        return self.head.backbone

    @property
    def target(self):
        return self._target[0]

    def compile(self):
        self.head.compile()

    def exact_field(self, x: Tensor) -> Tensor:
        """sigma * Delta_ij for i < j, mirrored to G[j,i] = -G[i,j]: the batched
        all-pairs form of FixedCompositionIsingTarget.swap_log_ratio at t=1."""
        A = self.target.A
        neighbour_sum = x @ A                                   # (B, d)
        diff = x.unsqueeze(1) - x.unsqueeze(2)                  # x_j - x_i
        field_difference = neighbour_sum.unsqueeze(2) - neighbour_sum.unsqueeze(1)  # h_i - h_j
        delta = 2.0 * diff * field_difference - 2.0 * diff * diff * A
        upper = torch.triu(self.target.sigma * delta, diagonal=1)
        return upper - upper.transpose(1, 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gain = self.gain_constant + self.gain_slope * t         # (B,)
        return self.head(x, t) + gain[:, None, None] * self.exact_field(x)
