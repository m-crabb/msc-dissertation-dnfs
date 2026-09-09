"""Exact-field channel: the closed-form Kawasaki energy change as a fixed
additive score, in front of any learned swap head.

    G(i, j | x) = G_head(i, j | x) + gain(t) * sigma * Delta_ij(x),
    sigma * Delta_ij = log p(swap2(x, i, j)) - log p(x)          (t = 1)
                     = sigma * [2 (x_j - x_i)(h_i - h_j) - 2 (x_j - x_i)^2 A_ij],
    h = x A  (neighbour sums),  gain(t) = gain_constant + gain_slope * t.

Under binary swap antisymmetry every score is (x_i - x_j) S_ij(x_{-ij}),
and the exact equilibrium log-ratio is the hole-excluded field difference
sigma (x_i - x_j)(h~_i - h~_j); the -2 diff^2 A_ij term excludes the
swap-invariant i-j bond for adjacent pairs. The 4x4 regression of trained
heads (2026-08-22) put ~half of Var S on this field, so the head only has
to learn the residual (the same move as MDNS's preconditioning for flips).

The channel is the t = 1 source direction sigma * Delta with a learned time
gain, not the log-ratio t * sigma * Delta: with one-way rates relu(+-G) a
nonzero G is transport, and along p_t ~ eta^{1-t} p^t mass must flow toward
lower energy from t = 0 (the KFE source -sigma E + const is nonzero there).

The gain starts at zero so the channel is bit-identical to the base head at
init (tests assert it) and archived cells' init telemetry is unchanged; at
sigma_c |sigma Delta| reaches ~1.8 against a trained mean per-pair rate of
~0.004, so a nonzero init would rescale the rates before the first step.

sigma * Delta is state-antisymmetric (sign flips at the swapped state) but
symmetric in the pair labels. The heads carry index antisymmetry
G[j,i] = -G[i,j] by the mirror (keep i < j, subtract the transpose) and the
loss reads i < j only; the channel gets the same mirror, so the sum is
index-antisymmetric with a zero diagonal. sigma and A are read live from
the target so the sigma-curriculum propagates.
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
        """sigma * Delta_ij for i < j, mirrored to G[j,i] = -G[i,j]: the target's
        all-pairs swap log-ratio at t=1 (target.base_swap_log_ratio)."""
        # Target-supplied closed form: Ising field difference or cluster
        # expansion -beta Delta E_swap (Eq. 3), for binary targets.
        upper = torch.triu(self.target.base_swap_log_ratio(x), diagonal=1)
        return upper - upper.transpose(1, 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gain = self.gain_constant + self.gain_slope * t  # (B,)
        return self.head(x, t) + gain[:, None, None] * self.exact_field(x)


class ExactFieldFlipModel(nn.Module):
    """The same channel for the soft chapter's FLIP process.

        G(i | x) <- G_model(i | x) + gain(t) * Delta_i(x),
        Delta_i  = x_i * [ -4 sigma h_i + 2 lambda (c_null_i - c*) + lambda/d ],
        h = x A  (Ising; an expansion supplies -beta Delta E_i instead),
        c_null_i = c(x) - (x_i + 1)/(2d),
        gain(t) = g0 + g1 t.

    The opt-in amortised correction adds a composition-dependent gain

        gain(t, c) = g0 + g1 t + (c - c0) (h0 + h1 t),

    c0 the target's scalar centre composition: the smallest family that
    expresses the critical specialists' observed composition-dependent
    gains. At c=c0 it is the archived channel; h0=h1=0 at init, so the
    step-zero forward is bit-identical, and the parameters are registered
    only when requested so archived checkpoints still load strictly.

    Delta_i is the exact soft flip log-ratio at t=1 (pinned against brute
    force in tests/test_soft_field_regression.py); written against the
    hole-excluded composition c_null it is odd in x_i, so it lives in the
    leTF's representable set G = -x_i S_i(x). The local-field regression
    put the two closed-form columns at ~95% of every trained lambda=50
    specialist's variance, dominated by the penalty column whose
    lambda^2 Var[delta_P] noise makes soft training fragile.

    Same conventions as the swap channel above: t=1 source direction with a
    learned time gain, gain zero at init, sigma, lambda and c* read live.

    Binary flips only: for S > 2 each destination token has its own
    log-ratio, so the constructor refuses rather than mis-scoring Potts.
    """

    def __init__(
        self,
        model: nn.Module,
        target,
        *,
        composition_conditioned_gain: bool = False,
    ):
        super().__init__()
        if getattr(model, "vocab_size", 2) != 2:
            raise ValueError(
                "ExactFieldFlipModel is derived for binary flips; "
                f"got vocab_size={model.vocab_size}"
            )
        if composition_conditioned_gain and not getattr(
            model, "condition_on_composition", False
        ):
            raise ValueError(
                "composition-conditioned gain requires a model built with "
                "condition_on_composition=True"
            )
        if composition_conditioned_gain and target.target_composition is None:
            raise ValueError(
                "composition-conditioned gain requires a scalar target "
                "composition to use as its centre"
            )
        self.model = model
        self._target = [target]  # list, not attribute: the target is not a Module
        self.composition_conditioned_gain = composition_conditioned_gain
        self.gain_constant = nn.Parameter(torch.zeros(()))
        self.gain_slope = nn.Parameter(torch.zeros(()))
        if composition_conditioned_gain:
            self.composition_gain_constant = nn.Parameter(torch.zeros(()))
            self.composition_gain_slope = nn.Parameter(torch.zeros(()))

    @property
    def target(self):
        return self._target[0]

    @property
    def is_locally_equivariant(self):
        # The trainers dispatch scores-vs-rates on this flag; the channel
        # adds a score, so it must ride the wrapped model's answer.
        return self.model.is_locally_equivariant

    # The trainers read these off the model object (rate diagnostics, the
    # amortised eval path); the wrapper must be transparent to them.
    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def condition_on_composition(self):
        return self.model.condition_on_composition

    def compile(self):
        self.model.compile()

    def exact_field(self, x: Tensor, composition: Tensor | None = None) -> Tensor:
        """Delta_i(x), shape (B, d) — the closed form above, live params.

        composition: per-row target composition (B,), the amortised route;
        None (specialist route) uses the target's scalar c*. In an amortised
        batch the penalty is bound per-row via target.composition_batch, so
        a channel left on the scalar would aim the field at the wrong
        composition (pinned by test_per_row_composition_matches_brute_force).
        """
        target = self.target
        d = x.shape[-1]
        lam = target.composition_penalty_strength
        c_star = (
            target.target_composition
            if composition is None
            else composition.unsqueeze(-1)  # (B, 1)
        )
        # No penalty or c*: use the bare energy log-ratio, avoiding
        # undefined penalty arithmetic when c* is None.
        if lam == 0.0 or c_star is None:
            return target.base_flip_log_ratio(x)
        c_hollow = ((x + 1.0) * 0.5).mean(-1, keepdim=True) - (x + 1.0) / (2.0 * d)
        # Target-supplied energy term: Ising -4 sigma x_i h_i or cluster
        # expansion -beta Delta E_i, for binary targets.
        return target.base_flip_log_ratio(x) + x * (
            2.0 * lam * (c_hollow - c_star) + lam / d
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        if self.composition_conditioned_gain and c is None:
            raise ValueError(
                "composition c must be supplied when the exact-field "
                "composition-conditioned gain is enabled"
            )
        G = self.model(x, t) if c is None else self.model(x, t, c)
        gain = self.gain_constant + self.gain_slope * t  # (B,)
        if self.composition_conditioned_gain:
            centred_c = c - float(self.target.target_composition)
            gain = gain + centred_c * (
                self.composition_gain_constant + self.composition_gain_slope * t
            )
        contribution = gain.unsqueeze(1) * self.exact_field(x, composition=c)
        # Only the flip slot moves; the current token's slot stays exactly
        # zero (the convention G.sum(-1) == flip score relies on).
        flip_slot = (1 - ((x + 1) / 2)).long().unsqueeze(-1)
        return G.scatter_add(-1, flip_slot, contribution.unsqueeze(-1).to(G.dtype))
