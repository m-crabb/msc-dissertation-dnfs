"""Factorised swap head: low-rank bilinear causal factors + hole-subtracted
global context. All-pairs G in ONE body pass with NO per-pair network.

Same readout as every swap head (DNFS Prop. 2 / Eq. (9), pair form):

    G(i, j | x) = < H_ij(x_-{i,j}),  omega_{x_i} - omega_{x_j} >

and the same blindness requirement: H_ij must not depend on the token VALUES
at sites i and j, at any layer -- one globally-mixing layer leaks x_i into
every representation (the two-hop leak, interval_swap_head.py). Where the
interval/masked-attention heads push all d^2 pair rows through a per-pair
readout MLP over concatenated summaries, here the pair context is itself
factorised into per-site pieces, so G assembles as batched matrix products
and the per-pair work drops to O(rank * factor_dim) multiply-adds:

    H_ij = sum_r  a_r(prefix_i, pos_i) * b_r(suffix_j, pos_j)   [bilinear]
         + rho( LN( c(x) - psi_i - psi_j ) )                    [global]
         + tau(t)                                               [time]

* bilinear -- prefix_i / suffix_j are the leTF causal-stream slice objects
  (blind to x_{>=i} / x_{<=j} by causality, so depth is free). The factor
  maps are LINEAR in the normalised streams: the nonlinearity lives in the
  causal stacks, and the rank R bounds the pair-interaction kernel. That
  bound is the expressivity price of the factorisation, and it is measured,
  not assumed (enumeration-gate at the smallest lattice before any scale
  run). The bilinear term alone cannot see the open interval (i, j) --
  prefix stops before i, suffix starts after j -- and the test suite pins
  that hole rather than hiding it.
* global -- psi_k = MLP(emb(x_k) ++ pos_k) is strictly per-site; the sum
  c = sum_k psi_k is a whole-lattice summary, and subtracting psi_i + psi_j
  removes the ONLY terms that touch the holes, so blindness is exact in
  real arithmetic (fp leaves an ~ulp cancellation residue, exactly as the
  interval head's prefix-sum band; suite bar 1e-5). This is the term that
  covers the interval interior. Its depth cap is structural: psi must stay
  per-site because any cross-site mixing BEFORE the subtraction
  re-introduces the leak; depth AFTER aggregation (rho) is safe. The
  LayerNorm before rho is load-bearing for scale: |c| grows linearly in d,
  and without the norm rho saturates at exactly the lattice sizes this head
  exists to reach.
* time -- a projected time embedding added to every pair's context: the
  additive analogue of the interval head's "context_norm(H) + time" line.
  There is deliberately NO norm over the assembled H: a post-sum LayerNorm
  is nonlinear across terms and would force H to be materialised, defeating
  the factorisation; scale control lives per-term instead.

forward assembles G without materialising the (B, d, d, f) context: the
readout distributes through the bilinear term,

    <a_i * b_j, w_i - w_j> = <a_i * w_i, b_j> - <a_i, b_j * w_j>,

so two batched (d x Rf)(Rf x d) matmuls give every pair at once, w being the
omega table projected to factor width. H is defined on i < j and mirrored
down (H_ji := H_ij, the interval head's label-symmetry convention), so
forward computes the upper triangle of the raw scores and mirrors
G[j,i] = -G[i,j] as an identity. The readable H-materialising path is kept
as `compute_pair_context`: the reference the tests hold forward to, and the
object the blindness probes flip spins at.

Rejected alternatives, for the record: full-context site features combined
into pairs break blindness at layer one, and repairing them by explicit
antisymmetrisation costs one swapped forward per ordered pair -- that is
`swap_readout.antisymmetrise`, the O(d^2)-pass test oracle, not a head. A
per-pair MLP over factorised summaries is the interval head; its d^2 MLP
rows are the cost this head removes.

The final contractions run inside an autocast-disabled fp32 block: einsum
and matmul are on the autocast lower-precision list, and G feeds fp32-only
diagnostics downstream -- the same dtype contract the interval and mask-one
heads keep through their mul+sum readouts.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.constraints.interval_swap_head import (
    causal_stream_summaries,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix


class FactorisedSwapHead(nn.Module):
    """One-pass doubly-hollow swap head with factorised pair assembly.

    Drop-in `head_kind` (forward: (B, d) state, (B,) time -> (B, d, d) pair
    scores), same contract as the heads in swap_readout.py.

    Args:
        backbone: the leTF rate model; reused for token/time embedders, the
            fwd/bwd causal stacks and the omega table. attention_readout is
            deliberately unused -- this head replaces it.
        bilinear_rank: number of rank-1 terms R in the bilinear pair kernel.
            The head's expressivity knob: R = d recovers (in principle) an
            arbitrary kernel over the causal features, small R is the bet
            the gate prices.
        factor_dim: width f of each factor vector and of the projected
            omega readout. The readout contracts in this width, not in the
            backbone's hidden width.
        global_feature_dim: channels of the per-site global features psi.
        position_dim: width of the head-owned site-position embedding
            (positions may enter H freely; blindness constrains only token
            values).
        use_bilinear / use_global: ablation switches. At least one term must
            be on -- a head with both off would score every pair from time
            and positions alone, which trains to nothing informative;
            reject at construction rather than at first flat loss.
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        bilinear_rank: int = 8,
        factor_dim: int = 32,
        global_feature_dim: int = 16,
        position_dim: int = 16,
        use_bilinear: bool = True,
        use_global: bool = True,
    ):
        super().__init__()
        if not (use_bilinear or use_global):
            raise ValueError(
                "FactorisedSwapHead needs at least one of use_bilinear / "
                "use_global: with both off, pair scores depend only on time "
                "and positions."
            )
        self.backbone = backbone
        self.d = backbone.d
        self.bilinear_rank = bilinear_rank
        self.factor_dim = factor_dim
        self.use_bilinear = use_bilinear
        self.use_global = use_global
        hidden = backbone.hidden_dim

        self.site_position_embedding = nn.Embedding(self.d, position_dim)
        # bias=False: a constant offset in the projected omega cancels in the
        # readout's omega-difference anyway, so the parameter would be dead.
        self.omega_projection = nn.Linear(hidden, factor_dim, bias=False)
        self.time_projection = nn.Linear(hidden, factor_dim)

        if use_bilinear:
            self.prefix_norm = nn.LayerNorm(hidden)
            self.suffix_norm = nn.LayerNorm(hidden)
            self.prefix_factors = nn.Linear(
                hidden + position_dim, bilinear_rank * factor_dim
            )
            self.suffix_factors = nn.Linear(
                hidden + position_dim, bilinear_rank * factor_dim
            )
        if use_global:
            self.global_site_features = nn.Sequential(
                nn.Linear(hidden + position_dim, global_feature_dim),
                nn.GELU(),
                nn.Linear(global_feature_dim, global_feature_dim),
            )
            self.global_context_norm = nn.LayerNorm(global_feature_dim)
            self.global_context_readout = nn.Sequential(
                nn.Linear(global_feature_dim, global_feature_dim),
                nn.GELU(),
                nn.Linear(global_feature_dim, factor_dim),
            )

    def _site_positions(self, x: Tensor) -> Tensor:
        """(B, d, position_dim) broadcast of the site-position embedding."""
        pos = self.site_position_embedding(
            torch.arange(self.d, device=x.device)
        )
        return pos.unsqueeze(0).expand(x.shape[0], -1, -1)

    def _factor_tensors(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Bilinear factors a, b, each (B, d, R, f).

        a[:, i] may depend only on {t, x_{<i}, i}; b[:, j] only on
        {t, x_{>j}, j} -- blindness by causality, inherited from the
        stream summaries, preserved because the factor maps are per-site.
        """
        prefix, suffix = causal_stream_summaries(self.backbone, x, t)
        positions = self._site_positions(x)
        a = self.prefix_factors(
            torch.cat([self.prefix_norm(prefix), positions], dim=-1)
        )
        b = self.suffix_factors(
            torch.cat([self.suffix_norm(suffix), positions], dim=-1)
        )
        shape = (x.shape[0], self.d, self.bilinear_rank, self.factor_dim)
        return a.view(shape), b.view(shape)

    def _global_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """rho(LN(c - psi_i - psi_j)) for every pair, (B, d, d, f).

        The subtraction removes the only terms of c that touch the holes,
        so entry [:, i, j] is blind to x_i and x_j exactly (up to the fp
        cancellation residue). Symmetric in (i, j) by construction, so it
        already respects the H_ji := H_ij mirror. `t` never enters: like
        the band heads, token statistics are time-free and time arrives
        through the dedicated time term.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        token_embedding = self.backbone.token_embedder(x_idx)   # (B, d, h)
        psi = self.global_site_features(
            torch.cat([token_embedding, self._site_positions(x)], dim=-1)
        )                                                       # (B, d, Fg)
        total = psi.sum(dim=1)                                  # (B, Fg)
        hole_subtracted = (
            total.view(x.shape[0], 1, 1, -1)
            - psi.unsqueeze(2)                                  # remove psi_i
            - psi.unsqueeze(1)                                  # remove psi_j
        )
        return self.global_context_readout(
            self.global_context_norm(hole_subtracted)
        )

    def compute_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """Reference path: materialise H, (B, d, d, f), mirrored to i > j.

        Exists so the blindness tests can probe H directly (strictly
        stronger than probing G's antisymmetry) and as the oracle `forward`
        must match. Not the efficient path -- forward never builds this
        tensor.
        """
        batch, d = x.shape
        H = x.new_zeros(batch, d, d, self.factor_dim)
        if self.use_bilinear:
            a, b = self._factor_tensors(x, t)
            H = H + torch.einsum("birf,bjrf->bijf", a, b)
        if self.use_global:
            H = H + self._global_pair_context(x, t)
        tau = self.time_projection(self.backbone.time_embedder(t))
        H = H + tau.view(batch, 1, 1, self.factor_dim)
        upper = torch.triu(
            torch.ones(d, d, dtype=torch.bool, device=x.device)
        ).view(1, d, d, 1)
        return torch.where(upper, H, H.transpose(1, 2))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), assembled without materialising H.

        Upper triangle first (the i < j definition), then the mirror
        G[j,i] = -G[i,j] applied as an identity, so index antisymmetry is
        exact by construction and the diagonal is exactly zero.
        """
        batch, d = x.shape
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)                      # (B, d, h)
        if self.use_bilinear:
            a, b = self._factor_tensors(x, t)
        if self.use_global:
            global_context = self._global_pair_context(x, t)    # (B, d, d, f)
        tau = self.time_projection(self.backbone.time_embedder(t))

        # fp32 contract on G (see module docstring): everything above may run
        # under autocast; the contractions below must not.
        with torch.autocast(device_type=x.device.type, enabled=False):
            omega_factor = self.omega_projection(omega.float()) # (B, d, f)
            scores = x.new_zeros(batch, d, d, dtype=torch.float32)
            if self.use_bilinear:
                a32, b32 = a.float(), b.float()
                omega_ranked = omega_factor.unsqueeze(2)        # (B, d, 1, f)
                scores = scores + torch.einsum(
                    "birf,bjrf->bij", a32 * omega_ranked, b32
                )
                scores = scores - torch.einsum(
                    "birf,bjrf->bij", a32, b32 * omega_ranked
                )
            if self.use_global:
                token_difference = omega_factor.unsqueeze(2) - omega_factor.unsqueeze(1)
                scores = scores + (global_context.float() * token_difference).sum(-1)
            time_alignment = (omega_factor * tau.float().unsqueeze(1)).sum(-1)
            scores = scores + time_alignment.unsqueeze(2) - time_alignment.unsqueeze(1)
            upper = torch.triu(scores, diagonal=1)
            return upper - upper.transpose(1, 2)
