"""Three-interval (leave-two-out) swap head: all-pairs H_ij in ONE body pass.

Direction (b) of the pair-equivariant spike (design walkthrough 2026-07-07;
literature scout: docs/findings/2026-07-07-pair-equivariant-readout-literature-
scout.md). The readout is unchanged from swap_readout.py (the pair form of
DNFS Prop. 2 / Eq. (9)):

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

Exact state-swap antisymmetry G(i,j|x) = -G(i,j|Swap2(x,i,j)) needs exactly
one property of the body: H_ij must be BLIND to the token values x_i and x_j
(the omega-difference flips sign under the swap; H must not move). Blindness
is strictly stronger than the antisymmetry it buys -- H_ij must be invariant
under ANY change to x_i or x_j, not just their exchange -- and the tests
falsify the stronger property directly.

Why blindness must hold at every layer (the two-hop leak): one attention
layer mixes x_i into every token's representation, so masking the i->j edge
only at a readout layer still leaks via x_i -> token k (layer 1) -> H_ij
(layer 2). The mask-one head buys layer-wise blindness with d anchor passes
(~99.5% of measured runtime). This head buys it STRUCTURALLY in one pass, by
the interval decomposition the two holes induce:

    [ x_0 .. x_{i-1} ]  x_i  ( x_{i+1} .. x_{j-1} )  x_j  [ x_{j+1} .. ]
       prefix P_i       HOLE       band  M_ij        HOLE    suffix S_j

* P and S are the leTF slice-trick objects and already exist in the
  backbone: with the cond_t prepend, the inclusive-causal fwd state at slot
  i depends only on x_{<i}, and the (un-flipped) bwd state at slot j+1
  depends only on x_{>j}. Deep and fully mixed within their interval, blind
  by causality -- and causality composes, so depth is safe.
* The band is the genuinely new object: an interval summary open at BOTH
  ends, which no causal sweep produces. Any feature shared across pairs
  must be blind for every pair that reads it, so band features cannot come
  from a globally-mixed deep stack -- they are LOCAL (unary and fixed-offset
  pair statistics of raw token embeddings), aggregated per pair with
  index-structural exclusion of any term touching a hole site.
* Cross-interval mixing (prefix <-> band <-> suffix) happens only in the
  per-pair readout MLP. This is the direction's declared capacity gamble;
  the D=4 gate (kill criterion K2) prices it.

Efficiency contract (R3): O(d) precompute (causal stacks, feature prefix
sums), O(1) assembly per pair, so all O(d^2) contexts cost one body pass
plus a batched per-pair MLP -- no d-anchor multiplier.

Numerical-exactness note (fp caveat to K1 "exact antisymmetry at init"):
prefix-sum band assembly (sum[i+1..j-1] = prefix[j-1] - prefix[i]) is blind
EXACTLY in real arithmetic, but both prefix sums contain the hole terms and
cancel them by subtraction, leaving an fp residue ~ ulp * |prefix| that
does depend on x_i. This is linear-scale cancellation (bounded features,
benign -- nothing like direction (c)'s exp-scale softmax denominators); the
test bar is the suite's ATOL=1e-5. If bit-exact blindness is ever needed,
assemble the band from hole-free partial sums instead (block decomposition:
whole blocks precomputed + O(sqrt d) boundary terms recomputed per pair --
every addend is then hole-free and blindness is structural, like the
mask-one head's).

The head is label-SYMMETRIC by construction (H_ji := H_ij, so the full
matrix is index-antisymmetric: G[j,i] = -G[i,j]); the downstream i<j
ordering convention is unaffected. The backbone's attention_readout is
deliberately unused -- this head replaces it.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.letf import LeTFRateMatrix


class IntervalSwapHead(nn.Module):
    """One-pass doubly-hollow swap head via three-interval assembly.

    Drop-in `head_kind` (forward: (B, d) state, (B,) time -> (B, d, d) pair
    scores), same contract as the heads in swap_readout.py.

    Args:
        backbone: the leTF rate model; this head reuses its token/time
            embedders and fwd/bwd causal stacks (P and S streams) and its
            omega table for the readout. attention_readout is unused.
        pair_offsets: sequence offsets delta for the band's pairwise-local
            features u^delta_k = f(x_k, x_{k+delta}). For a flattened D x D
            lattice pass (1, D): offset 1 is row adjacency, offset D column
            adjacency -- the interactions the Ising energy is built from.
            Unary features are always included.
        band_feature_dim: channels per band-feature family (unary + one per
            offset), concatenated into the pair readout input.
        position_dim: size of the head-owned site-position embedding fed to
            the pair readout (H_ij MAY depend on the positions i, j --
            blindness constrains only the token VALUES there).
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        pair_offsets: tuple[int, ...] = (1,),
        band_feature_dim: int = 16,
        position_dim: int = 16,
    ):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d
        self.pair_offsets = tuple(pair_offsets)
        hidden = backbone.hidden_dim

        def _feature_mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, band_feature_dim),
                nn.GELU(),
                nn.Linear(band_feature_dim, band_feature_dim),
            )

        self.band_unary_features = _feature_mlp(hidden)
        self.band_pair_features = nn.ModuleList(
            _feature_mlp(2 * hidden) for _ in self.pair_offsets
        )
        self.pair_position_embedding = nn.Embedding(self.d, position_dim)

        band_dim = band_feature_dim * (1 + len(self.pair_offsets))
        readout_in = 2 * hidden + band_dim + 2 * position_dim
        self.pair_readout = nn.Sequential(
            nn.Linear(readout_in, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        # Mirrors the letf readout's closing "output_norm(H) + time" line.
        self.context_norm = nn.LayerNorm(hidden)

    def causal_summaries(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Prefix/suffix interval summaries from the backbone's causal stacks.

        Returns (prefix_summary, suffix_summary), each (B, d, h):

            prefix_summary[:, i, :] depends ONLY on {t, x_0..x_{i-1}}  (x_{<i})
            suffix_summary[:, j, :] depends ONLY on {t, x_{j+1}..x_{d-1}} (x_{>j})

        These are the leTF slice-trick objects read one slot short of the
        hole. With cond_t prepended, the inclusive-causal fwd stack output
        at slot m depends on slots <= m, i.e. cond_t plus tokens 0..m-1, so
        slot i is the deepest state that has never seen x_i (i=0 gives the
        cond_t-only state -- the empty-prefix summary, no edge case needed).
        The bwd stack runs on the flipped sequence with the same prepend;
        after flipping its output back, slot k depends on x_{>=k}, so slot
        j+1 is the deepest state blind to x_{<=j} (slot d = empty suffix).

        The classic failure here is an off-by-one in either slice -- the
        blindness tests flip x at the boundary sites specifically to catch
        it. Follow `swap_readout._masked_body` for the exact stack-call
        pattern (embed -> prepend cond_t -> fwd_stack / flipped bwd_stack).
        """
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        x_emb = m.token_embedder(x_idx)                       # (B, d, h)
        cond_t = m.time_embedder(t).unsqueeze(1)              # (B, 1, h)
        fwd_x = m.fwd_stack(torch.cat([cond_t, x_emb], dim=1))            # (B, d+1, h)
        bwd_x = m.bwd_stack(torch.cat([cond_t, x_emb.flip(1)], dim=1)).flip(1)
        # fwd slot i has seen {cond_t, x_0..x_{i-1}}; flipped bwd slot k has
        # seen {cond_t, x_k..x_{d-1}}, so slot j+1 is blind to x_{<=j}.
        prefix_summary = fwd_x[:, : self.d, :]
        suffix_summary = bwd_x[:, 1:, :]
        return prefix_summary, suffix_summary

    def band_summaries(self, x: Tensor, t: Tensor) -> Tensor:
        """All-pairs middle-band statistics, (B, d, d, F); valid for i < j.

        F = band_feature_dim * (1 + len(pair_offsets)). Entry [:, i, j, :]
        aggregates LOCAL features over the open interval (i, j), containing
        no term that touches site i or site j:

            unary:            sum_{k = i+1 .. j-1}        v_k,
                              v_k = band_unary_features(emb(x_k))
            offset delta:     sum_{k = i+1 .. j-1-delta}  u^delta_k,
                              u^delta_k
                                = band_pair_features(emb(x_k) ++ emb(x_{k+delta}))

        The offset-delta index range is the straddle exclusion: u^delta_k
        touches sites (k, k+delta), so band-interior terms need k > i and
        k + delta < j. Exclusion is decided by INDEX arithmetic only --
        blindness cannot depend on the values being excluded.

        Efficiency: build each family's prefix-sum once, O(d); every (i, j)
        entry is then one subtraction, O(1)/pair. Empty ranges (adjacent
        pairs, j - i <= delta) must yield exact zeros. Only the i < j
        triangle is consumed (compute_pair_context symmetrises); the lower
        triangle's content is unspecified.

        fp caveat: subtractive assembly leaves ~ulp hole residue (module
        docstring); the hole-free block-decomposition variant restores
        bit-exact blindness if ATOL ever bites. (Sharper: x_j never enters
        these cumsums at all -- every family stops short of j -- so the
        residue is on the x_i side only.)

        `t` is unused by design: band features are token statistics; time
        dependence enters through the pair readout's summaries and time line.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)             # (B, d, h)
        d = self.d
        site_i = torch.arange(d, device=x.device).view(d, 1)  # broadcast over j
        site_j = torch.arange(d, device=x.device).view(1, d)  # broadcast over i

        def cumsum_with_zero(features: Tensor) -> Tensor:
            zero = features.new_zeros(features.shape[0], 1, features.shape[-1])
            return torch.cat([zero, features.cumsum(dim=1)], dim=1)

        families = []
        # Unary: sum_{k in [i+1, j-1]} v_k = c[j] - c[i+1]; empty when j-i < 2.
        unary_prefix = cumsum_with_zero(self.band_unary_features(emb))
        unary_sum = unary_prefix[:, site_j] - unary_prefix[:, site_i + 1]
        families.append(
            torch.where((site_j - site_i >= 2).view(1, d, d, 1), unary_sum, 0.0)
        )
        # Offset delta: u_k covers (k, k+delta), band-interior needs
        # k in [i+1, j-1-delta]  =>  cu[j-delta] - cu[i+1]; empty (and the
        # gather indices clamped-garbage, hence the mask) when j-i < delta+2.
        for delta, feature_mlp in zip(self.pair_offsets, self.band_pair_features):
            n_terms = d - delta
            pair_terms = feature_mlp(
                torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)
            )
            pair_prefix = cumsum_with_zero(pair_terms)
            pair_sum = (
                pair_prefix[:, (site_j - delta).clamp(min=0, max=n_terms)]
                - pair_prefix[:, (site_i + 1).clamp(max=n_terms)]
            )
            families.append(
                torch.where(
                    (site_j - site_i >= delta + 2).view(1, d, d, 1), pair_sum, 0.0
                )
            )
        return torch.cat(families, dim=-1)                    # (B, d, d, F)

    def compute_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """Assemble H, (B, d, d, h): the doubly-blind context for every pair.

        For i < j:

            H_ij = context_norm(pair_readout(
                       prefix[:, i] ++ suffix[:, j] ++ band[:, i, j]
                       ++ pos_emb(i) ++ pos_emb(j)
                   )) + time_embedder(t)

        then H[:, j, i] := H[:, i, j] (label symmetry). All O(d^2) rows go
        through pair_readout as one batched MLP over the trailing dim --
        broadcast prefix over j, suffix over i, positions over batch.

        Exposed separately from forward so the falsification tests can probe
        blindness on H directly (flip x_i / x_j / both: H_ij must not move)
        -- a strictly stronger check than G's antisymmetry, which a
        symmetric leak (H depending on x_i + x_j, say) would survive.
        """
        prefix_summary, suffix_summary = self.causal_summaries(x, t)
        band = self.band_summaries(x, t)                      # (B, d, d, F)
        batch, d, hidden = prefix_summary.shape
        position = self.pair_position_embedding(torch.arange(d, device=x.device))

        H = self.pair_readout(
            torch.cat(
                [
                    prefix_summary.unsqueeze(2).expand(batch, d, d, hidden),
                    suffix_summary.unsqueeze(1).expand(batch, d, d, hidden),
                    band,
                    position.view(1, d, 1, -1).expand(batch, d, d, -1),
                    position.view(1, 1, d, -1).expand(batch, d, d, -1),
                ],
                dim=-1,
            )
        )
        H = self.context_norm(H) + self.backbone.time_embedder(t).view(
            batch, 1, 1, hidden
        )
        # Label symmetry: the i<j triangle is the definition, mirror it down.
        upper = torch.triu(
            torch.ones(d, d, dtype=torch.bool, device=x.device)
        ).view(1, d, d, 1)
        return torch.where(upper, H, H.transpose(1, 2))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), via the swap readout.

            G[:, i, j] = < H_ij, omega_{x_i} - omega_{x_j} >

        (einsum over the hidden dim against the omega difference, exactly as
        the swap_readout.py heads). The diagonal and same-spin pairs vanish
        for free (zero token difference); index-antisymmetry G[j,i] = -G[i,j]
        follows from H's label symmetry; state-swap antisymmetry follows
        from H's value-blindness -- the property the whole head exists to
        provide, and the only one training never touches.
        """
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)                    # (B, d, h)
        token_difference = omega.unsqueeze(2) - omega.unsqueeze(1)
        H = self.compute_pair_context(x, t)
        # mul+sum, NOT einsum: einsum is on the autocast lower-precision
        # list, so under the Tier-2 eval_autocast_bf16 block it would emit
        # bf16 G (crashing the fp32-only quantile rate diagnostic and
        # departing from the dtype path Tier-2 was validated on); the
        # mask-one readout keeps G fp32 the same way.
        return (token_difference * H).sum(-1)
