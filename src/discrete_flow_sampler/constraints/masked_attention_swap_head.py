"""Exclusion-mask band-attention swap head: direction (a), the reported head.

Decision record (2026-07-07, explainability-first comparison): direction (b)
(`interval_swap_head.py`, prefix-sum band) passed all three spike kill
criteria first, but its band assembly cancels the hole terms by SUBTRACTION,
leaving an fp residue that needs a numerics caveat wherever the head is
claimed exact. This head keeps everything else -- the three-interval
decomposition, the causal P/S streams, the band feature families, the
per-pair readout, the omega-difference readout -- and swaps only the band
AGGREGATOR: masked attention over the same visible set the interval head
sums. Exclusion happens BEFORE the softmax, so a hole term never enters any
computed quantity and blindness is bit-exact, not exact-up-to-ulp. The
interval head stays in the tree as the O(1)-per-pair reserve; the
falsification suite is shared in structure (test_masked_attention_swap_head
mirrors test_interval_swap_head with the blindness bar tightened to
equality).

Same readout as swap_readout.py / interval_swap_head.py (DNFS Prop. 2 /
Eq. (9), pair form):

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

and the same blindness requirement on the body: H_ij must not depend on the
token VALUES at i and j (see interval_swap_head.py for why blindness must
hold at every layer -- the two-hop leak -- and why band content must
therefore be shallow/local in BOTH designs).

How the band works here, per feature family:

    * the family's per-term features (unary v_k, offset-delta u^delta_k) are
      unchanged from the interval head -- same MLPs, same meaning;
    * each pair (i, j) forms a QUERY from the two position embeddings (no
      token content: queries must be blind for every pair, and positions
      always are);
    * KEYS are term features plus the term's position embedding, so the
      pooling weights can address the interior both by content ("a domain
      wall") and by position ("the site next to the hole") -- the
      position-addressability the fixed-weight interval sum lacks;
    * visibility is the SAME index-arithmetic set as the interval head:
      unary term k needs i < k < j; offset-delta term k (touching sites k
      and k+delta) needs k > i and k + delta < j (straddle exclusion);
    * masked softmax pools the visible features. Masked scores are set to
      -1e9 (finite, not -inf): exp(-1e9 - max) underflows to EXACTLY +0.0
      in fp32, so an excluded term's weight -- and hence its contribution
      to numerator and denominator -- is exactly zero regardless of the
      token values at the holes. That is the bit-exactness claim, and the
      finite fill means a fully-masked row (empty interval: adjacent pairs,
      or j - i < delta + 2) yields a FINITE uniform softmax instead of NaN;
      its output is then overwritten with exact 0.0 by index mask, matching
      the interval head's empty-band convention, and its gradient
      contribution is zeroed by the same mask -- no NaN in forward or
      backward, with no special-case branch.

Cost contract: one backbone pass (P/S) + one masked attention of d^2 pair
queries over O(d) terms per family -- O(d) work per pair where the interval
head pays O(1), but the O(d)-backbone-pass anchor multiplier (the ~99.5%
term the mask-one head pays) is equally dead. The (B, d^2, n_terms) score
tensor is the price: fine through d = 64 unchunked; d = 256 wants pair
chunking (deferred to the D=16 repricing task, where the eval-path chunking
already exists).

Time is deliberately absent from the band, as in the interval head: band
features are token statistics; time enters H through the pair readout's
summaries and time line.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix

# Finite score for excluded terms: far enough below any live score that
# exp(fill - max) underflows to exactly +0.0 (fp32 underflows near -87), yet
# finite so fully-masked rows softmax to a discarded uniform, never NaN.
EXCLUDED_SCORE_FILL = -1e9


class MaskedAttentionSwapHead(IntervalSwapHead):
    """Doubly-hollow swap head with an exclusion-mask attention band.

    Subclass of IntervalSwapHead so the relationship is literal in code:
    `causal_summaries`, `compute_pair_context` and `forward` are inherited
    unchanged; only `band_summaries` -- the aggregator -- is overridden.
    Same (B, d, d, F) band contract, so the inherited pair readout applies.

    Args (beyond IntervalSwapHead's):
        attention_dim: query/key width of the per-family band attention.
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        pair_offsets: tuple[int, ...] = (1,),
        band_feature_dim: int = 16,
        position_dim: int = 16,
        attention_dim: int = 32,
    ):
        super().__init__(backbone, pair_offsets, band_feature_dim, position_dim)
        n_families = 1 + len(self.pair_offsets)
        self.attention_scale = attention_dim**-0.5
        self.band_query_projections = nn.ModuleList(
            nn.Linear(2 * position_dim, attention_dim) for _ in range(n_families)
        )
        self.band_key_projections = nn.ModuleList(
            nn.Linear(band_feature_dim + position_dim, attention_dim)
            for _ in range(n_families)
        )

    def _attend_band_family(
        self,
        family: int,
        pair_query_input: Tensor,
        term_features: Tensor,
        term_positions: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """Masked-softmax pool of one family's visible terms, (B, d, d, F).

        pair_query_input: (d, d, 2*position_dim); term_features: (B, n, F);
        term_positions: (1, n, position_dim); visible: (d, d, n) bool by
        index arithmetic only. Exclusion-before-softmax + exact-zero
        overwrite of empty rows per the module docstring.
        """
        query = self.band_query_projections[family](pair_query_input)  # (d, d, A)
        batch, n_terms, _ = term_features.shape
        keys = self.band_key_projections[family](
            torch.cat(
                [term_features, term_positions.expand(batch, n_terms, -1)], dim=-1
            )
        )  # (B, n, A)
        scores = torch.einsum("ija,bka->bijk", query, keys) * self.attention_scale
        scores = scores.masked_fill(~visible, EXCLUDED_SCORE_FILL)
        weights = scores.softmax(dim=-1)  # (B, d, d, n)
        pooled = torch.einsum("bijk,bkf->bijf", weights, term_features)
        return torch.where(visible.any(dim=-1, keepdim=True), pooled, 0.0)

    def band_summaries(self, x: Tensor, t: Tensor) -> Tensor:
        """All-pairs middle-band summaries via masked attention, (B, d, d, F).

        Same visible sets, features and output contract as the interval
        head's cumsum version; only the pooling differs (learned softmax
        weights instead of a fixed sum). Entry [:, i, j, :] contains no term
        touching site i or site j -- exactly, because excluded terms carry
        softmax weight +0.0 (module docstring). Empty intervals are exact
        zeros. Only the i < j triangle is consumed downstream; the lower
        triangle's visibility set is empty, so it holds zeros here.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)  # (B, d, h)
        d = self.d
        site = torch.arange(d, device=x.device)
        site_i = site.view(d, 1, 1)
        site_j = site.view(1, d, 1)

        position = self.pair_position_embedding(site)  # (d, P)
        pair_query_input = torch.cat(
            [
                position.view(d, 1, -1).expand(d, d, -1),
                position.view(1, d, -1).expand(d, d, -1),
            ],
            dim=-1,
        )  # (d, d, 2P)

        slot = site.view(1, 1, d)
        families = [
            self._attend_band_family(
                0,
                pair_query_input,
                self.band_unary_features(emb),
                position.unsqueeze(0),
                (slot > site_i) & (slot < site_j),
            )
        ]
        for family, (delta, feature_mlp) in enumerate(
            zip(self.pair_offsets, self.band_pair_features), start=1
        ):
            n_terms = d - delta
            term_features = feature_mlp(
                torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)
            )
            term_slot = site[:n_terms].view(1, 1, n_terms)
            visible = (term_slot > site_i) & (term_slot + delta < site_j)
            families.append(
                self._attend_band_family(
                    family,
                    pair_query_input,
                    term_features,
                    position[:n_terms].unsqueeze(0),
                    visible,
                )
            )
        return torch.cat(families, dim=-1)  # (B, d, d, F)
