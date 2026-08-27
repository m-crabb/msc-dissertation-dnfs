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
already exists). Half of that price is removed by `gather_triu_pairs`
(opt-in, default OFF): the band is DEFINED on i < j -- the lower triangle's
visible set is empty and holds zeros -- so scoring the full grid computes
d(d-1)/2 pair queries whose answer is known to be zero, plus a dead
diagonal. See `interval_swap_head.scatter_symmetric_pairs`.

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
        use_stencil: add the 5-point lattice-stencil band-feature family
            (2026-07-08). Off by default so every existing MA
            cell keeps building a byte-identical head. See `band_summaries`.
        lattice_side: side length D of the flattened D x D grid the stencil's
            column neighbours x_{k±D} address. None infers round(sqrt(d))
            and asserts squareness -- pass it explicitly for non-square d.
        readout_score_scale: muP readout compensation on the pair scores;
            see IntervalSwapHead.__init__ (the readout is inherited, so
            the knob is too). 1.0 = every archived MA cell, byte-identical.
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        pair_offsets: tuple[int, ...] = (1,),
        band_feature_dim: int = 16,
        position_dim: int = 16,
        attention_dim: int = 32,
        use_stencil: bool = False,
        lattice_side: int | None = None,
        readout_score_scale: float = 1.0,
        exterior_combiner: str = "mlp",
        bilinear_rank: int = 8,
        gather_triu_pairs: bool = False,
        attention_window: str = "interval",
    ):
        super().__init__(
            backbone, pair_offsets, band_feature_dim, position_dim,
            readout_score_scale=readout_score_scale,
            exterior_combiner=exterior_combiner, bilinear_rank=bilinear_rank,
            gather_triu_pairs=gather_triu_pairs,
        )
        if attention_window not in ("interval", "lattice"):
            raise ValueError(
                f"attention_window must be 'interval' or 'lattice', "
                f"got {attention_window!r}"
            )
        self.attention_window = attention_window
        self.use_stencil = use_stencil
        hidden = backbone.hidden_dim
        # The stencil is one extra band-feature family, so it gets its own
        # attention query/key projection alongside the unary + offset ones.
        n_families = 1 + len(self.pair_offsets) + (1 if use_stencil else 0)
        self.attention_scale = attention_dim**-0.5
        self.band_query_projections = nn.ModuleList(
            nn.Linear(2 * position_dim, attention_dim) for _ in range(n_families)
        )
        self.band_key_projections = nn.ModuleList(
            nn.Linear(band_feature_dim + position_dim, attention_dim)
            for _ in range(n_families)
        )
        if use_stencil:
            self.stencil_side = (
                lattice_side if lattice_side is not None else round(self.d**0.5)
            )
            if self.stencil_side**2 != self.d:
                raise ValueError(
                    f"stencil needs a square lattice: side {self.stencil_side} "
                    f"does not tile d={self.d}; pass lattice_side explicitly"
                )
            # Per-term feature over the 5-point neighbourhood {k, k±1, k±D}:
            # a depth-1 local 2D statistic (the object Ising energy diffs turn
            # on), still blind because exclusion is index arithmetic (below).
            self.band_stencil_features = nn.Sequential(
                nn.Linear(5 * hidden, band_feature_dim),
                nn.GELU(),
                nn.Linear(band_feature_dim, band_feature_dim),
            )
            # One extra family widens the band block, so the inherited
            # pair_readout's input is now band_feature_dim too narrow: rebuild
            # it. The narrow one the parent drew is discarded (one wasted init
            # draw -- harmless; use_stencil=False never enters here, so those
            # cells stay byte-identical to pre-stencil code).
            self.pair_readout = self._build_pair_readout(band_feature_dim * n_families)

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

        Under the triu-pair gather the pair axes collapse to a single list
        axis: pair_query_input (P, 2*position_dim), visible (P, n), result
        (B, P, F). Only the einsum subscripts move -- the DENSE subscripts are
        left literally as they were, so the archived path's contraction order
        (and with it its last bit) is untouched. This is where the lever pays
        most: the (B, d^2, n_terms) score tensor is this head's largest, and
        it is the "d = 256 wants pair chunking" price the module docstring
        names.
        """
        pair_axes = "ij" if pair_query_input.dim() == 3 else "p"
        query = self.band_query_projections[family](pair_query_input)  # (..., A)
        batch, n_terms, _ = term_features.shape
        keys = self.band_key_projections[family](
            torch.cat(
                [term_features, term_positions.expand(batch, n_terms, -1)], dim=-1
            )
        )  # (B, n, A)
        scores = torch.einsum(
            f"{pair_axes}a,bka->b{pair_axes}k", query, keys
        ) * self.attention_scale
        scores = scores.masked_fill(~visible, EXCLUDED_SCORE_FILL)
        weights = scores.softmax(dim=-1)  # (B, d, d, n) or (B, P, n)
        pooled = torch.einsum(
            f"b{pair_axes}k,bkf->b{pair_axes}f", weights, term_features
        )
        return torch.where(visible.any(dim=-1, keepdim=True), pooled, 0.0)

    def _family_visibility(self, slot, site_i, site_j, offsets):
        """Which terms of one band family the pair (i, j) may read.

        A term at slot k has support {k + o : o in `offsets`} -- {0} for the
        unary family, {0, delta} for the offset-delta bond family,
        {-side, -1, 0, 1, side} for the stencil. Two windows:

            interval  k + min(O) > i  and  k + max(O) < j    (strictly inside)
            lattice   k + o != i and k + o != j for every o  (touches neither)

        Blindness holds under BOTH, and for the same reason: what it
        requires is that exclusion remove every term whose support touches a
        hole, decided from the INDICES alone so the mask is
        value-independent. It does NOT require per-site or depth-0 features
        -- that was a rule stated on the depth axis when the live constraint
        is bounded support (corrected 2026-08-27). Exclusion is applied
        BEFORE the softmax either way, so an excluded term carries pooling
        weight exactly zero rather than a small one.

        The interval branch is written as min/max rather than as a per-offset
        conjunction because that is the archived semantics: for the stencil
        it is deliberately conservative, leaving an uncovered collar around
        each hole that the narrower families fill in.

        `lattice` is NOT simply the more general window. Its softmax
        normalises over ~d terms instead of ~|j - i|, which dilutes whatever
        mass the interval deserves, and a learned soft mask approximates the
        hard interval indicator without containing it -- so it can lose, and
        the arm exists to measure which.
        """
        if self.attention_window == "interval":
            return (
                (slot + min(offsets) > site_i) & (slot + max(offsets) < site_j)
            )
        visible = None
        for offset in offsets:
            support = slot + offset
            untouched = (support != site_i) & (support != site_j)
            visible = untouched if visible is None else visible & untouched
        return visible

    def band_summaries(
        self, x: Tensor, t: Tensor, pairs: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """All-pairs middle-band summaries via masked attention, (B, d, d, F).

        Same visible sets, features and output contract as the interval
        head's cumsum version; only the pooling differs (learned softmax
        weights instead of a fixed sum). Entry [:, i, j, :] contains no term
        touching site i or site j -- exactly, because excluded terms carry
        softmax weight +0.0 (module docstring). Empty intervals are exact
        zeros. Only the i < j triangle is consumed downstream; the lower
        triangle's visibility set is empty, so it holds zeros here.

        When use_stencil is set, a final 5-point lattice-stencil family is
        appended (trailing F channels); see the inline note below.

        `pairs` = (rows, cols) selects a LIST of pairs (the triu-pair gather)
        and returns (B, P, F). Visibility is index arithmetic in both forms,
        so exclusion -- and with it bit-exact blindness -- is unchanged; the
        hole terms are simply never scored for pairs nobody asked about.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)  # (B, d, h)
        d = self.d
        site = torch.arange(d, device=x.device)
        position = self.pair_position_embedding(site)  # (d, P)
        if pairs is None:
            site_i = site.view(d, 1, 1)
            site_j = site.view(1, d, 1)
            term_shape = (1, 1, -1)
            pair_query_input = torch.cat(
                [
                    position.view(d, 1, -1).expand(d, d, -1),
                    position.view(1, d, -1).expand(d, d, -1),
                ],
                dim=-1,
            )  # (d, d, 2P)
        else:
            rows, cols = pairs
            site_i, site_j = rows.view(-1, 1), cols.view(-1, 1)
            term_shape = (1, -1)
            pair_query_input = torch.cat(
                [position[rows], position[cols]], dim=-1
            )  # (P, 2P)

        slot = site.view(term_shape)
        families = [
            self._attend_band_family(
                0,
                pair_query_input,
                self.band_unary_features(emb),
                position.unsqueeze(0),
                self._family_visibility(slot, site_i, site_j, (0,)),
            )
        ]
        for family, (delta, feature_mlp) in enumerate(
            zip(self.pair_offsets, self.band_pair_features), start=1
        ):
            n_terms = d - delta
            term_features = feature_mlp(
                torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)
            )
            term_slot = site[:n_terms].view(term_shape)
            visible = self._family_visibility(
                term_slot, site_i, site_j, (0, delta)
            )
            families.append(
                self._attend_band_family(
                    family,
                    pair_query_input,
                    term_features,
                    position[:n_terms].unsqueeze(0),
                    visible,
                )
            )
        if self.use_stencil:
            # 5-point lattice-stencil family (2026-07-08). Per-term
            # feature s_k = MLP(emb(x_k) ++ emb(x_{k±1}) ++ emb(x_{k±side})) --
            # a 2D neighbourhood statistic, richer than the unary/offset terms
            # that capped the one-pass family at ~0.78 (H-shared).
            #
            # Centres exist only for side <= k < d - side (raster-boundary
            # sites lack a k±side neighbour); the range is empty when d = 2*side.
            # Straddle exclusion: centre k touches {k-side .. k+side}, all strictly
            # interior iff k - side > i AND k + side < j -- index arithmetic, so
            # blindness is value-independent, like every other family. The ±side
            # reach leaves an uncovered collar round each hole; the narrow
            # families above cover it, forming a locality ladder.
            side = self.stencil_side
            centres = torch.arange(side, d - side, device=x.device)
            stencil_features = self.band_stencil_features(
                torch.cat(
                    [
                        emb[:, centres],
                        emb[:, centres - 1],
                        emb[:, centres + 1],
                        emb[:, centres - side],
                        emb[:, centres + side],
                    ],
                    dim=-1,
                )
            )  # (B, n_centres, F); n_centres = 0 (empty) when d <= 2*side
            centre_slot = centres.view(term_shape)
            visible = self._family_visibility(
                centre_slot, site_i, site_j, (-side, -1, 0, 1, side)
            )
            families.append(
                self._attend_band_family(
                    len(families),  # stencil is the last family
                    pair_query_input,
                    stencil_features,
                    position[centres].unsqueeze(0),
                    visible,
                )
            )
        return torch.cat(families, dim=-1)  # (B, d, d, F)
