"""Exclusion-mask band-attention swap head, the reported head.

`interval_swap_head.py` assembles its band by prefix-sum subtraction, which
cancels the hole terms up to an fp residue. This head keeps its
three-interval decomposition, causal P/S streams, band feature families and
pair readout, and replaces only the band aggregator with masked attention
over the same visible set. Exclusion happens before the softmax, so a hole
term never enters any computed quantity and blindness is bit-exact.

Readout (DNFS Prop. 2 / Eq. (9), pair form), as in swap_readout.py:

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

H_ij must not depend on the token values at i and j; interval_swap_head.py
explains why blindness must hold at every layer (the two-hop leak) and why
band content is therefore shallow and local.

Band, per feature family: per-term features (unary v_k, offset-delta
u^delta_k) are the interval head's; each pair (i, j) forms a query from its
two position embeddings (no token content); keys are term features plus the
term's position embedding; visibility is the interval head's index
arithmetic (unary term k needs i < k < j; offset-delta term k needs k > i and
k + delta < j); a masked softmax pools the visible features. Masked scores
are filled with -1e9, not -inf: exp(-1e9 - max) underflows to exactly +0.0
in fp32, so an excluded term's weight is zero whatever the hole tokens are,
while a fully-masked row (empty interval, j - i < delta + 2) softmaxes to a
finite uniform instead of NaN; its output and gradient are then zeroed by
the index mask.

Cost: one backbone pass plus one masked attention of d^2 pair queries over
O(d) terms per family. The (B, d^2, n_terms) score tensor is the price; it is
halved by `gather_triu_pairs` (the band is defined on i < j, so the lower
triangle is known zeros; see `interval_swap_head.scatter_symmetric_pairs`)
and removed by `separable_band_scores`.

Time is absent from the band, as in the interval head: it enters H through
the pair readout's summaries and time line.
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

# Floor on the separable path's softmax normaliser. Only empty bands reach it
# (normaliser exactly 0); their pooled value is discarded, but `torch.where`
# carries NaN back through the unselected branch, so the division must not be
# 0/0. Well above fp32's 1.18e-38 smallest normal, so no live band is touched.
EMPTY_BAND_FLOOR = 1e-30


def _masked_exponential(scores: Tensor, mask: Tensor) -> Tensor:
    """`u * exp(scores)`: one half of a separable masked softmax, unshifted.

    The separable form would have to stabilise with `max_k A_ik + max_k B_jk`,
    and that row maximum ranges over `u_i`, a set containing the partner hole
    k = j; `exp(s - c)` rounds differently per `c`, so moving the token at j
    would move the pooled result in the last bits (measured 2.98e-8). Without
    the shift, exclusion is a multiplication by zero and blindness is exact.
    The price is exponent range: fp32 overflows near +88 and a product of two
    halves underflows near -87, against a trained 4x4 checkpoint's measured
    score range of [-6.65, 10.69]; `band_score_range` reports the live figure.
    """
    return torch.exp(scores) * mask


class MaskedAttentionSwapHead(IntervalSwapHead):
    """Doubly-hollow swap head with an exclusion-mask attention band.

    Subclass of IntervalSwapHead so the relationship is literal in code:
    `causal_summaries`, `compute_pair_context` and `forward` are inherited
    unchanged; only `band_summaries` -- the aggregator -- is overridden.
    Same (B, d, d, F) band contract, so the inherited pair readout applies.

    Args (beyond IntervalSwapHead's):
        attention_dim: query/key width of the per-family band attention.
        use_stencil: add the 5-point lattice-stencil band-feature family
            Off by default so every existing MA cell keeps building a
            byte-identical head. See `band_summaries`.
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
        pair_position_mode: str = "absolute",
        separable_band_scores: bool = False,
        site_orderings: tuple[str, ...] = ("row",),
    ):
        super().__init__(
            backbone,
            pair_offsets,
            band_feature_dim,
            position_dim,
            readout_score_scale=readout_score_scale,
            exterior_combiner=exterior_combiner,
            bilinear_rank=bilinear_rank,
            gather_triu_pairs=gather_triu_pairs,
            site_orderings=site_orderings,
            lattice_side=lattice_side,
        )
        if attention_window not in ("interval", "lattice"):
            raise ValueError(
                f"attention_window must be 'interval' or 'lattice', "
                f"got {attention_window!r}"
            )
        self.attention_window = attention_window
        if pair_position_mode not in ("absolute", "relative"):
            raise ValueError(
                f"pair_position_mode must be 'absolute' or 'relative', "
                f"got {pair_position_mode!r}"
            )
        self.pair_position_mode = pair_position_mode
        if separable_band_scores and pair_position_mode != "absolute":
            raise ValueError(
                "separable_band_scores needs pair_position_mode='absolute': a "
                "relative code emits ONE query vector per pair, so the score "
                f"is not an outer sum A_ik + B_jk (got {pair_position_mode!r})"
            )
        self.separable_band_scores = separable_band_scores
        self.use_stencil = use_stencil
        hidden = backbone.hidden_dim
        # The stencil is one extra band-feature family, so it gets its own
        # attention query/key projection alongside the unary + offset ones.
        n_families = 1 + len(self.pair_offsets) + (1 if use_stencil else 0)
        self.attention_scale = attention_dim**-0.5
        if self.pair_position_mode == "relative":
            self._init_relative_pair_positions(
                position_dim,
                lattice_side if lattice_side is not None else round(self.d**0.5),
            )
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
            # The extra family widens the band block, so the inherited
            # pair_readout is band_feature_dim too narrow: rebuild it.
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
        (B, P, F). Only the einsum subscripts move; the dense subscripts are
        left as they were so the archived path's contraction order is
        untouched.
        """
        pair_axes = "ij" if pair_query_input.dim() == 3 else "p"
        query = self.band_query_projections[family](pair_query_input)  # (..., A)
        batch, n_terms, _ = term_features.shape
        keys = self.band_key_projections[family](
            torch.cat(
                [term_features, term_positions.expand(batch, n_terms, -1)], dim=-1
            )
        )  # (B, n, A)
        scores = (
            torch.einsum(f"{pair_axes}a,bka->b{pair_axes}k", query, keys)
            * self.attention_scale
        )
        scores = scores.masked_fill(~visible, EXCLUDED_SCORE_FILL)
        weights = scores.softmax(dim=-1)  # (B, d, d, n) or (B, P, n)
        pooled = torch.einsum(
            f"b{pair_axes}k,bkf->b{pair_axes}f", weights, term_features
        )
        return torch.where(visible.any(dim=-1, keepdim=True), pooled, 0.0)

    def _band_family_halves(
        self,
        family: int,
        position: Tensor,
        term_features: Tensor,
        term_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """The band scores as an outer sum: (B, d, n) row part, (B, d, n) col.

        `s_ijk = A_ik + B_jk`. The query is one `nn.Linear` on the
        concatenation `[rho_i || rho_j]`, so its weight splits column-wise
        into `[W_row | W_col]` and the pair axes never meet. The bias rides
        with the row half; it is common to i and j, so it cancels in the
        softmax normalisation whichever half carries it.
        """
        query = self.band_query_projections[family]
        position_dim = position.shape[-1]
        batch, n_terms, _ = term_features.shape
        keys = self.band_key_projections[family](
            torch.cat(
                [term_features, term_positions.expand(batch, n_terms, -1)], dim=-1
            )
        )  # (B, n, A)
        row_query = position @ query.weight[:, :position_dim].T + query.bias
        col_query = position @ query.weight[:, position_dim:].T
        return (
            torch.einsum("ia,bka->bik", row_query, keys) * self.attention_scale,
            torch.einsum("ja,bka->bjk", col_query, keys) * self.attention_scale,
        )

    def _attend_band_family_separable(
        self,
        family: int,
        position: Tensor,
        term_features: Tensor,
        term_positions: Tensor,
        visible_row: Tensor,
        visible_col: Tensor,
        pairs: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        """`_attend_band_family` without the (B, d^2, n) score tensor.

        An exact rewrite of the same masked softmax pool, not the factorised
        swap head (which changes the function). Once `s_ijk = A_ik + B_jk` and
        `visible = u_ik w_jk`, both halves of the softmax are matrix products:

            Z_ij    = sum_k alpha_ik beta_jk
            out_ijf = (1/Z_ij) sum_k alpha_ik beta_jk feat_kf

        with `alpha = u e^{A}` and `beta = w e^{B}`, unshifted (see
        `_masked_exponential`). Peak memory falls to the (B, d, d, F) context;
        the 5.00 GB score slab at B=32, d=256 never exists.

        An empty band is a 0/0 here, not the dense path's discarded uniform
        row: the clamp before the division matters in backward, where
        `torch.where` propagates NaN from the unselected branch. Emptiness is
        decided from the mask (`u & w`), not from `Z > 0`, because the two
        agree only while nothing underflows.

        Under `gather_triu_pairs` the halves are gathered
        (`alpha[rows] * beta[cols]`) into a (B, P, n) product, so the two
        levers compose correctly but the gather buys nothing here.
        """
        row_scores, col_scores = self._band_family_halves(
            family, position, term_features, term_positions
        )
        alpha = _masked_exponential(row_scores, visible_row)
        beta = _masked_exponential(col_scores, visible_col)

        if pairs is None:
            normaliser = torch.einsum("bik,bjk->bij", alpha, beta)
            # Weight by the column half first: a three-operand einsum may
            # contract alpha with beta first, rebuilding the (B, d, d, n) slab.
            weighted = beta.unsqueeze(-1) * term_features.unsqueeze(1)
            numerator = torch.einsum("bik,bjkf->bijf", alpha, weighted)
            non_empty = (visible_row.float() @ visible_col.float().T > 0).unsqueeze(-1)
        else:
            rows, cols = pairs
            joint = alpha[:, rows] * beta[:, cols]  # (B, P, n)
            normaliser = joint.sum(dim=-1)
            numerator = torch.einsum("bpk,bkf->bpf", joint, term_features)
            non_empty = (visible_row[rows] & visible_col[cols]).any(
                dim=-1, keepdim=True
            )

        pooled = numerator / normaliser.clamp_min(EMPTY_BAND_FLOOR).unsqueeze(-1)
        return torch.where(non_empty, pooled, 0.0)

    def band_score_range(self, x: Tensor) -> float:
        """Worst-case exponent the unshifted separable pool can reach, in nats.

        `max|A| + max|B|` over pairs and families, a conservative two-sided
        bound. Compare against ~87 (fp32 overflows above +88, a product of two
        halves underflows below -87): the headroom the unshifted softmax in
        `_masked_exponential` spends. A trained 4x4 checkpoint measures
        [-6.65, 10.69], ~17 against a budget of 87. Reads only the (B, d, n)
        halves, never the (B, d^2, n) tensor.
        """
        emb = self.backbone.token_embedder(((x + 1) / 2).long())
        position = self.pair_position_embedding(torch.arange(self.d, device=x.device))
        widest = 0.0
        for family, (_, term_features, term_slot) in enumerate(
            self._band_families(emb)
        ):
            if term_features.shape[1] == 0:  # stencil is empty at d = 2*side
                continue
            row_scores, col_scores = self._band_family_halves(
                family, position, term_features, position[term_slot].unsqueeze(0)
            )
            widest = max(
                widest,
                row_scores.abs().max().item() + col_scores.abs().max().item(),
            )
        return widest

    def _init_relative_pair_positions(self, position_dim: int, side: int) -> None:
        """Signed torus displacement of j from i, one embedding row per
        displacement: the pair position code the two-hole patch head uses.

        `pair_position_embedding` is indexed by absolute site, so the query
        W_q(rho_i, rho_j) has to learn that sites 0 and d-1 are torus
        neighbours; indexing by displacement gives translation-equivalent
        pairs one code by construction. The RoPE experiment (24.0 ms vs
        leTF's 24.8 at d=256 on an A100, null on quality) swapped only the
        backbone's code, which reaches P_i and S_j, never this embedding.
        Output width is 2 * position_dim so the query projection's shape is
        unchanged. A relative code is one vector per pair, not a
        concatenation, so the outer-sum identity behind
        `separable_band_scores` does not hold with it.
        """
        site = torch.arange(self.d)
        rows, cols = site // side, site % side
        displacement = ((rows[None, :] - rows[:, None]) % side) * side + (
            cols[None, :] - cols[:, None]
        ) % side
        self.register_buffer("pair_displacement", displacement, persistent=False)
        self.relative_pair_embedding = nn.Embedding(self.d, 2 * position_dim)

    def _family_visibility(self, slot, site_i, site_j, offsets):
        """Which terms of one band family the pair (i, j) may read.

        A term at slot k has support {k + o : o in `offsets`} -- {0} for the
        unary family, {0, delta} for the offset-delta bond family,
        {-side, -1, 0, 1, side} for the stencil. Two windows:

            interval  k + min(O) > i  and  k + max(O) < j    (strictly inside)
            lattice   k + o != i and k + o != j for every o  (touches neither)

        Blindness holds under both: exclusion removes every term whose
        support touches a hole, decided from the indices alone, before the
        softmax. The requirement is bounded support, not depth-0 features.

        The interval branch is min/max rather than a per-offset conjunction
        because that is the archived semantics; for the stencil it leaves an
        uncovered collar round each hole that the narrower families fill.
        `lattice` normalises over ~d terms instead of ~|j - i| and can lose to
        the interval window; the option exists to measure which.
        """
        if self.attention_window == "interval":
            return (slot + min(offsets) > site_i) & (slot + max(offsets) < site_j)
        visible = None
        for offset in offsets:
            support = slot + offset
            untouched = (support != site_i) & (support != site_j)
            visible = untouched if visible is None else visible & untouched
        return visible

    def _family_visibility_halves(
        self, term_slot: Tensor, offsets: tuple[int, ...]
    ) -> tuple[Tensor, Tensor]:
        """The same predicate, split as visible(i, j, k) = u(i, k) & w(j, k).

        Both windows separate: `interval` is a pair of one-sided inequalities,
        one per hole; `lattice` is a product of per-hole disequalities.
        Returns two (d, n_terms) tables over the full site range, not the pair
        layout, because the separable path indexes them by row and column
        site (a list of pairs under the triu gather). A separable mask
        multiplies into alpha and beta, so exclusion is exactly zero rather
        than zero-up-to-underflow.
        """
        site = torch.arange(self.d, device=term_slot.device).view(-1, 1)
        slot = term_slot.view(1, -1)
        if self.attention_window == "interval":
            return slot + min(offsets) > site, slot + max(offsets) < site
        untouched = None
        for offset in offsets:
            touches = (slot + offset) != site
            untouched = touches if untouched is None else untouched & touches
        return untouched, untouched.clone()

    def _band_families(self, emb: Tensor):
        """Yield `(offsets, term_features, term_slot)` for every band family.

        Shared by the dense and separable paths so they differ only in the
        pool, never in what is pooled. `term_slot` is the term's index into
        the site axis (its row of the position table); the unary and offset
        families start at site 0, the stencil at `side`.
        """
        yield (
            (0,),
            self.band_unary_features(emb),
            torch.arange(self.d, device=emb.device),
        )
        for delta, feature_mlp in zip(self.pair_offsets, self.band_pair_features):
            n_terms = self.d - delta
            yield (
                (0, delta),
                feature_mlp(torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)),
                torch.arange(n_terms, device=emb.device),
            )
        if self.use_stencil:
            # 5-point stencil family, s_k = MLP(emb(x_k) ++ emb(x_{k±1}) ++
            # emb(x_{k±side})): a 2D neighbourhood statistic, richer than the
            # unary/offset terms that capped the one-pass family at ~0.78.
            # Centres exist only for side <= k < d - side (empty when
            # d = 2*side); centre k touches {k-side .. k+side}, interior iff
            # k - side > i and k + side < j.
            side = self.stencil_side
            centres = torch.arange(side, self.d - side, device=emb.device)
            yield (
                (-side, -1, 0, 1, side),
                self.band_stencil_features(
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
                ),  # (B, n_centres, F); n_centres = 0 when d <= 2*side
                centres,
            )

    def band_summaries(
        self, x: Tensor, t: Tensor, pairs: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """All-pairs middle-band summaries via masked attention, (B, d, d, F).

        Same visible sets, features and output contract as the interval head's
        cumsum version; only the pooling differs. Entry [:, i, j, :] contains
        no term touching site i or j (excluded terms carry softmax weight
        +0.0); empty intervals and the lower triangle are exact zeros. With
        use_stencil a stencil family is appended as trailing F channels.
        `pairs` = (rows, cols) selects a list of pairs and returns (B, P, F).
        Under `separable_band_scores` the pool is the exact rewrite in
        `_attend_band_family_separable`.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)  # (B, d, h)
        d = self.d
        site = torch.arange(d, device=x.device)
        position = self.pair_position_embedding(site)  # (d, P)
        if not self.separable_band_scores:
            if pairs is None:
                site_i = site.view(d, 1, 1)
                site_j = site.view(1, d, 1)
                term_shape = (1, 1, -1)
                pair_query_input = (
                    self.relative_pair_embedding(self.pair_displacement)
                    if self.pair_position_mode == "relative"
                    else torch.cat(
                        [
                            position.view(d, 1, -1).expand(d, d, -1),
                            position.view(1, d, -1).expand(d, d, -1),
                        ],
                        dim=-1,
                    )
                )  # (d, d, 2P)
            else:
                rows, cols = pairs
                site_i, site_j = rows.view(-1, 1), cols.view(-1, 1)
                term_shape = (1, -1)
                pair_query_input = (
                    self.relative_pair_embedding(self.pair_displacement[rows, cols])
                    if self.pair_position_mode == "relative"
                    else torch.cat([position[rows], position[cols]], dim=-1)
                )  # (P, 2P)

        families = []
        for family, (offsets, term_features, term_slot) in enumerate(
            self._band_families(emb)
        ):
            term_positions = position[term_slot].unsqueeze(0)
            if self.separable_band_scores:
                visible_row, visible_col = self._family_visibility_halves(
                    term_slot, offsets
                )
                families.append(
                    self._attend_band_family_separable(
                        family,
                        position,
                        term_features,
                        term_positions,
                        visible_row,
                        visible_col,
                        pairs,
                    )
                )
            else:
                visible = self._family_visibility(
                    term_slot.view(term_shape), site_i, site_j, offsets
                )
                families.append(
                    self._attend_band_family(
                        family,
                        pair_query_input,
                        term_features,
                        term_positions,
                        visible,
                    )
                )
        return torch.cat(families, dim=-1)  # (B, d, d, F)
