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

# Floor on the separable path's softmax normaliser. Only empty bands reach it
# (they normalise to exactly 0), and their pooled value is discarded -- but it
# must be discarded WITHOUT a division by zero, because `torch.where` carries
# NaN back through the branch it did not select. Well above fp32's 1.18e-38
# smallest normal, so it never perturbs a live band.
EMPTY_BAND_FLOOR = 1e-30


def _masked_exponential(scores: Tensor, mask: Tensor) -> Tensor:
    """`u * exp(scores)`: one half of a separable masked softmax, UNSHIFTED.

    NO STABILISING SHIFT, AND THAT IS THE POINT. A softmax normally subtracts
    a per-row maximum before exponentiating, and the separable form would have
    to subtract `max_k A_ik + max_k B_jk` -- an upper bound on the
    non-separable `max_k(A_ik + B_jk)`. Analytically a common shift cancels in
    the normalisation, so that costs nothing. Bit-for-bit it does not:
    `exp(s - c)` rounds differently for different `c`, and the row maximum is
    taken over `u_i`, a set that CONTAINS the partner hole k = j. Moving the
    token at j then moves the shift, and the pooled result changes in the last
    bits -- measured 2.98e-8, a residue exactly like the one this head was
    chosen over the interval head to avoid (module docstring).

    Blindness is worth more than the shift, because the shift is replaceable
    and blindness is not. Its only job is keeping `exp` inside the float's
    exponent budget, and that budget is measurable: fp32 overflows near +88
    and a product of two halves underflows near -87, against a trained 4x4
    checkpoint's measured score range of [-6.65, 10.69]. `band_score_range`
    reports the live figure so the margin is monitored rather than assumed.

    Exclusion is MULTIPLICATIVE, where the dense path fills scores with
    `EXCLUDED_SCORE_FILL` and leans on `exp(-1e9 - max)` underflowing. Both
    give an excluded term weight zero; only this one does so with no
    floating-point argument at all, which makes blindness here STRICTER than
    on the path it replaces.
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
        pair_position_mode: str = "absolute",
        separable_band_scores: bool = False,
        site_orderings: tuple[str, ...] = ("row",),
    ):
        super().__init__(
            backbone, pair_offsets, band_feature_dim, position_dim,
            readout_score_scale=readout_score_scale,
            exterior_combiner=exterior_combiner, bilinear_rank=bilinear_rank,
            gather_triu_pairs=gather_triu_pairs,
            site_orderings=site_orderings, lattice_side=lattice_side,
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
                lattice_side if lattice_side is not None else round(self.d ** 0.5),
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

    def _band_family_halves(
        self,
        family: int,
        position: Tensor,
        term_features: Tensor,
        term_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """The band scores as an outer SUM: (B, d, n) row part, (B, d, n) col.

        `s_ijk = A_ik + B_jk`. The query is one `nn.Linear` on the
        CONCATENATION `[rho_i || rho_j]`, so its weight splits column-wise
        into `[W_row | W_col]` and the pair axes never meet. The bias rides
        with the row half arbitrarily -- it is common to i and j, so it
        cancels in the softmax normalisation whichever half carries it.
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

        Computes the SAME masked softmax pool -- this is an algebraic
        identity, not an approximation, and it must not be confused with the
        factorised swap head, which changes the function and pays a measured
        variance price for it. Both halves of the softmax become matrix
        products once `s_ijk = A_ik + B_jk` and `visible = u_ik w_jk`:

            Z_ij    = sum_k alpha_ik beta_jk
            out_ijf = (1/Z_ij) sum_k alpha_ik beta_jk feat_kf

        with `alpha = u e^{A}` and `beta = w e^{B}`, unshifted for the reason
        in (1) below. Peak memory falls to the (B, d, d, F) context every all-pairs head
        returns anyway; the 5.00 GB slab at B=32, d=256 never exists.

        THREE THINGS THIS GUARDS AGAINST.

        1. THE SOFTMAX SHIFT IS GONE, DELIBERATELY. The separable form would
           have to stabilise with `max_k A_ik + max_k B_jk`, and that row
           maximum ranges over `u_i`, a set containing the partner hole k = j
           -- so a hole's token value reaches the answer in the last bits
           (measured 2.98e-8) and bit-exact blindness, the property this head
           was chosen for, is quietly lost. Unshifted, exclusion is a
           multiplication by zero and blindness is STRICTER here than on the
           dense path. The price is that the exponent budget is monitored
           rather than guaranteed: see `_masked_exponential` and
           `band_score_range`.

        2. AN EMPTY BAND IS A 0/0, NOT A UNIFORM ROW. The dense path fills
           masked scores with a finite -1e9 so a fully-masked row softmaxes
           to a discarded uniform; here `Z_ij` is exactly zero. The clamp
           before the division is load-bearing in BACKWARD, not forward:
           `torch.where` propagates NaN from the unselected branch.

        3. EMPTINESS IS DECIDED FROM THE MASK, NOT FROM `Z > 0`. The two
           agree only while nothing underflows, and underflow is exactly the
           failure mode (1) leaves live -- so the indicator is a boolean
           count over `u & w`, exact and batch-free, rather than a test on
           the normaliser it is meant to protect.

        Under `gather_triu_pairs` the pair axes collapse to a list, so the
        halves are GATHERED (`alpha[rows] * beta[cols]`) instead of outer-
        multiplied. That path keeps a (B, P, n) product, so it does NOT
        compose well with this lever: the gather removes half of a slab this
        removes entirely. Correct, but not the recommended pairing.
        """
        row_scores, col_scores = self._band_family_halves(
            family, position, term_features, term_positions
        )
        alpha = _masked_exponential(row_scores, visible_row)
        beta = _masked_exponential(col_scores, visible_col)

        if pairs is None:
            normaliser = torch.einsum("bik,bjk->bij", alpha, beta)
            # Weight by the column half FIRST. A single three-operand einsum
            # is free to contract alpha with beta first, which rebuilds the
            # (B, d, d, n) tensor this method exists to avoid.
            weighted = beta.unsqueeze(-1) * term_features.unsqueeze(1)
            numerator = torch.einsum("bik,bjkf->bijf", alpha, weighted)
            non_empty = (
                visible_row.float() @ visible_col.float().T > 0
            ).unsqueeze(-1)
        else:
            rows, cols = pairs
            joint = alpha[:, rows] * beta[:, cols]  # (B, P, n)
            normaliser = joint.sum(dim=-1)
            numerator = torch.einsum("bpk,bkf->bpf", joint, term_features)
            non_empty = (
                visible_row[rows] & visible_col[cols]
            ).any(dim=-1, keepdim=True)

        pooled = numerator / normaliser.clamp_min(EMPTY_BAND_FLOOR).unsqueeze(-1)
        return torch.where(non_empty, pooled, 0.0)

    def band_score_range(self, x: Tensor) -> float:
        """Worst-case exponent the unshifted separable pool can reach, in nats.

        `max|A| + max|B|`, maximised over pairs and families -- a conservative
        two-sided bound, since it dominates each half on its own as well as
        their sum. Compare against ~87: fp32 overflows above +88 and a product
        of the two halves underflows below -87, so this is the headroom the
        decision to drop the softmax shift is spending (see
        `_masked_exponential` for why the shift had to go). A trained 4x4
        checkpoint measures a score range of [-6.65, 10.69], i.e. ~17 against
        a budget of 87.

        Blind by construction, and cheap: it reads the same (B, d, n) halves
        the pool already builds, never the (B, d^2, n) tensor the lever
        exists to avoid.
        """
        emb = self.backbone.token_embedder(((x + 1) / 2).long())
        position = self.pair_position_embedding(
            torch.arange(self.d, device=x.device)
        )
        widest = 0.0
        for family, (_, term_features, term_slot) in enumerate(
            self._band_families(emb)
        ):
            if term_features.shape[1] == 0:      # stencil is empty at d = 2*side
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
        """Signed torus displacement of j from i, and one embedding row per
        displacement -- the pair position code the TWO-HOLE PATCH head uses
        and the raster heads do not.

        THE DEFECT THIS ADDRESSES. `pair_position_embedding` is
        `nn.Embedding(d, .)` indexed by ABSOLUTE site, so the query
        W_q(rho_i, rho_j) has no way to know that sites 0 and d-1 are torus
        neighbours; the wrap has to be learned from data. The patch head --
        the one that wins at every rung where both ran -- instead indexes
        `relative_position_embedding` by the signed displacement, so
        translation-equivalent pairs share a code by construction.

        WHY THIS IS NOT WHAT ROPE TESTED. The RoPE experiment swapped the
        BACKBONE's position code, which reaches only the causal-stream
        summaries P_i and S_j; it never touched this embedding, which is what
        the band's query and the pair readout actually consume. RoPE measured
        free on an A100 (24.0 ms against leTF's 24.8 at d=256) and read a
        null on quality -- consistent with having fixed the layer that
        matters least.

        The output width is 2 * position_dim so the query projection's shape
        is untouched and the two modes differ in nothing but the code.

        NOTE FOR THE BAND FACTORISATION: the absolute mode's query is linear
        on a CONCATENATION, which is what makes the score tensor an outer sum
        A_ik + B_jk. A relative code is one vector per pair, so that identity
        does NOT hold here and the two levers do not compose as written.
        """
        site = torch.arange(self.d)
        rows, cols = site // side, site % side
        displacement = (
            ((rows[None, :] - rows[:, None]) % side) * side
            + (cols[None, :] - cols[:, None]) % side
        )
        self.register_buffer("pair_displacement", displacement, persistent=False)
        self.relative_pair_embedding = nn.Embedding(self.d, 2 * position_dim)

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

    def _family_visibility_halves(
        self, term_slot: Tensor, offsets: tuple[int, ...]
    ) -> tuple[Tensor, Tensor]:
        """The same predicate, split as visible(i, j, k) = u(i, k) & w(j, k).

        Both windows separate, for different reasons: `interval` is a pair of
        one-sided inequalities, one per hole; `lattice` is a product of
        disequalities that is already per-hole term by term. Returns two
        (d, n_terms) tables over the FULL site range -- not the pair layout --
        because the separable path indexes them by row and column site, which
        under the triu gather is a list of pairs rather than a grid.

        Separability of the MASK is half of why the band factorises (the
        other half is that the query is linear on a concatenation): a
        separable mask multiplies into alpha and beta, where the dense path
        must fill scores with `EXCLUDED_SCORE_FILL` and rely on the softmax
        underflowing them. Multiplying makes exclusion exactly zero rather
        than zero-up-to-underflow, so blindness stops needing a numerical
        argument at all.
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

        The three families are built the same way whichever aggregator
        consumes them, so enumerating them once keeps the dense and separable
        paths honest: they differ only in the pool, never in what is pooled.
        `term_slot` is the term's index into the site axis, which is also its
        row of the position table -- the unary and offset families start at
        site 0, the stencil starts at `side`.
        """
        yield (0,), self.band_unary_features(emb), torch.arange(
            self.d, device=emb.device
        )
        for delta, feature_mlp in zip(self.pair_offsets, self.band_pair_features):
            n_terms = self.d - delta
            yield (
                (0, delta),
                feature_mlp(
                    torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)
                ),
                torch.arange(n_terms, device=emb.device),
            )
        if self.use_stencil:
            # 5-point lattice-stencil family (2026-07-08). Per-term feature
            # s_k = MLP(emb(x_k) ++ emb(x_{k±1}) ++ emb(x_{k±side})) -- a 2D
            # neighbourhood statistic, richer than the unary/offset terms that
            # capped the one-pass family at ~0.78 (H-shared).
            #
            # Centres exist only for side <= k < d - side (raster-boundary
            # sites lack a k±side neighbour); the range is empty when
            # d = 2*side. Straddle exclusion: centre k touches
            # {k-side .. k+side}, all strictly interior iff k - side > i AND
            # k + side < j -- index arithmetic, so blindness is
            # value-independent, like every other family. The ±side reach
            # leaves an uncovered collar round each hole; the narrow families
            # above cover it, forming a locality ladder.
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

        Same visible sets, features and output contract as the interval
        head's cumsum version; only the pooling differs (learned softmax
        weights instead of a fixed sum). Entry [:, i, j, :] contains no term
        touching site i or site j -- exactly, because excluded terms carry
        softmax weight +0.0 (module docstring). Empty intervals are exact
        zeros. Only the i < j triangle is consumed downstream; the lower
        triangle's visibility set is empty, so it holds zeros here.

        When use_stencil is set, a final 5-point lattice-stencil family is
        appended (trailing F channels); see `_band_families`.

        `pairs` = (rows, cols) selects a LIST of pairs (the triu-pair gather)
        and returns (B, P, F). Visibility is index arithmetic in both forms,
        so exclusion -- and with it bit-exact blindness -- is unchanged; the
        hole terms are simply never scored for pairs nobody asked about.

        Under `separable_band_scores` the pool is an exact rewrite that never
        builds the (B, d^2, n) score tensor; see
        `_attend_band_family_separable`. The two branches share `_band_families`
        precisely so the choice of aggregator cannot drift into a choice of
        features.
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
                    if self.pair_position_mode == "relative" else
                    torch.cat(
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
                    if self.pair_position_mode == "relative" else
                    torch.cat([position[rows], position[cols]], dim=-1)
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
                        family, position, term_features, term_positions,
                        visible_row, visible_col, pairs,
                    )
                )
            else:
                visible = self._family_visibility(
                    term_slot.view(term_shape), site_i, site_j, offsets
                )
                families.append(
                    self._attend_band_family(
                        family, pair_query_input, term_features,
                        term_positions, visible,
                    )
                )
        return torch.cat(families, dim=-1)  # (B, d, d, F)
