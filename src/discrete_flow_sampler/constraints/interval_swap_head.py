"""Three-interval (leave-two-out) swap head: all-pairs H_ij in one body pass.

Pair form of DNFS Prop. 2 / Eq. (9), as in swap_readout.py:

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

State-swap antisymmetry G(i,j|x) = -G(i,j|Swap2(x,i,j)) needs H_ij blind to
the token values x_i and x_j (the omega-difference flips sign; H must not
move). Blindness must hold at every layer: one attention layer mixes x_i
into every token, so masking only a readout edge leaks x_i -> token k -> H_ij.
The mask-one head buys layer-wise blindness with d anchor passes; this head
buys it structurally from the interval decomposition the two holes induce:

    [ x_0 .. x_{i-1} ]  x_i  ( x_{i+1} .. x_{j-1} )  x_j  [ x_{j+1} .. ]
       prefix P_i       HOLE       band  M_ij        HOLE    suffix S_j

P and S are the backbone's causal-stack states (fwd at slot i sees x_{<i},
un-flipped bwd at slot j+1 sees x_{>j}), blind by causality at any depth.
The band is an interval open at both ends, so it is built from local
features (unary and fixed-offset pair statistics of raw token embeddings)
with index-structural exclusion of every term touching a hole. The three
mix only in the per-pair readout MLP. Cost: O(d) precompute, O(1) per pair.

fp caveat: band assembly by prefix-sum subtraction (sum[i+1..j-1] =
prefix[j-1] - prefix[i]) is blind in real arithmetic but leaves an
~ulp * |prefix| residue that depends on x_i; the test bar is ATOL=1e-5. A
hole-free block decomposition would restore bit-exact blindness.

H_ji := H_ij, so G[j,i] = -G[i,j]. The backbone's attention_readout is unused.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.letf import LeTFRateMatrix

_TRIU_PAIR_CACHE: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}


def triu_pair_indices(d: int, device) -> tuple[Tensor, Tensor]:
    """Row/column index vectors of the d(d-1)/2 unordered pairs i < j.

    Shared by every label-symmetric head: H is computed on the i < j triangle
    and mirrored, halving the (B, d^2, F) activation slab. Cached per
    (d, device) like `_swap_neighbours.upper_tri_pairs` (kept separate so
    `constraints` does not import `samplers`); callers treat the result as
    read-only.
    """
    key = (d, torch.device(device))
    if key not in _TRIU_PAIR_CACHE:
        rows, cols = torch.triu_indices(d, d, offset=1, device=device)
        _TRIU_PAIR_CACHE[key] = (rows, cols)
    return _TRIU_PAIR_CACHE[key]


def scatter_symmetric_pairs(pair_values: Tensor, d: int) -> Tensor:
    """(B, P, F) per-pair values on i < j -> the symmetric (B, d, d, F) block.

    One tensor is written into both triangles, so H == H.transpose(1, 2)
    bit-exactly and index antisymmetry of G follows. The diagonal is left at
    zero: forward reads H against omega_{x_i} - omega_{x_j}, identically zero
    at i = j, so it never reaches G (`compute_pair_context` diagonals differ
    between the dense and gathered paths; nothing downstream reads them).
    Two index_put's rather than `out + out.transpose(1, 2)`, which costs an
    extra (B, d, d, F) tensor.
    """
    rows, cols = triu_pair_indices(d, pair_values.device)
    out = pair_values.new_zeros(pair_values.shape[0], d, d, pair_values.shape[-1])
    out[:, rows, cols] = pair_values
    out[:, cols, rows] = pair_values
    return out


def causal_stream_summaries(
    backbone: LeTFRateMatrix, x: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    """Prefix/suffix interval summaries from the backbone's causal stacks.

    Returns (prefix_summary, suffix_summary), each (B, d, h), with

        prefix_summary[:, i, :] depending only on {t, x_0..x_{i-1}}
        suffix_summary[:, j, :] depending only on {t, x_{j+1}..x_{d-1}}

    Slice derivation in IntervalSwapHead.causal_summaries. Shared with the
    factorised head.
    """
    x_idx = ((x + 1) / 2).long()
    x_emb = backbone.token_embedder(x_idx)  # (B, d, h)
    cond_t = backbone.time_embedder(t).unsqueeze(1)  # (B, 1, h)
    fwd_x = backbone.fwd_stack(torch.cat([cond_t, x_emb], dim=1))
    bwd_x = backbone.bwd_stack(torch.cat([cond_t, x_emb.flip(1)], dim=1)).flip(1)
    d = x.shape[1]
    return fwd_x[:, :d, :], bwd_x[:, 1:, :]


class IntervalSwapHead(nn.Module):
    """One-pass doubly-hollow swap head via three-interval assembly.

    Drop-in `head_kind` (forward: (B, d) state, (B,) time -> (B, d, d) pair
    scores), same contract as the heads in swap_readout.py.

    Args:
        backbone: the leTF rate model; reuses its token/time embedders,
            fwd/bwd causal stacks and omega table. attention_readout is unused.
        pair_offsets: sequence offsets delta for the band's pair features
            u^delta_k = f(x_k, x_{k+delta}). On a flattened D x D lattice,
            offset 1 is row adjacency and offset D column adjacency. Unary
            features are always included.
        band_feature_dim: channels per band-feature family (unary + one per
            offset).
        position_dim: head-owned site-position embedding fed to the pair
            readout (H_ij may depend on positions i, j; blindness constrains
            only the token values there).
        readout_score_scale: fixed multiplier on G, the muP readout
            compensation (MuReadout, Yang et al., arXiv:2203.03466).
            LayerNorm pins ||H_ij|| ~ sqrt(hidden) while omega's per-component
            std (0.002) is width-free, so G and the initial rates ReLU(G) grow
            as sqrt(hidden) (measured 2x at h32 -> h128); hidden_base/hidden
            (e.g. 32/128) cancels that. Not by shrinking omega's init std,
            which is shared with the backbone readout. ReLU is positively
            homogeneous, so this is exactly a rate scale; it changes the
            readout path's effective Adam step, not expressivity. Default 1.0
            skips the multiply and is a float, not a parameter, so archived
            state_dicts and RNG consumption are unchanged.
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        pair_offsets: tuple[int, ...] = (1,),
        band_feature_dim: int = 16,
        position_dim: int = 16,
        readout_score_scale: float = 1.0,
        exterior_combiner: str = "mlp",
        bilinear_rank: int = 8,
        gather_triu_pairs: bool = False,
        site_orderings: tuple[str, ...] = ("row",),
        lattice_side: int | None = None,
    ):
        """exterior_combiner: "mlp" is the archived head (per-pair readout
        over [prefix, suffix, band, positions]). "bilinear" moves the exterior
        out of that MLP into a rank-R product

            H_ij += sum_r a_r(prefix_i, i) * b_r(suffix_j, j),

        the factorised head's exterior at this width, so the MLP sees only
        [band, positions]; everything else is byte-identical, isolating what
        the factorisation itself costs. Blindness holds: the factor maps are
        per-site and prefix_i / suffix_j are causal.
        """
        super().__init__()
        if exterior_combiner not in ("mlp", "bilinear"):
            raise ValueError(
                "exterior_combiner must be 'mlp' or 'bilinear'; "
                f"got {exterior_combiner!r}"
            )
        # The bilinear exterior reads the row ordering's prefix/suffix only
        # (`_ordering_exterior_rows` runs on the "mlp" branch alone), so an
        # extra ordering would be invisible: ("row",) and ("row", "col") give a
        # bit-identical forward (d=16: mlp moves it 1.2e-2, bilinear 0). Refused
        # rather than silently dropping the +0.154 raw second-ordering lever.
        if exterior_combiner == "bilinear" and tuple(site_orderings)[1:]:
            raise ValueError(
                "exterior_combiner='bilinear' reads the row ordering only, so "
                f"the extra orderings in {tuple(site_orderings)!r} would be "
                "invisible; use exterior_combiner='mlp', or the factorised "
                "head, which builds per-ordering factor maps"
            )
        self.exterior_combiner = exterior_combiner
        self.bilinear_rank = bilinear_rank
        # Opt-in band/readout on the d(d-1)/2 pairs, mirrored back by
        # scatter_symmetric_pairs; no parameter or RNG draw, flag-off is archived.
        self.gather_triu_pairs = gather_triu_pairs
        self.site_orderings = tuple(site_orderings)
        self.position_dim = position_dim
        self.readout_score_scale = readout_score_scale
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

        # Registered before the readout only because `_build_pair_readout`
        # reads `site_orderings` for its width; no parameters, no RNG draw.
        self._register_site_orderings(lattice_side)

        band_dim = band_feature_dim * (1 + len(self.pair_offsets))
        self.pair_readout = self._build_pair_readout(band_dim)
        # Mirrors the letf readout's closing "output_norm(H) + time" line.
        self.context_norm = nn.LayerNorm(hidden)
        if exterior_combiner == "bilinear":
            # After the archived modules so the "mlp" path's init draws are
            # untouched; the factor maps mirror the factorised head's.
            self.prefix_norm = nn.LayerNorm(hidden)
            self.suffix_norm = nn.LayerNorm(hidden)
            self.prefix_factors = nn.Linear(
                hidden + position_dim, bilinear_rank * hidden
            )
            self.suffix_factors = nn.Linear(
                hidden + position_dim, bilinear_rank * hidden
            )

    def _build_pair_readout(self, band_dim: int) -> nn.Sequential:
        """Per-pair MLP; its input carries the exterior summaries only under
        the "mlp" combiner. Shared with the masked-attention stencil rebuild."""
        hidden = self.backbone.hidden_dim
        # One (prefix, suffix) pair per ordering under "mlp": an extra ordering
        # reuses the backbone stacks, so this widening is its entire parameter cost.
        exterior_dim = (
            2 * hidden * len(self.site_orderings)
            if self.exterior_combiner == "mlp"
            else 0
        )
        return nn.Sequential(
            nn.Linear(exterior_dim + band_dim + 2 * self.position_dim, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )

    def _register_site_orderings(self, lattice_side: int | None) -> None:
        """Permutations and per-pair (min, max) o-position grids, as buffers.

        For ordering o with `order`: o-position -> site and `inv` its inverse,
        the pair {i, j} sits at o-positions inv[i], inv[j]. The head reads the
        prefix stream at min(inv[i], inv[j]) and the suffix at max, so each is
        blind to x_i and x_j by the row ordering's causality argument; min and
        max are symmetric, so H_ji = H_ij needs no mirror. Non-persistent
        buffers (reproducible from the constructor args), as in the factorised
        head. "row" is the identity and registers nothing.
        """
        from discrete_flow_sampler.constraints.factorised_swap_head import (
            lattice_site_ordering,
        )

        if self.site_orderings[:1] != ("row",):
            raise ValueError(
                f"site_orderings must start with 'row' (the flattening "
                f"itself); got {self.site_orderings!r}"
            )
        side = lattice_side if lattice_side is not None else round(self.d**0.5)
        for name in self.site_orderings[1:]:
            if side * side != self.d:
                raise ValueError(
                    f"extra site_orderings need a square lattice: side {side} "
                    f"does not tile d={self.d}; pass lattice_side explicitly"
                )
            order = lattice_site_ordering(name, self.d, side)
            inverse = order.argsort()
            self.register_buffer(f"_order_{name}", order, persistent=False)
            rows, cols = inverse.view(-1, 1), inverse.view(1, -1)
            self.register_buffer(
                f"_order_lo_{name}", torch.minimum(rows, cols), persistent=False
            )
            self.register_buffer(
                f"_order_hi_{name}", torch.maximum(rows, cols), persistent=False
            )

    def _ordering_exterior_rows(
        self, x: Tensor, t: Tensor, pairs: tuple[Tensor, Tensor] | None
    ) -> list[Tensor]:
        """(prefix, suffix) grids for each extra ordering, in site space.

        Two tensors per extra ordering, (B, d, d, h) densely or (B, P, h)
        under the triu gather, ready to concatenate into the readout input.
        """
        rows_out: list[Tensor] = []
        for name in self.site_orderings[1:]:
            order = getattr(self, f"_order_{name}")
            prefix, suffix = causal_stream_summaries(self.backbone, x[:, order], t)
            lo, hi = (
                getattr(self, f"_order_lo_{name}"),
                getattr(self, f"_order_hi_{name}"),
            )
            if pairs is not None:
                pair_rows, pair_cols = pairs
                lo, hi = lo[pair_rows, pair_cols], hi[pair_rows, pair_cols]
            rows_out += [prefix[:, lo], suffix[:, hi]]
        return rows_out

    def _bilinear_exterior(
        self, prefix_summary: Tensor, suffix_summary: Tensor
    ) -> Tensor:
        """sum_r a_r(prefix_i, i) * b_r(suffix_j, j), (B, d, d, h)."""
        batch, d, hidden = prefix_summary.shape
        position = self.pair_position_embedding(
            torch.arange(d, device=prefix_summary.device)
        )
        position = position.unsqueeze(0).expand(batch, -1, -1)
        shape = (batch, d, self.bilinear_rank, hidden)
        a = self.prefix_factors(
            torch.cat([self.prefix_norm(prefix_summary), position], -1)
        ).view(shape)
        b = self.suffix_factors(
            torch.cat([self.suffix_norm(suffix_summary), position], -1)
        ).view(shape)
        return torch.einsum("birh,bjrh->bijh", a, b)

    def causal_summaries(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Prefix/suffix interval summaries from the backbone's causal stacks.

        Returns (prefix_summary, suffix_summary), each (B, d, h):

            prefix_summary[:, i, :] depends only on {t, x_0..x_{i-1}}  (x_{<i})
            suffix_summary[:, j, :] depends only on {t, x_{j+1}..x_{d-1}} (x_{>j})

        With cond_t prepended, the inclusive-causal fwd output at slot m
        depends on cond_t plus tokens 0..m-1, so slot i is the deepest state
        that has never seen x_i (i=0 is the cond_t-only empty-prefix state).
        The bwd stack runs on the flipped sequence with the same prepend;
        flipped back, slot k depends on x_{>=k}, so slot j+1 is the deepest
        state blind to x_{<=j} (slot d = empty suffix). The failure mode is an
        off-by-one in either slice; the blindness tests flip the boundary
        sites. Body in module-level `causal_stream_summaries`.
        """
        return causal_stream_summaries(self.backbone, x, t)

    def band_summaries(
        self, x: Tensor, t: Tensor, pairs: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """All-pairs middle-band statistics, (B, d, d, F); valid for i < j.

        F = band_feature_dim * (1 + len(pair_offsets)). Entry [:, i, j, :]
        aggregates local features over the open interval (i, j), with no term
        touching site i or j:

            unary:            sum_{k = i+1 .. j-1}        v_k,
                              v_k = band_unary_features(emb(x_k))
            offset delta:     sum_{k = i+1 .. j-1-delta}  u^delta_k,
                              u^delta_k
                                = band_pair_features(emb(x_k) ++ emb(x_{k+delta}))

        The offset range is the straddle exclusion: u^delta_k touches
        (k, k+delta), so interior terms need k > i and k + delta < j.
        Exclusion is index arithmetic only; blindness cannot depend on the
        excluded values. Each family is one prefix-sum, O(d), then one
        subtraction per pair; empty ranges (j - i <= delta + 1) must give
        exact zeros. Only the i < j triangle is consumed; the lower triangle
        is unspecified. The fp residue (module docstring) is on the x_i side
        only, since x_j never enters the cumsums. `t` is unused: time enters
        via the pair readout. `pairs` = (rows, cols) selects a list of pairs
        and returns (B, P, F); only the index shapes change.
        """
        del t
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)  # (B, d, h)
        d = self.d
        if pairs is None:
            site_i = torch.arange(d, device=x.device).view(d, 1)  # over j
            site_j = torch.arange(d, device=x.device).view(1, d)  # over i
            mask_shape = (1, d, d, 1)
        else:
            site_i, site_j = pairs  # (P,), (P,)
            mask_shape = (1, -1, 1)

        def cumsum_with_zero(features: Tensor) -> Tensor:
            zero = features.new_zeros(features.shape[0], 1, features.shape[-1])
            return torch.cat([zero, features.cumsum(dim=1)], dim=1)

        families = []
        # Unary: sum_{k in [i+1, j-1]} v_k = c[j] - c[i+1]; empty when j-i < 2.
        unary_prefix = cumsum_with_zero(self.band_unary_features(emb))
        unary_sum = unary_prefix[:, site_j] - unary_prefix[:, site_i + 1]
        families.append(
            torch.where((site_j - site_i >= 2).view(mask_shape), unary_sum, 0.0)
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
                    (site_j - site_i >= delta + 2).view(mask_shape), pair_sum, 0.0
                )
            )
        return torch.cat(families, dim=-1)  # (B, d, d, F) or (B, P, F)

    def hole_free_bond_totals(
        self, x: Tensor, pairs: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """Whole-lattice bond sums with every hole-touching bond removed,
        (B, d, d, F) or (B, P, F), F = band_feature_dim * len(pair_offsets).

        Entry [:, i, j] is

            sum over k in [0, d - delta)  of  u^delta_k,
            restricted to  k not in {i, j}  and  k + delta not in {i, j},   (*)

        per offset delta, concatenated. `band_summaries` sums the same
        features over the open interval (i, j); a head carrying both can
        express their difference, the exterior bond sum, which neither gives
        alone. Exclusion in (*) is index arithmetic, so the result is blind.

        Failure mode guarded: u^delta_k touches k and k + delta, so (*) drops
        k in {i, i-delta, j, j-delta}, and that set collides when |i - j| =
        delta, which with pair_offsets (1, D) is exactly the nearest-neighbour
        pairs. Subtracting all four removes one term twice and leaves
        -u^delta, which depends on the hole spins, so blindness fails; hence
        the inclusion-exclusion add-back below. Boundary cases (k < 0, or
        u^delta_i not existing) use the clamp-and-mask idiom of
        `band_summaries`. Symmetric in (i, j) by construction, so no mirror.
        Valid for i != j; the diagonal is unspecified and never reaches G
        (omega_{x_i} - omega_{x_j} = 0 there).
        """
        x_idx = ((x + 1) / 2).long()
        emb = self.backbone.token_embedder(x_idx)  # (B, d, h)
        d = self.d
        if pairs is None:
            hole_i = torch.arange(d, device=x.device).view(d, 1)
            hole_j = torch.arange(d, device=x.device).view(1, d)
            mask_shape = (1, d, d, 1)
        else:
            hole_i, hole_j = pairs  # (P,), (P,)
            mask_shape = (1, -1, 1)

        families = []
        for delta, feature_mlp in zip(self.pair_offsets, self.band_pair_features):
            n_terms = d - delta
            terms = feature_mlp(
                torch.cat([emb[:, :n_terms], emb[:, delta:]], dim=-1)
            )  # (B, n_terms, F)

            def term_at(index: Tensor) -> Tensor:
                """u^delta_index, or exact zero where no such bond exists.

                The validity mask is shaped from the index ((d, 1)/(1, d)
                dense, (P,) gathered) so it broadcasts against the pair shape;
                `mask_shape` is for the pair-shaped adjacency test.
                """
                inside = ((index >= 0) & (index < n_terms)).unsqueeze(0).unsqueeze(-1)
                return torch.where(inside, terms[:, index.clamp(0, n_terms - 1)], 0.0)

            total = terms.sum(dim=1)
            total = total.view(x.shape[0], *([1] * (len(mask_shape) - 2)), -1)
            hole_free = (
                total
                - term_at(hole_i)
                - term_at(hole_i - delta)
                - term_at(hole_j)
                - term_at(hole_j - delta)
            )
            # Inclusion-exclusion: when the pair is a delta-bond, one term was
            # reached by two gathers above. Both signs keep (i, j) symmetry.
            adjacent_forward = (hole_j - hole_i == delta).view(mask_shape)
            adjacent_backward = (hole_i - hole_j == delta).view(mask_shape)
            hole_free = (
                hole_free
                + torch.where(adjacent_forward, term_at(hole_i), 0.0)
                + torch.where(adjacent_backward, term_at(hole_j), 0.0)
            )
            families.append(hole_free)
        return torch.cat(families, dim=-1)

    def compute_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """Assemble H, (B, d, d, h): the doubly-blind context for every pair.

        For i < j:

            H_ij = context_norm(pair_readout(
                       prefix[:, i] ++ suffix[:, j] ++ band[:, i, j]
                       ++ pos_emb(i) ++ pos_emb(j)
                   )) + time_embedder(t)

        then H[:, j, i] := H[:, i, j]. All d^2 rows go through pair_readout
        as one batched MLP. Exposed separately from forward so the tests can
        probe blindness on H directly (flip x_i / x_j / both), which is
        stronger than G's antisymmetry. Under `gather_triu_pairs` the same
        assembly runs on the i < j pairs and is mirrored by
        `scatter_symmetric_pairs`; only the diagonal differs (zero).
        """
        prefix_summary, suffix_summary = self.causal_summaries(x, t)
        batch, d, hidden = prefix_summary.shape
        position = self.pair_position_embedding(torch.arange(d, device=x.device))
        if self.gather_triu_pairs:
            return self._gathered_pair_context(
                x, t, prefix_summary, suffix_summary, position
            )
        band = self.band_summaries(x, t)  # (B, d, d, F)

        exterior_rows = (
            [
                prefix_summary.unsqueeze(2).expand(batch, d, d, hidden),
                suffix_summary.unsqueeze(1).expand(batch, d, d, hidden),
            ]
            + self._ordering_exterior_rows(x, t, None)
            if self.exterior_combiner == "mlp"
            else []
        )
        H = self.pair_readout(
            torch.cat(
                exterior_rows
                + [
                    band,
                    position.view(1, d, 1, -1).expand(batch, d, d, -1),
                    position.view(1, 1, d, -1).expand(batch, d, d, -1),
                ],
                dim=-1,
            )
        )
        if self.exterior_combiner == "bilinear":
            H = H + self._bilinear_exterior(prefix_summary, suffix_summary)
        H = self.context_norm(H) + self.backbone.time_embedder(t).view(
            batch, 1, 1, hidden
        )
        # Label symmetry: the i<j triangle is the definition, mirror it down.
        upper = torch.triu(torch.ones(d, d, dtype=torch.bool, device=x.device)).view(
            1, d, d, 1
        )
        return torch.where(upper, H, H.transpose(1, 2))

    def _gathered_pair_context(
        self,
        x: Tensor,
        t: Tensor,
        prefix_summary: Tensor,
        suffix_summary: Tensor,
        position: Tensor,
    ) -> Tensor:
        """`compute_pair_context` on the i < j pairs only (the memory lever).

        Every readout tensor drops from (B, d^2, .) to (B, d(d-1)/2, .),
        including the masked-attention subclass's attention scores. The
        bilinear exterior stays dense and is indexed afterwards: one
        (d x Rh)(Rh x d) matmul is faster whole than a gathered product.
        """
        batch, d, hidden = prefix_summary.shape
        rows, cols = triu_pair_indices(d, x.device)
        exterior_rows = (
            [prefix_summary[:, rows], suffix_summary[:, cols]]
            + self._ordering_exterior_rows(x, t, (rows, cols))
            if self.exterior_combiner == "mlp"
            else []
        )
        H = self.pair_readout(
            torch.cat(
                exterior_rows
                + [
                    self.band_summaries(x, t, (rows, cols)),
                    position[rows].unsqueeze(0).expand(batch, -1, -1),
                    position[cols].unsqueeze(0).expand(batch, -1, -1),
                ],
                dim=-1,
            )
        )
        if self.exterior_combiner == "bilinear":
            H = (
                H
                + self._bilinear_exterior(prefix_summary, suffix_summary)[:, rows, cols]
            )
        H = self.context_norm(H) + self.backbone.time_embedder(t).view(batch, 1, hidden)
        return scatter_symmetric_pairs(H, d)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), via the swap readout.

            G[:, i, j] = < H_ij, omega_{x_i} - omega_{x_j} >

        Diagonal and same-spin pairs vanish (zero token difference);
        G[j,i] = -G[i,j] follows from H's label symmetry, state-swap
        antisymmetry from H's value-blindness.
        """
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)  # (B, d, h)
        token_difference = omega.unsqueeze(2) - omega.unsqueeze(1)
        H = self.compute_pair_context(x, t)
        # mul+sum, not einsum: einsum is on the autocast lower-precision list,
        # so under eval_autocast_bf16 it would emit bf16 G and crash the
        # fp32-only quantile rate diagnostic; the mask-one readout does the same.
        scores = (token_difference * H).sum(-1)
        if self.readout_score_scale != 1.0:
            # muP readout compensation (see __init__); guarded so the
            # default path stays byte-identical to the archived readout.
            scores = scores * self.readout_score_scale
        return scores
