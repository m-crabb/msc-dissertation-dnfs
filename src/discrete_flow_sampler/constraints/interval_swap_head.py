"""Three-interval (leave-two-out) swap head: all-pairs H_ij in ONE body pass.

Uses swap_readout.py's pair form of DNFS Prop. 2 / Eq. (9):

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
  per-pair readout MLP. This is the direction's capacity gamble; the D=4
  gate prices it.

Efficiency contract: O(d) precompute (causal stacks, feature prefix
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

_TRIU_PAIR_CACHE: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}


def triu_pair_indices(d: int, device) -> tuple[Tensor, Tensor]:
    """Row/column index vectors of the d(d-1)/2 unordered pairs i < j.

    Shared by every head whose per-pair work is label-symmetric (this head,
    the masked-attention band, the factorised global term): those heads
    compute H on all d^2 ordered pairs and then mirror the i < j triangle
    down, so half the nonlinear work -- and half of the (B, d^2, F)
    activation slab that is the heads' memory footprint at d = 256 -- is
    the mirror image of the other half.

    Cached per (d, device) exactly like the sampler's own pair table
    (`_swap_neighbours.upper_tri_pairs`, kept separate so `constraints`
    does not import `samplers`): every forward re-reads the same indices and
    the list grows as d^2, so rebuilding it per call is pure dispatch
    overhead. Callers treat the result as read-only.
    """
    key = (d, torch.device(device))
    if key not in _TRIU_PAIR_CACHE:
        rows, cols = torch.triu_indices(d, d, offset=1, device=device)
        _TRIU_PAIR_CACHE[key] = (rows, cols)
    return _TRIU_PAIR_CACHE[key]


def scatter_symmetric_pairs(pair_values: Tensor, d: int) -> Tensor:
    """(B, P, F) per-pair values on i < j -> the symmetric (B, d, d, F) block.

    The inverse of the triu gather, and the point at which the heads' label-
    symmetry convention H_ji := H_ij becomes an identity rather than a
    numerical property: ONE tensor is written into both triangles, so
    H == H.transpose(1, 2) bit-exactly and index antisymmetry of G follows.

    The DIAGONAL is left at zero rather than recomputed. forward reads H
    against omega_{x_i} - omega_{x_j}, which is identically zero at i = j, so
    the dense path's diagonal never reaches G; reproducing it would reinstate
    d of the rows this lever exists to drop. (Consequence, stated because it
    is a real difference: `compute_pair_context` diagonals differ between the
    two paths. Nothing downstream reads them.)

    Written as two index_put's into one freshly-allocated output. The two
    rejected alternatives -- `out + out.transpose(1, 2)`, and a gather
    through a pad-slot index map -- each cost an extra (B, d, d, F)
    tensor, which is precisely what the lever exists to save.
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

    Shared by every head that assembles pair contexts from the leTF
    slice-trick objects (this head and the factorised head). Returns
    (prefix_summary, suffix_summary), each (B, d, h), with

        prefix_summary[:, i, :] depending ONLY on {t, x_0..x_{i-1}}
        suffix_summary[:, j, :] depending ONLY on {t, x_{j+1}..x_{d-1}}

    -- see IntervalSwapHead.causal_summaries for the full derivation of the
    slice indices; the classic failure is an off-by-one in either slice, and
    the blindness tests probe the boundary sites specifically to catch it.
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
        readout_score_scale: fixed multiplier on the pair scores G — the
            muP readout compensation (MuReadout's output multiplier, Yang
            et al., arXiv:2203.03466). The score chain
            `context_norm -> <H, omega_diff>` has no fan-in compensation:
            LayerNorm pins ||H_ij|| ~ sqrt(hidden) while omega's
            per-component std (0.002) is width-free, so G — and with it
            the initial rates ReLU(G) — grows as sqrt(hidden) (verified
            2x at h32 -> h128). Setting hidden_base/hidden (e.g. 32/128)
            cancels that growth with muP's extra 1/sqrt(width) margin, so
            rates start small and unclipped at any width. A multiplier is
            used rather than the two rejected alternatives: zero-init of
            pair_readout's last layer feeds context_norm an exactly-zero
            input (LayerNorm's 1/sqrt(eps) gradient there), and shrinking
            omega's init std touches a table shared with the backbone
            readout, breaking archived parity for every head. ReLU is
            positively homogeneous, so the scale is exactly a rate scale;
            expressivity is untouched (the model can learn to undo it) —
            what changes is the readout path's effective step size under
            Adam, which is the intended muP dynamics change. Default 1.0
            = every archived cell: the multiply is skipped entirely, and
            the attribute is a float, not a parameter, so state_dict and
            RNG consumption are unchanged either way.
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
        """exterior_combiner: "mlp" is the archived head, the
        per-pair readout over [prefix, suffix, band, positions]. "bilinear"
        moves the deep exterior OUT of that MLP and into a rank-R product

            H_ij += sum_r a_r(prefix_i, i) * b_r(suffix_j, j),

        the factorised head's exterior at this head's hidden width, so the
        per-pair MLP sees only [band, positions]. Everything else -- band,
        context_norm, time line, H width -- is byte-identical, which makes
        this the single-variable test of what the factorisation itself
        costs: the factorised-head cells change the chassis at the same
        time (global sum, per-term scaling, factor width). Blindness is
        unchanged: the factor maps are per-site, prefix_i is blind to
        x_{>=i} and suffix_j to x_{<=j} by causality, and the post-sum
        LayerNorm is admissible here because H is materialised anyway.
        """
        super().__init__()
        if exterior_combiner not in ("mlp", "bilinear"):
            raise ValueError(
                f"exterior_combiner must be 'mlp' or 'bilinear'; got {exterior_combiner!r}"
            )
        # The bilinear exterior reads the ROW ordering's prefix/suffix only:
        # `_ordering_exterior_rows` is called on the "mlp" branch alone, and
        # `_bilinear_exterior` takes the row summaries. So an extra ordering
        # under this combiner is not merely dead weight, it is INVISIBLE --
        # ("row",) and ("row", "col") build the same parameters and return a
        # bit-identical forward. Measured at d=16: mlp gains 256 parameters
        # and moves the forward by 1.2e-2, bilinear gains 0 and moves it by 0.
        # Without this raise, `ivmo2ef` + bilinear reads as a single-variable
        # test of the factorisation while silently also deleting the second
        # ordering, worth +0.154 raw and disjoint -- the largest lever on this
        # axis. FactorisedSwapHead carries the mirror check because it builds
        # per-ordering factor maps (`extra_ordering_modules`); this head does
        # not, so the combination is refused rather than approximated.
        if exterior_combiner == "bilinear" and tuple(site_orderings)[1:]:
            raise ValueError(
                "exterior_combiner='bilinear' reads the row ordering only, so "
                f"the extra orderings in {tuple(site_orderings)!r} would be "
                "invisible; use exterior_combiner='mlp', or the factorised "
                "head, which builds per-ordering factor maps"
            )
        self.exterior_combiner = exterior_combiner
        self.bilinear_rank = bilinear_rank
        # Opt-in band/readout on d(d-1)/2 unordered pairs, then mirror via
        # scatter_symmetric_pairs. No parameter, buffer or RNG draw: flag-off
        # heads stay byte-identical to archived ones; only GEMM shape differs.
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

        # Extra orderings register BEFORE the readout is drawn only because
        # `_build_pair_readout` reads `site_orderings` for its width; they add
        # no parameters and no RNG draw, so ('row',) stays byte-identical to
        # every archived raster cell.
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
        # One (prefix, suffix) pair PER ORDERING under the "mlp" combiner: an
        # extra ordering owns no modules of its own -- it reuses the
        # backbone's causal stacks on a permuted sequence -- so this widening
        # is the extra ordering's ENTIRE parameter cost, which keeps a lift from
        # being confounded with capacity.
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

        For ordering o with `order` mapping o-position -> site and `inv` the
        inverse, the pair {i, j} sits at o-positions inv[i], inv[j]. The head
        reads the prefix stream at min(inv[i], inv[j]) and the suffix stream
        at max: the prefix has then seen only sites earlier in o than BOTH
        holes, and the suffix only sites later than both, so each is blind to
        x_i and x_j by exactly the causality argument the row ordering uses.
        min and max of an UNORDERED pair are symmetric, so label symmetry
        H_ji = H_ij comes for free rather than needing a mirror.

        Index arithmetic reproducible from the constructor args, so these ride
        as NON-persistent buffers and never enter a checkpoint -- the
        convention the factorised head already follows. "row" is the identity
        and registers nothing, which is what keeps the archived path clean.
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
        """(prefix, suffix) grids for each EXTRA ordering, in site space.

        Returns 2 tensors per extra ordering, shaped (B, d, d, h) densely or
        (B, P, h) under the triu gather -- ready to concatenate into the pair
        readout's input alongside the row ordering's own pair.
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
        The body lives in module-level `causal_stream_summaries` so the
        factorised head can share it without inheriting this head's band.
        """
        return causal_stream_summaries(self.backbone, x, t)

    def band_summaries(
        self, x: Tensor, t: Tensor, pairs: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
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

        `pairs` = (rows, cols) selects a LIST of pairs instead of the grid and
        returns (B, P, F). Only the index tensors change shape -- every
        prefix-sum and every exclusion mask below is written once and reads
        (d, 1)/(1, d) broadcast indices or (P,) list indices interchangeably --
        because the band is O(1) per pair either way. This is the entry point
        the triu-pair gather uses (the band is DEFINED on i < j, so the
        gathered form needs no mirror at all).
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
        """WHOLE-LATTICE bond sums with every hole-touching bond removed,
        (B, d, d, F) or (B, P, F), F = band_feature_dim * len(pair_offsets).

        Entry [:, i, j] is

            sum over k in [0, d - delta)  of  u^delta_k,
            restricted to  k not in {i, j}  and  k + delta not in {i, j},   (*)

        for each offset delta, concatenated. `band_summaries` gives the same
        features summed over the OPEN INTERVAL (i, j); this gives them summed
        over the whole lattice. Neither recovers the other -- a part is not a
        total -- so a head carrying both can express their difference, the
        EXTERIOR bond sum, which neither gives alone. That is already the
        situation on the unary side, where the global term's per-site sum and
        the band's unary sum coexist; this restores the missing basis vector
        on the bond side.

        Blindness. Exclusion in (*) is decided by INDEX arithmetic alone, so
        the result cannot depend on the values excluded -- the same argument
        as `band_summaries`, and the reason this is safe to feed the global
        term's per-pair path.

        THE FAILURE MODE THIS GUARDS AGAINST. u^delta_k touches sites k and
        k + delta, so (*) drops k in {i, i-delta, j, j-delta} -- four gathers,
        not the global term's two. That set COLLIDES when |i - j| = delta, and
        with pair_offsets (1, D) those are exactly the nearest-neighbour pairs
        the Ising energy is built from. Subtracting all four blindly removes
        one term TWICE, which leaves -u^delta in the residual; u^delta depends
        on the hole spins, so BLINDNESS FAILS, on the pairs that matter most.
        Hence the inclusion-exclusion add-back below. The two boundary cases
        (k = i - delta < 0, and i >= d - delta so u^delta_i does not exist)
        are handled by the same clamp-and-mask idiom `band_summaries` uses for
        its empty ranges.

        Symmetric in (i, j) by construction -- the four gathers treat the two
        holes identically -- which the global term requires, so unlike the
        band this needs no mirror on either path.

        VALID FOR i != j; the diagonal is unspecified, as `band_summaries`
        leaves its lower triangle unspecified. At i == j the four gathers
        reduce to two distinct indices, each subtracted twice, and the
        add-back below does not fire (j - i = 0 is not an offset). Correcting
        it would put two more masked adds on the per-pair path for entries
        that cannot reach a result: the readout multiplies H by
        omega_{x_i} - omega_{x_j}, which is identically zero on the diagonal,
        and the head's exact antisymmetry pins G_ii = 0 for every input.
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

                The validity mask is shaped from the INDEX, not from the pair
                grid: a single hole's index is (d, 1) / (1, d) on the dense
                path and (P,) on the gathered one, so it broadcasts against
                the pair shape rather than filling it. `mask_shape` below is
                for the adjacency test, which is a function of BOTH holes and
                so is pair-shaped already.
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
            # Inclusion-exclusion: when the pair IS a delta-bond, one term was
            # reached by two of the four gathers above. Both signs, so the
            # result stays symmetric in (i, j).
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

        then H[:, j, i] := H[:, i, j] (label symmetry). All O(d^2) rows go
        through pair_readout as one batched MLP over the trailing dim --
        broadcast prefix over j, suffix over i, positions over batch.

        Exposed separately from forward so the falsification tests can probe
        blindness on H directly (flip x_i / x_j / both: H_ij must not move)
        -- a strictly stronger check than G's antisymmetry, which a
        symmetric leak (H depending on x_i + x_j, say) would survive.

        Under `gather_triu_pairs` the same assembly runs on the d(d-1)/2
        pairs with i < j and is mirrored by `scatter_symmetric_pairs`: the
        mirror is what the dense path does anyway, so the lower triangle's
        readout rows were always thrown away. Only the diagonal differs
        (left at zero; it never reaches G -- see `scatter_symmetric_pairs`).
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

        Every tensor the pair readout touches drops from (B, d^2, .) to
        (B, d(d-1)/2, .): the band -- for the masked-attention subclass, the
        (B, d^2, n_terms) attention SCORES its docstring flags as the d = 256
        price -- the readout's concatenated input, its hidden activation and
        the LayerNorm. The bilinear exterior stays DENSE and is indexed after
        the fact: it is a single (d x Rh)(Rh x d) matmul, which is faster
        whole than a gathered elementwise product, and its output is the same
        size as the (B, d, d, h) result the head must return regardless.
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

        (einsum over the hidden dim against the omega difference, exactly as
        the swap_readout.py heads). The diagonal and same-spin pairs vanish
        for free (zero token difference); index-antisymmetry G[j,i] = -G[i,j]
        follows from H's label symmetry; state-swap antisymmetry follows
        from H's value-blindness -- the property the whole head exists to
        provide, and the only one training never touches.
        """
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)  # (B, d, h)
        token_difference = omega.unsqueeze(2) - omega.unsqueeze(1)
        H = self.compute_pair_context(x, t)
        # mul+sum, NOT einsum: einsum is on the autocast lower-precision
        # list, so under the Tier-2 eval_autocast_bf16 block it would emit
        # bf16 G (crashing the fp32-only quantile rate diagnostic and
        # departing from the dtype path Tier-2 was validated on); the
        # mask-one readout keeps G fp32 the same way.
        scores = (token_difference * H).sum(-1)
        if self.readout_score_scale != 1.0:
            # muP readout compensation (see __init__); guarded so the
            # default path stays byte-identical to the archived readout.
            scores = scores * self.readout_score_scale
        return scores
