"""Three-interval (leave-two-out) swap head: all-pairs H_ij in ONE body pass.

Direction (b) of the pair-equivariant spike (2026-07-07). The readout is
unchanged from swap_readout.py (the pair form of DNFS Prop. 2 / Eq. (9)):

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
    x_emb = backbone.token_embedder(x_idx)                    # (B, d, h)
    cond_t = backbone.time_embedder(t).unsqueeze(1)           # (B, 1, h)
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
            et al., arXiv:2203.03466), 2026-08-18. The score chain
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
    ):
        """exterior_combiner (2026-08-23): "mlp" is the archived head, the
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
            raise ValueError(f"exterior_combiner must be 'mlp' or 'bilinear'; got {exterior_combiner!r}")
        self.exterior_combiner = exterior_combiner
        self.bilinear_rank = bilinear_rank
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

        band_dim = band_feature_dim * (1 + len(self.pair_offsets))
        self.pair_readout = self._build_pair_readout(band_dim)
        # Mirrors the letf readout's closing "output_norm(H) + time" line.
        self.context_norm = nn.LayerNorm(hidden)
        if exterior_combiner == "bilinear":
            # After the archived modules so the "mlp" path's init draws are
            # untouched; the factor maps mirror the factorised head's.
            self.prefix_norm = nn.LayerNorm(hidden)
            self.suffix_norm = nn.LayerNorm(hidden)
            self.prefix_factors = nn.Linear(hidden + position_dim, bilinear_rank * hidden)
            self.suffix_factors = nn.Linear(hidden + position_dim, bilinear_rank * hidden)

    def _build_pair_readout(self, band_dim: int) -> nn.Sequential:
        """Per-pair MLP; its input carries the exterior summaries only under
        the "mlp" combiner. Shared with the masked-attention stencil rebuild."""
        hidden = self.backbone.hidden_dim
        exterior_dim = 2 * hidden if self.exterior_combiner == "mlp" else 0
        return nn.Sequential(
            nn.Linear(exterior_dim + band_dim + 2 * self.position_dim, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )

    def _bilinear_exterior(self, prefix_summary: Tensor, suffix_summary: Tensor) -> Tensor:
        """sum_r a_r(prefix_i, i) * b_r(suffix_j, j), (B, d, d, h)."""
        batch, d, hidden = prefix_summary.shape
        position = self.pair_position_embedding(torch.arange(d, device=prefix_summary.device))
        position = position.unsqueeze(0).expand(batch, -1, -1)
        shape = (batch, d, self.bilinear_rank, hidden)
        a = self.prefix_factors(torch.cat([self.prefix_norm(prefix_summary), position], -1)).view(shape)
        b = self.suffix_factors(torch.cat([self.suffix_norm(suffix_summary), position], -1)).view(shape)
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

        exterior_rows = (
            [
                prefix_summary.unsqueeze(2).expand(batch, d, d, hidden),
                suffix_summary.unsqueeze(1).expand(batch, d, d, hidden),
            ]
            if self.exterior_combiner == "mlp" else []
        )
        H = self.pair_readout(
            torch.cat(
                exterior_rows + [
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
        scores = (token_difference * H).sum(-1)
        if self.readout_score_scale != 1.0:
            # muP readout compensation (see __init__); guarded so the
            # default path stays byte-identical to the archived readout.
            scores = scores * self.readout_score_scale
        return scores
