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

Multi-order causal streams (the design's A-prime extension, opt-in via
`site_orderings`): the bilinear term under the row-major ordering is
structurally blind to the whole raster interval between its holes -- prefix
stops before i, suffix starts after j -- so the interval interior is covered
only by the shallow global term. Running the causal stacks under EXTRA site
orderings (column-major, anti-diagonal) gives every pair a second split,

    H_ij += sum_o sum_r a^o_r(P^o_first, first) * b^o_r(S^o_last, last),

first/last taken in ordering o, so a site interior to the row interval but
exterior in ordering o gains DEEP coverage; what remains invisible to the
bilinear terms shrinks to the intersection of the per-ordering intervals.
This adds INFORMATION, not capacity -- the rank/width axes were measured
and refuted as the deficit's cause at the 4x4 gate, while the interior is
where the trained field's deviation from the reference concentrates -- and
blindness stays bit-exact by causality in every ordering (the earlier hole
bounds the prefix, the later hole the suffix). The backbone stacks are
SHARED across orderings (one extra forward pass each, the cheap term);
only the factor maps are per-ordering. `("row",)` is byte-identical to the
single-ordering head: no extra modules, no persistent state, archived
checkpoints load unchanged.

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


def lattice_site_ordering(name: str, d: int, lattice_side: int) -> Tensor:
    """o-position -> site map for a flattened lattice_side^2 lattice.

    "row" is the identity (the flattening itself); "col" walks columns; "diag"
    walks anti-diagonals r+c = 0, 1, ... with row-major tie-break, the third
    independent sweep direction on a square lattice. A pair far apart in one
    sweep is often close in another, which is what the multi-order extension
    trades on.
    """
    if name == "row":
        return torch.arange(d)
    rows = torch.arange(d) // lattice_side
    cols = torch.arange(d) % lattice_side
    if name == "col":
        return (cols * lattice_side + rows).argsort(stable=True)
    if name == "diag":
        # sort sites by (r + c, r): smaller anti-diagonal first, then row.
        key = (rows + cols) * lattice_side + rows
        return key.argsort(stable=True)
    raise ValueError(f"unknown site ordering {name!r}; known: row, col, diag")


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
        site_orderings: causal-sweep directions for the bilinear factors
            (the A-prime extension; see the module docstring). Must start
            with "row" -- the archived-checkpoint module tree -- and the
            default ("row",) adds nothing: no extra modules, no persistent
            state. Extras from {"col", "diag"} each add one shared-backbone
            pass and their own factor maps.
        lattice_side: D of the flattened D x D lattice; required by (and
            only by) the extra orderings, whose permutations are index
            arithmetic on (row, col).
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
        site_orderings: tuple[str, ...] = ("row",),
        lattice_side: int | None = None,
    ):
        super().__init__()
        if not (use_bilinear or use_global):
            raise ValueError(
                "FactorisedSwapHead needs at least one of use_bilinear / "
                "use_global: with both off, pair scores depend only on time "
                "and positions."
            )
        site_orderings = tuple(site_orderings)
        if not site_orderings or site_orderings[0] != "row":
            raise ValueError(
                "site_orderings must start with 'row' (the base ordering "
                f"whose modules archived checkpoints carry); got "
                f"{site_orderings!r}"
            )
        if len(set(site_orderings)) != len(site_orderings):
            raise ValueError(f"duplicate site ordering in {site_orderings!r}")
        if site_orderings[1:] and not use_bilinear:
            raise ValueError(
                "extra site_orderings extend the bilinear term; with "
                "use_bilinear=False they would be dead weight"
            )
        if site_orderings[1:]:
            if lattice_side is None or lattice_side * lattice_side != backbone.d:
                raise ValueError(
                    "extra site_orderings need lattice_side with "
                    f"lattice_side**2 == d; got lattice_side={lattice_side} "
                    f"for d={backbone.d}"
                )
        self.backbone = backbone
        self.d = backbone.d
        self.bilinear_rank = bilinear_rank
        self.factor_dim = factor_dim
        self.use_bilinear = use_bilinear
        self.use_global = use_global
        self.site_orderings = site_orderings
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

        # Extra orderings LAST, so the base modules' init draws are identical
        # for any k at a fixed seed (and absent entirely at k=1, keeping the
        # module tree byte-compatible with pre-extension checkpoints). The
        # permutations ride as NON-persistent buffers: index arithmetic,
        # reproducible from the constructor args, invisible to state_dict.
        for name in site_orderings[1:]:
            order = lattice_site_ordering(name, self.d, lattice_side)
            self.register_buffer(f"_order_{name}", order, persistent=False)
            self.register_buffer(
                f"_order_inverse_{name}", order.argsort(), persistent=False
            )
        if site_orderings[1:]:
            self.extra_ordering_modules = nn.ModuleDict({
                name: nn.ModuleDict({
                    "prefix_norm": nn.LayerNorm(hidden),
                    "suffix_norm": nn.LayerNorm(hidden),
                    "prefix_factors": nn.Linear(
                        hidden + position_dim, bilinear_rank * factor_dim
                    ),
                    "suffix_factors": nn.Linear(
                        hidden + position_dim, bilinear_rank * factor_dim
                    ),
                })
                for name in site_orderings[1:]
            })

    def ordering_permutation(self, name: str) -> Tensor:
        """o-position -> site map for one of this head's orderings."""
        if name not in self.site_orderings:
            raise KeyError(f"{name!r} not in {self.site_orderings!r}")
        if name == "row":
            return torch.arange(self.d)
        return getattr(self, f"_order_{name}")

    def _extra_factor_tensors(
        self, name: str, x: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Bilinear factors for one extra ordering, (B, d, R, f) in o-POSITION
        space: entry [:, p] belongs to site order[p].

        a[:, p] may depend only on {t, x[order[<p]], order[p]}; b[:, q] only
        on {t, x[order[>q]], order[q]} -- causality in the permuted sequence,
        which is what makes every ordering's factors hole-blind for any pair:
        the earlier-in-o hole bounds the usable prefix, the later one the
        suffix.
        """
        order = getattr(self, f"_order_{name}")
        modules = self.extra_ordering_modules[name]
        prefix, suffix = causal_stream_summaries(self.backbone, x[:, order], t)
        positions = self.site_position_embedding(order)
        positions = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        a = modules["prefix_factors"](
            torch.cat([modules["prefix_norm"](prefix), positions], dim=-1)
        )
        b = modules["suffix_factors"](
            torch.cat([modules["suffix_norm"](suffix), positions], dim=-1)
        )
        shape = (x.shape[0], self.d, self.bilinear_rank, self.factor_dim)
        return a.view(shape), b.view(shape)

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
        H = torch.where(upper, H, H.transpose(1, 2))
        # Extra orderings: each term is defined on o-ordered pairs, mirrored
        # in ITS OWN o-space (the label-symmetry convention per ordering),
        # then un-permuted back to site indices and summed.
        if self.use_bilinear:
            for name in self.site_orderings[1:]:
                factor_a, factor_b = self._extra_factor_tensors(name, x, t)
                inverse = getattr(self, f"_order_inverse_{name}")
                H_extra = torch.einsum("birf,bjrf->bijf", factor_a, factor_b)
                H_extra = torch.where(upper, H_extra, H_extra.transpose(1, 2))
                H = H + H_extra[:, inverse][:, :, inverse]
        return H

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), assembled without materialising H.

        Upper triangle first (the i < j definition), then the mirror
        G[j,i] = -G[i,j] applied as an identity, so index antisymmetry is
        exact by construction and the diagonal is exactly zero.
        """
        batch, d = x.shape
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)                      # (B, d, h)
        extra_factors = {}
        if self.use_bilinear:
            a, b = self._factor_tensors(x, t)
            extra_factors = {
                name: self._extra_factor_tensors(name, x, t)
                for name in self.site_orderings[1:]
            }
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
            pair_scores = upper - upper.transpose(1, 2)
            # Extra orderings: raw o-space scores via the same distributed
            # readout, mirrored in o-space (each term exactly antisymmetric),
            # un-permuted, summed -- IEEE negation distributes over the sum,
            # so exact index antisymmetry survives the addition.
            for name, (factor_a, factor_b) in extra_factors.items():
                order = getattr(self, f"_order_{name}")
                inverse = getattr(self, f"_order_inverse_{name}")
                omega_ranked = omega_factor[:, order].unsqueeze(2)
                a32, b32 = factor_a.float(), factor_b.float()
                extra = torch.einsum("birf,bjrf->bij", a32 * omega_ranked, b32)
                extra = extra - torch.einsum(
                    "birf,bjrf->bij", a32, b32 * omega_ranked
                )
                extra_upper = torch.triu(extra, diagonal=1)
                mirrored = extra_upper - extra_upper.transpose(1, 2)
                pair_scores = pair_scores + mirrored[:, inverse][:, :, inverse]
            return pair_scores
