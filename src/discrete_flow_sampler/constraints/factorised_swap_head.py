"""Factorised swap head: low-rank bilinear causal factors + hole-subtracted
global context. All-pairs G in one body pass with no per-pair pooling.

Same readout as every swap head (DNFS Prop. 2 / Eq. (9), pair form):

    G(i, j | x) = < H_ij(x_-{i,j}),  omega_{x_i} - omega_{x_j} >

and the same blindness requirement: H_ij must not depend on the token values
at sites i and j, at any layer (one globally-mixing layer leaks x_i into
every representation; see interval_swap_head.py). The pair context is built
from per-site pieces, so no pair pools over O(d) lattice terms:

    H_ij = sum_r  a_r(prefix_i, pos_i) * b_r(suffix_j, pos_j)   [bilinear]
         + rho( LN( c(x) - psi_i - psi_j ) )                    [global]
         + tau(t)                                               [time]

* bilinear -- prefix_i / suffix_j are the leTF causal-stream summaries
  (blind to x_{>=i} / x_{<=j} by causality). The factor maps are linear in
  the normalised streams; the rank R bounds the pair kernel. This term
  cannot see the open interval (i, j): prefix stops before i, suffix starts
  after j.
* global -- psi_k = MLP(emb(x_k) ++ pos_k) is per-site, c = sum_k psi_k,
  and subtracting psi_i + psi_j removes the only terms touching the holes,
  so blindness is exact up to fp cancellation (suite bar 1e-5). The rule is
  that every term's support be a bounded index set the subtraction can
  reach; a term downstream of an unmasked layer has whole-lattice support
  and cannot be subtracted, which confines depth to after aggregation (rho).
  The LayerNorm before rho matters: |c| grows linearly in d and rho would
  saturate at large lattices without it.
* time -- a projected time embedding added to every pair. There is no norm
  over the assembled H: a post-sum LayerNorm would force H to be
  materialised term-by-term; scale control is per-term instead.

The bilinear and time readouts distribute (<a_i * b_j, w_i - w_j> =
<a_i * w_i, b_j> - <a_i, b_j * w_j>; <tau, w_i - w_j> = <tau, w_i> -
<tau, w_j>); the global term does not (LN and rho are nonlinear in
psi_i + psi_j), so it is a per-pair map over a materialised (B, d, d, F_g)
tensor, and those tensors are the head's memory footprint. Cost class is
O(d^2) against masked attention's O(d^3). `gather_triu_pairs` (default off)
runs LN and rho on the d(d-1)/2 pairs with i < j and mirrors back
(`scatter_symmetric_pairs`), a memory lever at d=256.

forward materialises H once (`compute_pair_context`, where the blindness
probes flip spins) and reads G off it. A distributed bilinear readout via
two (d x Rf)(Rf x d) matmuls was removed: with the global term the
(B, d, d, f) intermediate exists anyway and the distributed form cost ~2Rf
per pair against Rf + f (0.401 vs ~0.35 GFLOP at d=256, B=2). H is defined
on i < j and mirrored down (H_ji := H_ij); G's upper triangle is mirrored as
G[j,i] = -G[i,j], so index antisymmetry is an identity.

Interior band (opt-in via `interior_band`): the interval head's prefix-sum
band or the masked-attention head's attention band (both blind by index
exclusion, both (B, d, d, F_b), valid for i < j) is concatenated into the
global term's input,

    rho( LN( [ c - psi_i - psi_j ; M_ij ] ) ),

which holds the interior mechanism fixed while only the exterior combiner
changes. The provider head is used for `band_summaries` alone: its pair
readout is deleted and its backbone reference kept off the module tree so
the shared backbone is not checkpointed twice. `interior_band=None`
constructs nothing extra, in the same RNG order.

Multi-order causal streams (opt-in via `site_orderings`): the row-major
bilinear term is blind to the whole raster interval between its holes.
Running the causal stacks under extra orderings (column-major,
anti-diagonal) gives every pair a second split,

    H_ij += sum_o sum_r a^o_r(P^o_first, first) * b^o_r(S^o_last, last),

first/last taken in ordering o, so what stays invisible to the bilinear
terms shrinks to the intersection of the per-ordering intervals. Blindness
holds by causality in every ordering. The backbone stacks are shared across
orderings; only the factor maps are per-ordering. `("row",)` is
byte-identical to the single-ordering head.

Full-context site features repaired by explicit antisymmetrisation cost one
swapped forward per ordered pair; that is `swap_readout.antisymmetrise`,
the test oracle, not a head.

The final contractions run in an autocast-disabled fp32 block: einsum and
matmul are on the autocast lower-precision list, and G feeds fp32-only
diagnostics, the same dtype contract as the interval and mask-one heads.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.constraints.interval_swap_head import (
    IntervalSwapHead,
    causal_stream_summaries,
    scatter_symmetric_pairs,
    triu_pair_indices,
)
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
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
            unused -- this head replaces it.
        bilinear_rank: number of rank-1 terms R in the bilinear pair kernel.
        factor_dim: width f of each factor vector and of the projected
            omega readout.
        global_feature_dim: channels of the per-site global features psi.
        position_dim: width of the head-owned site-position embedding
            (positions may enter H freely; blindness constrains only token
            values).
        use_bilinear / use_global: ablation switches. At least one term must
            be on; with both off every pair is scored from time and
            positions alone.
        site_orderings: causal-sweep directions for the bilinear factors
            (module docstring). Must start with "row" (the archived-checkpoint
            module tree); extras from {"col", "diag"} each add one
            shared-backbone pass and their own factor maps.
        lattice_side: D of the flattened D x D lattice; required only by the
            extra orderings.
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
        interior_band: str | None = None,
        band_feature_dim: int = 16,
        pair_offsets: tuple[int, ...] | None = None,
        attention_dim: int = 32,
        site_orderings: tuple[str, ...] = ("row",),
        lattice_side: int | None = None,
        gather_triu_pairs: bool = False,
        global_bond_features: bool = False,
    ):
        super().__init__()
        if not (use_bilinear or use_global or interior_band):
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
        if interior_band not in (None, "prefix", "attention"):
            raise ValueError(
                f"interior_band must be None, 'prefix' or 'attention'; got {interior_band!r}"
            )

        if global_bond_features and interior_band is None:
            raise ValueError(
                "global_bond_features SHARES the band provider's "
                "band_pair_features rather than owning a copy, so it "
                "needs interior_band set; that sharing is what makes "
                "global - band the exterior bond sum in one basis"
            )
        self.backbone = backbone
        self.d = backbone.d
        self.bilinear_rank = bilinear_rank
        self.factor_dim = factor_dim
        self.use_bilinear = use_bilinear
        self.use_global = use_global
        self.site_orderings = site_orderings
        # Opt-in LN + MLP on d(d-1)/2 unordered pairs. No parameter, buffer
        # or RNG draw: flag-off heads stay byte-identical to archived ones.
        self.gather_triu_pairs = gather_triu_pairs
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
        # Band width decided up front so the per-pair map is built at its
        # final width (no discarded init draws); 0 keeps the archived shapes.
        self.interior_band = interior_band
        if interior_band is not None:
            pair_offsets = pair_offsets or (1, lattice_side or round(self.d**0.5))
        band_dim = (
            0 if interior_band is None else band_feature_dim * (1 + len(pair_offsets))
        )
        # Bond-carrying global term: whole-lattice bond sums ride the same
        # per-pair path, one family per offset. The features are the band
        # provider's own modules, so the option costs only this widening
        # (576 of the head's 145,778 parameters at production width, +0.4%).
        self.global_bond_features = global_bond_features
        if global_bond_features:
            band_dim += band_feature_dim * len(pair_offsets)
        if use_global:
            self.global_site_features = nn.Sequential(
                nn.Linear(hidden + position_dim, global_feature_dim),
                nn.GELU(),
                nn.Linear(global_feature_dim, global_feature_dim),
            )
        # The per-pair readout is shared by the global term and the band and
        # exists whenever either does. The module names stay `global_*` even
        # with no global term: they are state_dict keys in 101 archived
        # factorised cells.
        if use_global or interior_band is not None:
            context_dim = (global_feature_dim if use_global else 0) + band_dim
            self.global_context_norm = nn.LayerNorm(context_dim)
            self.global_context_readout = nn.Sequential(
                nn.Linear(context_dim, global_feature_dim),
                nn.GELU(),
                nn.Linear(global_feature_dim, factor_dim),
            )

        # Extra orderings last, so the base modules' init draws are identical
        # at a fixed seed and the module tree matches archived checkpoints.
        # Permutations are non-persistent buffers, invisible to state_dict.
        for name in site_orderings[1:]:
            order = lattice_site_ordering(name, self.d, lattice_side)
            self.register_buffer(f"_order_{name}", order, persistent=False)
            self.register_buffer(
                f"_order_inverse_{name}", order.argsort(), persistent=False
            )
        if site_orderings[1:]:
            self.extra_ordering_modules = nn.ModuleDict(
                {
                    name: nn.ModuleDict(
                        {
                            "prefix_norm": nn.LayerNorm(hidden),
                            "suffix_norm": nn.LayerNorm(hidden),
                            "prefix_factors": nn.Linear(
                                hidden + position_dim, bilinear_rank * factor_dim
                            ),
                            "suffix_factors": nn.Linear(
                                hidden + position_dim, bilinear_rank * factor_dim
                            ),
                        }
                    )
                    for name in site_orderings[1:]
                }
            )
        # Band provider last for the same reason as the orderings: absent at
        # interior_band=None, and never ahead of the archived modules' draws.
        if interior_band is not None:
            self._build_interior_band_provider(
                backbone,
                interior_band,
                pair_offsets,
                band_feature_dim,
                attention_dim,
                lattice_side,
            )

    def _build_interior_band_provider(
        self,
        backbone,
        interior_band,
        pair_offsets,
        band_feature_dim,
        attention_dim,
        lattice_side,
    ) -> None:
        """Own a band head for its `band_summaries` only (module docstring)."""
        if interior_band == "prefix":
            provider = IntervalSwapHead(
                backbone, pair_offsets=pair_offsets, band_feature_dim=band_feature_dim
            )
        else:
            provider = MaskedAttentionSwapHead(
                backbone,
                pair_offsets=pair_offsets,
                band_feature_dim=band_feature_dim,
                attention_dim=attention_dim,
                lattice_side=lattice_side,
            )
        del provider.pair_readout, provider.context_norm
        if interior_band == "prefix":
            # Only the attention queries and the deleted readout use pair
            # positions; the prefix-sum band would carry them as dead weight.
            del provider.pair_position_embedding
        # Keep the shared backbone reachable but off the provider's module
        # tree: a registered submodule would re-emit every backbone tensor
        # under `interior_band_provider.backbone.*` in state_dict.
        del provider._modules["backbone"]
        provider.__dict__["backbone"] = backbone
        self.interior_band_provider = provider

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
        """Bilinear factors for one extra ordering, (B, d, R, f) in o-position
        space: entry [:, p] belongs to site order[p].

        a[:, p] may depend only on {t, x[order[<p]], order[p]}; b[:, q] only
        on {t, x[order[>q]], order[q]} -- causality in the permuted sequence,
        so the earlier-in-o hole bounds the prefix and the later the suffix.
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
        pos = self.site_position_embedding(torch.arange(self.d, device=x.device))
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

    def _interior_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """The per-pair interior term, (B, d, d, f).

        Assembled from whichever the head carries: the hole-subtracted global
        sum rho(LN(c - psi_i - psi_j)), the interior band, and the
        whole-lattice bond totals; the global term and the band can each
        stand alone. Entry [:, i, j] is blind to x_i and x_j up to fp
        cancellation, and symmetric in (i, j), so it already respects the
        H_ji := H_ij mirror. `t` only reaches the band provider's signature;
        time enters through the dedicated time term.

        Under `gather_triu_pairs` the subtraction, LayerNorm and rho run on
        the d(d-1)/2 pairs with i < j and are mirrored back
        (`scatter_symmetric_pairs`); band summaries are natively on i < j,
        so the dense path's `torch.where` mirror is not needed there.
        """
        if self.use_global:
            x_idx = ((x + 1) / 2).long()
            token_embedding = self.backbone.token_embedder(x_idx)  # (B, d, h)
            psi = self.global_site_features(
                torch.cat([token_embedding, self._site_positions(x)], dim=-1)
            )  # (B, d, Fg)
            total = psi.sum(dim=1)  # (B, Fg)
        if self.gather_triu_pairs:
            rows, cols = triu_pair_indices(self.d, x.device)
            if self.use_global:
                hole_subtracted = (
                    total.unsqueeze(1) - psi[:, rows] - psi[:, cols]
                )  # (B, P, Fg)
            if self.interior_band is not None:
                band = self.interior_band_provider.band_summaries(x, t, (rows, cols))
                hole_subtracted = (
                    torch.cat([hole_subtracted, band.to(hole_subtracted.dtype)], dim=-1)
                    if self.use_global
                    else band
                )
            if self.global_bond_features:
                # Symmetric in (i, j) already; unlike the band, no mirror.
                bonds = self.interior_band_provider.hole_free_bond_totals(
                    x, (rows, cols)
                )
                hole_subtracted = torch.cat(
                    [hole_subtracted, bonds.to(hole_subtracted.dtype)], dim=-1
                )
            return scatter_symmetric_pairs(
                self.global_context_readout(self.global_context_norm(hole_subtracted)),
                self.d,
            )
        if self.use_global:
            hole_subtracted = (
                total.view(x.shape[0], 1, 1, -1)
                - psi.unsqueeze(2)  # remove psi_i
                - psi.unsqueeze(1)  # remove psi_j
            )
        if self.interior_band is not None:
            # Band summaries are defined on i < j; mirror to the label-
            # symmetry convention so the whole global block stays symmetric.
            band = self.interior_band_provider.band_summaries(x, t)
            upper = torch.triu(
                torch.ones(self.d, self.d, dtype=torch.bool, device=x.device)
            ).view(1, self.d, self.d, 1)
            band = torch.where(upper, band, band.transpose(1, 2))
            hole_subtracted = (
                torch.cat([hole_subtracted, band.to(hole_subtracted.dtype)], dim=-1)
                if self.use_global
                else band
            )
        if self.global_bond_features:
            bonds = self.interior_band_provider.hole_free_bond_totals(x)
            hole_subtracted = torch.cat(
                [hole_subtracted, bonds.to(hole_subtracted.dtype)], dim=-1
            )
        return self.global_context_readout(self.global_context_norm(hole_subtracted))

    def compute_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """The pair context H, (B, d, d, f), mirrored to i > j.

        forward reads G off it and the blindness tests flip hole spins at it.
        The global and time terms are symmetric in (i, j) already; only the
        bilinear terms are defined on ordered pairs and need the mirror, each
        ordering mirrored in its own o-space before un-permuting.
        Contractions run in fp32 (module docstring).
        """
        batch, d = x.shape
        bilinear_factors = []
        if self.use_bilinear:
            bilinear_factors.append((self._factor_tensors(x, t), None))
            bilinear_factors += [
                (self._extra_factor_tensors(name, x, t), name)
                for name in self.site_orderings[1:]
            ]
        interior_context = (
            self._interior_pair_context(x, t)
            if self.use_global or self.interior_band is not None
            else None
        )
        tau = self.time_projection(self.backbone.time_embedder(t))
        upper = torch.triu(torch.ones(d, d, dtype=torch.bool, device=x.device)).view(
            1, d, d, 1
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            H = (
                tau.float()
                .view(batch, 1, 1, self.factor_dim)
                .expand(batch, d, d, self.factor_dim)
            )
            if interior_context is not None:
                H = H + interior_context.float()
            for (factor_a, factor_b), name in bilinear_factors:
                term = torch.einsum(
                    "birf,bjrf->bijf", factor_a.float(), factor_b.float()
                )
                term = torch.where(upper, term, term.transpose(1, 2))
                if name is not None:
                    inverse = getattr(self, f"_order_inverse_{name}")
                    term = term[:, inverse][:, :, inverse]
                H = H + term
        return H

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d): <H_ij, omega_i - omega_j>.

        Upper triangle first (the i < j definition), then the mirror
        G[j,i] = -G[i,j] applied as an identity, so index antisymmetry is
        exact by construction and the diagonal is exactly zero.
        """
        H = self.compute_pair_context(x, t)
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)  # (B, d, h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            omega_factor = self.omega_projection(omega.float())  # (B, d, f)
            token_difference = omega_factor.unsqueeze(2) - omega_factor.unsqueeze(1)
            scores = (H * token_difference).sum(-1)
            upper = torch.triu(scores, diagonal=1)
            return upper - upper.transpose(1, 2)
