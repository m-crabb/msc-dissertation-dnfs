"""Two-hole patch swap head: blindness by LOCALITY, not by ordering.

Same readout as every swap head (DNFS Prop. 2 / Eq. (9), pair form):

    G(i, j | x) = < H_ij(x_-{i,j}),  omega_{x_i} - omega_{x_j} >,

so exact state-swap antisymmetry needs H_ij blind to the token values at
both holes. The leTF heads get that from a raster ordering (prefix/suffix
streams blind by causality) and pay for it with the interval pathology: the
sites between the holes in raster order are invisible to the deep streams.
This head has no ordering at all. Every term is a function of a bounded
neighbourhood or a linear pool, and the two holes are removed from each
term by construction:

    H_ij = P_ij - P_ji,     P_ij = rho( LN( z_ij ) ),
    z_ij = W_first C_i^{(-j)} + W_second C_j^{(-i)} + e(j - i) + tau(t),
    C_i^{(-j)} = f_i^{(-j)} + sum_l c^l_i{(-j)}   [context of hole i with hole j removed]

  f   = hollow patch feature with the partner zeroed,
  c^l = pooled level l with both holes subtracted,
  e   = torus-relative position of j from i.

* Patch term. P_i is the (2R+1)^2 - 1 spins around i on the torus, centre
  EXCLUDED (the LEAPS zero-centre kernel, made explicit as a gather), and
  f_i = phi(P_i, t) is a small MLP. f_i is blind to x_i by the hollow
  window but sees x_j whenever |j - i|_inf <= R, so for those pairs the
  partner's entry is overwritten with 0 (not a token value) and phi is
  re-run: f_i^{(-j)} = phi(P_i with x_j := 0). That recompute is d * K
  extra phi evaluations with K = (2R+1)^2 - 1 -- O(d R^2) patch MLPs of
  width O(R^2), i.e. O(d R^4) MACs, the R^4 law of the deep-LEAPS recompute
  but on a ONE-layer patch map, so at R = 1..2 it is a few MMAC. For
  |j - i|_inf > R the per-site f_i is already blind to x_j and is reused
  unchanged. With R = 1 and phi linear, f_i^{(-j)} - f_j^{(-i)} IS the
  partner-excluded field difference of the Kawasaki log-ratio, so the
  exact equilibrium rate is in the function class at the smallest radius.

* Pooled levels. psi_k = emb(x_k) is strictly per-site, v^l = W_l psi, and
  c^l_i is the mean of v^l over the centred (2r_l+1)^2 torus box around i
  with the two holes' terms subtracted (the global level is the whole
  lattice). Subtraction is exact in real arithmetic because nothing mixes
  sites before the pool -- the factorised head's global-term argument,
  here at every scale. Centred circular boxes, not a quadtree: block
  boundaries would break torus translation equivariance, which the tests
  demand of the pair output. For S = 2 each level carries the local
  magnetisation at that scale around each hole; it is the low-rank far
  field of an H-matrix split (near field exact in the patch, far field
  pooled), the prior that scale-free critical correlations want.

* Relative position. e depends only on the torus displacement (j - i) mod
  D in each axis, so it is translation-invariant by construction; it is
  NOT tied over C4v (the patch MLP is not either), so the head is exactly
  translation-equivariant and only approximately rotation-equivariant.

Label parity, which the first draft got wrong. The physical rate of the
unordered pair {i, j} is one number, so G must be label-SYMMETRIC and
S_ij = G_ij / (x_i - x_j) label-ODD: the exact Kawasaki rate has S_ij
proportional to the partner-excluded field DIFFERENCE h~_j - h~_i. A
label-symmetric z_ij = u_i + u_j can only produce even S, and that draft
could not fit the exact field even supervised (4x4 MSE flat at the
target's variance). Two consequences are built in here: the holes enter
through DIFFERENT linear maps (otherwise z depends on C_i + C_j only), and
the readout is antisymmetrised explicitly, H_ij = P_ij - P_ji, which is
free because P is computed for every ordered pair anyway (time enters
INSIDE z for the same reason: an additive time line would be an even,
i.e. unphysical, contribution to S). The explicit oddness is what makes the stored i < j convention translation-equivariant:
a lattice shift can move the lower index to the other hole, and only an
odd S gives the same physical rate whichever hole is called first. Index
antisymmetry of the stored matrix is then the usual triangle-and-mirror
identity. State-swap antisymmetry needs none of this, only blindness:
H(x) = H(swap2(x, i, j)) and the omega difference flips sign. rho and LN
act AFTER the holes are removed, which is why depth there is free: the
blindness constraint binds only the site-level maps (phi per-patch, psi
per-site), never the per-pair readout.

Cost: O(d K) patch work, O(d) pooling, O(d^2 f) assembly and O(d^2 f^2) for
the per-pair rho -- the same O(d^2) class as the factorised head with no
(B, d^2, n_terms) band tensor and no causal stacks (the backbone's leTF
stacks are NOT run; only its token/time embedders and omega are reused, so
the head drops into the existing trainer, EMA and exact-field wrapper).

What it gives up, stated up front: no nonlinear function of the far field
(only pooled means of per-site embeddings), no bond/energy density beyond
the patch radius, and neither phi nor e is C4v-tied. The trained-head
regression at 4x4 put 80-95% of Var S on local functions of the holes'
neighbourhoods, which is exactly the patch term's reach; the remaining
non-local share is what the pooled levels must carry.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.models.letf import LeTFRateMatrix


def torus_neighbour_offsets(radius: int) -> list[tuple[int, int]]:
    """Row-major (dr, dc) offsets of the (2R+1)^2 window with the centre
    removed: the hollow patch. Closed under negation, which the partner
    correction relies on (the offset from j back to i is -(j - i))."""
    return [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if (dr, dc) != (0, 0)
    ]


def _torus_displacements(lattice_side: int) -> tuple[Tensor, Tensor]:
    """Minimal-image |dr|, |dc| in [0, D/2] for every ordered site pair, (d, d)."""
    sites = torch.arange(lattice_side * lattice_side)
    rows, cols = sites // lattice_side, sites % lattice_side
    dr = (rows[:, None] - rows[None, :]).abs()
    dc = (cols[:, None] - cols[None, :]).abs()
    return torch.minimum(dr, lattice_side - dr), torch.minimum(dc, lattice_side - dc)


class TwoHolePatchSwapHead(nn.Module):
    """Ordering-free doubly-hollow swap head (module docstring).

    Drop-in `head_kind` (forward: (B, d) state, (B,) time -> (B, d, d) pair
    scores), same contract as the heads in swap_readout.py.

    Args:
        backbone: the leTF rate model, reused ONLY for token_embedder,
            time_embedder and omega (shared readout convention); its causal
            stacks and attention readout are never run.
        lattice_side: D of the D x D torus (d = D^2).
        patch_radius: R of the hollow (2R+1)^2 window. Needs 2R+1 <= D so
            two window entries never alias to one site through the wrap.
        feature_dim: width f of every pair-context term and of the
            projected omega readout.
        patch_hidden_dim: hidden width of the patch MLP phi.
        pooling_radii: radii of the centred pooled levels; None = powers of
            two while the box fits the torus (2r+1 <= D). The whole-lattice
            level is always appended.
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        lattice_side: int,
        patch_radius: int = 1,
        feature_dim: int = 32,
        patch_hidden_dim: int = 32,
        pooling_radii: tuple[int, ...] | None = None,
    ):
        super().__init__()
        d = lattice_side * lattice_side
        if d != backbone.d:
            raise ValueError(f"lattice_side {lattice_side}^2 != backbone.d {backbone.d}")
        if 2 * patch_radius + 1 > lattice_side:
            raise ValueError(
                f"patch_radius {patch_radius} needs 2R+1 <= D={lattice_side}: "
                "window entries would alias through the wrap"
            )
        if pooling_radii is None:
            pooling_radii = tuple(
                2**level for level in range(int(math.log2(lattice_side)) + 1)
                if 2 * 2**level + 1 <= lattice_side
            )
        if any(2 * r + 1 > lattice_side for r in pooling_radii):
            raise ValueError(f"pooling box must fit the torus: {pooling_radii} at D={lattice_side}")
        self.backbone = backbone
        self.d = d
        self.lattice_side = lattice_side
        self.patch_radius = patch_radius
        self.feature_dim = feature_dim
        self.pooling_radii = tuple(pooling_radii)
        hidden = backbone.hidden_dim

        offsets = torus_neighbour_offsets(patch_radius)
        self.n_patch = len(offsets)
        sites = torch.arange(d)
        rows, cols = sites // lattice_side, sites % lattice_side
        neighbour_site = torch.stack([
            ((rows + dr) % lattice_side) * lattice_side + (cols + dc) % lattice_side
            for dr, dc in offsets
        ], dim=1)                                                   # (d, K)
        opposite = torch.tensor([offsets.index((-dr, -dc)) for dr, dc in offsets])
        self.register_buffer("neighbour_site", neighbour_site, persistent=False)
        self.register_buffer("opposite_offset", opposite, persistent=False)

        dr, dc = _torus_displacements(lattice_side)
        for level, radius in enumerate(self.pooling_radii):
            in_box = (torch.maximum(dr, dc) <= radius).float()      # (d, d)
            self.register_buffer(f"level_mask_{level}", in_box, persistent=False)
        # Signed torus displacement of j from i, one embedding row per (dr, dc).
        displacement = (
            ((rows[None, :] - rows[:, None]) % lattice_side) * lattice_side
            + (cols[None, :] - cols[:, None]) % lattice_side
        )                                                           # (d, d)
        self.register_buffer("pair_displacement", displacement, persistent=False)
        self.relative_position_embedding = nn.Embedding(d, feature_dim)

        self.patch_mlp = nn.Sequential(
            nn.Linear(self.n_patch + hidden, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, feature_dim),
        )
        # One projection per pooled level plus the global level, bias-free:
        # a bias is hole-invariant and already lives in the pair MLP.
        self.level_projections = nn.ModuleList([
            nn.Linear(hidden, feature_dim, bias=False)
            for _ in range(len(self.pooling_radii) + 1)
        ])
        self.first_hole_map = nn.Linear(feature_dim, feature_dim, bias=False)
        self.second_hole_map = nn.Linear(feature_dim, feature_dim, bias=False)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.time_projection = nn.Linear(hidden, feature_dim)
        self.omega_projection = nn.Linear(hidden, feature_dim, bias=False)

    # ---- site-level pieces -------------------------------------------------

    def _patches(self, x: Tensor) -> Tensor:
        """Hollow torus patches, (B, d, K): entry [i, k] = x at site i + offset_k."""
        return x[:, self.neighbour_site]

    def _patch_mlp(self, patches: Tensor, t: Tensor) -> Tensor:
        """phi(patch, t) over any leading patch layout (..., K) -> (..., f)."""
        time_embedding = self.backbone.time_embedder(t)             # (B, h)
        shape = patches.shape[:-1] + (time_embedding.shape[-1],)
        time_embedding = time_embedding.view(
            (patches.shape[0],) + (1,) * (patches.ndim - 2) + (-1,)
        ).expand(shape)
        return self.patch_mlp(torch.cat([patches, time_embedding], dim=-1))

    def site_patch_features(self, x: Tensor, t: Tensor) -> Tensor:
        """f_i = phi(P_i, t), (B, d, f): hollow in x_i, NOT blind to neighbours."""
        return self._patch_mlp(self._patches(x), t)

    def partner_zeroed_patch_features(self, x: Tensor, t: Tensor) -> Tensor:
        """f_i^{(-offset_k)} for every site and window entry, (B, d, K, f):
        phi on P_i with entry k overwritten by 0 (an unconditional override,
        so the result cannot depend on the token that was there)."""
        patches = self._patches(x)                                   # (B, d, K)
        keep = 1.0 - torch.eye(self.n_patch, device=x.device, dtype=x.dtype)
        zeroed = patches.unsqueeze(2) * keep                         # (B, d, K, K)
        return self._patch_mlp(zeroed, t)

    def _level_box_mean(self, values: Tensor, radius: int) -> Tensor:
        """Mean of `values` (B, d, f) over the centred (2r+1)^2 torus box, (B, d, f)."""
        batch, d, f = values.shape
        D = self.lattice_side
        grid = values.transpose(1, 2).reshape(batch, f, D, D)
        padded = F.pad(grid, (radius,) * 4, mode="circular")
        k = 2 * radius + 1
        box_sum = F.conv2d(padded, grid.new_ones(f, 1, k, k), groups=f)
        return box_sum.reshape(batch, f, d).transpose(1, 2) / (k * k)

    def _level_values(self, x: Tensor) -> list[tuple[Tensor, float]]:
        """(v^l, n_l) per level: projected per-site embeddings and box size."""
        psi = self.backbone.token_embedder(((x + 1) / 2).long())     # (B, d, h)
        boxes = [(2 * r + 1) ** 2 for r in self.pooling_radii] + [self.d]
        return [
            (projection(psi), float(n))
            for projection, n in zip(self.level_projections, boxes)
        ]

    # ---- pair assembly -----------------------------------------------------

    def compute_pair_context(self, x: Tensor, t: Tensor) -> Tensor:
        """H, (B, d, d, f), blind to both holes of every pair and exactly
        label-odd: H_ij = P_ij - P_ji.

        The role maps are linear, so they are applied to the per-site pieces
        (own term, level values, patch deltas) and the d^2 assembly is adds
        only: z_ij = [W_first own]_i + [W_second own]_j - level partner
        terms - patch deltas for near pairs + e(j - i) + tau(t). The
        per-pair oracle `pair_context_reference` applies the maps to the
        assembled contexts instead, which is the same thing by linearity and
        is what the tests compare against.
        """
        batch, d = x.shape
        f_site = self.site_patch_features(x, t)                      # (B, d, f)
        own = f_site.clone()
        levels = []
        for level, (v, n) in enumerate(self._level_values(x)):
            if level < len(self.pooling_radii):
                mean = self._level_box_mean(v, self.pooling_radii[level])
                in_box = getattr(self, f"level_mask_{level}").view(1, d, d, 1)
            else:
                mean = v.sum(dim=1, keepdim=True).expand_as(v) / n
                in_box = None
            own = own + mean - v / n                                 # drop own term
            levels.append((v / n, in_box))
        tau = self.time_projection(self.backbone.time_embedder(t))
        z = (
            self.first_hole_map(own).unsqueeze(2)                    # hole i's context
            + self.second_hole_map(own).unsqueeze(1)                 # hole j's context
            + self.relative_position_embedding(self.pair_displacement)
            + tau.view(batch, 1, 1, -1)
        )
        # Hole i loses partner j from its boxes, hole j loses partner i.
        for v, in_box in levels:
            partner = (
                self.first_hole_map(v).unsqueeze(1)                  # v_j seen from i
                + self.second_hole_map(v).unsqueeze(2)               # v_i seen from j
            )
            z = z - (partner if in_box is None else partner * in_box)
        # Near pairs (j = i + offset): swap in phi with the partner zeroed on
        # both sides; i sits at the OPPOSITE offset inside j's window.
        delta = self.partner_zeroed_patch_features(x, t) - f_site.unsqueeze(2)
        site = torch.arange(d, device=x.device).repeat_interleave(self.n_patch)
        offset = torch.arange(self.n_patch, device=x.device).repeat(d)
        partner_site = self.neighbour_site.reshape(-1)
        z[:, site, partner_site] = (
            z[:, site, partner_site]
            + self.first_hole_map(delta)[:, site, offset]
            + self.second_hole_map(delta)[:, partner_site, self.opposite_offset[offset]]
        )
        pair = self.pair_mlp(self.context_norm(z))
        return pair - pair.transpose(1, 2)

    def pair_context_reference(self, x: Tensor, t: Tensor, i: int, j: int) -> Tensor:
        """Slow per-pair oracle of H_ij, (B, f), by explicit masking: the
        readable form of the math and the test oracle for the scatter."""
        patches = self._patches(x)
        neighbours = self.neighbour_site
        contexts = []
        for hole, partner in ((i, j), (j, i)):  # C_i^{(-j)}, C_j^{(-i)}
            patch = patches[:, hole].clone()
            patch[:, neighbours[hole] == partner] = 0.0
            context = self._patch_mlp(patch, t)
            for level, (v, n) in enumerate(self._level_values(x)):
                if level < len(self.pooling_radii):
                    box = getattr(self, f"level_mask_{level}")[hole].clone()
                else:
                    box = torch.ones(self.d, device=x.device)
                box[i] = 0.0
                box[j] = 0.0
                context = context + (box @ v) / n
            contexts.append(context)
        first, second = contexts
        tau = self.time_projection(self.backbone.time_embedder(t))
        z_ij = (
            self.first_hole_map(first) + self.second_hole_map(second)
            + self.relative_position_embedding(self.pair_displacement[i, j]) + tau
        )
        z_ji = (
            self.first_hole_map(second) + self.second_hole_map(first)
            + self.relative_position_embedding(self.pair_displacement[j, i]) + tau
        )
        return (
            self.pair_mlp(self.context_norm(z_ij))
            - self.pair_mlp(self.context_norm(z_ji))
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d): <H_ij, omega_i - omega_j>, upper
        triangle mirrored so index antisymmetry is an identity.

        DO NOT HAND-FUSE THIS READOUT (measured dead end, 2026-08-27). In
        source terms it materialises three (B, d, d, f) tensors -- H, the
        omega difference, and their product -- which at d=400, batch 512,
        f=32 reads as 10.5 GB each against a (B, d, d) output of 328 MB.
        There is an identity that removes all three: with
        S_ij = <P_ij, D_ij> and D_ji = -D_ij, the score matrix is S + S^T,
        and S splits into two contractions against the per-site omega.
        Implemented and measured at d=400 R=3 on an A100-80GB, it is a real
        win EAGER (0.2998 s / 20.58 GB against 0.3124 / 23.11 at batch 128)
        and a real LOSS COMPILED (0.1175 / 18.13 against 0.1044 / 15.92),
        which is the configuration every cell trains under. Inductor already
        fuses the broadcast-difference-times-difference-summed-over-f
        pattern and never materialises those tensors -- the compiled
        baseline is 31% under the eager one on exactly that account -- while
        the hand-fused einsum needs omega indexed by the second spatial axis
        and forces a permuted contiguous copy it cannot fuse through. The
        slabs are a property of the source, not of the executed kernel.
        """
        H = self.compute_pair_context(x, t)
        omega = self.backbone.omega(((x + 1) / 2).long())
        with torch.autocast(device_type=x.device.type, enabled=False):
            omega_factor = self.omega_projection(omega.float())
            token_difference = omega_factor.unsqueeze(2) - omega_factor.unsqueeze(1)
            scores = (H.float() * token_difference).sum(-1)
            upper = torch.triu(scores, diagonal=1)
            return upper - upper.transpose(1, 2)
