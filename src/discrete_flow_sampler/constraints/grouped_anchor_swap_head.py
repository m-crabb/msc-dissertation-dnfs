"""Grouped-anchor swap head: mask a group of sites per pass, not a single one.

`LeTFMaskOneSwapHead` runs d masked body passes (one anchor site each; ESS
frac 0.9103 at the d=64 sigma_c rung); the one-pass heads
(`interval_swap_head.py`, `masked_attention_swap_head.py`) run none
(0.78-0.80). Nothing forces the anchor count to equal d.

Partition the sites into k groups. For group a, one pass with every site in
that group content-free returns H^a; read at j,

    H_ij := H^a[:, j, :]   for   a = group(i)

is blind to x_i (zeroed at the input, so it enters no layer) and hollow in
x_j (the leTF readout at j ignores j's own input). Taking a = group(i) covers
every ordered pair in k passes. k = d with the "strided" grouping recovers
mask_one bit-exactly; k is otherwise free.

Same readout as swap_readout.py / interval_swap_head.py (DNFS Prop. 2 /
Eq. (9), pair form):

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

so state-swap antisymmetry G(i,j|x) = -G(i,j|Swap2(x,i,j)) and trivial-swap
vanishing (x_i = x_j => G = 0) hold at random init, untrained.

Masking is an unconditional embedding override at the input, so the two-hop
leak that caps the one-pass heads' band depth (x_i -> token k -> H_ij) does
not arise: blindness is bit-exact and structural, with no band, collar or
straddle exclusion.

Trade: each pass destroys d/k sites' content when only x_i and x_j had to
go, but keeps full depth and global mixing over the survivors, the opposite
of the one-pass heads (every site kept, band depth 1). At d=64, k=8 masks
12.5% of sites per pass. On the raster-flattened lattice "strided" and
"contiguous" at k=D are a column and a row; "diagonal" disperses the masked
sites so each keeps live neighbours (see `site_groups`). `grouping` exists to
test dispersed vs line-shaped, not to assume it.

Cost: k body passes instead of d, so mask_one's anchor multiplier is cut by
d/k (d passes is ~56 s/forward at d=256). The (B, d, d, h) readout is
inherent to a (B, d, d) score matrix; `group_chunk_size` bounds the
(n_groups*B, n_heads, d, 2d) readout attention buffer as mask_one's
`anchor_chunk_size` does. Activation memory at d=64, B=128: 877 MiB at k=8,
1690 at k=16, 6566 at k=64 (mask_one), linear in k.

Not label-symmetric (H_ij != H_ji, different group passes), as for mask_one:
the downstream swap residual must order each unordered pair by site index
(i < j), never by spin.
"""

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.constraints.swap_readout import _keep_masked_bodies
from discrete_flow_sampler.models.letf import LeTFRateMatrix

GROUPINGS = ("diagonal", "strided", "contiguous")


def site_groups(
    d: int, n_groups: int, grouping: str = "diagonal", lattice_side: int | None = None
) -> Tensor:
    """Assign each of the d sites to one of `n_groups` groups; returns (d,) long.

    Correctness needs only a total function site -> group (else some pair
    (i, j) has no pass masking i); balance and shape are quality choices, so
    the head validates coverage and permits any of the three shapes.

        "diagonal":   anti-diagonal dispersal on the D x D raster. At
                      n_groups = D it is (row + col) mod D: one masked site
                      per row and per column, and the masked set is an
                      independent set of the nearest-neighbour graph (no two
                      members differ by offset 1 or D), so every masked site
                      keeps all four live neighbours. Not maximally spread
                      under a second-neighbour metric (members are diagonally
                      adjacent, Chebyshev distance 1 for k <= D, 2, 4, 8
                      above); describe it as "no masked site loses a
                      neighbour". Above D only 2D-1 anti-diagonals exist, so
                      each class is split further by row:

                          k = columns * rows,  columns = min(k, D),
                          group = ((row + col) mod columns) * rows + row % rows

                      which stays balanced (d/k sites per group) and reduces
                      to (row + col) mod k for k <= D. `columns` and `rows`
                      must divide D or the split is ragged.
        "strided":    site mod n_groups; a column at n_groups = D, and
                      n_groups = d gives group(i) = i (mask_one).
        "contiguous": balanced consecutive blocks, site * n_groups // d; a
                      row at n_groups = D. The worst-shaped control.

    `lattice_side` is D; defaults to sqrt(d) and is read only by "diagonal".
    """
    if grouping not in GROUPINGS:
        raise ValueError(f"Unknown grouping: {grouping!r}; expected one of {GROUPINGS}")
    if not 1 <= n_groups <= d:
        raise ValueError(f"n_groups must be in [1, {d}], got {n_groups}")
    site = torch.arange(d)
    if grouping == "strided":
        return site % n_groups
    if grouping == "contiguous":
        return site * n_groups // d
    side = lattice_side if lattice_side is not None else int(round(d**0.5))
    if side * side != d:
        raise ValueError(
            f"diagonal grouping needs a square lattice: side {side} vs d {d}"
        )
    columns = min(n_groups, side)
    rows = n_groups // columns
    if columns * rows != n_groups or side % columns or side % rows:
        raise ValueError(
            f"diagonal grouping cannot balance n_groups={n_groups} on a "
            f"{side}x{side} raster: it factorises as {columns} anti-diagonal "
            f"classes x {rows} row classes, and both must divide {side}"
        )
    row, col = site // side, site % side
    return ((row + col) % columns) * rows + row % rows


class GroupedAnchorSwapHead(nn.Module):
    """Swap head with k masked body passes for arbitrary k <= d.

    Drop-in `head_kind` (forward: (B, d) state, (B,) time -> (B, d, d) pair
    scores), same contract as the heads in swap_readout.py.

    Args:
        backbone: the leTF rate model, reused untouched (its token/time
            embedders, fwd/bwd stacks, attention_readout and omega table).
        n_groups: k, the number of masked passes. k = d with grouping
            "strided" reproduces LeTFMaskOneSwapHead bit-exactly; smaller k
            trades masked-site count for passes.
        grouping: how sites are assigned to groups; see `site_groups`.
        lattice_side: D for the raster, needed by "diagonal"; defaults to
            sqrt(d).
        group_chunk_size: groups per stacked pass, bounding the readout
            attention buffer (n_groups*B, n_heads, d, 2d) exactly as mask_one's
            `anchor_chunk_size` does. None = all k groups in one pass.

    Raises if any group is empty: an empty group spends a body pass covering
    no pairs, which is always a misconfiguration (e.g. "diagonal" with
    n_groups > 2*D - 1, where the high residues are unreachable).
    """

    def __init__(
        self,
        backbone: LeTFRateMatrix,
        n_groups: int,
        grouping: str = "diagonal",
        lattice_side: int | None = None,
        group_chunk_size: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d
        self.n_groups = n_groups
        self.grouping = grouping
        self.group_chunk_size = group_chunk_size

        group_of_site = site_groups(self.d, n_groups, grouping, lattice_side)
        occupancy = torch.bincount(group_of_site, minlength=n_groups)
        if int(occupancy.min()) == 0:
            raise ValueError(
                f"grouping {grouping!r} with n_groups={n_groups} leaves groups "
                f"{(occupancy == 0).nonzero().flatten().tolist()} empty: those "
                "passes would cover no pairs"
            )
        self.register_buffer("group_of_site", group_of_site, persistent=False)
        # keep[a, s] = 0.0 exactly when site s is in group a, so pass a is blind
        # to group a; the partition of `group_of_site` makes k rows cover all pairs.
        keep = (group_of_site.view(1, -1) != torch.arange(n_groups).view(-1, 1)).float()
        self.register_buffer("group_keep", keep, persistent=False)

    def masked_bodies(self, x: Tensor, t: Tensor, groups: Tensor) -> Tensor:
        """Bodies for the given group ids, (n, B, d, h), in one stacked pass.

        Separate from forward so the tests can probe blindness on H directly
        (flip x_i for i in the masked group: H must move by exactly 0.0), a
        stronger check than G's antisymmetry, which a symmetric leak survives.
        """
        return _keep_masked_bodies(self.backbone, x, t, self.group_keep[groups])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), via the swap readout.

            G[:, i, j] = < H^{group(i)}[:, j, :],  omega_{x_i} - omega_{x_j} >

        Pass a fills every row i in group a at once (shared body H^a, different
        omega_{x_i}); the diagonal and same-spin pairs vanish for free.
        mul+sum rather than einsum: einsum is on the autocast bf16 list and
        would emit bf16 G under eval_autocast_bf16, crashing the fp32-only
        quantile rate diagnostic.
        """
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)  # (B, d, h)
        batch = x.shape[0]
        chunk = self.group_chunk_size or self.n_groups
        G = omega.new_zeros(batch, self.d, self.d)
        for start in range(0, self.n_groups, chunk):
            groups = torch.arange(
                start, min(start + chunk, self.n_groups), device=x.device
            )
            H = self.masked_bodies(x, t, groups)  # (n, B, d, h)
            for offset, group_id in enumerate(groups.tolist()):
                sites = (self.group_of_site == group_id).nonzero().flatten()
                # difference[:, p, j, :] = omega_{x_i} - omega_{x_j}, i = sites[p]
                difference = omega[:, sites].unsqueeze(2) - omega.unsqueeze(1)
                G[:, sites, :] = (difference * H[offset].unsqueeze(1)).sum(-1)
        return G
