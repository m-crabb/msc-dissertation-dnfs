"""Grouped-anchor swap head: mask a GROUP of sites per pass, not a single one.

Motivation (2026-07-22, after the stencil verdict). The head
family had only ever been sampled at its two endpoints: `LeTFMaskOneSwapHead`
runs **d** masked body passes (one anchor site each, ESS frac 0.9103 at the
d=64 sigma_c rung), and the one-pass heads (`interval_swap_head.py`,
`masked_attention_swap_head.py`) run **zero** extra passes (0.78-0.80).
Nothing forces the anchor count to equal d.

Partition the sites into k groups. For group a, ONE pass with every site in
that group content-free returns H^a; read at j,

    H_ij := H^a[:, j, :]   for   a = group(i)

is blind to x_i (site i is zeroed at the input, so its value never enters any
layer) and hollow in x_j (the same single-site leTF hollowness mask_one
already relies on: the readout at j ignores j's own input). Taking a =
group(i) covers every ordered pair in **k passes**. k = d with the "strided"
grouping recovers mask_one bit-exactly; k is otherwise free.

Same readout as swap_readout.py / interval_swap_head.py (DNFS Prop. 2 /
Eq. (9), pair form):

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

so exact state-swap antisymmetry G(i,j|x) = -G(i,j|Swap2(x,i,j)) and free
trivial-swap vanishing (x_i = x_j => G = 0) follow at random init, untrained,
exactly as for the other heads.

WHY THIS IS THE SIMPLEST CORRECTNESS ARGUMENT IN THE FAMILY. Masking happens
at the INPUT (an unconditional embedding override, independent of the true
token), so no masked site's value enters any computed quantity at any depth.
The two-hop leak that forces the one-pass heads' band content to be shallow --
one attention layer mixes x_i into every token, so masking only at a readout
layer still leaks via x_i -> token k -> H_ij -- simply does not arise. There
is no band, no collar, no straddle exclusion by index arithmetic, and no
prefix-sum cancellation residue: blindness is bit-exact and structural.

THE TRADE, STATED HONESTLY. Each pass destroys the content of d/k sites when
only x_i and x_j had to go, so H_ij sees less than mask_one's H_ij does. What
it keeps is full DEPTH and full GLOBAL MIXING over the surviving sites -- the
exact opposite trade to the one-pass heads, which keep every site's content
but cap band depth at 1 and never mix prefix with suffix outside the 2-layer
pair readout. At d=64, k=8 masks 12.5% of sites per pass.

GROUP SHAPE IS A DESIGN DECISION, NOT A DETAIL. d = D*D is a raster-flattened
lattice, so on an 8x8 grid the naive choices are both lines: "strided" (site %
k) with k=8 is a whole COLUMN, "contiguous" is a whole ROW. Either cuts the
correlation structure along a line, removing a coherent slab of the lattice.
"diagonal" -- group(site) = (row + col) mod k -- disperses the masked sites so
that at k = D exactly one lands in each row and each column (a Latin-square
diagonal), leaving every masked site surrounded by live neighbours. In a
correlated configuration near sigma_c much of a dispersed site's information
survives in its neighbourhood, so the prediction is dispersed >> line-shaped;
`grouping` exists to test that rather than to assume it.

Cost contract: k body passes instead of d, so the ~99.5%-of-runtime anchor
multiplier the mask-one head pays is cut by d/k. The readout is (B, d, d, h)
of work either way -- that is inherent to producing a (B, d, d) score matrix --
and `group_chunk_size` bounds the transient exactly as mask_one's
`anchor_chunk_size` does, since the stacked pass builds a
(n_groups*B, n_heads, d, 2d) readout attention buffer.

WHY D=16 NEEDS THIS -- COMPUTE, NOT MEMORY. The memory framing is tempting and
wrong, so state it precisely: mask_one's readout buffer is
(A*B, n_heads, d, 2d) with A = `anchor_chunk_size`, NOT A = d. Chunking
already bounds it, and test_swap_head_vectorised.py exercises exactly that at
d=256 with a ragged tail chunk. What chunking cannot bound is the NUMBER of
body passes, which is d by construction (~56 s/forward at d=256; see
LeTFMaskOneSwapHead). Grouped anchors cut that count to k, chosen
independently of d -- that is the lever that moves D=16 into budget. Activation
memory does fall too (measured at d=64, B=128: 877 MiB at k=8, 1690 at k=16,
6566 at mask_one's k=64, i.e. linear in k), but as a CONSEQUENCE of running
fewer passes, not because mask_one lacks a bound this head has.

NOT label-symmetric (H_ij != H_ji, since the two come from different group
passes), exactly as for mask_one -- the downstream swap residual must order
each unordered pair by site index (i < j), never by spin.
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

    Correctness needs only that this be a TOTAL function site -> group: every
    site must land in exactly one group, or some pair (i, j) would have no
    pass whose mask covers i. Balance and shape are quality choices, not
    correctness ones -- which is why the head validates coverage but permits
    any of the three shapes.

        "diagonal":   anti-diagonal dispersal on the D x D raster. At
                      n_groups = D it is (row + col) mod D, a Latin-square
                      diagonal: exactly one masked site per row and per column.
                      Precisely: the masked set is an INDEPENDENT SET in the
                      nearest-neighbour graph (no two members differ by offset
                      1 or D, so every masked site keeps all four live
                      neighbours) -- which is the property that matters, since
                      the Ising coupling is nearest-neighbour only. It is not
                      maximally spread under a second-neighbour metric: group
                      members are diagonally adjacent (min Chebyshev distance
                      1 for k <= D, rising to 2, 4, 8 above it), where a
                      spaced sublattice would be 2. Do not write it up as
                      "maximally dispersed"; write it up as "no masked site
                      loses a neighbour". Above D
                      there are only 2D-1 distinct anti-diagonals, so the
                      residue alone cannot address k > D groups; the general
                      form splits each anti-diagonal class further by row,

                          k = columns * rows,  columns = min(k, D),
                          group = ((row + col) mod columns) * rows + row % rows

                      which stays exactly balanced (d/k sites per group) and
                      keeps the dispersal, and reduces to (row + col) mod k
                      whenever k <= D. Requires `columns` and `rows` to divide
                      D, else the split would be ragged and some passes would
                      carry more masked sites than others.
        "strided":    site mod n_groups. On a raster this is a COLUMN when
                      n_groups = D. Also the mask_one-equivalent grouping:
                      n_groups = d makes group(i) = i.
        "contiguous": balanced consecutive blocks, site * n_groups // d. A ROW
                      when n_groups = D. Included as the deliberately
                      worst-shaped control for the dispersal hypothesis.

    `lattice_side` is D (the raster side); defaults to sqrt(d) and is only
    consulted by "diagonal", the one shape that reads 2D structure.
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
        # keep[a, s] = 0.0 exactly when site s belongs to group a, so pass a is
        # blind to every site in group a. One row per pass; the partition
        # property of `group_of_site` is what makes the k rows cover all pairs.
        keep = (group_of_site.view(1, -1) != torch.arange(n_groups).view(-1, 1)).float()
        self.register_buffer("group_keep", keep, persistent=False)

    def masked_bodies(self, x: Tensor, t: Tensor, groups: Tensor) -> Tensor:
        """Bodies for the given group ids, (n, B, d, h), via ONE stacked pass.

        Exposed separately from forward so the falsification tests can probe
        blindness on H directly (flip x_i for any i in the masked group: H must
        not move by EXACTLY 0.0) -- a strictly stronger check than G's
        antisymmetry, which a symmetric leak would survive.
        """
        return _keep_masked_bodies(self.backbone, x, t, self.group_keep[groups])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Pair-score matrix G, (B, d, d), via the swap readout.

            G[:, i, j] = < H^{group(i)}[:, j, :],  omega_{x_i} - omega_{x_j} >

        Rows are filled group by group: pass a supplies every row i in group a
        at once, since those rows share the same body H^a and differ only in
        which omega_{x_i} they read against. The diagonal and same-spin pairs
        vanish for free (the token difference is zero there).

        mul+sum, NOT einsum: einsum is on the autocast lower-precision list, so
        under the Tier-2 eval_autocast_bf16 block it would emit bf16 G,
        crashing the fp32-only quantile rate diagnostic -- the mask-one and
        masked-attention readouts keep G fp32 the same way.
        """
        x_idx = ((x + 1) / 2).long()
        omega = self.backbone.omega(x_idx)                     # (B, d, h)
        batch = x.shape[0]
        chunk = self.group_chunk_size or self.n_groups
        G = omega.new_zeros(batch, self.d, self.d)
        for start in range(0, self.n_groups, chunk):
            groups = torch.arange(
                start, min(start + chunk, self.n_groups), device=x.device
            )
            H = self.masked_bodies(x, t, groups)               # (n, B, d, h)
            for offset, group_id in enumerate(groups.tolist()):
                sites = (self.group_of_site == group_id).nonzero().flatten()
                # difference[:, p, j, :] = omega_{x_i} - omega_{x_j}, i = sites[p]
                difference = omega[:, sites].unsqueeze(2) - omega.unsqueeze(1)
                G[:, sites, :] = (difference * H[offset].unsqueeze(1)).sum(-1)
        return G
