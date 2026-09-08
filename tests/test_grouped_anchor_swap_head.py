"""Falsification suite for GroupedAnchorSwapHead (2026-07-22).

Mirrors the structure of the existing head suites (test_swap_readout,
test_masked_attention_swap_head) with the blindness bar at equality: masking
here is an unconditional input override, so a masked site's value cannot enter
any computed quantity at any depth and the residue is exactly 0.0.

Equality is the bar for every claim the data flow enforces (blindness,
hollowness, antisymmetry, trivial-swap vanishing). Chunking is the exception:
`group_chunk_size` changes the stacked pass's row count, hence BLAS blocking
and reduction order, so it is pinned with ATOL, as LeTFMaskOneSwapHead's
`anchor_chunk_size` is in test_swap_head_vectorised.py.

Contracts pinned here:
  1. blindness: H^{group(i)} read at j moves by exactly 0.0 under any change
     to x_i (i in the masked group) or to x_j (leTF hollowness at the read
     site), on both readout kernels -- the manual matmul/softmax path and the
     fused SDPA path every d64 cell runs (use_sdpa_readout=True);
  2. exact state-swap antisymmetry and trivial-swap vanishing of G, untrained;
  3. G is non-degenerate at the k the cells run: it responds to sites that are
     neither masked nor read, the one contract an all-masked head (blind,
     antisymmetric, information-free) would fail;
  4. n_groups = d with "strided" grouping reproduces LeTFMaskOneSwapHead
     bit-exactly, since that grouping makes group(i) = i and the head
     degenerates to per-anchor masking;
  5. the head runs exactly k body passes (the efficiency claim);
  6. group chunking is numerically inert to ATOL (incl. a ragged tail chunk);
  7. every grouping is a partition, and "diagonal" at n_groups = D is the
     Latin-square dispersal (one masked site per row and per column);
  8. gradients are finite and reach the backbone.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.grouped_anchor_swap_head import (
    GroupedAnchorSwapHead,
    site_groups,
)
from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
    swap2,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

ATOL = 1e-5


def _backbone(d, seed=42, hidden_dim=8, n_heads=2, n_layers=2, use_sdpa=False):
    torch.manual_seed(seed)
    m = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        use_sdpa_readout=use_sdpa,
    )
    m.eval()
    return m


def _assert_matches(got, want, label):
    """Bit-exact where the backend allows it, ATOL as the actual contract.

    Same helper as test_swap_head_vectorised.py: re-batching applies identical
    kernels, but the batch shape selects the BLAS blocking, so a chunked and
    an unchunked reduction can differ in the last few ulps on some backends
    without either being wrong."""
    if not torch.equal(got, want):
        diff = (got - want).abs().max().item()
        assert diff < ATOL, f"{label}: max diff {diff:.2e}"


def _state_batch(d, batch, seed=7):
    """A (batch, d) +-1 state batch; every row guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    for b in range(batch):
        if bool((x[b] > 0).all()) or bool((x[b] < 0).all()):
            x[b, 0] *= -1
    return x


def _time(batch, value=0.3):
    return torch.full((batch,), value)


# --------------------------------------------------------------------------
# Grouping: partition property and the dispersal claim
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grouping", ["diagonal", "strided", "contiguous"])
@pytest.mark.parametrize("n_groups", [2, 4, 8, 16, 32])
def test_grouping_is_a_balanced_partition(grouping, n_groups):
    """Correctness needs a total site -> group map: every site in exactly one
    group, else some pair has no pass covering it. Balance is a separate
    requirement: unequal groups mean unequal masked fractions across passes,
    so H_ij would see systematically more of the lattice for some i than
    others. Pinned at every k the cells use -- the k=16 diagonal case is where
    the naive (row+col) mod k silently leaves a group empty on an 8x8
    raster."""
    d = 64
    groups = site_groups(d, n_groups, grouping, lattice_side=8)
    assert groups.shape == (d,)
    assert int(groups.min()) >= 0 and int(groups.max()) < n_groups
    occupancy = torch.bincount(groups, minlength=n_groups)
    assert int(occupancy.sum()) == d
    assert int(occupancy.min()) == int(occupancy.max()) == d // n_groups


def test_diagonal_grouping_disperses_one_site_per_row_and_column():
    """At n_groups = D each "diagonal" group hits every row once and every
    column once (Latin-square diagonal), so no masked site is adjacent to
    another. The line-shaped alternatives are the contrast: "strided" is a
    column, "contiguous" a row."""
    side = 8
    d = side * side
    groups = site_groups(d, side, "diagonal", lattice_side=side)
    site = torch.arange(d)
    rows, cols = site // side, site % side
    for group_id in range(side):
        member = groups == group_id
        assert int(member.sum()) == side
        assert sorted(rows[member].tolist()) == list(range(side))
        assert sorted(cols[member].tolist()) == list(range(side))

    strided = site_groups(d, side, "strided")
    contiguous = site_groups(d, side, "contiguous")
    # A column: one site per row, all in the same column.
    assert len(set(cols[strided == 0].tolist())) == 1
    # A row: one site per column, all in the same row.
    assert len(set(rows[contiguous == 0].tolist())) == 1


def test_diagonal_grouping_above_side_splits_by_row_and_stays_dispersed():
    """Above k = D the 2D-1 anti-diagonals cannot address k groups on their
    own, so the general form splits each class by row. The result must still
    be balanced (pinned above) and must still spread every group across
    multiple rows and columns, or "diagonal" degrades into a line shape at
    large k."""
    side = 8
    d = side * side
    groups = site_groups(d, 16, "diagonal", lattice_side=side)
    site = torch.arange(d)
    rows, cols = site // side, site % side
    for group_id in range(16):
        member = groups == group_id
        assert int(member.sum()) == 4
        assert len(set(rows[member].tolist())) > 1
        assert len(set(cols[member].tolist())) > 1


def test_grouping_rejects_bad_configurations():
    with pytest.raises(ValueError, match="Unknown grouping"):
        site_groups(64, 8, "spiral")
    with pytest.raises(ValueError, match="n_groups"):
        site_groups(64, 65, "strided")
    with pytest.raises(ValueError, match="square lattice"):
        site_groups(63, 8, "diagonal")
    # k that cannot factorise into row/anti-diagonal classes dividing the side
    # would give ragged groups; refuse rather than mask unequal fractions.
    with pytest.raises(ValueError, match="cannot balance"):
        site_groups(64, 24, "diagonal", lattice_side=8)
    with pytest.raises(ValueError, match="cannot balance"):
        GroupedAnchorSwapHead(_backbone(16), n_groups=3, grouping="diagonal")


# --------------------------------------------------------------------------
# Blindness: the property the head exists to provide
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("grouping", ["diagonal", "strided", "contiguous"])
def test_body_is_exactly_blind_to_every_masked_site(grouping, use_sdpa):
    """H^a must not move -- by exactly zero -- under any change to a site in
    group a: stronger than the antisymmetry it buys, since H must be invariant
    to arbitrary changes at those sites, not merely to their exchange.

    Run on both readout kernels. use_sdpa_readout=True is what every d64
    curriculum cell sets, so the fused path is the one the GPU runs.
    Blindness should be kernel-independent (the masked embedding is zero
    before any attention happens); this pins it rather than assuming it."""
    d, batch = 16, 3
    backbone = _backbone(d, use_sdpa=use_sdpa)
    head = GroupedAnchorSwapHead(backbone, n_groups=4, grouping=grouping)
    x, t = _state_batch(d, batch), _time(batch)

    for group_id in range(head.n_groups):
        groups = torch.tensor([group_id])
        reference = head.masked_bodies(x, t, groups)
        masked_sites = (head.group_of_site == group_id).nonzero().flatten()
        for site in masked_sites.tolist():
            perturbed = x.clone()
            perturbed[:, site] *= -1
            assert torch.equal(head.masked_bodies(perturbed, t, groups), reference), (
                f"group {group_id} leaked the value of its masked site {site}"
            )


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_body_is_hollow_at_the_read_site(use_sdpa):
    """The second hole is free: the leTF readout at j ignores x_j, so H^a[:, j]
    is blind to x_j for every j, whether or not j is in the masked group. That
    is what lets one pass serve a whole row of pairs.

    This hole is inherited from leTF's slice-and-mask construction, so unlike
    the masking hole it depends on the attention mask being honoured. The SDPA
    branch negates the mask (`attn_mask=~joint_mask`) for the fused kernel's
    inverted convention, so both kernels are pinned here."""
    d, batch = 16, 3
    head = GroupedAnchorSwapHead(
        _backbone(d, use_sdpa=use_sdpa), n_groups=4, grouping="strided"
    )
    x, t = _state_batch(d, batch), _time(batch)
    groups = torch.tensor([0])
    reference = head.masked_bodies(x, t, groups)[0]

    for site in range(d):
        perturbed = x.clone()
        perturbed[:, site] *= -1
        moved = head.masked_bodies(perturbed, t, groups)[0]
        assert torch.equal(moved[:, site, :], reference[:, site, :]), (
            f"body read at {site} depends on x_{site}"
        )


# --------------------------------------------------------------------------
# The readout properties that blindness buys, at random init
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grouping", ["diagonal", "strided", "contiguous"])
def test_exact_swap_antisymmetry_untrained(grouping):
    """G(i, j | x) = -G(i, j | Swap2(x, i, j)) for every pair, at random init;
    the property training never touches."""
    d, batch = 16, 2
    head = GroupedAnchorSwapHead(_backbone(d), n_groups=4, grouping=grouping)
    x, t = _state_batch(d, batch), _time(batch)
    base = head(x, t)

    for i in range(d):
        for j in range(i + 1, d):
            swapped = head(swap2(x, i, j), t)
            assert torch.allclose(swapped[:, i, j], -base[:, i, j], atol=1e-6), (
                f"antisymmetry broken at pair ({i}, {j})"
            )


def test_trivial_swaps_and_diagonal_vanish():
    """Same-spin pairs and the diagonal read exactly zero: the token difference
    omega_{x_i} - omega_{x_j} vanishes there, so the swap CTMC never proposes a
    move that would leave the state unchanged."""
    d, batch = 16, 4
    head = GroupedAnchorSwapHead(_backbone(d), n_groups=4)
    x, t = _state_batch(d, batch), _time(batch)
    G = head(x, t)

    assert torch.equal(torch.diagonal(G, dim1=1, dim2=2), torch.zeros(batch, d))
    same_spin = x.unsqueeze(2) == x.unsqueeze(1)  # (B, d, d)
    assert torch.equal(G[same_spin], torch.zeros(int(same_spin.sum())))


def test_G_responds_to_sites_that_are_neither_masked_nor_read():
    """Non-degeneracy at the k the cells run.

    Every other contract here says what G must not depend on, and a head whose
    keep-mask zeroed the whole lattice would satisfy all of them while
    carrying no information about the configuration. Only the `n_groups = d`
    mask_one equivalence would catch that, and it is the degenerate limit
    where every group is a singleton, not the k = 8 the run uses.

    The positive direction: for a pair (i, j), a site s that is neither masked
    (s not in group(i)) nor read (s != j) is fully visible to H^{group(i)},
    and flipping it must move G[:, i, j]. omega_{x_i} and omega_{x_j} are
    untouched by that flip, so the body is the only channel the change can
    travel through, making this a measurement of how much of the lattice each
    pass still sees."""
    d, batch, n_groups = 64, 2, 8
    head = GroupedAnchorSwapHead(_backbone(d), n_groups=n_groups, grouping="diagonal")
    x, t = _state_batch(d, batch), _time(batch)

    masked = (head.group_of_site == 0).nonzero().flatten().tolist()
    row = masked[0]
    # A read site whose spin differs from the row's in every batch element, so
    # the token difference -- and hence G[:, row, read] -- is nonzero to begin with.
    read = next(
        s for s in range(d) if s not in masked and bool((x[:, s] != x[:, row]).all())
    )
    base = head(x, t)
    assert not torch.equal(base[:, row, read], torch.zeros(batch))

    unmoved = []
    for site in range(d):
        if site in masked or site == read:
            continue
        perturbed = x.clone()
        perturbed[:, site] *= -1
        if torch.equal(head(perturbed, t)[:, row, read], base[:, row, read]):
            unmoved.append(site)
    assert not unmoved, (
        f"G[{row}, {read}] ignores unmasked, unread sites {unmoved}: the pass "
        "is destroying more of the lattice than group 0"
    )


# --------------------------------------------------------------------------
# Faithfulness to the endpoint it generalises, and the efficiency claim
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d", [16, 64])
def test_n_groups_equal_d_reproduces_mask_one_bit_exactly(d):
    """k = d with "strided" grouping makes group(i) = i, so every pass masks
    exactly one site and the head degenerates to LeTFMaskOneSwapHead. Bit-exact
    equality (not ATOL) is the bar: both paths call _keep_masked_bodies with
    the same diagonal keep-mask and reduce in the same order, so any drift
    means the generalisation changed the arithmetic rather than extending it."""
    batch = 2
    backbone = _backbone(d)
    grouped = GroupedAnchorSwapHead(backbone, n_groups=d, grouping="strided")
    reference = LeTFMaskOneSwapHead(backbone)
    x, t = _state_batch(d, batch), _time(batch)

    assert torch.equal(grouped(x, t), reference(x, t))


def test_runs_exactly_k_body_passes():
    """k stacked passes, each carrying n_groups_in_chunk * B rows, so the work
    is k/d of mask_one's."""
    d, batch, n_groups = 64, 3, 8
    backbone = _backbone(d)
    head = GroupedAnchorSwapHead(backbone, n_groups=n_groups, grouping="diagonal")
    x, t = _state_batch(d, batch), _time(batch)

    rows_per_call = []
    original = backbone.fwd_stack.forward

    def counting_forward(sequence, *args, **kwargs):
        rows_per_call.append(sequence.shape[0])
        return original(sequence, *args, **kwargs)

    backbone.fwd_stack.forward = counting_forward
    try:
        head(x, t)
    finally:
        backbone.fwd_stack.forward = original

    assert len(rows_per_call) == 1  # unchunked: one pass
    assert rows_per_call[0] == n_groups * batch
    assert sum(rows_per_call) == n_groups * batch < d * batch


@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_group_chunking_is_numerically_inert(chunk):
    """Chunking bounds the (n_groups*B, n_heads, d, 2d) readout attention
    buffer and must not change the result, including when the last chunk is
    ragged (8 groups / chunk 3).

    ATOL, not torch.equal: chunk size sets the stacked pass's row count, which
    selects the BLAS blocking and hence the reduction order. Observed residue
    is ~6e-9 against a |G| of ~7e-3 (about 1e-6 relative) at chunks 1-3 on
    CPU, and mask_one's `anchor_chunk_size` shows the same at the same
    scale."""
    d, batch = 64, 2
    backbone = _backbone(d)
    x, t = _state_batch(d, batch), _time(batch)
    unchunked = GroupedAnchorSwapHead(backbone, n_groups=8, grouping="diagonal")
    chunked = GroupedAnchorSwapHead(
        backbone, n_groups=8, grouping="diagonal", group_chunk_size=chunk
    )
    _assert_matches(chunked(x, t), unchunked(x, t), f"group_chunk_size={chunk}")


def test_gradients_are_finite_and_reach_the_backbone():
    """The head must train: a scalar built from G has finite gradient at every
    backbone parameter that participates. Catches masked-out paths silently
    detaching the graph."""
    d, batch = 16, 2
    backbone = _backbone(d)
    head = GroupedAnchorSwapHead(backbone, n_groups=4, grouping="diagonal")
    x, t = _state_batch(d, batch), _time(batch)

    head(x, t).pow(2).sum().backward()
    touched = [
        (name, p) for name, p in backbone.named_parameters() if p.grad is not None
    ]
    assert touched, "no backbone parameter received a gradient"
    for name, parameter in touched:
        assert torch.isfinite(parameter.grad).all(), f"non-finite grad at {name}"
