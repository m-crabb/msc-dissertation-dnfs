"""Falsification tests for cross-size warm starting of the swap head.

The mechanism under test: take a swap head trained on an L_src x L_src torus
and produce a state dict that loads into the SAME architecture built for an
L_dst x L_dst torus, so a large-lattice run starts inside a trained basin
instead of at init noise.

What makes the transfer legitimate is a structural fact about the
architecture, pinned by `test_size_dependent_tensors_are_exactly_the_known_set`:
every Linear/attention/LayerNorm tensor is (hidden_dim, hidden_dim)-shaped and
so carries over verbatim, and the band-feature MLPs are indexed by RELATIVE
offset -- the delta=L "column neighbour" family means column neighbour at
both sizes. Only learned POSITIONAL tables depend on d. Those live on the
L x L grid and are resampled.

The three failure modes these tests exist to catch:

  * the conditioning row. Causal-stack tables are (1 + d, hidden): row 0 is
    the prepended cond_t token, NOT a lattice site. Feeding it to the
    resampler would smear a conditioning vector into the corner sites and
    silently corrupt every downstream position.
  * the raster reversal. The bwd stack sees x.flip(1), so its table's site
    rows are in REVERSED raster order -- a 180-degree rotation of the grid.
    Resampling it as if it were an ordinary grid is only correct because the
    resampler commutes with that rotation, which is asserted rather than
    assumed.
  * blindness. The swap heads buy exact state-swap antisymmetry from a body
    that is blind to the token VALUES at the two swapped sites. Position
    embeddings are explicitly allowed to depend on POSITION, so resampling
    them must not touch blindness -- but "must not" is worth falsifying,
    because a warm-started head is a head no falsification suite has run on
    before.
"""

import sys

import pytest
import torch
from scripts.warm_start_swap_head import _resize_grid_rows, build_transfer

from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
    swap2,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

HIDDEN_DIM, N_HEADS, N_LAYERS = 8, 2, 2


def _head(lattice_side, kind="masked_attention", seed=0):
    """A small swap head on a lattice_side x lattice_side torus."""
    torch.manual_seed(seed)
    d = lattice_side * lattice_side
    backbone = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        use_sdpa_readout=False,
    )
    if kind == "mask_one":
        head = LeTFMaskOneSwapHead(backbone)
    else:
        head = MaskedAttentionSwapHead(
            backbone, pair_offsets=(1, lattice_side), lattice_side=lattice_side
        )
    head.eval()
    return head


def _transfer(side_src, side_dst, kind="masked_attention", **kwargs):
    """Transfer dict plus the two state dicts it was built from."""
    source = _head(side_src, kind, seed=1).state_dict()
    target = _head(side_dst, kind, seed=2).state_dict()
    transfer, interpolated, skipped = build_transfer(
        source, target, side_src**2, side_dst**2, **kwargs
    )
    return transfer, source, target, interpolated, skipped


def _state(d, seed=1):
    """A (1, d) +-1 state guaranteed to contain both spins."""
    torch.manual_seed(seed)
    spins = (torch.randint(0, 2, (1, d)) * 2 - 1).float()
    if bool((spins > 0).all()) or bool((spins < 0).all()):
        spins[0, 0] *= -1
    return spins


# --------------------------------------------------------------------------
# What is size-dependent at all: the premise the whole transfer rests on.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected",
    [
        (
            "masked_attention",
            {
                "backbone.fwd_stack.blocks.0.pos_embed",
                "backbone.fwd_stack.blocks.1.pos_embed",
                "backbone.bwd_stack.blocks.0.pos_embed",
                "backbone.bwd_stack.blocks.1.pos_embed",
                "backbone.attention_readout.pos_embed",
                "pair_position_embedding.weight",
            },
        ),
        (
            "mask_one",
            {
                "backbone.fwd_stack.blocks.0.pos_embed",
                "backbone.fwd_stack.blocks.1.pos_embed",
                "backbone.bwd_stack.blocks.0.pos_embed",
                "backbone.bwd_stack.blocks.1.pos_embed",
                "backbone.attention_readout.pos_embed",
            },
        ),
    ],
)
def test_size_dependent_tensors_are_exactly_the_known_set(kind, expected):
    """Everything else must be shape-identical across lattice size.

    If this fails, a new d-dependent parameter has been added and the
    transfer is silently dropping or mis-resampling it.
    """
    small, large = _head(3, kind).state_dict(), _head(6, kind).state_dict()
    assert set(small) == set(large)
    size_dependent = {k for k in small if small[k].shape != large[k].shape}
    assert size_dependent == expected


# --------------------------------------------------------------------------
# Identity: L -> L must be a no-op.
# --------------------------------------------------------------------------


def test_identity_transfer_is_bit_exact_on_every_tensor():
    transfer, source, _, interpolated, _ = _transfer(4, 4)
    assert interpolated == [], "identity must not resample anything"
    assert set(transfer) == set(source)
    for key, value in transfer.items():
        assert torch.equal(value, source[key]), key


def test_identity_transfer_loads_strictly_and_reproduces_the_source():
    transfer, source, _, _, _ = _transfer(4, 4)
    head = _head(4, seed=99)
    missing, unexpected = head.load_state_dict(transfer, strict=True)
    assert not missing and not unexpected
    for key, value in head.state_dict().items():
        assert torch.equal(value, source[key]), key


def test_identity_transfer_is_bit_exact_for_the_mask_one_head():
    """The mask_one head's live attention_readout must survive identity too."""
    transfer, source, _, interpolated, _ = _transfer(4, 4, kind="mask_one")
    assert interpolated == []
    assert set(transfer) == set(source)
    for key, value in transfer.items():
        assert torch.equal(value, source[key]), key


# --------------------------------------------------------------------------
# Cross-size: shapes, coverage, and what must NOT move.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["masked_attention", "mask_one"])
def test_cross_size_transfer_loads_with_strict_true(kind):
    """L=4 -> L=8 must produce a complete, exactly-shaped state dict."""
    transfer, _, target, _, skipped = _transfer(4, 8, kind=kind)
    assert skipped == [], f"nothing should be dropped: {skipped}"
    head = _head(8, kind, seed=7)
    missing, unexpected = head.load_state_dict(transfer, strict=True)
    assert not missing and not unexpected
    for key, value in transfer.items():
        assert value.shape == target[key].shape, key


def test_cross_size_leaves_non_positional_tensors_bit_identical():
    """Only positional tables may move; every other tensor is copied."""
    transfer, source, target, interpolated, _ = _transfer(4, 8)
    shape_identical = {k for k in source if source[k].shape == target[k].shape}
    assert shape_identical, "sanity: most tensors are size-independent"
    assert set(interpolated) == set(source) - shape_identical
    for key in shape_identical:
        assert torch.equal(transfer[key], source[key]), key


def test_mask_one_transfer_carries_its_live_attention_readout():
    """attention_readout is the mask_one head's compute path, not dead weight.

    Dropping it would leave the one module that actually reads the lattice at
    fresh init on top of a fully transferred trunk -- the exact defect that
    makes a warm start worse than useless.
    """
    transfer, source, _, _, _ = _transfer(4, 8, kind="mask_one")
    readout_keys = {k for k in source if "attention_readout" in k}
    assert readout_keys
    assert readout_keys <= set(transfer)
    assert transfer["backbone.attention_readout.pos_embed"].shape == (64, 4)


# --------------------------------------------------------------------------
# The conditioning row: present in the causal-stack tables, not a site.
# --------------------------------------------------------------------------


def test_conditioning_row_is_carried_across_verbatim():
    transfer, source, _, _, _ = _transfer(4, 8)
    for key in transfer:
        if "stack" in key and key.endswith("pos_embed"):
            assert torch.equal(transfer[key][0], source[key][0]), key


def test_conditioning_row_is_not_mixed_into_any_site_row():
    """Sentinel probe: a huge cond row over zero sites must leave sites zero.

    This is the sharp version of the previous test -- carrying row 0 across
    correctly is worthless if row 0 also leaked into the resampled block.
    """
    source, target = _head(4, seed=1).state_dict(), _head(8, seed=2).state_dict()
    key = "backbone.fwd_stack.blocks.0.pos_embed"
    source = dict(source)
    probe = torch.zeros_like(source[key])
    probe[0] = 1e6
    source[key] = probe

    transfer, _, _ = build_transfer(source, target, 16, 64)

    assert torch.equal(transfer[key][0], probe[0])
    assert torch.equal(transfer[key][1:], torch.zeros_like(transfer[key][1:]))


# --------------------------------------------------------------------------
# Properties of the resampler itself, on bare grids (no model needed).
# --------------------------------------------------------------------------


def test_constant_field_resamples_to_the_same_constant():
    """Partition of unity: a featureless table must stay featureless."""
    rows = torch.full((64, 5), 0.37)
    resized = _resize_grid_rows(rows, 64, 256)
    assert resized.shape == (256, 5)
    assert torch.allclose(resized, torch.full_like(resized, 0.37), atol=1e-6)


def test_resampler_is_identity_at_equal_size():
    rows = torch.randn(64, 5)
    assert torch.equal(_resize_grid_rows(rows, 64, 64), rows)


def test_a_single_site_bump_stays_local():
    """A one-hot at (r, c) must land on (2r, 2c), not scatter over the grid."""
    side_src, side_dst, row, col = 8, 16, 3, 5
    rows = torch.zeros(side_src * side_src, 1)
    rows[row * side_src + col] = 1.0

    grid = _resize_grid_rows(rows, side_src**2, side_dst**2).reshape(side_dst, side_dst)

    peak = divmod(int(grid.abs().argmax()), side_dst)
    # Source pixel centre r maps between destination rows 2r and 2r+1.
    assert peak[0] in (2 * row, 2 * row + 1)
    assert peak[1] in (2 * col, 2 * col + 1)
    # Bulk of the absolute mass inside the 6x6 window around the image of the
    # bump: bicubic's negative lobes reach 2 destination pixels, no further.
    window = grid[2 * row - 2 : 2 * row + 4, 2 * col - 2 : 2 * col + 4]
    assert window.abs().sum() > 0.9 * grid.abs().sum()


def test_resampling_is_periodic_on_the_torus():
    """A circular shift of the source shifts the destination, exactly.

    This is the property that PINS circular padding. Under the default
    edge-replicate behaviour a shifted field and a shifted resampling differ
    at the boundary collar, because replicate invents a distinguished edge on
    a lattice where every site is equivalent.
    """
    side_src, side_dst = 8, 16
    scale = side_dst // side_src
    grid = torch.randn(side_src, side_src, 3)

    plain = _resize_grid_rows(grid.reshape(-1, 3), 64, 256)
    shifted_source = grid.roll(1, dims=0).reshape(-1, 3)
    shifted_then_resized = _resize_grid_rows(shifted_source, 64, 256)
    resized_then_shifted = (
        plain.reshape(side_dst, side_dst, 3).roll(scale, dims=0).reshape(-1, 3)
    )

    assert torch.allclose(shifted_then_resized, resized_then_shifted, atol=1e-5)


def test_resampling_commutes_with_the_180_degree_raster_reversal():
    """Justifies resampling the bwd stack's table as an ordinary grid.

    The bwd stack reads x.flip(1), so its table's site rows run in reversed
    raster order -- i.e. the grid rotated 180 degrees. Treating those rows as
    a plain L x L grid is correct precisely because resampling commutes with
    that rotation; if it did not, every bwd table would be transferred
    upside down.
    """
    rows = torch.randn(64, 3)
    reversed_first = _resize_grid_rows(rows.flip(0), 64, 256)
    resized_first = _resize_grid_rows(rows, 64, 256).flip(0)
    assert torch.allclose(reversed_first, resized_first, atol=1e-6)


# --------------------------------------------------------------------------
# The load-bearing correctness property: blindness must survive the transfer.
# --------------------------------------------------------------------------


def _warm_started_head(side_src=3, side_dst=6):
    transfer, _, _, _, _ = _transfer(side_src, side_dst)
    head = _head(side_dst, seed=5)
    head.load_state_dict(transfer, strict=True)
    head.eval()
    return head


@pytest.mark.parametrize("pair", [(0, 35), (4, 5), (0, 1), (7, 20)])
def test_pair_context_of_a_warm_started_head_is_blind_to_both_holes(pair):
    """H_ij must not move when the tokens at i and j change. Exactly.

    The masked-attention band excludes hole terms BEFORE the softmax, so the
    bar is equality, not a tolerance -- the same bar the head's own
    falsification suite holds it to at fresh init.
    """
    head = _warm_started_head()
    site_i, site_j = pair
    spins = _state(36)
    time = torch.rand(1)

    reference = head.compute_pair_context(spins, time)[0, site_i, site_j]
    for sites in [(site_i,), (site_j,), (site_i, site_j)]:
        flipped = spins.clone()
        for site in sites:
            flipped[0, site] *= -1
        probe = head.compute_pair_context(flipped, time)[0, site_i, site_j]
        assert torch.equal(probe, reference), f"leak flipping {sites}"


def test_warm_started_blindness_probe_has_teeth():
    """The probe above must be capable of failing.

    Flipping a site OUTSIDE the pair has to move H_ij; otherwise the
    blindness assertions would be satisfied by a context that ignores the
    state entirely, and would pin nothing.
    """
    head = _warm_started_head()
    site_i, site_j, outside = 4, 20, 12
    spins = _state(36)
    time = torch.rand(1)

    reference = head.compute_pair_context(spins, time)[0, site_i, site_j]
    flipped = spins.clone()
    flipped[0, outside] *= -1
    moved = head.compute_pair_context(flipped, time)[0, site_i, site_j]
    assert (moved - reference).abs().max() > 1e-6


def test_warm_started_head_differs_from_fresh_initialisation():
    """Sanity: the transfer must actually change the destination weights.

    LayerNorm scales and shifts are excluded: they initialise to constant
    ones/zeros regardless of seed, so source and destination agree there for
    reasons that have nothing to do with the transfer. Every RANDOMLY
    initialised tensor must move.
    """
    head, fresh = _warm_started_head(), _head(6, seed=5)
    warm_sd, fresh_sd = head.state_dict(), fresh.state_dict()
    randomly_initialised = [
        k for k in warm_sd if fresh_sd[k].std() > 0 and warm_sd[k].numel() > 1
    ]
    assert len(randomly_initialised) > 0.5 * len(warm_sd)
    unchanged = [
        k for k in randomly_initialised if torch.equal(warm_sd[k], fresh_sd[k])
    ]
    assert unchanged == []


def test_warm_started_head_keeps_exact_state_swap_antisymmetry():
    """G(i,j|x) = -G(i,j|Swap2(x,i,j)) -- the property the head exists for."""
    head = _warm_started_head()
    spins = _state(36)
    time = torch.rand(1)
    scores = head(spins, time)

    for site_i, site_j in [(0, 35), (2, 3), (10, 25)]:
        swapped = head(swap2(spins, site_i, site_j), time)
        assert torch.equal(scores[0, site_i, site_j], -swapped[0, site_i, site_j])


def test_warm_started_head_keeps_exact_index_antisymmetry_and_zero_diagonal():
    head = _warm_started_head()
    scores = head(_state(36), torch.rand(1))
    assert torch.equal(scores, -scores.transpose(1, 2))
    assert torch.equal(torch.diagonal(scores, dim1=1, dim2=2), torch.zeros(1, 36))


def test_warm_started_head_produces_finite_gradients():
    head = _warm_started_head()
    head(_state(36), torch.rand(1)).pow(2).sum().backward()
    trained = [p for p in head.parameters() if p.grad is not None]
    assert trained
    assert all(torch.isfinite(p.grad).all() for p in trained)


# --------------------------------------------------------------------------
# Why the archived d64 -> d256 post-mortem cannot be right.
# --------------------------------------------------------------------------


def test_attention_readout_is_dead_weight_on_the_masked_attention_head():
    """The band heads replace the backbone's single-site readout entirely.

    Pinned because a 2026-08-13 post-mortem attributed a warm-started
    d256 run's huge gradient norm to those tensors being freshly
    initialised. They receive no gradient, so they cannot contribute to a
    gradient norm at all, and the attribution cannot stand.
    """
    head = _head(4)
    head(_state(16), torch.rand(1)).pow(2).sum().backward()
    dead = {n for n, p in head.named_parameters() if p.grad is None}
    assert {n for n in dead if "attention_readout" in n} == {
        n for n, _ in head.named_parameters() if "attention_readout" in n
    }


# --------------------------------------------------------------------------
# The runner wiring.
# --------------------------------------------------------------------------


def test_same_size_init_from_reproduces_the_source_head_exactly(tmp_path):
    """The pre-existing same-size --init-from path must stay byte-identical."""
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    cfg = CONFIGS["H2_d16_c50_s010_letf_ma_10k"]
    _, source_head = build_target_and_head(cfg, "cpu")
    checkpoint = tmp_path / "final.pt"
    torch.save(source_head.state_dict(), checkpoint)

    _, fresh_head = build_target_and_head(cfg, "cpu")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = fresh_head.load_state_dict(loaded, strict=False)

    assert not missing and not unexpected
    for key, value in fresh_head.state_dict().items():
        assert torch.equal(value, source_head.state_dict()[key]), key


def test_cli_passes_init_from_through(monkeypatch):
    import experiments.constrained_hard_03.run as hard_run

    seen = {}
    monkeypatch.setattr(hard_run, "train", lambda cfg, **kw: seen.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--cfg",
            "H2_d16_c50_s010_letf_ma_10k",
            "--no-wandb",
            "--init-from",
            "some/transfer.pt",
        ],
    )
    hard_run.main()
    assert seen["init_from"] == "some/transfer.pt"
