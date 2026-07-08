"""Falsification tests for the exclusion-mask band-attention swap head.

Ported from test_interval_swap_head.py (the probes are aggregator-agnostic
by design) with the bars TIGHTENED: exclusion happens before the softmax, so
hole terms never enter any computed quantity and blindness/antisymmetry are
asserted at exactly 0.0 -- not the interval head's ATOL, which priced its
fp cancellation residue. A nonzero residual here is a leak, not noise.

Two (a)-specific additions: empty visible sets (adjacent pairs, bands
shorter than an offset) must yield exact-zero band blocks AND finite
gradients -- the fully-masked-softmax row is the one place this head could
NaN, and the finite EXCLUDED_SCORE_FILL + index-mask overwrite is the
designed guard.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import _masked_body, swap2
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)


def _head(d=9, offsets=(1, 3), seed=42, hidden_dim=8, n_heads=2, n_layers=2):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=hidden_dim, n_layers=n_layers,
        n_heads=n_heads, use_sdpa_readout=False,
    )
    head = MaskedAttentionSwapHead(backbone, pair_offsets=offsets)
    head.eval()
    return head


def _state(d=9, seed=1):
    """A (1, d) ±1 state guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (1, d)) * 2 - 1).float()
    if bool((x > 0).all()) or bool((x < 0).all()):
        x[0, 0] *= -1
    return x


def _flip(x, *sites):
    y = x.clone()
    for s in sites:
        y[0, s] *= -1
    return y


def _active_pairs(x):
    d = x.shape[1]
    return [(i, j) for i in range(d) for j in range(i + 1, d) if x[0, i] != x[0, j]]


def _drift(a, b):
    return (a - b).abs().max().item()


# Edge-heavy pair list for d=9: empty prefix+suffix, empty band (adjacent),
# band shorter than the largest offset, and a generic interior pair.
PROBE_PAIRS = [(0, 8), (3, 4), (0, 1), (7, 8), (2, 5)]


@torch.no_grad()
def test_pair_context_blind_to_both_holes_exactly():
    """K1 core, exact form: H_ij must not move AT ALL under any change to
    x_i or x_j. Excluded terms carry softmax weight +0.0, so the residual
    is zero in exact arithmetic AND in floating point -- assert equality."""
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    H = head.compute_pair_context(x, t)

    for i, j in PROBE_PAIRS:
        base = H[:, i, j, :]
        for flipped_x, label in (
            (_flip(x, i), f"x_{i}"),
            (_flip(x, j), f"x_{j}"),
            (_flip(x, i, j), f"x_{i} and x_{j}"),
        ):
            drift = _drift(
                head.compute_pair_context(flipped_x, t)[:, i, j, :], base
            )
            assert drift == 0.0, f"H_[{i},{j}] leaks {label}: {drift:.2e}"


@torch.no_grad()
def test_band_summaries_blind_exactly():
    """Sharper probe, directly on the band (the component that differs from
    the interval head): flip either hole, band[:, i, j] must be identical."""
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    band = head.band_summaries(x, t)

    for i, j in PROBE_PAIRS:
        for flipped_x, label in ((_flip(x, i), f"x_{i}"), (_flip(x, j), f"x_{j}")):
            drift = _drift(head.band_summaries(flipped_x, t)[:, i, j, :],
                           band[:, i, j, :])
            assert drift == 0.0, f"band[{i},{j}] leaks {label}: {drift:.2e}"


@torch.no_grad()
def test_band_empty_visible_sets_are_exact_zero():
    """Empty intervals must be exact zeros, per family: an adjacent pair has
    no interior at all; a 3-wide gap admits unary and offset-1 terms but no
    offset-3 term. Channel layout is [unary | offset per pair_offsets]."""
    head = _head(d=9, offsets=(1, 3))
    x = _state(d=9)
    band = head.band_summaries(x, torch.rand(1))
    F = band.shape[-1] // 3

    assert (band[:, 3, 4, :] == 0.0).all(), "adjacent pair: whole band nonzero"
    offset3_block = band[:, 2, 5, 2 * F:]
    assert (offset3_block == 0.0).all(), "j-i=3 < delta+2: offset-3 block nonzero"
    assert (band[:, 2, 5, : 2 * F] != 0.0).any(), (
        "unary/offset-1 blocks empty on a 3-wide gap: over-masking"
    )


@torch.no_grad()
def test_pair_context_sensitive_to_context():
    """Anti-triviality control: a head blind to EVERYTHING passes the
    blindness probes. H_ij must actually depend on each visible interval."""
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = 2, 5  # prefix {0,1}, band {3,4}, suffix {6,7,8}
    base = head.compute_pair_context(x, t)[:, i, j, :]

    for site, interval in ((0, "prefix"), (3, "band"), (7, "suffix")):
        drift = _drift(
            head.compute_pair_context(_flip(x, site), t)[:, i, j, :], base
        )
        assert drift > 1e-7, f"H_[{i},{j}] ignores its {interval} (site {site})"


@torch.no_grad()
def test_blindness_probe_has_teeth():
    """Negative control for the TEST: an unmasked-body context must register
    loudly under the same flip probe, pinning the probe's sensitivity."""
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = 2, 5
    leaky = _masked_body(head.backbone, x, t, ())[:, j, :]
    leaky_flipped = _masked_body(head.backbone, _flip(x, i), t, ())[:, j, :]
    assert _drift(leaky_flipped, leaky) > 1e-4, (
        "flip probe cannot distinguish a leaky context; blindness tests are void"
    )


@torch.no_grad()
@pytest.mark.parametrize("d,offsets", [(9, (1, 3)), (16, (1, 4))])
def test_antisymmetric_at_init_exactly(d, offsets):
    """K1, exact form: G(i,j|x) = -G(i,j|Swap2(x,i,j)) with residual 0.0.
    H is exactly blind and the omega difference negates exactly, so the
    einsum negates term-by-term -- no tolerance needed."""
    head = _head(d=d, offsets=offsets)
    x = _state(d=d)
    t = torch.rand(1)
    G = head(x, t)
    worst = 0.0
    for i, j in _active_pairs(x):
        G_swapped = head(swap2(x, i, j), t)
        worst = max(worst, (G[0, i, j] + G_swapped[0, i, j]).abs().item())
    assert worst == 0.0, f"d={d}: antisymmetry residual {worst:.2e}"


@torch.no_grad()
def test_trivial_swap_vanishes_exactly():
    """Same-spin pair => zero token difference => G == 0.0, untrained."""
    head = _head(d=9)
    x = _state(d=9)
    G = head(x, torch.rand(1))
    same = [(i, j) for i in range(9) for j in range(i + 1, 9) if x[0, i] == x[0, j]]
    assert same, "fixture must contain at least one same-spin pair"
    worst = max(G[0, i, j].abs().item() for (i, j) in same)
    assert worst == 0.0, f"trivial-swap nonzero: {worst:.2e}"


@torch.no_grad()
def test_index_antisymmetry_pinned_exactly():
    """G[j,i] == -G[i,j]: pins the label-SYMMETRY convention (H_ji := H_ij),
    inherited from the interval head. Exact for the same reason as the
    state-swap antisymmetry."""
    head = _head(d=9)
    x = _state(d=9)
    G = head(x, torch.rand(1))
    worst = (G + G.transpose(1, 2)).abs().max().item()
    assert worst == 0.0, f"index-antisymmetry broken: {worst:.2e}"


@torch.no_grad()
def test_shapes_finite_and_pair_gather():
    """Drop-in contract: batched shapes, finiteness (including the
    fully-masked adjacent-pair rows -- the NaN risk this head must guard),
    and the upper-triangle gather the sampler/loss actually consume."""
    d, batch = 9, 3
    head = _head(d=d)
    torch.manual_seed(7)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    for b in range(batch):
        if bool((x[b] > 0).all()) or bool((x[b] < 0).all()):
            x[b, 0] *= -1
    t = torch.rand(batch)

    H = head.compute_pair_context(x, t)
    assert H.shape == (batch, d, d, head.backbone.hidden_dim)
    assert torch.isfinite(H).all()
    G = head(x, t)
    assert G.shape == (batch, d, d)
    assert torch.isfinite(G).all()

    pairs = upper_tri_pairs(d, x.device)
    scores = gather_pair_scores(G, pairs)
    assert scores.shape == (batch, pairs.shape[0])
    assert torch.isfinite(scores).all()


def test_head_parameters_receive_grad_and_grads_finite():
    """Every head-owned module live in the graph -- including the new band
    query/key projections -- with FINITE gradients despite the fully-masked
    rows in the batch (adjacent pairs are forced into the fixture: an -inf
    fill would send NaN through the softmax backward even where the forward
    output is discarded). attention_readout stays pinned dead."""
    head = _head(d=9)
    x = _state(d=9)
    x[0, 3], x[0, 4] = 1.0, -1.0  # active adjacent pair => empty band row
    head(x, torch.rand(1)).sum().backward()

    for name, param in head.named_parameters():
        if "backbone" in name:
            continue
        assert param.grad is not None, f"head module dead in graph: {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad: {name}"

    stacks = list(head.backbone.fwd_stack.parameters()) + list(
        head.backbone.bwd_stack.parameters()
    )
    assert any(p.grad is not None for p in stacks), "causal stacks dead: no P/S"
    for p in stacks:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite grad in causal stacks"
    readout_grads = [
        p.grad for p in head.backbone.attention_readout.parameters()
    ]
    assert all(g is None for g in readout_grads), (
        "attention_readout unexpectedly live; the one-pass design routed "
        "through the machinery it exists to replace"
    )


def test_blindness_holds_at_tuned_band_capacity():
    """The band-capacity knobs (design 2026-07-08) must not perturb the
    exclusion logic: H_ij stays EXACTLY unchanged under any flip of x_i /
    x_j at non-default widths and offsets, including an offset (4) that is
    neither row nor column adjacency."""
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(
        d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2,
        use_sdpa_readout=False,
    )
    head = MaskedAttentionSwapHead(
        backbone, pair_offsets=(1, 2, 4, 8), band_feature_dim=32, attention_dim=64
    )
    head.eval()
    x = torch.randint(0, 2, (3, 16)).float() * 2 - 1
    t = torch.rand(3)
    H = head.compute_pair_context(x, t)
    for i, j in ((0, 5), (4, 11), (14, 15)):
        for flip_sites in ((i,), (j,), (i, j)):
            x_flipped = x.clone()
            for site in flip_sites:
                x_flipped[:, site] = -x_flipped[:, site]
            H_flipped = head.compute_pair_context(x_flipped, t)
            assert torch.equal(H[:, i, j], H_flipped[:, i, j])
