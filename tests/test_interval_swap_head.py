"""Falsification tests for the three-interval (leave-two-out) swap head.

The workhorse is the blindness probe -- flip a held-out spin and demand H_ij
unchanged -- which is strictly stronger than antisymmetry (a symmetric leak
survives the swap test but not the flip test), and it is paired with a
sensitivity control (a head that ignores x entirely is perfectly blind) and a
leaky negative control (proving the probe has teeth). No numeric oracle
against DoublyHollowSwapHead: the architecture changed, so property tests +
the D=4 gate are the bar.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import _masked_body, swap2
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)

ATOL = 1e-5  # suite bar; subtractive band assembly leaves ~1e-6 fp residue,
#              hole-free (blocked) assembly should sit at exactly 0.0


def _head(
    d=9,
    offsets=(1, 3),
    seed=42,
    hidden_dim=8,
    n_heads=2,
    n_layers=2,
    gather_triu_pairs=False,
):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        use_sdpa_readout=False,
    )
    head = IntervalSwapHead(
        backbone, pair_offsets=offsets, gather_triu_pairs=gather_triu_pairs
    )
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
def test_causal_summaries_blindness():
    """prefix[:, i] blind to x_{>=i}; suffix[:, j] blind to x_{<=j}.

    Probes the boundary site itself (k = i, k = j) -- the off-by-one in the
    slice-trick indexing is the single most likely implementation bug.
    """
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    prefix, suffix = head.causal_summaries(x, t)
    assert prefix.shape == suffix.shape == (1, 9, head.backbone.hidden_dim)

    for i in (0, 1, 4, 8):
        for k in {i, min(i + 2, 8)}:
            prefix_flipped, _ = head.causal_summaries(_flip(x, k), t)
            drift = _drift(prefix_flipped[:, i, :], prefix[:, i, :])
            assert drift < ATOL, f"prefix[{i}] leaks x_{k}: {drift:.2e}"
    for j in (0, 4, 7, 8):
        for k in {j, max(j - 2, 0)}:
            _, suffix_flipped = head.causal_summaries(_flip(x, k), t)
            drift = _drift(suffix_flipped[:, j, :], suffix[:, j, :])
            assert drift < ATOL, f"suffix[{j}] leaks x_{k}: {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("gather_triu_pairs", [False, True], ids=["dense", "triu"])
def test_pair_context_blind_to_both_holes(gather_triu_pairs):
    """Blindness core: H_ij invariant under any change to x_i, x_j (not just
    swap). Run on both assembly paths: the triu-pair gather re-indexes the
    per-pair work, and blindness is a property of which terms enter each
    pair's row, so it must survive the re-indexing untouched."""
    head = _head(d=9, gather_triu_pairs=gather_triu_pairs)
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
            drift = _drift(head.compute_pair_context(flipped_x, t)[:, i, j, :], base)
            assert drift < ATOL, f"H_[{i},{j}] leaks {label}: {drift:.2e}"


@torch.no_grad()
def test_pair_context_sensitive_to_context():
    """Anti-triviality control: a head blind to everything passes the
    blindness probes. H_ij must actually depend on each visible interval."""
    head = _head(d=9)
    x = _state(d=9)
    t = torch.rand(1)
    i, j = 2, 5  # prefix {0,1}, band {3,4}, suffix {6,7,8}
    base = head.compute_pair_context(x, t)[:, i, j, :]

    for site, interval in ((0, "prefix"), (3, "band"), (7, "suffix")):
        drift = _drift(head.compute_pair_context(_flip(x, site), t)[:, i, j, :], base)
        assert drift > 1e-7, f"H_[{i},{j}] ignores its {interval} (site {site})"


@torch.no_grad()
def test_blindness_probe_has_teeth():
    """Negative control for the test: an unmasked-body context (the leak the
    mask-one head spends d passes preventing) must register loudly under the
    same flip probe, pinning the probe's sensitivity floor."""
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
def test_antisymmetric_at_init(d, offsets):
    """Antisymmetry: G(i,j|x) = -G(i,j|Swap2(x,i,j)) at random init, all active
    pairs."""
    head = _head(d=d, offsets=offsets)
    x = _state(d=d)
    t = torch.rand(1)
    G = head(x, t)
    worst = 0.0
    for i, j in _active_pairs(x):
        G_swapped = head(swap2(x, i, j), t)
        worst = max(worst, (G[0, i, j] + G_swapped[0, i, j]).abs().item())
    assert worst < ATOL, f"d={d}: antisymmetry residual {worst:.2e}"


@torch.no_grad()
def test_trivial_swap_vanishes():
    """Same-spin pair => zero token difference => G == 0, untrained."""
    head = _head(d=9)
    x = _state(d=9)
    G = head(x, torch.rand(1))
    same = [(i, j) for i in range(9) for j in range(i + 1, 9) if x[0, i] == x[0, j]]
    assert same, "fixture must contain at least one same-spin pair"
    worst = max(G[0, i, j].abs().item() for (i, j) in same)
    assert worst < ATOL, f"trivial-swap nonzero: {worst:.2e}"


@torch.no_grad()
def test_index_antisymmetry_pinned():
    """G[j,i] == -G[i,j]: pins the label-symmetry convention (H_ji := H_ij),
    in deliberate contrast with the mask-one head's pinned label asymmetry.
    The downstream i<j gather is indifferent, but the convention must not
    change silently.
    """
    head = _head(d=9)
    x = _state(d=9)
    G = head(x, torch.rand(1))
    worst = (G + G.transpose(1, 2)).abs().max().item()
    assert worst < ATOL, f"index-antisymmetry broken: {worst:.2e}"


@torch.no_grad()
def test_shapes_finite_and_pair_gather():
    """Drop-in contract: batched shapes, finiteness, and the downstream
    upper-triangle gather the sampler/loss actually consume."""
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
    G = head(x, t)
    assert G.shape == (batch, d, d)
    assert torch.isfinite(G).all()

    pairs = upper_tri_pairs(d, x.device)
    scores = gather_pair_scores(G, pairs)
    assert scores.shape == (batch, pairs.shape[0])
    assert torch.isfinite(scores).all()


def test_head_parameters_receive_grad():
    """Every head-owned module must be live in the graph (catches a band or
    position feature silently dropped from the readout concat), and the
    backbone's causal stacks must be live through P/S. attention_readout is
    pinned dead: this head deliberately replaces it."""
    head = _head(d=9)
    x = _state(d=9)
    head(x, torch.rand(1)).sum().backward()

    for name, param in head.named_parameters():
        if "backbone" in name:
            continue
        assert param.grad is not None, f"head module dead in graph: {name}"

    stacks = list(head.backbone.fwd_stack.parameters()) + list(
        head.backbone.bwd_stack.parameters()
    )
    assert any(p.grad is not None for p in stacks), "causal stacks dead: no P/S"
    readout_grads = [p.grad for p in head.backbone.attention_readout.parameters()]
    assert all(g is None for g in readout_grads), (
        "attention_readout unexpectedly live; the one-pass design routed "
        "through the machinery it exists to replace"
    )


# --- bilinear exterior combiner ----------------------------------------------
# [prefix, suffix] leave the per-pair MLP for a rank-R product added to H
# before context_norm. Pins: the default is byte-identical to the archived
# head; the bilinear head keeps blindness and exact index antisymmetry for
# both band aggregators; its MLP input is the band and positions only; the
# factor maps receive gradient.


def _combiner_head(head_cls, exterior_combiner, d=9, seed=42):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
        use_sdpa_readout=False,
    )
    kw = (
        {"attention_dim": 6, "lattice_side": 3}
        if head_cls is MaskedAttentionSwapHead
        else {}
    )
    head = head_cls(
        backbone,
        pair_offsets=(1, 3),
        band_feature_dim=5,
        position_dim=4,
        exterior_combiner=exterior_combiner,
        bilinear_rank=3,
        **kw,
    )
    head.eval()
    return head


@pytest.mark.parametrize("head_cls", [IntervalSwapHead, MaskedAttentionSwapHead])
def test_mlp_combiner_is_byte_identical_to_archived_head(head_cls):
    torch.manual_seed(7)
    backbone = LeTFRateMatrix(
        d=9, vocab_size=2, hidden_dim=8, n_layers=2, n_heads=2, use_sdpa_readout=False
    )
    kw = (
        {"attention_dim": 6, "lattice_side": 3}
        if head_cls is MaskedAttentionSwapHead
        else {}
    )
    archived = head_cls(
        backbone, pair_offsets=(1, 3), band_feature_dim=5, position_dim=4, **kw
    )
    torch.manual_seed(7)
    backbone = LeTFRateMatrix(
        d=9, vocab_size=2, hidden_dim=8, n_layers=2, n_heads=2, use_sdpa_readout=False
    )
    explicit = head_cls(
        backbone,
        pair_offsets=(1, 3),
        band_feature_dim=5,
        position_dim=4,
        exterior_combiner="mlp",
        **kw,
    )
    assert archived.state_dict().keys() == explicit.state_dict().keys()
    assert "prefix_factors.weight" not in archived.state_dict()
    x, t = _state(), torch.rand(1)
    assert torch.equal(archived.eval()(x, t), explicit.eval()(x, t))


@pytest.mark.parametrize("head_cls", [IntervalSwapHead, MaskedAttentionSwapHead])
def test_bilinear_combiner_context_blind_to_both_holes(head_cls):
    head = _combiner_head(head_cls, "bilinear")
    x, t = _state(), torch.rand(1)
    H = head.compute_pair_context(x, t)
    for i in range(9):
        for j in range(i + 1, 9):
            for flips in ((i,), (j,), (i, j)):
                y = x.clone()
                y[:, list(flips)] *= -1
                drift = (
                    (head.compute_pair_context(y, t)[:, i, j] - H[:, i, j])
                    .abs()
                    .max()
                    .item()
                )
                assert drift < ATOL, (head_cls.__name__, i, j, flips, drift)


@pytest.mark.parametrize("head_cls", [IntervalSwapHead, MaskedAttentionSwapHead])
def test_bilinear_combiner_readout_sees_band_and_positions_only(head_cls):
    head = _combiner_head(head_cls, "bilinear")
    band_dim = 5 * (1 + 2)
    assert head.pair_readout[0].in_features == band_dim + 2 * 4
    assert (
        _combiner_head(head_cls, "mlp").pair_readout[0].in_features
        == 2 * 8 + band_dim + 2 * 4
    )


@pytest.mark.parametrize("head_cls", [IntervalSwapHead, MaskedAttentionSwapHead])
def test_bilinear_combiner_antisymmetric_and_exterior_sensitive(head_cls):
    head = _combiner_head(head_cls, "bilinear")
    x, t = _state(), torch.rand(1)
    G = head(x, t)
    assert (G + G.transpose(1, 2)).abs().max().item() < ATOL
    # the exterior must still reach H: flip a site outside (i, j) = (3, 5)
    y = x.clone()
    y[:, 0] *= -1
    assert (
        head.compute_pair_context(y, t)[:, 3, 5]
        - head.compute_pair_context(x, t)[:, 3, 5]
    ).abs().max() > ATOL
    head.train()
    head(x, t).sum().backward()
    for name in ("prefix_factors", "suffix_factors"):
        assert getattr(head, name).weight.grad is not None
