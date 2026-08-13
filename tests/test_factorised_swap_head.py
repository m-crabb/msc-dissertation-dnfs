"""Falsification tests for the factorised (low-rank bilinear + hole-subtracted
global) swap head.

Written BEFORE the head body: these encode what correct looks like for the
factorised one-pass design, independently of its implementation. The head's
claim is that the pair context can be assembled from per-site pieces --

    H_ij = sum_r a_r(prefix_i, pos_i) * b_r(suffix_j, pos_j)      (bilinear)
         + rho( c(x) - psi_i - psi_j )                            (global)
         + tau(t)                                                 (time)

with every piece blind to the token values at i and j, so the shared readout
G(i,j|x) = <H_ij, omega_{x_i} - omega_{x_j}> keeps exact state-swap
antisymmetry with NO per-pair network evaluation.

The workhorse is the same blindness probe as the interval-head suite (flip a
hole spin, demand H unchanged -- strictly stronger than antisymmetry), plus
two pins specific to this head:

  * the bilinear-only ablation must be blind to the ENTIRE interval interior
    (prefix_i stops before i, suffix_j starts after j, and nothing else looks
    at x) -- the coverage hole the global term exists to fill, stated as a
    test rather than prose;
  * forward (the einsum assembly that never materialises H) must agree with
    the readable reference that does materialise H.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import _masked_body, swap2
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)

ATOL = 1e-5  # suite bar, matching the interval suite: the global term's
#              subtractive hole removal leaves an ~ulp fp residue; the causal
#              streams should sit far below this.


def _head(
    d=9,
    seed=42,
    hidden_dim=8,
    n_heads=2,
    n_layers=2,
    bilinear_rank=3,
    factor_dim=4,
    global_feature_dim=6,
    position_dim=5,
    use_bilinear=True,
    use_global=True,
):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=hidden_dim, n_layers=n_layers,
        n_heads=n_heads, use_sdpa_readout=False,
    )
    head = FactorisedSwapHead(
        backbone,
        bilinear_rank=bilinear_rank,
        factor_dim=factor_dim,
        global_feature_dim=global_feature_dim,
        position_dim=position_dim,
        use_bilinear=use_bilinear,
        use_global=use_global,
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


# Edge-heavy pair list for d=9: empty prefix+suffix, adjacent (empty interior),
# and a generic interior pair. No empty-band special case exists in this head
# (the global term is defined for every pair), so adjacency is exercised only
# as an ordinary input, not a convention.
PROBE_PAIRS = [(0, 8), (3, 4), (0, 1), (7, 8), (2, 5)]

ABLATIONS = [
    pytest.param(True, True, id="bilinear+global"),
    pytest.param(True, False, id="bilinear-only"),
    pytest.param(False, True, id="global-only"),
]


@torch.no_grad()
@pytest.mark.parametrize("use_bilinear,use_global", ABLATIONS)
def test_pair_context_blind_to_both_holes(use_bilinear, use_global):
    """Core claim: H_ij invariant under ANY change to x_i, x_j, in every
    ablation arm -- blindness is per-term, so no arm may leak."""
    head = _head(use_bilinear=use_bilinear, use_global=use_global)
    x = _state()
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
            assert drift < ATOL, f"H_[{i},{j}] leaks {label}: {drift:.2e}"


@torch.no_grad()
def test_pair_context_sensitive_to_all_coverage_regions():
    """Anti-triviality control: with both terms on, H_ij must move when any
    visible region moves -- prefix (bilinear left), suffix (bilinear right),
    and interval interior (reachable ONLY through the global term)."""
    head = _head()
    x = _state()
    t = torch.rand(1)
    i, j = 2, 5  # prefix {0,1}, interior {3,4}, suffix {6,7,8}
    base = head.compute_pair_context(x, t)[:, i, j, :]

    for site, region in ((0, "prefix"), (3, "interior"), (7, "suffix")):
        drift = _drift(
            head.compute_pair_context(_flip(x, site), t)[:, i, j, :], base
        )
        assert drift > 1e-7, f"H_[{i},{j}] ignores its {region} (site {site})"


@torch.no_grad()
def test_bilinear_only_blind_to_interval_interior():
    """The coverage hole, pinned: with the global term off, NOTHING in the
    head sees the open interval (i, j) -- prefix_i stops before i, suffix_j
    starts after j. This is the structural fact that makes the global term a
    necessary component rather than an enrichment."""
    head = _head(use_bilinear=True, use_global=False)
    x = _state()
    t = torch.rand(1)
    i, j = 2, 6
    base = head.compute_pair_context(x, t)[:, i, j, :]

    for interior_site in (3, 4, 5):
        drift = _drift(
            head.compute_pair_context(_flip(x, interior_site), t)[:, i, j, :],
            base,
        )
        assert drift < ATOL, (
            f"bilinear-only H_[{i},{j}] sees interior x_{interior_site}: "
            f"{drift:.2e} -- a causal-stream slice is off by one"
        )


@torch.no_grad()
def test_blindness_probe_has_teeth():
    """Negative control for the TEST: an unmasked-body context must register
    loudly under the same flip probe, pinning the probe's sensitivity floor."""
    head = _head()
    x = _state()
    t = torch.rand(1)
    i, j = 2, 5
    leaky = _masked_body(head.backbone, x, t, ())[:, j, :]
    leaky_flipped = _masked_body(head.backbone, _flip(x, i), t, ())[:, j, :]
    assert _drift(leaky_flipped, leaky) > 1e-4, (
        "flip probe cannot distinguish a leaky context; blindness tests are void"
    )


@torch.no_grad()
@pytest.mark.parametrize("use_bilinear,use_global", ABLATIONS)
@pytest.mark.parametrize("d", [9, 16])
def test_antisymmetric_at_init(d, use_bilinear, use_global):
    """G(i,j|x) = -G(i,j|Swap2(x,i,j)) at random init, all active pairs,
    every ablation arm."""
    head = _head(d=d, use_bilinear=use_bilinear, use_global=use_global)
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
    head = _head()
    x = _state()
    G = head(x, torch.rand(1))
    same = [(i, j) for i in range(9) for j in range(i + 1, 9) if x[0, i] == x[0, j]]
    assert same, "fixture must contain at least one same-spin pair"
    worst = max(G[0, i, j].abs().item() for (i, j) in same)
    assert worst < ATOL, f"trivial-swap nonzero: {worst:.2e}"


@torch.no_grad()
def test_index_antisymmetry_pinned():
    """G[j,i] == -G[i,j] exactly: pins the label-SYMMETRY convention
    (H_ji := H_ij, as the interval head) AND the upper-triangle-then-mirror
    assembly, which makes index antisymmetry an identity rather than a
    numerical property."""
    head = _head()
    x = _state()
    G = head(x, torch.rand(1))
    worst = (G + G.transpose(1, 2)).abs().max().item()
    assert worst == 0.0, f"index-antisymmetry not exact: {worst:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("use_bilinear,use_global", ABLATIONS)
def test_forward_matches_context_readout(use_bilinear, use_global):
    """The efficient assembly (einsum over factors, H never materialised)
    must agree with the readable reference <H_ij, omega_i - omega_j> built
    from compute_pair_context -- the mask-one forward/forward_looped pattern,
    here guarding the factorised regrouping."""
    head = _head(use_bilinear=use_bilinear, use_global=use_global)
    x = _state()
    t = torch.rand(1)
    omega = head.backbone.omega(((x + 1) / 2).long())
    omega_f = head.omega_projection(omega)
    token_difference = omega_f.unsqueeze(2) - omega_f.unsqueeze(1)
    H = head.compute_pair_context(x, t)
    G_reference = (token_difference * H).sum(-1)
    assert _drift(head(x, t), G_reference) < ATOL


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
    assert H.shape == (batch, d, d, head.factor_dim)
    G = head(x, t)
    assert G.shape == (batch, d, d)
    assert torch.isfinite(G).all()

    pairs = upper_tri_pairs(d, x.device)
    scores = gather_pair_scores(G, pairs)
    assert scores.shape == (batch, pairs.shape[0])
    assert torch.isfinite(scores).all()


def test_head_parameters_receive_grad():
    """Every head-owned module must be live in the graph, the backbone's
    causal stacks must be live through the factor streams, and
    attention_readout is pinned DEAD (this head replaces it)."""
    head = _head()
    x = _state()
    head(x, torch.rand(1)).sum().backward()

    for name, param in head.named_parameters():
        if "backbone" in name:
            continue
        assert param.grad is not None, f"head module dead in graph: {name}"

    stacks = list(head.backbone.fwd_stack.parameters()) + list(
        head.backbone.bwd_stack.parameters()
    )
    assert any(p.grad is not None for p in stacks), "causal stacks dead"
    readout_grads = [
        p.grad for p in head.backbone.attention_readout.parameters()
    ]
    assert all(g is None for g in readout_grads), (
        "attention_readout unexpectedly live; the one-pass design routed "
        "through the machinery it exists to replace"
    )


def test_flags_must_enable_at_least_one_term():
    """A head with both terms off scores every pair from time and positions
    alone -- reject at construction, not at first NaN."""
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(
        d=9, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2,
    )
    with pytest.raises(ValueError):
        FactorisedSwapHead(backbone, use_bilinear=False, use_global=False)


@torch.no_grad()
def test_G_stays_fp32_under_bf16_autocast():
    """The pair-score tensor feeds fp32-only diagnostics downstream; the
    interval/mask-one heads keep G fp32 under the eval autocast block and
    this head must too, even though its assembly uses einsum (which autocast
    would otherwise emit in bf16)."""
    head = _head()
    x = _state()
    t = torch.rand(1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        G = head(x, t)
    assert G.dtype == torch.float32, f"G downcast to {G.dtype} under autocast"
