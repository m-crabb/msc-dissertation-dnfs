"""Falsification tests for multiple site orderings on the RASTER heads.

Written BEFORE the extension: these encode what correct looks like
independently of how the extra streams are wired.

WHY THIS EXISTS. `site_orderings` was factorised-head-only. The recorded
reason was a coverage argument -- orderings shrink the region no bilinear
term sees to the INTERSECTION of the per-ordering intervals, and the raster
heads already tile the lattice (prefix, band, suffix partition everything but
the two holes), so there was nothing left to shrink.

THAT ARGUMENT IS REFUTED BY MEASUREMENT. `fimo2ef` carries a prefix band, so
it tiles the lattice too, and it still loses 0.144 raw -- DISJOINT -- when its
second ordering is removed. Coverage cannot be the mechanism.

What survives is DEPTH. The band is constrained to be shallow: deep band
content would leak a hole's value through the two-hop path, so blindness
forces it to local, shallow features. Causal streams carry no such
constraint -- causality does the work, so P_i and S_j are deep and fully
mixed within their intervals. A second ordering therefore buys a DEEP read of
a region the band can only read SHALLOWLY, and that is as true of the
attention band as of the prefix sum.

THE CONSTRUCTION, and why it is blind. For ordering o with permutation
`order` (o-position -> site) and inverse `inv` (site -> o-position), the pair
{i, j} occupies o-positions inv[i] and inv[j]. Feed

    P^o at min(inv[i], inv[j])      S^o at max(inv[i], inv[j])

computed by running the causal stacks on the PERMUTED sequence. P^o at the
minimum has seen only sites earlier than BOTH holes in o; S^o at the maximum
only sites later than both. So each is blind to x_i and x_j, for every
ordering, by the same causality argument the row ordering already uses -- and
because min and max of an UNORDERED pair are symmetric, label symmetry
H_ji = H_ij survives for free.

WHAT IS PINNED HARDEST BELOW:

  * BLINDNESS, because it is the whole design rule and the new streams are a
    new way to break it. An off-by-one in either slice, or using inv[i] where
    min is meant, leaks a hole into its own context.

  * THAT THE FLAG IS NOT INERT. 13 archived cells already carry
    `site_orderings=('row','col')` on heads that IGNORE it, inherited through
    `replace(...)`. Adding real support to those heads means a config that
    used to be a no-op now CHANGES the model, so the test that the extra
    ordering moves the output is a provenance guard as much as a correctness
    one.

  * THAT ('row',) IS BYTE-IDENTICAL to the archived head. Every masked-
    attention and interval cell reported was trained without this, and the
    single-ordering path must draw the same parameters in the same order.

  * THAT THE ARM IS CHEAP IN PARAMETERS. An extra ordering adds NO modules --
    it reuses the backbone's causal stacks on a permuted sequence -- so the
    only parameter change is the pair readout's wider input. If that is not
    exactly what the count moves by, something grew that should not have.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

LATTICE_SIDE = 4
D = LATTICE_SIDE * LATTICE_SIDE
HIDDEN = 8

# Blindness here is an exclusion identity, not a cancellation: the streams are
# read one slot short of the hole, so a leak is a bug rather than fp residue.
BLIND_ATOL = 1e-6


def _backbone(seed=42):
    torch.manual_seed(seed)
    return LeTFRateMatrix(
        d=D,
        vocab_size=2,
        hidden_dim=HIDDEN,
        n_layers=2,
        n_heads=2,
        use_sdpa_readout=False,
    )


def _interval(orderings=("row",), **kw):
    torch.manual_seed(0)
    return IntervalSwapHead(
        _backbone(),
        pair_offsets=(1, LATTICE_SIDE),
        band_feature_dim=6,
        position_dim=5,
        lattice_side=LATTICE_SIDE,
        site_orderings=orderings,
        **kw,
    ).eval()


def _masked_attention(orderings=("row",), **kw):
    torch.manual_seed(0)
    return MaskedAttentionSwapHead(
        _backbone(),
        pair_offsets=(1, LATTICE_SIDE),
        band_feature_dim=6,
        position_dim=5,
        attention_dim=6,
        lattice_side=LATTICE_SIDE,
        site_orderings=orderings,
        **kw,
    ).eval()


BUILDERS = {"interval": _interval, "masked_attention": _masked_attention}
MULTI = [("row", "col"), ("row", "col", "diag")]


def _state(batch=2, seed=7):
    torch.manual_seed(seed)
    half = torch.cat([torch.ones(D // 2), -torch.ones(D - D // 2)])
    return torch.stack([half[torch.randperm(D)] for _ in range(batch)])


# --------------------------------------------------------------------------
# Blindness: the design rule, and the thing the new streams could break.
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_stay_blind(head_kind, orderings):
    """H_ij must not move when the tokens AT the holes move -- for EVERY pair,
    not a sampled few, because a permutation reorders which pairs are near a
    boundary and an off-by-one may only show at one of them."""
    head = BUILDERS[head_kind](orderings)
    x, t = _state(), torch.rand(2)
    H = head.compute_pair_context(x, t)
    for i in range(D):
        for j in range(i + 1, D):
            moved = x.clone()
            moved[:, i] *= -1
            moved[:, j] *= -1
            drift = (
                (head.compute_pair_context(moved, t)[:, i, j] - H[:, i, j])
                .abs()
                .max()
                .item()
            )
            assert drift < BLIND_ATOL, (
                f"{head_kind} {orderings}: H_{i},{j} moved by {drift:.2e}"
            )


@torch.no_grad()
@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_keep_exact_antisymmetry(head_kind, orderings):
    """min/max over an unordered pair are symmetric, so label symmetry -- and
    with it G[j,i] = -G[i,j] and a zero diagonal -- must hold at EXACTLY 0.0,
    not to a tolerance."""
    head = BUILDERS[head_kind](orderings)
    G = head(_state(), torch.rand(2))
    assert torch.equal(G, -G.transpose(1, 2))
    assert torch.equal(torch.diagonal(G, dim1=1, dim2=2), torch.zeros(2, D))


# --------------------------------------------------------------------------
# The flag must do something, and must cost only what it should.
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_change_the_scores(head_kind, orderings):
    """PROVENANCE GUARD as much as a correctness one: 13 archived cells carry
    `site_orderings=('row','col')` on heads that ignored it. Once these heads
    honour the field, a config that used to be a no-op changes the model, so
    an inert implementation would be invisible exactly where it matters."""
    single = BUILDERS[head_kind](("row",))
    multi = BUILDERS[head_kind](orderings)
    x, t = _state(), torch.rand(2)
    assert not torch.allclose(multi(x, t), single(x, t))


@pytest.mark.parametrize("head_kind", list(BUILDERS))
def test_single_ordering_is_byte_identical_to_the_archived_head(head_kind):
    """Every reported raster cell trained without this. `('row',)` must draw
    the same parameters in the same order and produce the same scores."""
    explicit = BUILDERS[head_kind](("row",))
    default = BUILDERS[head_kind]()
    assert list(explicit.state_dict()) == list(default.state_dict())
    for name, parameter in default.state_dict().items():
        assert torch.equal(parameter, explicit.state_dict()[name]), name
    with torch.no_grad():
        x, t = _state(), torch.rand(2)
        assert torch.equal(explicit(x, t), default(x, t))


@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_add_only_the_readout_widening(head_kind, orderings):
    """An extra ordering reuses the BACKBONE's causal stacks on a permuted
    sequence, so it owns no modules of its own. The only parameter change is
    the pair readout's first Linear taking 2*hidden more inputs per extra
    ordering -- and its bias does not move. Anything else means a module grew
    that should not have, which would confound a lift with capacity."""
    single = BUILDERS[head_kind](("row",))
    multi = BUILDERS[head_kind](orderings)
    grew = sum(p.numel() for p in multi.parameters()) - sum(
        p.numel() for p in single.parameters()
    )
    expected = (len(orderings) - 1) * 2 * HIDDEN * (2 * HIDDEN)
    assert grew == expected, f"{head_kind} {orderings}: grew {grew}, want {expected}"


@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_register_no_persistent_state(head_kind, orderings):
    """The permutations are index arithmetic, reproducible from the
    constructor args, so they ride as NON-persistent buffers and never enter a
    checkpoint -- the convention the factorised head already follows."""
    head = BUILDERS[head_kind](orderings)
    for key in head.state_dict():
        assert "_order" not in key, f"{key} leaked into the state_dict"


@pytest.mark.parametrize("head_kind", list(BUILDERS))
@pytest.mark.parametrize("orderings", MULTI)
def test_extra_orderings_train_every_head_parameter(head_kind, orderings):
    """`sum(G**2)`, not `sum(G)`: G is exactly antisymmetric, so its plain sum
    is identically zero as a function of the parameters and every gradient
    would vanish -- an objective that passes a dead module."""
    head = BUILDERS[head_kind](orderings).train()
    head(_state(), torch.rand(2)).pow(2).sum().backward()
    for name, parameter in head.named_parameters():
        if name.startswith("backbone."):
            continue
        assert parameter.grad is not None, f"{head_kind}: {name} got no grad"
        assert parameter.grad.abs().sum() > 0, f"{head_kind}: {name} is dead"


@torch.no_grad()
@pytest.mark.parametrize("head_kind", list(BUILDERS))
def test_extra_orderings_compose_with_the_triu_gather(head_kind):
    """The gather runs the pair-side map on the i<j list, so the per-ordering
    min/max gathers have to follow it there too."""
    dense = BUILDERS[head_kind](("row", "col"))
    gathered = BUILDERS[head_kind](("row", "col"), gather_triu_pairs=True)
    x, t = _state(), torch.rand(2)
    drift = (gathered(x, t) - dense(x, t)).abs().max().item()
    assert drift < 1e-6, f"{head_kind}: gather moved G by {drift:.2e}"


def test_unknown_ordering_is_refused():
    """A typo must not silently fall back to the identity permutation, which
    would look like a working second ordering and measure nothing."""
    with pytest.raises((ValueError, KeyError)):
        _masked_attention(("row", "spiral"))


def test_config_flag_reaches_the_raster_heads():
    """The field has ridden INERTLY on 13 cells. Once these heads honour it,
    the config path must actually deliver it."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    cfg = replace(CONFIGS["H2_d16_c50_s010_letf_ma_10k"], site_orderings=("row", "col"))
    head = build_swap_head(cfg, _backbone())
    assert head.site_orderings == ("row", "col")


@pytest.mark.parametrize("head_kind", list(BUILDERS))
def test_bilinear_exterior_refuses_extra_orderings(head_kind):
    """The bilinear combiner cannot carry a second ordering, so asking for
    both must raise rather than quietly return the one-ordering head.

    `_ordering_exterior_rows` runs on the "mlp" branch alone and
    `_bilinear_exterior` reads the row summaries, so under "bilinear" the
    extra ordering is INVISIBLE, not merely inert: measured at d=16, mlp
    gains 256 parameters and moves G by 1.2e-2 when "col" is added, while
    bilinear gains 0 and returns a bit-identical G. Left unguarded, the
    `ivmo2ef` + bilinear cell would read as the single-variable test of the
    factorisation while also deleting the second ordering -- worth +0.154
    raw and disjoint, the largest lever on this axis -- and the regression
    would be attributed to the wrong knob.
    """
    with pytest.raises(ValueError, match="row ordering only"):
        BUILDERS[head_kind](("row", "col"), exterior_combiner="bilinear")


@pytest.mark.parametrize("head_kind", list(BUILDERS))
def test_bilinear_exterior_still_builds_at_one_ordering(head_kind):
    """The guard is about the COMBINATION; the archived `mab` / `ivb` cells
    ran at ("row",) and must keep building unchanged."""
    head = BUILDERS[head_kind](("row",), exterior_combiner="bilinear")
    assert head.exterior_combiner == "bilinear"
    assert head(_state(), torch.rand(2)).isfinite().all()
