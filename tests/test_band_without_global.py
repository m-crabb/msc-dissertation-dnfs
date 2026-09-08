"""Tests for an interior band with no global term.

The factorised head's interior is a 2x2: what the pooling sees (the open
interval between the holes, or the whole lattice bar the two holes) crossed
with how it weights what it sees (uniformly, or by a learned softmax). One of
those cells could not be built alone, because the band had no readout of its
own -- it was concatenated onto the hole-subtracted global vector and shared
`global_context_readout`, so every head carrying a band also carried a global
term. At 16x16 the only interior mechanism measured in isolation was the
learned one (the masked-attention head); both uniform-weight interiors arrive
fused inside `fimo2ef`, which sets `interior_band='prefix'` and
`use_global=True`.

`fbil` (bilinear exterior, no global term) was seed-unstable at the 4x4 gate
and the reading on record is "the global term stabilises", measured with no
interior mechanism at all. If a prefix band alone stabilises the bilinear
exterior just as well, the global term is one of two interchangeable interior
suppliers. Neither is a cost lever -- both are O(1) per pair (one cumsum plus
two gathers against one lattice sum plus four gathers).

Pinned hardest below:

  * `global_site_features`, `global_context_norm` and `global_context_readout`
    keep those names even when no global term exists: 101 archived factorised
    cells carry those keys in their state_dict.
  * the construction order of the init draws survives, so a `use_global=True`
    head built at a fixed seed matches the pre-change one exactly, not to a
    tolerance.
  * with bilinear, global and band all absent the pair score depends only on
    time, so the guard widens rather than disappears.
  * blindness is unconditional: with no hole subtraction, index exclusion on
    the band is the only thing carrying it, so it is tested directly rather
    than inherited.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

LATTICE_SIDE = 4
D = LATTICE_SIDE * LATTICE_SIDE

# Blindness is an exclusion identity on the band path, so it is exact up to
# the fp cancellation the hole subtraction introduces -- and with no global
# term there is no subtraction, so the bar is the suite's tight one.
BLIND_ATOL = 1e-6


def _backbone(seed=42):
    torch.manual_seed(seed)
    return LeTFRateMatrix(
        d=D,
        vocab_size=2,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
        use_sdpa_readout=False,
    )


def _head(**kw):
    torch.manual_seed(0)
    defaults = dict(
        bilinear_rank=3,
        factor_dim=4,
        global_feature_dim=6,
        position_dim=5,
        band_feature_dim=6,
        attention_dim=6,
        pair_offsets=(1, LATTICE_SIDE),
        lattice_side=LATTICE_SIDE,
    )
    return FactorisedSwapHead(_backbone(), **{**defaults, **kw}).eval()


def _state(batch=2, seed=7):
    torch.manual_seed(seed)
    half = torch.cat([torch.ones(D // 2), -torch.ones(D - D // 2)])
    return torch.stack([half[torch.randperm(D)] for _ in range(batch)])


# The new cell across every axis it has to survive: both band kinds, the
# triu-pair gather (which runs the interior on the i<j list), and a second
# site ordering (which adds a bilinear term but no interior machinery).
BAND_ONLY_CASES = {
    "prefix": {"interior_band": "prefix", "use_global": False},
    "attention": {"interior_band": "attention", "use_global": False},
    "prefix_gathered": {
        "interior_band": "prefix",
        "use_global": False,
        "gather_triu_pairs": True,
    },
    "prefix_two_orderings": {
        "interior_band": "prefix",
        "use_global": False,
        "site_orderings": ("row", "col"),
    },
}


@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_band_without_global_builds_and_scores(case):
    """The cell builds at all: the guard that forbade it is gone."""
    head = _head(**BAND_ONLY_CASES[case])
    G = head(_state(), torch.rand(2))
    assert G.shape == (2, D, D)
    assert torch.isfinite(G).all()


@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_band_without_global_owns_no_global_site_features(case):
    """Dropping the global term must drop its per-site encoder too, not merely
    stop calling it: a module built and never used is a dead init draw that
    shifts every later draw, breaking byte-identity for any head built
    after."""
    head = _head(**BAND_ONLY_CASES[case])
    assert not hasattr(head, "global_site_features")
    assert hasattr(head, "global_context_readout")  # the band's readout now


@torch.no_grad()
@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_band_without_global_stays_blind(case):
    """H_ij must not move when the tokens at the holes move. With no global
    term there is no hole subtraction, so index exclusion on the band is the
    only thing carrying blindness."""
    head = _head(**BAND_ONLY_CASES[case])
    x, t = _state(), torch.rand(2)
    H = head.compute_pair_context(x, t)
    for i, j in ((3, 11), (0, 15), (5, 6)):
        moved = x.clone()
        moved[:, i] *= -1
        moved[:, j] *= -1
        drift = (head.compute_pair_context(moved, t)[:, i, j] - H[:, i, j]).abs().max()
        assert drift < BLIND_ATOL, f"{case}: H_{i}{j} moved by {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_band_without_global_keeps_exact_antisymmetry(case):
    """G[j,i] = -G[i,j] and G_ii = 0 at exactly 0.0: identities of the mirror,
    which the swap CTMC's reverse rate leans on."""
    G = _head(**BAND_ONLY_CASES[case])(_state(), torch.rand(2))
    assert torch.equal(G, -G.transpose(1, 2))
    assert torch.equal(torch.diagonal(G, dim1=1, dim2=2), torch.zeros(2, D))


@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_band_without_global_trains_every_parameter(case):
    """`sum(G**2)`, not `sum(G)`: G is exactly antisymmetric, so its plain sum
    is identically zero as a function of the parameters and every gradient
    would vanish -- an objective that passes a dead module."""
    head = _head(**BAND_ONLY_CASES[case]).train()
    head(_state(), torch.rand(2)).pow(2).sum().backward()
    for name, parameter in head.named_parameters():
        # Backbone params are out of scope by the suite's convention, and
        # `attention_readout` is pinned dead for this head: the one-pass design
        # routes around it, which its own suite asserts.
        if name.startswith("backbone."):
            continue
        assert parameter.grad is not None, f"{case}: {name} got no grad"
        assert torch.isfinite(parameter.grad).all(), f"{case}: {name} non-finite"
        assert parameter.grad.abs().sum() > 0, f"{case}: {name} is dead"


def test_head_still_refuses_an_empty_context():
    """Widening the guard must not delete it: with bilinear, global and band
    all absent the pair score is a function of time alone."""
    with pytest.raises(ValueError, match="use_global"):
        _head(use_bilinear=False, use_global=False, interior_band=None)


def test_global_bond_features_still_need_a_band():
    """The bond totals share the band provider's feature modules, and it is
    that sharing which makes global-minus-band the exterior bond sum in one
    basis."""
    with pytest.raises(ValueError, match="band"):
        _head(use_global=False, interior_band=None, global_bond_features=True)


@torch.no_grad()
@pytest.mark.parametrize("case", list(BAND_ONLY_CASES))
def test_dropping_the_global_term_is_visible_in_the_scores(case):
    """The band-only head must not reproduce its global-carrying twin: if it
    did, either the global term never contributed or the flag is not wired."""
    knobs = BAND_ONLY_CASES[case]
    band_only = _head(**knobs)
    with_global = _head(**{**knobs, "use_global": True})
    x, t = _state(), torch.rand(2)
    assert not torch.allclose(band_only(x, t), with_global(x, t))


def test_config_flag_reaches_the_head():
    """`use_global=False` with a band must survive the config path."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    parent = CONFIGS["H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2"]
    torch.manual_seed(42)
    d64 = LeTFRateMatrix(
        d=64,
        vocab_size=2,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
        use_sdpa_readout=False,
    )
    # The exact-field channel is off so the assertion lands on the factorised
    # head itself: with it on, build_swap_head returns the wrapper.
    head = build_swap_head(
        replace(parent, use_global=False, exact_field_channel=False), d64
    )
    assert head.use_global is False
    assert head.interior_band == "prefix"
