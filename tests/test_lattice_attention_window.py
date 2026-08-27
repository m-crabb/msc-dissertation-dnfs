"""Arm A: the masked-attention head over the WHOLE LATTICE, not the interval.

WHAT CHANGES AND WHY IT IS NEARLY FREE. The head already computes every
band-feature family for every pair and then masks before the softmax, so the
whole-lattice variant moves ONE predicate and adds no FLOPs, no parameter,
no buffer and no RNG draw. A term at slot k with support offsets O is
visible to the pair (i, j) when

    interval:  k + min(O) > i  AND  k + max(O) < j      (strictly inside)
    lattice:   for every o in O,  k + o != i  and  k + o != j
               (equivalently: its support does not touch either hole)

THE LEGALITY RULE (corrected 2026-08-27, see the band-interior memory). What
blindness requires is NOT that features be per-site or depth-0; it is that
exclusion remove EVERY term touching a hole, decided from the INDICES alone
so it is value-independent. Both windows satisfy that; they differ only in
how much of the lattice survives the mask.

WHAT THE ARM IS FOR. It is the single-variable test of the WINDOW at fixed
feature family and fixed weights, against the masked-attention head already
in print. It is one of two cells left unbuilt in the head construction's
2 x 2 x 2 (feature family x window x weights).

IT IS NOT SIMPLY THE MORE GENERAL HEAD, and this is the reason to measure
rather than assume: normalising the softmax over ~d terms instead of ~|j-i|
dilutes whatever mass the interval deserves, and a learned soft mask
approximates the hard interval indicator without containing it. So the
lattice window can lose, and the experiment is what settles it.
"""
import pytest
import torch

from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.models.letf import LeTFRateMatrix

ATOL = 1e-5


def _head(d=16, window="interval", seed=42, pair_offsets=(1,), **kw):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(d=d, vocab_size=2, hidden_dim=8, n_layers=1,
                              n_heads=2)
    head = MaskedAttentionSwapHead(
        backbone, pair_offsets=pair_offsets, band_feature_dim=6,
        position_dim=6, attention_dim=8, attention_window=window, **kw
    )
    head.eval()
    return head


def _state(d, batch=1, seed=0):
    torch.manual_seed(seed)
    half = torch.cat([torch.ones(d // 2), -torch.ones(d - d // 2)])
    return torch.stack([half[torch.randperm(d)] for _ in range(batch)])


def test_interval_window_is_the_default_and_is_byte_identical():
    """Every archived cell must be untouched: the default window reproduces
    the head exactly, so the flag adds nothing until it is asked for."""
    torch.manual_seed(1)
    x, t = _state(16, batch=2), torch.rand(2)
    assert torch.equal(_head(window="interval")(x, t), _head()(x, t))


@pytest.mark.parametrize("window", ["interval", "lattice"])
def test_pair_context_is_blind_to_both_holes(window):
    """THE guarantee, under both windows: changing x_i or x_j must not move
    the context of the pair (i, j). Under the lattice window this is a real
    test rather than a formality -- almost every term is visible, so an
    off-by-one in the touch predicate would leak a hole immediately."""
    d = 16
    head = _head(d=d, window=window, pair_offsets=(1, 4))
    x = _state(d, batch=1)
    t = torch.rand(1)
    base = head.compute_pair_context(x, t)
    for i, j in ((0, 1), (0, 15), (3, 9), (7, 8), (2, 6)):
        for hole in (i, j):
            flipped = x.clone()
            flipped[0, hole] = -flipped[0, hole]
            moved = (head.compute_pair_context(flipped, t)[:, i, j]
                     - base[:, i, j]).abs().max().item()
            assert moved < ATOL, (window, i, j, hole, moved)


def test_lattice_window_sees_strictly_more_than_the_interval():
    """The arm has to actually do something: for a near pair the interval is
    empty and the lattice window still has ~d terms, so the two contexts must
    differ. Guards against a predicate that silently reduces to the old one."""
    d = 16
    x, t = _state(d, batch=1), torch.rand(1)
    interval = _head(d=d, window="interval", pair_offsets=(1, 4))
    lattice = _head(d=d, window="lattice", pair_offsets=(1, 4))
    lattice.load_state_dict(interval.state_dict())
    a = interval.compute_pair_context(x, t)
    b = lattice.compute_pair_context(x, t)
    # Adjacent pair: the open interval (i, i+1) is EMPTY.
    assert (b[:, 3, 4] - a[:, 3, 4]).abs().max().item() > 1e-3


@pytest.mark.parametrize("window", ["interval", "lattice"])
def test_swap_antisymmetry_survives_the_window(window):
    """G_ji = -G_ij exactly, and G(Swap2(x,i,j))_ij = -G(x)_ij to tolerance:
    the window may not cost the property the whole construction exists for."""
    d = 16
    head = _head(d=d, window=window, pair_offsets=(1, 4))
    x, t = _state(d, batch=1), torch.rand(1)
    G = head(x, t)
    assert (G + G.transpose(1, 2)).abs().max().item() == 0.0
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)
             if x[0, i] != x[0, j]]
    worst = max((G[0, i, j] + head(swap2(x, i, j), t)[0, i, j]).abs().item()
                for i, j in pairs)
    assert worst < ATOL, (window, worst)


def test_lattice_window_costs_no_parameters():
    """'Nearly free' is a claim about the parameter count as well as the
    FLOPs: the flag must not add a tensor."""
    a = _head(window="interval", pair_offsets=(1, 4))
    b = _head(window="lattice", pair_offsets=(1, 4))
    assert {n: p.shape for n, p in a.named_parameters()} == \
           {n: p.shape for n, p in b.named_parameters()}


def test_lattice_window_is_blind_with_the_stencil_family_too():
    """The stencil is a bounded-support family like the others (support
    {k-side, k-1, k, k+1, k+side}), so the touch predicate must cover all
    five offsets. Off by default, but legal, and a wrong predicate here
    would leak a hole through the widest family."""
    d = 16
    head = _head(d=d, window="lattice", use_stencil=True, lattice_side=4)
    x, t = _state(d, batch=1), torch.rand(1)
    base = head.compute_pair_context(x, t)
    for i, j in ((2, 9), (0, 15)):
        for hole in (i, j):
            flipped = x.clone()
            flipped[0, hole] = -flipped[0, hole]
            moved = (head.compute_pair_context(flipped, t)[:, i, j]
                     - base[:, i, j]).abs().max().item()
            assert moved < ATOL, (i, j, hole, moved)


def test_mal_gate_cells_are_their_ma_twins_plus_the_window():
    """The twin relationship arm A is read through: each `mal` gate cell
    must differ from its `ma` sibling in `attention_window` and NOTHING
    else, so a window effect is chargeable to the window."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    from experiments.constrained_hard_03.configs import _MAL_TWINS

    assert len(_MAL_TWINS) == 4, "gate + 8x8 rung, both couplings"
    for parent, name in _MAL_TWINS.items():
        ma, mal = CONFIGS[parent], CONFIGS[name]
        assert ma.attention_window == "interval", name
        assert mal.attention_window == "lattice", name
        assert replace(mal, name=ma.name, attention_window="interval") == ma, name


def test_mal_gate_cells_build_their_heads():
    """Construction check at the gate lattice, so a knob typo fails here
    rather than after a launch."""
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget
    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    from experiments.constrained_hard_03.configs import _MAL_TWINS

    for name in _MAL_TWINS.values():
        cfg = CONFIGS[name]
        backbone = LeTFRateMatrix(d=cfg.ising.D ** 2, vocab_size=2,
                                  hidden_dim=16, n_layers=2, n_heads=2)
        target = FixedCompositionIsingTarget(
            D=cfg.ising.D, sigma=cfg.ising.sigma, target_composition=0.5
        )
        head = build_swap_head(cfg, backbone, target=target)
        assert head.attention_window == "lattice", name
