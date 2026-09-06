"""What correct looks like for the slice-mixture target on a cluster expansion.

Same three algebraic facts as the Ising mixture (test_mixture_composition_target.py):
swaps conserve n_plus row-wise, so `base_log_eta` read off x makes the per-slice
geometric path exact, and `swap_log_ratio` (Eq. (3) on the expansion) is unchanged
because both states of a swap share a slice. Pinned on the 16-site Cu-Au cell.
"""

import math

import pytest
import torch

from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    FixedCompositionClusterExpansionTarget,
    MixtureCompositionClusterExpansionTarget,
)

SPEC = "data/ce/cuau_fcc_2x2x4.json"
COMPOSITIONS = (0.5, 0.4375, 0.375, 0.3125, 0.25)  # n_Au = 8, 7, 6, 5, 4 of 16
BETA = 1.0 / (8.617333262e-5 * 500.0)


def _mixture(compositions=COMPOSITIONS):
    return MixtureCompositionClusterExpansionTarget(
        BinaryExpansionSpec.from_json(SPEC), beta=BETA, compositions=compositions
    )


def _n_plus(x):
    return ((x + 1) * 0.5).sum(dim=-1)


def test_sample_base_lands_only_on_registered_slices_and_covers_all():
    x = _mixture().sample_base(1000, device="cpu")
    allowed = {round(c * 16) for c in COMPOSITIONS}
    counts = _n_plus(x)
    assert set(counts.tolist()) == allowed
    for n_plus in allowed:
        assert (counts == n_plus).sum() > 120


def test_base_log_eta_is_the_per_row_slice_constant():
    tgt = _mixture()
    x = tgt.sample_base(200, device="cpu")
    expected = torch.tensor(
        [
            -(math.lgamma(17) - math.lgamma(n + 1) - math.lgamma(17 - n))
            for n in _n_plus(x).long().tolist()
        ]
    )
    assert torch.allclose(tgt.base_log_eta(x), expected.to(x.dtype), atol=1e-5)


def test_swap_log_ratio_matches_the_single_slice_expansion_target_per_row():
    tgt = _mixture()
    x = tgt.sample_base(64, device="cpu")
    pairs = torch.tensor([[0, 5], [3, 12], [7, 8]])
    t = torch.full((64,), 0.7)
    got = tgt.swap_log_ratio(x, t, pairs)
    for c in COMPOSITIONS:
        single = FixedCompositionClusterExpansionTarget(
            BinaryExpansionSpec.from_json(SPEC), beta=BETA, target_composition=c
        )
        rows = _n_plus(x) == single.n_plus_target
        assert torch.allclose(
            got[rows], single.swap_log_ratio(x[rows], t[rows], pairs), atol=1e-6
        )


def test_anchor_slice_is_the_first_composition_and_energy_is_the_expansions():
    tgt = _mixture()
    assert tgt.n_plus_target == 8 and tgt.compositions == COMPOSITIONS
    x = tgt.sample_base(8, device="cpu")
    same_slice = _n_plus(x) == 8
    assert same_slice.sum() >= 2
    x = x[same_slice][:2]
    delta_log_prob = tgt.log_prob(x[1:]) - tgt.log_prob(x[:1])
    delta_energy = tgt.spec.energy(x[1:].double()) - tgt.spec.energy(x[:1].double())
    assert torch.allclose(delta_log_prob.double(), -BETA * delta_energy, atol=1e-4)


def test_assert_on_manifold_accepts_members_rejects_others():
    tgt = _mixture()
    tgt.assert_on_manifold(tgt.sample_base(50, device="cpu"))
    off = torch.full((1, 16), -1.0)
    off[0, :3] = 1.0  # n_Au = 3, not on the grid
    with pytest.raises(AssertionError):
        tgt.assert_on_manifold(off)


def test_non_integral_grid_raises():
    with pytest.raises(ValueError):
        _mixture((0.5, 0.3))
