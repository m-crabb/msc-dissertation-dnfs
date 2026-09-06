"""What correct looks like for the slice-mixture target, written before it.

Composition amortisation trains ONE head on a mixture of
fixed-composition slices. The whole design
rests on three algebraic facts this file pins:

  * swaps conserve n_plus row-wise, so a trajectory never leaves the slice
    it started on and every element's weights are exact against ITS OWN
    slice conditional;
  * `base_log_eta` must therefore be computed FROM x (-log C(d, n_plus(x)))
    rather than stored as a single constant -- that is the one change that
    makes the per-slice geometric path exact under the mixture;
  * `swap_log_ratio`'s closed form survives unchanged, because the swapped
    and unswapped states share a slice and the base constant cancels
    pairwise exactly as it does for the single-slice target.
"""
import math

import pytest
import torch

from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    MixtureCompositionIsingTarget,
)

# The d64 mixture grid is n+/64 = 32/30/28/24/20; the D=4 twins here use
# the same kind of spread at d=16.
COMPOSITIONS = (0.5, 0.375, 0.25)


def _mixture(D=4, sigma=0.1, compositions=COMPOSITIONS):
    return MixtureCompositionIsingTarget(
        D=D, sigma=sigma, compositions=compositions)


def _n_plus(x):
    return ((x + 1) * 0.5).sum(dim=-1)


def test_sample_base_lands_only_on_registered_slices_and_covers_all():
    tgt = _mixture()
    x = tgt.sample_base(600, device="cpu")
    allowed = {round(c * 16) for c in COMPOSITIONS}
    counts = _n_plus(x)
    assert set(counts.tolist()) == allowed
    # Uniform slice choice: each slice gets roughly a third of 600 draws.
    for n_plus in allowed:
        assert (counts == n_plus).sum() > 120


def test_base_log_eta_is_the_per_row_slice_constant():
    """-log C(d, n_plus(x)) per row: on each slice it must agree exactly
    with the single-slice target's constant, because the mixture's
    per-slice conditional IS that target's."""
    tgt = _mixture()
    x = tgt.sample_base(300, device="cpu")
    eta = tgt.base_log_eta(x)
    for c in COMPOSITIONS:
        single = FixedCompositionIsingTarget(
            D=4, sigma=0.1, target_composition=c)
        rows = _n_plus(x) == single.n_plus_target
        assert rows.any()
        expected = single.base_log_eta(x[rows])
        assert torch.allclose(eta[rows], expected, atol=1e-6)


def test_swap_log_ratio_matches_the_single_slice_target_per_row():
    """The closed form must be byte-level the same computation: swaps stay
    on-slice, so the mixture changes nothing about any ratio."""
    tgt = _mixture()
    x = tgt.sample_base(64, device="cpu")
    pairs = torch.tensor([[0, 5], [3, 12], [7, 8]])
    t = torch.full((64,), 0.7)
    got = tgt.swap_log_ratio(x, t, pairs)
    for c in COMPOSITIONS:
        single = FixedCompositionIsingTarget(
            D=4, sigma=0.1, target_composition=c)
        rows = _n_plus(x) == single.n_plus_target
        expected = single.swap_log_ratio(x[rows], t[rows], pairs)
        assert torch.allclose(got[rows], expected, atol=1e-6)


def test_geometric_path_is_per_slice_exact_at_the_endpoints():
    """log p~_t must interpolate each row's OWN slice base and the shared
    energy: t=0 gives -log C(d, n_plus(x)) per row, t=1 gives log_prob."""
    tgt = _mixture()
    x = tgt.sample_base(120, device="cpu")
    t0 = tgt.log_p_tilde_t(x, torch.zeros(120))
    assert torch.allclose(t0, tgt.base_log_eta(x), atol=1e-6)
    t1 = tgt.log_p_tilde_t(x, torch.ones(120))
    assert torch.allclose(t1, tgt.log_prob(x), atol=1e-6)


def test_assert_on_manifold_accepts_members_rejects_others():
    tgt = _mixture()
    x = tgt.sample_base(64, device="cpu")
    tgt.assert_on_manifold(x)
    off = x.clone()
    off[0] = torch.full((16,), 1.0)  # n_plus=16, on no registered slice
    with pytest.raises(AssertionError, match="off-manifold"):
        tgt.assert_on_manifold(off)


def test_non_integral_or_empty_grid_raises():
    with pytest.raises(ValueError, match="not integral"):
        _mixture(compositions=(0.5, 0.4))  # 0.4*16 = 6.4
    with pytest.raises(ValueError, match="at least one"):
        _mixture(compositions=())


def test_single_composition_mixture_degenerates_to_the_fixed_target():
    """A one-slice mixture must be statistically the fixed target: same
    base support, same constants -- the degenerate case that keeps the two
    classes honest against each other."""
    mix = _mixture(compositions=(0.5,))
    single = FixedCompositionIsingTarget(D=4, sigma=0.1,
                                         target_composition=0.5)
    x = mix.sample_base(128, device="cpu")
    assert torch.all(_n_plus(x) == 8)
    assert torch.allclose(mix.base_log_eta(x), single.base_log_eta(x))
    # base_log_eta returns x.dtype (float32), so exactness is fp32-level.
    assert math.isclose(
        mix.base_log_eta(x)[0].item(), -single._log_slice_size,
        rel_tol=1e-6)
