import math

import pytest
import torch

from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _target(D=4, sigma=0.1, c=0.5):
    return FixedCompositionIsingTarget(D=D, sigma=sigma, target_composition=c)


def test_n_plus_target_derived_from_c():
    assert _target(D=4, c=0.5).n_plus_target == 8
    assert _target(D=4, c=0.375).n_plus_target == 6


def test_non_integral_composition_raises():
    with pytest.raises(ValueError, match="not integral"):
        # 0.4 * 16 = 6.4, not integral
        FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.4)


def test_sample_base_is_exactly_on_manifold():
    tgt = _target(D=4, c=0.5)
    x = tgt.sample_base(256, device="cpu")
    n_plus = ((x + 1) * 0.5).sum(dim=-1)
    assert torch.all(n_plus == 8)
    assert set(x.unique().tolist()) <= {-1.0, 1.0}


def test_sample_base_covers_many_configs():
    # Uniform-over-slice should not collapse to one arrangement.
    tgt = _target(D=4, c=0.5)
    x = tgt.sample_base(512, device="cpu")
    assert x.unique(dim=0).shape[0] > 50


def test_base_log_eta_is_constant_minus_log_slice_size():
    tgt = _target(D=4, c=0.5)
    x = tgt.sample_base(16, device="cpu")
    expected = -(math.lgamma(17) - math.lgamma(9) - math.lgamma(9))  # -log C(16,8)
    eta = tgt.base_log_eta(x)
    assert eta.shape == (16,)
    assert torch.allclose(eta, torch.full((16,), expected), atol=1e-6)


def test_assert_on_manifold():
    tgt = _target(D=4, c=0.5)
    tgt.assert_on_manifold(tgt.sample_base(8, device="cpu"))  # no raise
    off = tgt.sample_base(8, device="cpu")
    off[0, 0] = -off[0, 0]  # break composition on row 0
    with pytest.raises(AssertionError):
        tgt.assert_on_manifold(off)


def test_no_soft_penalty():
    tgt = _target(D=4, c=0.5)
    assert tgt.composition_penalty_strength == 0.0
    x = tgt.sample_base(4, device="cpu")
    assert torch.allclose(tgt.log_prob(x), tgt.base_log_prob(x))
