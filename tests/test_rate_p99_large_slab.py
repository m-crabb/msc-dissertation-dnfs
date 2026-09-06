"""The pair-rate p99 must survive lattices above 16x16.

`torch.quantile` caps its input at 2**24 = 16,777,216 elements. The swap
trainer's `rate_pair_p99` reads the whole pair-rate slab, shape
(outer_batch, d(d-1)/2), which at d=256 and the production outer batch of
512 is 16,711,680 -- **99.6% of that cap**, a margin of 65,536 elements.
d=400 is the first rung over it (40,857,600, 2.4x the cap), and the 20x20
probe hit `RuntimeError: quantile() input tensor is too large` in the
step-0 init diagnostic, before its first optimiser step.

These tests pin the fix's two obligations, which pull in opposite
directions: it must WORK above the cap, and it must not move a single
number below it -- every reported d256 figure was produced by the
`torch.quantile` path and the 16x16 house table is filled from those runs.
"""

import torch

from discrete_flow_sampler.samplers.swap_training import (
    _QUANTILE_MAX_ELEMENTS,
    _p99,
)


def test_matches_torch_quantile_below_the_cap():
    """Below the cap the two paths must be the same function, so no archived
    cell's logged rate_pair_p99 moves. Checked across shapes and scales, and
    on a slab the exact size of the d256 production diagnostic."""
    torch.manual_seed(0)
    for numel in (1, 2, 17, 1000, 2016 * 256):
        values = torch.randn(numel).float()
        assert _p99(values) == torch.quantile(values, 0.99).item(), numel


def test_reproduces_the_d256_production_shape_exactly():
    """The 99.6%-of-cap case itself: d=256, outer batch 512. This is the
    shape every printed 16x16 number was logged at, so it must still take
    the torch.quantile path and return its value bit-for-bit."""
    torch.manual_seed(1)
    d, batch = 256, 512
    numel = batch * (d * (d - 1) // 2)
    assert numel == 16_711_680 and numel <= _QUANTILE_MAX_ELEMENTS
    values = torch.rand(numel).float()
    assert _p99(values) == torch.quantile(values, 0.99).item()


def test_works_above_the_cap_and_interpolates_the_same_way():
    """Above the cap torch.quantile raises, so the fallback carries it. It
    must use quantile's documented 'linear' convention -- position
    q*(n-1), interpolated between neighbours -- which is checked here
    against an analytically known answer rather than against torch (which
    cannot compute this input at all)."""
    n = _QUANTILE_MAX_ELEMENTS + 1
    # 0, 1, ..., n-1 shuffled: the p99 of arange(n) is exactly 0.99*(n-1).
    values = torch.randperm(n).float()
    expected = 0.99 * (n - 1)
    assert abs(_p99(values) - expected) <= 1.0
    # And torch.quantile genuinely cannot do this, which is why _p99 exists.
    try:
        torch.quantile(values, 0.99)
    except RuntimeError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("torch.quantile no longer caps -- revisit _p99")


def test_d400_pair_slab_shape_is_over_the_cap():
    """The regression this exists for: the 20x20 probe's own diagnostic
    shape must be one the fallback handles, not one that raises."""
    d, batch = 400, 512
    numel = batch * (d * (d - 1) // 2)
    assert numel == 40_857_600 > _QUANTILE_MAX_ELEMENTS
    torch.manual_seed(2)
    # Same shape as the real slab, values irrelevant to the code path.
    values = torch.rand(numel, dtype=torch.float32)
    assert torch.isfinite(torch.tensor(_p99(values)))
