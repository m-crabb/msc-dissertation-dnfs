"""Closed-form swap log-ratio: FixedComposition override + generic fallback.

Encodes "what correct looks like" for `swap_log_ratio(x, t, pairs) -> (B, P)`
before the closed form is implemented (TDD). The oracle is the already-tested
neighbour helper `_log_p_tilde_at_swap_neighbours`, so a shared bug cannot pass
both sides. Closed form:

    log p̃_t(Swap2(x, i, j)) − log p̃_t(x)
        = t·σ·[ 2(x_j − x_i)(h_i − h_j) − 2(x_j − x_i)²·A_ij ],   h = x·A

exact on the fixed-N slice because base_log_eta is constant there (the (1−t)
term cancels) and bias·Σx is swap-invariant.
"""

import pytest
import torch

from discrete_flow_sampler.samplers._swap_neighbours import (
    _log_p_tilde_at_swap_neighbours,
    upper_tri_pairs,
)
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)


def _oracle_log_ratio(x, t, target):
    """log p̃_t(Swap2) − log p̃_t(x) over all i<j pairs, via the tested helper."""
    neighbours = _log_p_tilde_at_swap_neighbours(x, t, target)
    return neighbours - target.log_p_tilde_t(x, t)[:, None]


# (D, sigma, c, batch) — four numerically-verified cells.
CLOSED_FORM_CELLS = [
    (4, 0.1, 0.5, 32),
    (4, 0.3, 0.375, 32),  # Z2-broken composition
    (8, 0.223, 0.5, 16),  # near-critical, probe scale
    (16, 0.1, 0.5, 4),  # paper scale
]


@pytest.mark.parametrize("D, sigma, c, batch", CLOSED_FORM_CELLS)
def test_closed_form_matches_neighbour_oracle(D, sigma, c, batch):
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=D, sigma=sigma, target_composition=c)
    x = target.sample_base(batch, device="cpu")
    t = torch.rand(batch)
    pairs = upper_tri_pairs(target.d, "cpu")

    got = target.swap_log_ratio(x, t, pairs)

    assert got.shape == (batch, pairs.shape[0])
    assert torch.allclose(got, _oracle_log_ratio(x, t, target), atol=1e-4)


def test_generic_fallback_matches_neighbour_oracle():
    """Base-class build-and-evaluate fallback on a non-fixed-composition target.

    Non-uniform base + bias exercise the general log_p_tilde_t through the
    fallback; a swap still preserves n_plus and Σx, so both sides agree.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=4, sigma=0.3, bias=0.2, base_composition=0.4)
    x = target.sample_base(16, device="cpu")
    t = torch.rand(16)
    pairs = upper_tri_pairs(target.d, "cpu")

    got = target.swap_log_ratio(x, t, pairs)

    assert torch.allclose(got, _oracle_log_ratio(x, t, target), atol=1e-6)


def test_same_spin_pairs_give_exactly_zero():
    """diff = x_j − x_i = 0 ⇒ the closed form is identically 0 (trivial swap)."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)
    x = target.sample_base(8, device="cpu")
    t = torch.rand(8)
    pairs = upper_tri_pairs(target.d, "cpu")

    got = target.swap_log_ratio(x, t, pairs)
    same_spin = x[:, pairs[:, 0]] == x[:, pairs[:, 1]]
    assert torch.all(got[same_spin] == 0.0)


def test_pair_order_invariant():
    """Swap2(x, i, j) = Swap2(x, j, i) ⇒ the log-ratio ignores pair order."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.223, target_composition=0.5)
    x = target.sample_base(16, device="cpu")
    t = torch.rand(16)
    pairs = upper_tri_pairs(target.d, "cpu")

    assert torch.allclose(
        target.swap_log_ratio(x, t, pairs),
        target.swap_log_ratio(x, t, pairs.flip(1)),
        atol=1e-6,
    )


def test_pair_columns_cache_serves_hits_and_recomputes_on_new_pairs():
    """B4 (2026-08-24): the (site_i, site_j, A_ij) gather is cached by
    pairs-tensor IDENTITY — the same object must serve bit-equal results,
    and a different pairs tensor must recompute, never serve stale columns."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.223, target_composition=0.5)
    x = target.sample_base(8, device="cpu")
    t = torch.rand(8)
    pairs = upper_tri_pairs(target.d, "cpu")

    first = target.swap_log_ratio(x, t, pairs)
    # Same object again — the cache-hit path.
    assert torch.equal(target.swap_log_ratio(x, t, pairs), first)

    # A different tensor (reversed prefix of the pair list) — the miss
    # path; column r of the result must be column 6-r of the full grid.
    subset = pairs[:7].flip(0).clone()
    assert torch.allclose(
        target.swap_log_ratio(x, t, subset),
        first[:, :7].flip(1),
        atol=1e-6,
    )
