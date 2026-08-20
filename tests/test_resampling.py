"""Tests for the eval-time SMC resampling upgrade (samplers/resampling.py).

Written before the bodies — they encode "what correct looks like":

    1) systematic resampling: counts_i within 1 of B·w̄_i (the low-variance
       floor/ceil property), uniform weights -> every particle exactly once.
    2) resample_if_needed: no-fire path returns inputs unchanged and
       consumes NO RNG (the bit-exact parity guarantee); fire path banks
       logmeanexp(log_w) and resets weights to zero.
    3) sampler wiring: never-firing config replays the plain sampler
       bit-exactly; aggressive config stays on the composition manifold
       (swap) and returns the (state, log_w, stats) contract (both samplers).
    4) estimator unbiasedness on a toy problem: SMC product-form log Ẑ
       matches the enumerated log(Z_1/Z_0) on the 2x2 fixed-composition
       slice, as does plain IS.
"""

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    ess_from_log_weights,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.resampling import (
    ResamplingConfig,
    log_mean_exp,
    resample_if_needed,
    smc_log_z_estimate,
    systematic_resample_indices,
)
from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)


def _head_and_target(D=2, seed=42, sigma=0.3):
    torch.manual_seed(seed)
    tgt = FixedCompositionIsingTarget(D=D, sigma=sigma, target_composition=0.5)
    backbone = LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    return DoublyHollowSwapHead(backbone), tgt


class ConstantRateModel:
    """Constant flip rate per site — enough to exercise the flip-CTMC wiring."""

    def __init__(self, flip_rate: float):
        self.flip_rate = flip_rate

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.flip_rate, dtype=torch.float)


# ---------------------------------------------------------------- primitives


def test_systematic_counts_within_one_of_expected():
    # Floor/ceil property: an interval of length B·w̄_i on the scaled CDF
    # axis contains ⌊B·w̄_i⌋ or ⌈B·w̄_i⌉ unit-grid points, for ANY jitter u.
    log_weights = torch.tensor([0.0, -1.0, 2.0, 0.5, -3.0, 1.0, 0.0, -0.5])
    batch_size = log_weights.shape[0]
    expected_counts = batch_size * torch.softmax(log_weights, dim=0)
    for u in (0.0, 0.25, 0.5, 0.99):
        ancestors = systematic_resample_indices(log_weights, uniform=u)
        assert ancestors.shape == (batch_size,)
        assert ancestors.dtype == torch.long
        counts = torch.bincount(ancestors, minlength=batch_size).float()
        assert ((counts - expected_counts).abs() < 1.0).all()


def test_systematic_uniform_weights_keeps_every_particle_once():
    # Equal weights: every interval has length exactly 1 on the scaled CDF
    # axis, so every particle survives exactly once — resampling adds zero
    # noise in the already-balanced case.
    log_weights = torch.full((16,), -2.3)
    for u in (0.0, 0.37, 0.99):
        ancestors = systematic_resample_indices(log_weights, uniform=u)
        counts = torch.bincount(ancestors, minlength=16)
        assert (counts == 1).all()


def test_resample_if_needed_no_fire_leaves_inputs_and_rng_untouched():
    state = torch.randn(8, 4)
    log_weights = torch.zeros(8)                    # ESS = B -> never below τ·B
    rng_before = torch.get_rng_state()
    new_state, new_log_w, increment, fired = resample_if_needed(
        state, log_weights, ess_threshold_fraction=0.5
    )
    assert not fired
    assert torch.equal(new_state, state)
    assert torch.equal(new_log_w, log_weights)
    assert increment.item() == 0.0
    # No RNG consumed on the no-fire path: this is what makes a never-firing
    # config bit-identical to the plain sampler under the same seed.
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_resample_if_needed_fires_banks_log_z_and_resets_weights():
    # Degenerate weights: particle 0 carries everything -> ESS ≈ 1 fires the
    # τ=0.5 trigger, all ancestors collapse to row 0, weights restart at 0,
    # and the banked increment is exactly logmeanexp of the old weights.
    torch.manual_seed(0)
    state = torch.randn(4, 6)
    log_weights = torch.tensor([0.0, -50.0, -50.0, -50.0])
    new_state, new_log_w, increment, fired = resample_if_needed(
        state, log_weights, ess_threshold_fraction=0.5
    )
    assert fired
    assert torch.equal(new_log_w, torch.zeros(4))
    assert torch.isclose(increment, log_mean_exp(log_weights))
    assert torch.equal(new_state, state[0].expand(4, 6))


def test_smc_log_z_reduces_to_plain_is_with_zero_events():
    from discrete_flow_sampler.samplers.resampling import ResamplingStats

    log_weights = torch.randn(32)
    stats = ResamplingStats(log_z_increment=torch.zeros(()))
    assert torch.isclose(
        smc_log_z_estimate(stats, log_weights), log_mean_exp(log_weights)
    )


# ------------------------------------------------------------ sampler wiring


@torch.no_grad()
def test_swap_sampler_never_firing_config_is_bit_exact_parity():
    head, tgt = _head_and_target(D=4)
    ts = torch.linspace(0.0, 1.0, 20)

    torch.manual_seed(7)
    x0 = tgt.sample_base(16, device="cpu")
    x_plain, log_w_plain = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt
    )

    torch.manual_seed(7)
    x0 = tgt.sample_base(16, device="cpu")
    x_smc, log_w_smc, stats = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt,
        resampling=ResamplingConfig(ess_threshold_fraction=0.0),
    )
    assert stats.n_events == 0 and stats.event_steps == []
    assert stats.log_z_increment.item() == 0.0
    assert torch.equal(x_smc, x_plain)
    assert torch.equal(log_w_smc, log_w_plain)


@torch.no_grad()
def test_swap_sampler_aggressive_resampling_contract_and_manifold():
    head, tgt = _head_and_target(D=4)
    torch.manual_seed(11)
    x0 = tgt.sample_base(32, device="cpu")
    ts = torch.linspace(0.0, 1.0, 30)
    # τ=1.0 fires whenever the weights are not exactly uniform, i.e. at
    # (almost) every checkpoint after the first weight update.
    x_final, log_w, stats = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt,
        resampling=ResamplingConfig(ess_threshold_fraction=1.0),
    )
    assert x_final.shape == (32, 16) and log_w.shape == (32,)
    tgt.assert_on_manifold(x_final)      # resampling only duplicates slice rows
    assert stats.n_events > 0
    assert len(stats.event_steps) == stats.n_events
    assert torch.isfinite(stats.log_z_increment)
    assert torch.isfinite(smc_log_z_estimate(stats, log_w))


@torch.no_grad()
def test_flip_sampler_resampling_contract():
    # Sampler-agnosticism: the identical hook drives the unconstrained /
    # soft-target Euler loop.
    torch.manual_seed(3)
    target = IsingTarget(D=2, sigma=0.1)
    x0 = torch.randint(0, 2, (16, 4)).float() * 2 - 1
    ts = torch.linspace(0.0, 1.0, 25)
    x_final, log_w, stats = sample_ctmc(
        ConstantRateModel(flip_rate=0.5), x0, ts,
        return_log_weights=True, target=target,
        resampling=ResamplingConfig(ess_threshold_fraction=1.0),
    )
    assert x_final.shape == (16, 4) and log_w.shape == (16,)
    assert torch.isfinite(log_w).all()
    assert torch.isfinite(smc_log_z_estimate(stats, log_w))


def test_resampling_requires_log_weights():
    head, tgt = _head_and_target(D=2)
    x0 = tgt.sample_base(4, device="cpu")
    ts = torch.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        sample_swap_ctmc(head, x0, ts, resampling=ResamplingConfig())
    with pytest.raises(ValueError):
        sample_ctmc(ConstantRateModel(0.1), x0, ts, resampling=ResamplingConfig())


# --------------------------------------------------------- toy unbiasedness


@torch.no_grad()
def test_smc_log_z_matches_enumeration_on_2x2_slice():
    # E[Ẑ] = Z_1/Z_0 for ANY valid rates (here: random init, no training),
    # so both the plain-IS estimator and the SMC product form must land on
    # the enumerated log(Z_1^C/Z_0^C) of the d=4, N_A=2 slice (6 states),
    # up to Euler bias (O(dt), 48-step grid) and MC error (B=1024).
    # τ=1.0 forces resampling at every checkpoint — at this toy scale the
    # d=4 weight variance is so small that τ=0.5 never fires (measured
    # 2026-07-24), which would leave the banked-increment bookkeeping
    # untested. Calibration across seeds 456/789/1011 at these sizes:
    # |smc_err| ≤ 0.044 with 47 events, |plain_err| ≤ 0.005.
    head, tgt = _head_and_target(D=2, seed=42)
    slice_states = enumerate_states(4).float()
    n_plus = ((slice_states + 1) * 0.5).sum(dim=-1)
    slice_states = slice_states[n_plus == tgt.n_plus_target]
    n_slice = slice_states.shape[0]
    exact_log_z_ratio = (
        torch.logsumexp(
            tgt.log_p_tilde_t(slice_states, torch.ones(n_slice)), dim=0
        )
        - torch.logsumexp(
            tgt.log_p_tilde_t(slice_states, torch.zeros(n_slice)), dim=0
        )
    )

    ts = torch.linspace(0.0, 1.0, 48)
    n_particles = 1024

    torch.manual_seed(123)
    x0 = tgt.sample_base(n_particles, device="cpu")
    _, log_w_plain = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt
    )
    plain_estimate = log_mean_exp(log_w_plain)

    torch.manual_seed(456)
    x0 = tgt.sample_base(n_particles, device="cpu")
    _, log_w_final, stats = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt,
        resampling=ResamplingConfig(ess_threshold_fraction=1.0),
    )
    smc_estimate = smc_log_z_estimate(stats, log_w_final)

    assert stats.n_events > 0          # the product form is actually exercised
    assert torch.isclose(plain_estimate, exact_log_z_ratio, atol=0.1)
    assert torch.isclose(smc_estimate, exact_log_z_ratio, atol=0.1)


@torch.no_grad()
def test_resampling_lifts_final_segment_ess():
    # The point of the exercise: killing the weight-variance backlog leaves
    # the final segment's ESS fraction at or above the plain-IS one.
    # Sizes calibrated 2026-07-24: B=32/30 steps gives plain ESS/B ≈ 0.15 vs
    # SMC ≈ 0.51 (one event) — a wide margin at ~35 s; B=256/40 steps showed
    # the same picture (0.14 vs 0.66) at 200 s, not worth the wall-clock.
    head, tgt = _head_and_target(D=4, seed=42, sigma=0.2)
    ts = torch.linspace(0.0, 1.0, 30)
    n_particles = 32

    torch.manual_seed(21)
    x0 = tgt.sample_base(n_particles, device="cpu")
    _, log_w_plain = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt
    )

    torch.manual_seed(22)
    x0 = tgt.sample_base(n_particles, device="cpu")
    _, log_w_final, stats = sample_swap_ctmc(
        head, x0, ts, return_log_weights=True, target=tgt,
        resampling=ResamplingConfig(ess_threshold_fraction=0.5),
    )
    ess_plain = ess_from_log_weights(log_w_plain)
    ess_smc = ess_from_log_weights(log_w_final)
    assert ess_smc >= ess_plain


def test_systematic_ancestors_are_sorted_so_a_prefix_is_not_a_uniform_subset():
    """Ancestor indices come back in CDF order, so `ancestors[:k]` is a
    contiguous low-CDF block, NOT an exchangeable subset.

    This is the property that makes `swap_training`'s c_t-batch prefix
    slice unsafe after a resample fires: with `c_t_batch > outer_batch` the
    rollout draws n_rollout rows, resamples all of them, and the replay
    buffer then keeps the FIRST outer_batch. When the rows are iid base
    draws a prefix is a uniform subset and that is free; once systematic
    resampling has ordered the rows by ancestor, the prefix over-represents
    the low-index end of the CDF and clusters duplicate lineages together.
    The trainer must therefore shuffle before slicing. This test pins the
    sortedness so the requirement cannot silently lapse.
    """
    torch.manual_seed(0)
    log_weights = torch.randn(16) * 2.0
    ancestors = systematic_resample_indices(log_weights, uniform=0.5)

    assert bool((ancestors[1:] >= ancestors[:-1]).all()), "ancestors not sorted"
    # The concrete harm: the prefix carries fewer distinct lineages than the
    # same-sized uniform subset would, because clones sit adjacent.
    prefix_lineages = len(set(ancestors[:8].tolist()))
    assert prefix_lineages < len(set(ancestors.tolist()))
