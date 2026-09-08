"""Toy-input tests for the 4x4 gate analysis helpers.

Only the pure-tensor analysis functions are exercised here: histogram/TV, the
on-slice free-energy reference, and the within-level uniformity metric. The
full gate (`run_gate`/`main`) instantiates the leTF backbone and runs the swap
CTMC, so it is not tested locally -- the controller runs the real gate.
"""

import itertools

import torch
from experiments.constrained_hard_03.gate_4x4 import (
    energy_marginal_tv,
    on_slice_free_energy_reference,
    slice_energy_hist,
    within_level_uniformity,
)

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

# --- energy-marginal TV basics ---


def test_energy_marginal_tv_is_zero_for_identical():
    states = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    w = torch.tensor([0.5, 0.5])
    bins = torch.linspace(-3, 3, 7)
    a = slice_energy_hist(states, w, adj, bins)
    b = slice_energy_hist(states, w, adj, bins)
    assert abs(energy_marginal_tv(a, b)) < 1e-9


def test_energy_marginal_tv_detects_shift():
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    bins = torch.linspace(-3, 3, 7)
    # xAx=+2
    up = slice_energy_hist(torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]), adj, bins)
    # xAx=-2
    anti = slice_energy_hist(
        torch.tensor([[1.0, -1.0]]), torch.tensor([1.0]), adj, bins
    )
    assert energy_marginal_tv(up, anti) > 0.9


# --- on-slice free-energy reference ---


def test_on_slice_free_energy_reference_uses_2sigma_d_normalisation():
    # The reference must divide -logsumexp(log p over the slice) by (2*sigma*d),
    # the exact per-site convention of `free_energy_lb_estimate`, so the DNFS
    # ELBO estimate and this reference are directly differenceable.
    tgt = FixedCompositionIsingTarget(D=2, sigma=0.3, target_composition=0.5)
    states = enumerate_states(4).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == tgt.n_plus_target]
    ref = on_slice_free_energy_reference(tgt, slice_states)
    expected = -torch.logsumexp(tgt.log_prob(slice_states), dim=0) / (2 * 0.3 * 4)
    assert torch.isclose(ref, expected, atol=1e-6)


# --- within-level uniformity ---


def _one_level_slice():
    # Four distinct d=4 configs, all declared to sit at one energy level.
    slice_states = torch.tensor(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    slice_energies = torch.zeros(4)
    return slice_states, slice_energies


def test_within_level_uniform_coverage_has_nonpositive_excess():
    # Exactly-uniform coverage: TV_k = 0, so excess = -TV_ref <= 0 (the matched
    # perfect-sampler baseline is subtracted off).
    slice_states, slice_energies = _one_level_slice()
    samples = slice_states.repeat_interleave(100, dim=0)  # 100 of each -> uniform
    weights = torch.ones(samples.shape[0])
    sample_energies = torch.zeros(samples.shape[0])
    levels = within_level_uniformity(
        samples,
        weights,
        sample_energies,
        slice_states,
        slice_energies,
        min_count=4,
        seed=0,
    )
    assert len(levels) == 1
    level = levels[0]
    assert level["n_k"] == 400 and level["g_k"] == 4
    assert abs(level["n_eff_k"] - 400) < 0.1  # equal weights: n_eff_k == n_k
    assert level["tv_k"] < 1e-5  # exactly uniform up to float32 accumulation
    assert level["excess"] <= 1e-5


def test_within_level_concentration_shows_large_excess():
    # All mass on one of four states: TV_k = 0.5*(0.75 + 3*0.25) = 0.75, and the
    # matched baseline is small, so the excess is clearly positive.
    slice_states, slice_energies = _one_level_slice()
    samples = slice_states[0:1].repeat(400, 1)
    weights = torch.ones(400)
    sample_energies = torch.zeros(400)
    levels = within_level_uniformity(
        samples,
        weights,
        sample_energies,
        slice_states,
        slice_energies,
        min_count=4,
        seed=0,
    )
    level = levels[0]
    assert abs(level["tv_k"] - 0.75) < 1e-6
    assert level["excess"] > 0.3


def test_within_level_skewed_weights_uniform_states_excess_near_zero():
    # Weight-matching pin: the states cover the level
    # exactly uniformly, but the IS weights are heavily skewed (lognormal,
    # n_eff_k ~ 37 << n_k = 3200). Weight dispersion alone floors the raw TV_k
    # at ~0.30; the weight-matched null (observed weights on uniform draws)
    # models exactly that, so the excess must sit ~0. Under the old unweighted
    # baseline (tv_ref ~ 0.04) the excess would be ~0.26 >> the 0.05 gate
    # threshold -- a spurious failure at any low-within-level-ESS rung.
    rows = list(itertools.product([-1.0, 1.0], repeat=6))[:32]
    slice_states = torch.tensor(rows)  # g_k = 32 states
    slice_energies = torch.zeros(32)
    samples = slice_states.repeat(100, 1)  # exactly uniform coverage
    torch.manual_seed(0)
    weights = torch.exp(2.0 * torch.randn(3200))  # heavy-tailed IS weights
    levels = within_level_uniformity(
        samples,
        weights,
        torch.zeros(3200),
        slice_states,
        slice_energies,
        min_count=4,
        seed=0,
    )
    level = levels[0]
    assert level["n_eff_k"] < 100  # skew is visible
    assert level["tv_k"] > 0.1  # raw TV floored by skew
    assert abs(level["excess"]) < 0.05  # null absorbs the floor


def test_within_level_skips_sparse_levels():
    # Levels with fewer than min_count raw samples are dropped (the TVD-floor
    # trap: a raw TV_k is uninterpretable when n_k << g_k).
    slice_states, slice_energies = _one_level_slice()
    samples = slice_states[0:1].repeat(3, 1)
    levels = within_level_uniformity(
        samples,
        torch.ones(3),
        torch.zeros(3),
        slice_states,
        slice_energies,
        min_count=100,
        seed=0,
    )
    assert levels == []
