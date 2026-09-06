"""Tests for the (sigma_c, 8x8) headline-cell probe analysis, written before
the analysis script bodies.

What is under test here is exactly the NEW research-bearing computation the
probe analysis adds over the already-tested demo_4x4 machinery:

1. batch_means_tau_int — the competitor burn-in rule is "max(1e4 sweeps,
   20*tau_int(energy)) with tau_int from batch means at block length
   >= 10*tau_int". Batch means estimates tau_int from the variance
   inflation of block averages,
   Var(block_mean) ~= tau_int * Var(x) / L for block length L >> tau_int,
   so tau_hat = L * Var(block means) / Var(x). The estimator must certify
   its own block length (L >= 10 * tau_hat), the self-consistency
   condition.
2. kawasaki_burn_in_sweeps — the max() rule itself.
3. ratio_with_ci — the delta-method 95% CI on the per-compute N_eff ratio
   that the GO margin rule reads: CI excluding 1 AND point >= 1.5.
4. frozen_verdict — the three-way GO/PARTIAL/NO-GO mapping with the
   PARTIAL narratives; the outcome is decided by it mechanically, so every
   branch is pinned.
"""

import numpy as np
import pytest

from experiments.constrained_hard_03.probe_analysis_8x8 import (
    batch_means_tau_int,
    frozen_verdict,
    kawasaki_burn_in_sweeps,
    ratio_with_ci,
    ratio_with_f_ci,
    total_variation,
    tv_noise_floor,
)


# ---------------------------------------------------------------------------
# batch_means_tau_int
# ---------------------------------------------------------------------------


def _ar1(rho, n, seed):
    """AR(1) trace with known integrated autocorrelation time.

    For x_t = rho * x_{t-1} + eps_t, the autocorrelation is rho^k, so
    tau_int = 1 + 2 * sum_k rho^k = (1 + rho) / (1 - rho).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    trace = np.empty(n)
    trace[0] = noise[0]
    for i in range(1, n):
        trace[i] = rho * trace[i - 1] + noise[i]
    return trace


def test_batch_means_tau_int_recovers_ar1():
    rho = 0.9
    tau_true = (1 + rho) / (1 - rho)  # = 19
    trace = _ar1(rho, 200_000, seed=0)
    tau, block_length, n_blocks = batch_means_tau_int(trace)
    assert tau == pytest.approx(tau_true, rel=0.25)


def test_batch_means_tau_int_iid_is_about_one():
    rng = np.random.default_rng(1)
    tau, _, _ = batch_means_tau_int(rng.standard_normal(100_000))
    assert 0.7 < tau < 1.4


def test_batch_means_block_length_self_consistent():
    """The rule requires block length >= 10 * tau_hat; the estimator
    must return a block length that certifies its own estimate."""
    trace = _ar1(0.9, 200_000, seed=2)
    tau, block_length, n_blocks = batch_means_tau_int(trace)
    assert block_length >= 10 * tau
    assert n_blocks >= 20  # enough blocks for a stable variance-of-means


# ---------------------------------------------------------------------------
# kawasaki_burn_in_sweeps
# ---------------------------------------------------------------------------


def test_burn_in_floor_dominates_short_tau():
    # 20 * 100 = 2_000 < 10_000 -> the 1e4-sweep floor wins
    assert kawasaki_burn_in_sweeps(tau_int_sweeps=100.0) == 10_000


def test_burn_in_tau_dominates_long_tau():
    # 20 * 1_000 = 20_000 > 10_000 -> the tau term wins
    assert kawasaki_burn_in_sweeps(tau_int_sweeps=1_000.0) == 20_000


# ---------------------------------------------------------------------------
# ratio_with_ci
# ---------------------------------------------------------------------------


def test_ratio_point_estimate_is_per_compute():
    # neural: 1500 N_eff for 1e6 units; kawasaki: 500 N_eff for 1e6 units
    # -> per-compute ratio exactly 3, regardless of the SEs.
    ratio = ratio_with_ci(1500.0, 10.0, 1e6, 500.0, 10.0, 1e6)
    assert ratio["point"] == pytest.approx(3.0)


def test_ratio_ci_excludes_one_when_ses_are_tiny():
    ratio = ratio_with_ci(1500.0, 1.0, 1e6, 1000.0, 1.0, 1e6)
    assert ratio["lo"] > 1.0
    assert ratio["excludes_parity"]


def test_ratio_ci_contains_one_when_ses_are_large():
    # equal efficiency with sloppy error bars: parity must NOT be excluded
    ratio = ratio_with_ci(1000.0, 400.0, 1e6, 1000.0, 400.0, 1e6)
    assert ratio["lo"] < 1.0 < ratio["hi"]
    assert not ratio["excludes_parity"]


def test_ratio_ci_is_symmetric_in_log_space():
    ratio = ratio_with_ci(1000.0, 200.0, 1e6, 1000.0, 200.0, 1e6)
    assert np.log(ratio["hi"]) + np.log(ratio["lo"]) == pytest.approx(
        2 * np.log(ratio["point"]), abs=1e-9
    )


# ---------------------------------------------------------------------------
# ratio_with_f_ci — the variance-ratio (F) construction
# ---------------------------------------------------------------------------


def test_ratio_f_ci_point_matches_delta_point():
    delta = ratio_with_ci(1500.0, 10.0, 1e6, 500.0, 10.0, 1e6)
    f = ratio_with_f_ci(1500.0, 1e6, 8, 500.0, 1e6, 8)
    assert f["point"] == pytest.approx(delta["point"])


def test_ratio_f_ci_bounds_are_f_quantile_factors():
    # N_eff ratio is an MSE ratio; with R=8 replicates each side and
    # mean-zero normal errors the ratio is F(8,8)-distributed around truth,
    # so the 95% CI is the point divided/multiplied by F_0.975(8,8) = 4.433
    f = ratio_with_f_ci(1500.0, 1e6, 8, 500.0, 1e6, 8)
    assert f["lo"] == pytest.approx(3.0 / 4.4333, rel=1e-3)
    assert f["hi"] == pytest.approx(3.0 * 4.4333, rel=1e-3)


def test_ratio_f_ci_resolves_large_effects_only():
    # 10x effect: resolvable at R=8; 2x effect: not — the construction's
    # power is exactly what separates the two CI methods
    assert ratio_with_f_ci(5000.0, 1e6, 8, 500.0, 1e6, 8)["excludes_parity"]
    assert not ratio_with_f_ci(1000.0, 1e6, 8, 500.0, 1e6, 8)[
        "excludes_parity"]


# ---------------------------------------------------------------------------
# tv_noise_floor — coverage TV must be read against the finite-sample
# floor a PERFECT sampler would show at the same effective sample size
# ---------------------------------------------------------------------------


def test_tv_noise_floor_rarely_exceeded_by_the_law_itself():
    rng = np.random.default_rng(3)
    law = np.ones(33) / 33
    floor = tv_noise_floor(law, 1000, rng)
    exceedances = sum(
        total_variation(rng.multinomial(1000, law), law) > floor
        for _ in range(100)
    )
    assert exceedances <= 15  # ~5% nominal, slack for bootstrap noise


def test_tv_noise_floor_flags_a_genuinely_distorted_sampler():
    rng = np.random.default_rng(4)
    law = np.ones(33) / 33
    distorted = law.copy()
    distorted[:16] *= 0.5  # half the mass gone from one mode side
    distorted /= distorted.sum()
    floor = tv_noise_floor(law, 1000, rng)
    assert total_variation(rng.multinomial(1000, distorted), law) > floor


def test_tv_noise_floor_shrinks_with_sample_size():
    rng = np.random.default_rng(5)
    law = np.ones(33) / 33
    assert tv_noise_floor(law, 100_000, rng) < tv_noise_floor(law, 1000, rng)


# ---------------------------------------------------------------------------
# frozen_verdict — the GO / PARTIAL / NO-GO / PROVISIONAL mapping
# ---------------------------------------------------------------------------


def _margin(point, excludes_parity):
    return {"point": point, "excludes_parity": excludes_parity}


def test_verdict_go_requires_everything():
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(2.0, True),
        network_pass_ratio=_margin(1.8, True),
        floor_not_worse=True,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] == "GO"


def test_verdict_marginal_win_is_partial_real_but_marginal():
    # significant in both currencies but point < 1.5x -> the
    # "real-but-marginal" narrative, NOT GO (magnitude bar) and NOT NO-GO
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(1.2, True),
        network_pass_ratio=_margin(1.3, True),
        floor_not_worse=True,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] == "PARTIAL"
    assert verdict["narrative"] == "real_but_marginal"


def test_verdict_one_currency_is_partial_amortisation():
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(0.8, False),
        network_pass_ratio=_margin(2.5, True),
        floor_not_worse=True,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] == "PARTIAL"
    assert verdict["narrative"] == "one_currency"


def test_verdict_no_go_when_losing_both_currencies():
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(0.6, True),
        network_pass_ratio=_margin(0.7, True),
        floor_not_worse=True,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=False,
    )
    assert verdict["verdict"] == "NO-GO"


def test_verdict_beats_local_only_is_partial_move_set():
    # loses to non-local in both currencies but beats the local variant:
    # the "non-local move set does the work" narrative, not a bare NO-GO
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(0.7, True),
        network_pass_ratio=_margin(0.8, True),
        floor_not_worse=True,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] == "PARTIAL"
    assert verdict["narrative"] == "move_set_does_the_work"


def test_verdict_pending_floor_blocks_go():
    # GO requires "not worse at the floor"; with the floor replicates not yet
    # drawn the result must be PROVISIONAL, never GO.
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(2.0, True),
        network_pass_ratio=_margin(1.8, True),
        floor_not_worse=None,
        coverage_ok=True,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] == "PROVISIONAL"
    assert verdict["pending"] == ["floor_not_worse"]


def test_verdict_coverage_failure_blocks_go():
    # a win on speed that loses modes is not a win
    verdict = frozen_verdict(
        energy_eval_ratio=_margin(2.0, True),
        network_pass_ratio=_margin(1.8, True),
        floor_not_worse=True,
        coverage_ok=False,
        gate_holds=True,
        beats_local_variant=True,
    )
    assert verdict["verdict"] != "GO"
