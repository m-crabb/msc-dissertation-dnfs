"""Training-FLOP accounting: the derived count, and the measured extrapolation.

The house table's FLOP/es column prices sampling only. Training cost is what
the amortisation argument turns on — "one training run serves N targets" is
unpriceable without it — so it gets its own two independent instruments:

  derived   `training_run_flops` multiplies a measured per-forward count by the
            forward count the loop structure implies.
  measured  run the real loop at several horizons under FlopCounterMode and fit
            a line; the slope is the per-cycle cost.

Neither is trusted alone. The derived figure has to assume backward = 2x forward
(a convention, not a measurement) and has to assume nobody is doing forwards the
recipe does not mention. The measured figure cannot be run to the full horizon,
so it has to assume linearity. Each covers the other's assumption, and the two
are compared rather than reconciled — a gap is a finding about the loop, not a
number to be averaged away.

What these tests pin:

1. The fit recovers slope and intercept on exactly-linear data, and reports a
   residual large enough to refuse data that is not linear.
2. The forward count matches the loop structure by hand, including the c_t grid
   recompute being skipped in the mode the production cells run.
3. Horizons that are not whole cycles, or that straddle the periodic
   in-training eval, are refused rather than fitted — both make total FLOPs a
   step function of n_steps.
"""

import math

import pytest

from discrete_flow_sampler.diagnostics.flops import (
    fit_flop_scaling,
    training_forward_counts,
    training_run_flops,
    valid_measurement_horizons,
)

# --- 1. the fit and its refusal to extrapolate off bad data -------------------


def test_fit_recovers_an_exactly_linear_run():
    """Startup and buffer fill are a fixed cost; the per-cycle rate is the
    slope. Differencing across horizons is what removes the first
    `replay_buffer_cycles` cycles, during which the buffer is still filling and
    the loop is genuinely unrepresentative."""
    fixed, per_cycle = 5.0e12, 3.0e12
    horizons = (10, 20, 40)
    totals = [fixed + per_cycle * n for n in horizons]
    fit = fit_flop_scaling(horizons, totals)
    assert fit["fixed_flops"] == pytest.approx(fixed, rel=1e-9)
    assert fit["flops_per_outer_cycle"] == pytest.approx(per_cycle, rel=1e-9)
    assert fit["max_relative_residual"] < 1e-9
    assert fit["extrapolate"](1000) == pytest.approx(fixed + per_cycle * 1000)


def test_fit_reports_a_large_residual_on_nonlinear_data():
    """The residual is the extrapolation's licence. If per-cycle cost drifts —
    a curriculum stage that changes shapes, an eval that fires irregularly — the
    fit must say so rather than return a plausible slope."""
    horizons = (10, 20, 40)
    totals = [1.0e12 * n**1.5 for n in horizons]
    fit = fit_flop_scaling(horizons, totals)
    assert fit["max_relative_residual"] > 0.05


def test_fit_needs_at_least_three_horizons():
    """Two points fit any line exactly, so a residual computed from two points
    is identically zero and certifies nothing. The third point is what makes
    the linearity claim falsifiable."""
    with pytest.raises(ValueError, match="at least three"):
        fit_flop_scaling((10, 20), (1.0, 2.0))


# --- 2. the derived forward count ---------------------------------------------


def test_forward_counts_follow_the_loop_structure():
    """Per outer cycle: one rollout of n_euler head forwards, then
    inner_steps_per_outer updates. Hand-checked against the production recipe
    (100k steps, inner 100, n_euler 128) — 1,000 cycles, 128,000 rollout
    forwards, 100,000 update forwards."""
    counts = training_forward_counts(
        n_steps=100_000, inner_steps_per_outer=100, n_euler_steps=128
    )
    assert counts["n_outer"] == 1_000
    assert counts["rollout_forwards"] == 128_000
    assert counts["update_forwards"] == 100_000


def test_c_t_grid_recompute_is_charged_only_when_it_runs():
    """With c_t_from_rollout in control-variate mode the rollout hands back its
    own xi_t and the grid pass is skipped entirely — that is the mode the d256
    cells run. Charging it anyway would overstate training by a second full
    128-forward rollout per cycle, i.e. double the rollout term."""
    shared = dict(n_steps=1_000, inner_steps_per_outer=100, n_euler_steps=128)
    reused = training_forward_counts(**shared, c_t_from_rollout=True)
    recomputed = training_forward_counts(**shared, c_t_from_rollout=False)
    assert reused["c_t_grid_forwards"] == 0
    assert recomputed["c_t_grid_forwards"] == 128 * 10
    assert recomputed["rollout_forwards"] == reused["rollout_forwards"], (
        "the rollout itself is unchanged; only the extra grid pass appears"
    )


def test_run_flops_prices_backward_against_forward():
    """The backward multiplier is the derived figure's one soft assumption, so
    it is a named parameter rather than a buried constant: the measured leg
    exists precisely to test it."""
    kwargs = dict(
        rollout_forward_flops=10,
        update_forward_flops=100,
        n_steps=1_000,
        inner_steps_per_outer=100,
        n_euler_steps=128,
    )
    # 10 cycles: 1280 rollout forwards at 10, 1000 updates at 100 x (1 + bwd).
    assert training_run_flops(**kwargs, backward_multiplier=2.0) == (
        1280 * 10 + 1000 * 100 * 3
    )
    assert training_run_flops(**kwargs, backward_multiplier=0.0) == (
        1280 * 10 + 1000 * 100
    )


def test_partial_outer_cycles_are_refused():
    """train_swap validates n_steps % inner_steps_per_outer == 0 and so must
    this, or the count silently prices a cycle that never ran."""
    with pytest.raises(ValueError, match="whole outer cycles"):
        training_forward_counts(
            n_steps=150, inner_steps_per_outer=100, n_euler_steps=128
        )


# --- 3. choosing horizons that are actually fittable --------------------------


def test_horizons_are_whole_cycles_and_whole_eval_periods():
    """Total FLOPs step up whenever the periodic in-training eval fires, so a
    horizon that straddles one lands off the line for a reason that has nothing
    to do with the training loop. Valid horizons are multiples of the lcm."""
    horizons = valid_measurement_horizons(
        inner_steps_per_outer=100, eval_every=250, n_horizons=3
    )
    period = math.lcm(100, 250)
    assert all(h % period == 0 for h in horizons)
    assert len(set(horizons)) == 3
    assert horizons == sorted(horizons)


def test_horizon_period_is_the_lcm_not_the_larger_of_the_two():
    """500, not 250: a multiple of eval_every alone can still cut a cycle in
    half, and a multiple of inner_steps_per_outer alone can still straddle an
    eval."""
    horizons = valid_measurement_horizons(
        inner_steps_per_outer=100, eval_every=250, n_horizons=2
    )
    assert horizons[0] == 500
    assert horizons[1] == 1000


def test_eval_every_none_falls_back_to_the_cycle():
    """With the periodic eval off, whole cycles are the only constraint."""
    horizons = valid_measurement_horizons(
        inner_steps_per_outer=100, eval_every=None, n_horizons=3
    )
    assert horizons == [100, 200, 300]


def test_measurement_curriculum_is_truncated_to_the_horizon():
    """The production ladder's later stages start beyond every measurement
    horizon and the trainer's validator refuses them (`curriculum start_step
    must be < n_steps`), which killed the first live run (Modal, 2026-08-31).
    The harness hands the trainer only the stages the horizon can reach, which
    for every valid horizon is the first stage."""
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.measure_training_flops import (
        curriculum_within,
    )

    cfg = CONFIGS["H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3"]
    stages = curriculum_within(cfg.curriculum, horizon=500)
    assert len(stages) == 1
    assert stages[0].start_step == 0
    assert curriculum_within(None, horizon=500) is None
    full = curriculum_within(cfg.curriculum, horizon=cfg.train.n_steps)
    assert full == tuple(cfg.curriculum.stages)


def test_diagnostic_eval_flops_price_the_severable_instrument():
    """The in-training frozen-ESS eval is instrumentation, not the algorithm:
    frozen weights, no gradients, severable by turning eval_every off. It is
    priced as its own term, never folded into training-proper — the d64 thp
    certification measured it at 37% of the as-instrumented bill (gap
    reconciled to 0.8%). No backward is charged (the draws are no-grad), and
    draws scale the update-batch forward linearly."""
    from discrete_flow_sampler.diagnostics.flops import diagnostic_eval_flops

    # 50k steps, eval every 200 -> 250 evals; each draws 512 samples
    # through 128 CTMC steps, priced off a batch-128 forward of 1e9:
    # 250 * 128 * 1e9 * (512/128) = 1.28e14.
    total = diagnostic_eval_flops(
        update_forward_flops=1e9,
        update_batch_size=128,
        n_euler_steps=128,
        n_steps=50_000,
        eval_every=200,
        n_eval_draws=512,
    )
    assert total == 250 * 128 * 1e9 * 4.0
    # Instrument off -> nothing charged.
    assert (
        diagnostic_eval_flops(
            update_forward_flops=1e9,
            update_batch_size=128,
            n_euler_steps=128,
            n_steps=50_000,
            eval_every=None,
            n_eval_draws=512,
        )
        == 0.0
    )
