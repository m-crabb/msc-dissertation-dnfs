"""Tests for the c_t grid EMA (M2 of the 2026-08-14 M-campaign).

Pre-registered in docs/plans/2026-08-14-m-scaling-experiments.md Task 2;
diagnosis in docs/design/2026-08-14-scaling-assessment.md §5/§8.

What correct looks like, independent of implementation:

1. **Off is byte-identical.** `c_t_ema_halflife_cycles = 0.0` (the default,
   and the value every archived run implicitly carries) must leave the
   training trajectory untouched — the knob defaults to OFF, so the
   falsification record of every archived cell stays valid.
2. **No init contamination.** The EMA must pass the FIRST outer cycle's
   raw c_t grid through exactly (re-seeded at it), never mix in a zero or
   stale initial state — the failure mode `ema.py`'s warmup schedule exists
   to prevent, avoided here by construction.
3. **Fixed point is unchanged.** A constant integrand sequence must be
   reproduced exactly at every cycle: smoothing is pure variance reduction
   on a drifting target, not a shift of what it converges to (the Eq.-8
   identity E[xi] = dt log Z_t holds for the model's own law, so recent
   cycles estimate the same slowly-drifting quantity).
4. **Tracking lag is the advertised halflife.** After a step change in the
   target, the state must close >= half the gap within `halflife` cycles.
5. **Curriculum transitions reset.** c_t = d_t log Z_t is a function of
   sigma; smoothing must never mix estimates across a sigma boundary. At
   the transition cycle the logged `c_t_ema_rms_delta` must be exactly 0.0
   (passthrough), growing again afterwards.
6. **Resume bit-exactness.** The EMA state and its validity must travel in
   resume.pt: an interrupt-and-resume under the EMA must reproduce the
   uninterrupted run's training log bit-for-bit (excluding wall-clock),
   the same contract test_swap_training_resume.py pins for the base state.
7. **Resume re-homes onto the live device.** resume.pt is deliberately
   device-portable (the state is stored on CPU, as replay chunks and the
   parameter shadow are), so a restored state must end up back on whatever
   device the run is using before it is folded into the next cycle's grid.
   Unlike the parameter EMA there are no parameters to read a device from,
   so the contract is checked at the fold. This is the only contract in
   this file that CPU-only runs cannot falsify — it is the resume path of
   every GPU cell that arms the knob.
"""
import csv
import io
from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.ema import CTGridEMA
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

from experiments.dnfs_baseline_01.configs import CurriculumStageCfg


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


TWO_STAGE_CURRICULUM = (
    CurriculumStageCfg(start_step=0, sigma=0.1, lr=1e-3),
    CurriculumStageCfg(start_step=2, sigma=0.223, lr=3e-4),
)


def _head(init_seed: int) -> LeTFMaskOneSwapHead:
    # Mask-one, not the doubly-hollow oracle: the c_t EMA contracts are
    # head-agnostic, and the O(d^2)-pass head costs ~15x per call for no
    # extra coverage here (first version of this file used it; 95s for the
    # file against the suite's 80s total).
    torch.manual_seed(init_seed)
    return LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _cfgs(n_steps: int, c_t_ema_halflife_cycles: float):
    train_cfg = _Cfg(n_steps=n_steps, batch_size=8, outer_batch_size=8,
                     inner_steps_per_outer=2, lr=1e-3, seed=0,
                     replay_buffer_cycles=2, grad_clip_max_norm=500.0,
                     warmup_steps=0, resume_every_outer=1,
                     c_t_ema_halflife_cycles=c_t_ema_halflife_cycles)
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=16)
    return train_cfg, ctmc_cfg, eval_cfg


def _run(run_dir: Path, n_steps: int, head, halflife: float) -> None:
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(n_steps, halflife)
    train_swap(head, target, train_cfg, ctmc_cfg, eval_cfg, run_dir,
               use_wandb=False, estimator_mode="control_variate",
               sigma_curriculum=TWO_STAGE_CURRICULUM)


def _log_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


# --------------------------------------------------------------------------
# Helper-level contracts


def test_disabled_value_is_rejected_and_detectable():
    """The trainer constructs the helper only when the knob is on; the
    helper itself refuses the disabled value rather than silently
    smoothing with a meaningless rate."""
    assert not CTGridEMA.is_enabled(0.0)
    assert CTGridEMA.is_enabled(4.0)
    try:
        CTGridEMA(n_grid=8, halflife_cycles=0.0)
        raise AssertionError("halflife 0.0 must be rejected")
    except ValueError:
        pass


def test_first_cycle_is_passthrough():
    ema = CTGridEMA(n_grid=8, halflife_cycles=4.0)
    grid = torch.randn(8)
    out = ema.update(grid)
    assert torch.equal(out, grid)


def test_constant_integrand_fixed_point():
    """A constant integrand is the EMA's fixed point: exact in real
    arithmetic, and in fp32 the fold's rounding must not compound into a
    drift — bounded at ulp level even after 100 cycles."""
    ema = CTGridEMA(n_grid=8, halflife_cycles=4.0)
    grid = torch.randn(8)
    for _ in range(100):
        out = ema.update(grid)
    drift = (out - grid).abs().max().item()
    assert drift <= 1e-6, f"fixed-point drift {drift} exceeds ulp level"


def test_step_change_tracks_with_halflife_lag():
    ema = CTGridEMA(n_grid=1, halflife_cycles=4.0)
    ema.update(torch.zeros(1))
    for _ in range(4):  # one halflife at the new level
        out = ema.update(torch.ones(1))
    assert out.item() >= 0.5
    for _ in range(12):  # three further halflives
        out = ema.update(torch.ones(1))
    assert out.item() >= 1.0 - 2.0 ** -4 - 1e-6  # fp rounding around exact


def test_reset_re_seeds_at_next_cycle():
    ema = CTGridEMA(n_grid=2, halflife_cycles=4.0)
    ema.update(torch.full((2,), 3.0))
    ema.update(torch.full((2,), 3.0))
    ema.reset()
    fresh = torch.full((2,), -1.0)
    out = ema.update(fresh)
    assert torch.equal(out, fresh)


def test_state_dict_roundtrip_continuation():
    ema = CTGridEMA(n_grid=4, halflife_cycles=4.0)
    ema.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    ema.update(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    state = ema.state_dict()

    grids = [torch.full((4,), float(k)) for k in range(3, 6)]
    expected = [ema.update(g) for g in grids]

    restored = CTGridEMA(n_grid=4, halflife_cycles=4.0)
    restored.load_state_dict(state)
    actual = [restored.update(g) for g in grids]
    for ref, got in zip(expected, actual):
        assert torch.equal(ref, got)


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_state_dict_roundtrip_re_homes_off_cpu():
    """Contract 7. state_dict() stores on CPU for portability, so a resumed
    run on an accelerator must not be left folding a CPU state into a device
    grid. Exercised through torch.save/torch.load because that (not a bare
    dict handoff) is what resume.pt actually does."""
    device = _accelerator()
    if device is None:
        import pytest
        pytest.skip("no accelerator: the CPU path cannot falsify contract 7")

    ema = CTGridEMA(n_grid=4, halflife_cycles=4.0)
    grids = [torch.full((4,), float(k), device=device) for k in range(1, 6)]
    for grid in grids[:2]:
        ema.update(grid)
    buffer = io.BytesIO()
    torch.save(ema.state_dict(), buffer)
    expected = [ema.update(grid) for grid in grids[2:]]

    buffer.seek(0)
    restored = CTGridEMA(n_grid=4, halflife_cycles=4.0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = [restored.update(grid) for grid in grids[2:]]
    for got in actual:
        assert got.device.type == device
    for ref, got in zip(expected, actual):
        assert torch.equal(ref, got)


# --------------------------------------------------------------------------
# Trainer-level contracts


def test_off_run_is_bit_identical_to_unknobbed(tmp_path):
    """0.0 (and the getattr default) must be the archived behaviour."""
    knobbed_dir = tmp_path / "knobbed"
    default_dir = tmp_path / "default"
    _run(knobbed_dir, n_steps=4, head=_head(init_seed=0), halflife=0.0)

    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(4, 0.0)
    del train_cfg.c_t_ema_halflife_cycles  # the getattr-default path
    train_swap(_head(init_seed=0), target, train_cfg, ctmc_cfg, eval_cfg,
               default_dir, use_wandb=False, estimator_mode="control_variate",
               sigma_curriculum=TWO_STAGE_CURRICULUM)

    knobbed_rows = _log_rows(knobbed_dir)
    default_rows = _log_rows(default_dir)
    assert len(knobbed_rows) == len(default_rows) == 4
    for knobbed, default in zip(knobbed_rows, default_rows):
        knobbed.pop("wall_clock_step_s")
        default.pop("wall_clock_step_s")
        assert knobbed == default
    for row in knobbed_rows:
        assert row["c_t_ema_rms_delta"] == "nan"


def test_ema_run_curriculum_reset_visible_in_log(tmp_path):
    """At the sigma-transition cycle the EMA re-seeds: rms delta is exactly
    0.0 there (passthrough), positive on cycles that smooth a drifting
    target, and the run stays finite throughout."""
    run_dir = tmp_path / "run"
    _run(run_dir, n_steps=8, head=_head(init_seed=0), halflife=4.0)
    rows = _log_rows(run_dir)
    assert len(rows) == 8
    deltas = [float(row["c_t_ema_rms_delta"]) for row in rows]
    # Logged once per inner step but constant within an outer cycle.
    # inner_steps_per_outer=2 -> deltas at even indices are per-cycle;
    # the curriculum transition at step 2 is outer cycle 1.
    cycle = deltas[::2]
    assert cycle[0] == 0.0                      # cycle 0: first-cycle passthrough
    assert cycle[1] == 0.0                      # cycle 1: sigma transition reset
    assert cycle[2] > 0.0                       # cycle 2: smoothing resumed
    assert all(d == d for d in deltas)          # no NaNs when enabled
    assert all(float(row["loss"]) == float(row["loss"]) for row in rows)


def test_resumed_ema_run_is_bit_exact_with_uninterrupted(tmp_path):
    """The EMA state must ride resume.pt, or the continuation diverges."""
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"

    _run(uninterrupted_dir, n_steps=8, head=_head(init_seed=0), halflife=4.0)

    _run(interrupted_dir, n_steps=4, head=_head(init_seed=0), halflife=4.0)
    (interrupted_dir / "checkpoints" / "final.pt").unlink()
    _run(interrupted_dir, n_steps=8, head=_head(init_seed=999), halflife=4.0)

    reference_rows = _log_rows(uninterrupted_dir)
    resumed_rows = _log_rows(interrupted_dir)
    assert len(reference_rows) == len(resumed_rows) == 8
    for reference, resumed in zip(reference_rows, resumed_rows):
        reference.pop("wall_clock_step_s")
        resumed.pop("wall_clock_step_s")
        assert reference == resumed
