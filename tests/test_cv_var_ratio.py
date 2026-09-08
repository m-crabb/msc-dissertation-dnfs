"""CV-inversion observer + tripwire (adversarial panel, 2026-08-18).

The panel validated the controlled/naive integrand variance ratio as a 5/5
in-run classifier of the d256 cold-CV failure (cold: ratio never < 1.5
across 5k steps; healthy warm: crosses below 1 within ~1000 steps) — yet
the two variances, both already computed every outer cycle
(var_estimator_integrand and var_dt_log_p_tilde over the same rollout
rows), were compared nowhere in code.

1. The ratio is logged every step as `cv_var_ratio`. In naive mode the
   active integrand is ∂_t log p̃, evaluated row-locally on the same rows,
   so the ratio must be exactly 1.0 — a free wiring self-check.
2. In CV mode the column is the division of its two parent columns on every
   row — no separate estimator pass, no new randomness.
3. The halt guard defaults off (None — every archived cell's behaviour).
   When armed it must stop the run gracefully — marker file written, loop
   exited, final checkpoint still saved — and only on a sustained
   inversion: the ratio above 1.0 for a full trailing window of outer
   cycles at/after the arming step. A transient inversion (the healthy
   warm-start pattern) must never trip it, which is what the window and
   the arming step are for.
"""

import csv
from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_training
from discrete_flow_sampler.samplers.swap_training import (
    _cv_inversion_sustained,
    train_swap,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _head(init_seed: int = 0) -> LeTFMaskOneSwapHead:
    torch.manual_seed(init_seed)
    return LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _run(
    run_dir: Path,
    estimator_mode: str,
    n_steps: int = 4,
    halt_after=None,
    halt_window: int = 2,
) -> None:
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg = _Cfg(
        n_steps=n_steps,
        batch_size=8,
        outer_batch_size=8,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=2,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
        halt_on_cv_inversion_after=halt_after,
        halt_cv_inversion_window=halt_window,
    )
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=16)
    train_swap(
        _head(),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        run_dir,
        use_wandb=False,
        estimator_mode=estimator_mode,
    )


def _rows(run_dir: Path) -> list[dict]:
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


# --------------------------------------------------------------------------
# Observer column


def test_cv_var_ratio_is_exactly_one_in_naive_mode(tmp_path):
    """Naive mode: the estimator integrand is the naive integrand, computed
    row-locally on the same rollout rows, so the logged ratio is 1.0. This
    pins the wiring end to end: a value != 1.0 here means the two variance
    columns no longer share their row set."""
    _run(tmp_path / "naive", estimator_mode="naive_mc")
    rows = _rows(tmp_path / "naive")
    assert len(rows) == 4
    for row in rows:
        assert abs(float(row["cv_var_ratio"]) - 1.0) < 1e-9


def test_cv_var_ratio_is_the_division_of_its_parent_columns(tmp_path):
    """CV mode: the column must equal var_estimator_integrand /
    var_dt_log_p_tilde on every row — an observer formed from quantities
    the trainer already logs, never a new estimator pass."""
    _run(tmp_path / "cv", estimator_mode="control_variate")
    for row in _rows(tmp_path / "cv"):
        expected = float(row["var_estimator_integrand"]) / float(
            row["var_dt_log_p_tilde"]
        )
        assert abs(float(row["cv_var_ratio"]) - expected) < 1e-9 * max(
            1.0, abs(expected)
        )


# --------------------------------------------------------------------------
# Halt-window logic (pure function)


def test_sustained_inversion_requires_a_full_window_above_one():
    assert not _cv_inversion_sustained([], window=3)
    assert not _cv_inversion_sustained([1.2, 1.2], window=3)
    assert _cv_inversion_sustained([1.2, 1.1, 1.01], window=3)
    # One healthy cycle inside the window resets the case: the warm-start
    # pattern (inversion healing within ~1000 steps) must never trip.
    assert not _cv_inversion_sustained([1.2, 0.9, 1.2], window=3)
    # The boundary is strict: exactly 1.0 is not an inversion.
    assert not _cv_inversion_sustained([1.2, 1.0, 1.2], window=3)
    # Only the trailing window counts; ancient inversions are forgiven.
    assert _cv_inversion_sustained([0.5, 1.3, 1.2, 1.1], window=3)


# --------------------------------------------------------------------------
# Tripwire wiring


def test_halt_default_off_and_naive_ratio_never_trips(tmp_path):
    """Armed guard + naive mode: ratio is exactly 1.0 (not > 1), so the run
    must complete every step and write no marker."""
    _run(
        tmp_path / "armed_naive", estimator_mode="naive_mc", halt_after=0, halt_window=1
    )
    assert len(_rows(tmp_path / "armed_naive")) == 4
    assert not (tmp_path / "armed_naive" / "cv_inversion_halt.json").exists()


def test_halt_stops_gracefully_when_detector_fires(tmp_path, monkeypatch):
    """Wiring: with the detector forced positive, an armed CV run must stop
    before its first inner step, write the marker, and still save final.pt
    (graceful stop, not a crash — artefacts stay judgeable)."""
    monkeypatch.setattr(swap_training, "_cv_inversion_sustained", lambda *a, **k: True)
    run_dir = tmp_path / "tripped"
    _run(run_dir, estimator_mode="control_variate", halt_after=0, halt_window=1)
    assert (run_dir / "cv_inversion_halt.json").exists()
    assert len(_rows(run_dir)) == 0
    assert (run_dir / "checkpoints" / "final.pt").exists()


def test_unarmed_cv_run_is_unchanged(tmp_path, monkeypatch):
    """Default (halt_on_cv_inversion_after=None): even a permanently-firing
    detector must never be consulted — the archived CV cells' behaviour."""
    monkeypatch.setattr(swap_training, "_cv_inversion_sustained", lambda *a, **k: True)
    run_dir = tmp_path / "unarmed"
    _run(run_dir, estimator_mode="control_variate", halt_after=None)
    assert len(_rows(run_dir)) == 4
    assert not (run_dir / "cv_inversion_halt.json").exists()
