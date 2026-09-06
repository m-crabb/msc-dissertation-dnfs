"""Replay-buffer flushing at curriculum sigma boundaries.

What correct looks like, written before the flag existed:

- DEFAULT (``flush_replay_on_stage`` absent or True) reproduces every
  archived run: crossing a sigma boundary empties the retention window, so
  the outer cycle that follows the boundary trains on ONE cycle of states
  and the window has to refill over ``replay_buffer_cycles`` cycles.
- FLAG OFF retains the window across the boundary: the buffer never
  shrinks, because the loss recomputes both target terms at the live sigma
  (samplers/swap_kolmogorov.loss_swap), so a retained state is an
  evaluation point under the NEW target rather than a stale label.
- The flag governs the STATES only. ``c_t`` is a function of sigma, so its
  cross-cycle EMA must reset at the boundary either way; a run that skipped
  that reset would be smoothing two different quantities together.

The observable is the buffer row count handed to the inner loop at each
outer cycle, captured by wrapping the trainer's ``_append_replay_buffer``.
Rows are counted in units of one outer cycle's contribution, so the
expected sequences are exact integers and independent of grid size.
"""

from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_training
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _tiny_head():
    return DoublyHollowSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


# Four retained cycles and a boundary at outer cycle 4 (step 8, with two
# inner steps per cycle), so the window is exactly full when the boundary
# arrives -- the case where flushing and retaining differ most.
_REPLAY_CYCLES = 4
_INNER_PER_OUTER = 2
_N_OUTER = 8

_TWO_STAGE_CURRICULUM = [
    _Cfg(start_step=0, sigma=0.10),
    _Cfg(start_step=_INNER_PER_OUTER * 4, sigma=0.14),
]


def _tiny_cfgs(**train_overrides):
    train_kwargs = dict(
        n_steps=_INNER_PER_OUTER * _N_OUTER,
        batch_size=8,
        outer_batch_size=8,
        inner_steps_per_outer=_INNER_PER_OUTER,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=_REPLAY_CYCLES,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
    )
    train_kwargs.update(train_overrides)
    return (
        _Cfg(**train_kwargs),
        _Cfg(n_euler_steps=8),
        _Cfg(eval_every=1000, n_eval_samples=8),  # eval off: not under test
    )


def _buffer_cycles_per_outer_step(tmp_path, monkeypatch, **train_overrides):
    """Run the tiny trainer and return the retained buffer size, in whole
    outer cycles, as seen by each outer step's inner loop."""
    original_append = swap_training._append_replay_buffer
    retained_rows: list[int] = []

    def recording_append(*args, **kwargs):
        x_buffer, t_idx_buffer = original_append(*args, **kwargs)
        retained_rows.append(x_buffer.shape[0])
        return x_buffer, t_idx_buffer

    monkeypatch.setattr(swap_training, "_append_replay_buffer", recording_append)
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(**train_overrides)
    train_swap(
        _tiny_head(),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        Path(tmp_path),
        use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=_TWO_STAGE_CURRICULUM,
    )
    rows_per_cycle = retained_rows[0]
    assert all(rows % rows_per_cycle == 0 for rows in retained_rows)
    return [rows // rows_per_cycle for rows in retained_rows]


def test_flush_at_sigma_boundary_is_the_default(tmp_path, monkeypatch):
    """Unset flag = archived behaviour: the window refills from empty after
    the boundary at outer cycle 4."""
    cycles = _buffer_cycles_per_outer_step(tmp_path, monkeypatch)
    assert cycles == [1, 2, 3, 4, 1, 2, 3, 4]


def test_flush_true_matches_the_default(tmp_path, monkeypatch):
    cycles = _buffer_cycles_per_outer_step(
        tmp_path, monkeypatch, flush_replay_on_stage=True
    )
    assert cycles == [1, 2, 3, 4, 1, 2, 3, 4]


def test_no_flush_retains_the_window_across_the_boundary(tmp_path, monkeypatch):
    """Flag off: the boundary is invisible to the buffer, which stays at its
    retention cap throughout."""
    cycles = _buffer_cycles_per_outer_step(
        tmp_path, monkeypatch, flush_replay_on_stage=False
    )
    assert cycles == [1, 2, 3, 4, 4, 4, 4, 4]


def test_no_flush_still_resets_the_c_t_ema_at_the_boundary(tmp_path, monkeypatch):
    """c_t = dt log Z_t is a function of sigma, so its cross-cycle EMA must
    be reset at the transition whatever the buffer does. Guarded because the
    reset shares the same `if sigma changed` block as the flush and could be
    disabled by accident along with it."""
    reset_steps: list[int] = []
    original_ema_class = swap_training.CTGridEMA

    class RecordingCTGridEMA(original_ema_class):
        def reset(self):
            reset_steps.append(1)
            return super().reset()

    monkeypatch.setattr(swap_training, "CTGridEMA", RecordingCTGridEMA)
    cycles = _buffer_cycles_per_outer_step(
        tmp_path,
        monkeypatch,
        flush_replay_on_stage=False,
        c_t_ema_halflife_cycles=4.0,
    )
    assert cycles == [1, 2, 3, 4, 4, 4, 4, 4]
    assert reset_steps, "the c_t grid EMA was never reset at the boundary"
