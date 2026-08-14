"""Tests for the decoupled c_t rollout batch (M3 of the 2026-08-14
M-campaign).

Pre-registered in docs/plans/2026-08-14-m-scaling-experiments.md Task 3;
diagnosis in docs/design/2026-08-14-scaling-assessment.md §5/§8.

What correct looks like, independent of implementation:

1. **Off is byte-identical.** `c_t_batch = None` (the default, and the value
   every archived run implicitly carries) must leave the training trajectory
   untouched — same RNG stream, same c_t, same buffer — so the falsification
   record of every archived cell stays valid.
2. **Explicit-equals-default is a no-op.** `c_t_batch == outer_batch` must
   also be bit-identical: the knob only has content when it ENLARGES the
   rollout set (otherwise it is the off path wearing a name).
3. **c_t uses the larger set.** With c_t_batch > outer_batch, the base draw
   and the c_t grid must both see c_t_batch rows — c_t = mean_m xi_t over
   the enlarged set (the Eq.-8 identity holds for the model's own law, so a
   bigger M is a purer estimate of the same quantity, standard error ~ 1/M).
4. **The buffer is unchanged in size and composition.** The replay buffer
   must receive exactly the first `outer_batch` rows of the enlarged rollout
   (a uniform subset — no selection bias), so inner-step sampling sees the
   same buffer_size as an off run; the whole point is that ONLY the no-grad
   c_t phase scales, never the inner-update batch or buffer.
5. **Under-supply is rejected.** c_t_batch < outer_batch would starve the
   buffer (only the first outer_batch rows feed it); refuse loudly.
6. **Resume stays bit-exact under the knob.** The knob is stateless, so an
   interrupt-and-resume with c_t_batch on must reproduce the uninterrupted
   log bit-for-bit (same contract test_swap_training_resume.py pins for the
   base state) — this pins that the enlarged draw's RNG consumption is
   checkpointed via the RNG state, not reconstructed.
"""
import csv
from pathlib import Path

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_training
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

OUTER_BATCH = 8


def _head(init_seed: int) -> LeTFMaskOneSwapHead:
    # Mask-one, not the doubly-hollow oracle: the rollout-accounting
    # contracts are head-agnostic and the oracle costs ~15x per call for no
    # extra coverage here (same reasoning as tests/test_c_t_ema.py).
    torch.manual_seed(init_seed)
    return LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _cfgs(n_steps: int, c_t_batch, n_eval_samples: int = 16):
    train_cfg = _Cfg(n_steps=n_steps, batch_size=8, outer_batch_size=8,
                     inner_steps_per_outer=2, lr=1e-3, seed=0,
                     replay_buffer_cycles=2, grad_clip_max_norm=500.0,
                     warmup_steps=0, resume_every_outer=1,
                     c_t_batch=c_t_batch)
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=n_eval_samples)
    return train_cfg, ctmc_cfg, eval_cfg


def _run(run_dir: Path, n_steps: int, head, c_t_batch,
         n_eval_samples: int = 16) -> None:
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(n_steps, c_t_batch, n_eval_samples)
    train_swap(head, target, train_cfg, ctmc_cfg, eval_cfg, run_dir,
               use_wandb=False, estimator_mode="control_variate",
               sigma_curriculum=TWO_STAGE_CURRICULUM)


def _log_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def _assert_logs_bit_identical(dir_a: Path, dir_b: Path, n_rows: int) -> None:
    rows_a, rows_b = _log_rows(dir_a), _log_rows(dir_b)
    assert len(rows_a) == len(rows_b) == n_rows
    for a, b in zip(rows_a, rows_b):
        a.pop("wall_clock_step_s")
        b.pop("wall_clock_step_s")
        assert a == b


# --------------------------------------------------------------------------
# Trainer-level contracts


def test_off_and_explicit_equal_are_bit_identical_to_unknobbed(tmp_path):
    """None (the getattr default and the explicit value) and
    c_t_batch == outer_batch must all three be the archived behaviour."""
    none_dir = tmp_path / "explicit_none"
    default_dir = tmp_path / "unknobbed"
    equal_dir = tmp_path / "explicit_equal"

    _run(none_dir, n_steps=4, head=_head(init_seed=0), c_t_batch=None)
    _run(equal_dir, n_steps=4, head=_head(init_seed=0), c_t_batch=OUTER_BATCH)

    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(4, None)
    del train_cfg.c_t_batch  # the getattr-default path
    train_swap(_head(init_seed=0), target, train_cfg, ctmc_cfg, eval_cfg,
               default_dir, use_wandb=False, estimator_mode="control_variate",
               sigma_curriculum=TWO_STAGE_CURRICULUM)

    _assert_logs_bit_identical(none_dir, default_dir, n_rows=4)
    _assert_logs_bit_identical(none_dir, equal_dir, n_rows=4)


def test_c_t_uses_the_larger_rollout_set(tmp_path, monkeypatch):
    """With c_t_batch = 2x outer_batch, the base draw and the c_t grid must
    both see the enlarged row count (the instrumented call count)."""
    seen = {"base_batches": [], "c_t_row_counts": []}

    real_sample_base = FixedCompositionIsingTarget.sample_base

    def base_spy(self, batch_size, device=None):
        seen["base_batches"].append(batch_size)
        return real_sample_base(self, batch_size, device=device)

    real_c_t = swap_training.compute_c_t_grid_swap

    def c_t_spy(t_grid, x_traj, target, head, *, mode):
        seen["c_t_row_counts"].append(x_traj.shape[1])
        return real_c_t(t_grid, x_traj, target, head, mode=mode)

    monkeypatch.setattr(FixedCompositionIsingTarget, "sample_base", base_spy)
    monkeypatch.setattr(swap_training, "compute_c_t_grid_swap", c_t_spy)

    # n_eval_samples deliberately != c_t_batch so the spied base draws are
    # unambiguous: init diagnostics draw outer_batch (8), eval draws 12,
    # the c_t cycle draws the enlarged 16.
    run_dir = tmp_path / "run"
    _run(run_dir, n_steps=4, head=_head(init_seed=0), c_t_batch=2 * OUTER_BATCH,
         n_eval_samples=12)

    # n_steps=4 -> two outer cycles, hence two c_t calls, both enlarged.
    assert seen["c_t_row_counts"] == [2 * OUTER_BATCH, 2 * OUTER_BATCH]
    assert seen["base_batches"].count(2 * OUTER_BATCH) == 2


def test_buffer_size_and_composition_unchanged(tmp_path, monkeypatch):
    """The replay buffer must receive exactly the first outer_batch rows of
    the enlarged rollout (uniform subset, no selection bias): same row
    count, same t_idx range, same contents as a slice of the big run."""
    captured = {}

    real_c_t = swap_training.compute_c_t_grid_swap

    def c_t_spy(t_grid, x_traj, target, head, *, mode):
        captured["full_traj"] = x_traj
        return real_c_t(t_grid, x_traj, target, head, mode=mode)

    real_append = swap_training._append_replay_buffer

    def append_spy(x_chunks, t_idx_chunks, x_traj, t_idx_buffer, cycles):
        captured["buffer_traj"] = x_traj
        captured["t_idx_buffer"] = t_idx_buffer
        x_buffer, t_idx = real_append(
            x_chunks, t_idx_chunks, x_traj, t_idx_buffer, cycles
        )
        captured["buffer_size"] = x_buffer.shape[0]
        return x_buffer, t_idx

    monkeypatch.setattr(swap_training, "compute_c_t_grid_swap", c_t_spy)
    monkeypatch.setattr(swap_training, "_append_replay_buffer", append_spy)

    run_dir = tmp_path / "run"
    _run(run_dir, n_steps=8, head=_head(init_seed=0), c_t_batch=2 * OUTER_BATCH)

    full = captured["full_traj"]           # last cycle: (T, 2*outer_batch, D)
    buf = captured["buffer_traj"]          # (T, outer_batch, D)
    n_grid = full.shape[0]
    assert full.shape[1] == 2 * OUTER_BATCH
    assert buf.shape[1] == OUTER_BATCH
    assert torch.equal(buf, full[:, :OUTER_BATCH])
    assert captured["t_idx_buffer"].numel() == n_grid * OUTER_BATCH
    # replay_buffer_cycles=2 -> the live buffer holds two cycles of the
    # OFF-run row count, never c_t_batch rows.
    assert captured["buffer_size"] == 2 * n_grid * OUTER_BATCH


def test_c_t_batch_below_outer_batch_rejected(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="c_t_batch"):
        _run(run_dir, n_steps=2, head=_head(init_seed=0),
             c_t_batch=OUTER_BATCH - 1)
    # A non-integer row count is nonsense for a rollout draw.
    with pytest.raises((ValueError, TypeError)):
        _run(tmp_path / "run2", n_steps=2, head=_head(init_seed=0),
             c_t_batch=OUTER_BATCH + 0.5)


def test_resumed_run_bit_exact_with_knob_on(tmp_path):
    """The knob is stateless; resume bit-exactness must survive it (this
    pins that the enlarged base draw's RNG consumption is carried by the
    checkpointed RNG state)."""
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"

    _run(uninterrupted_dir, n_steps=8, head=_head(init_seed=0),
         c_t_batch=2 * OUTER_BATCH)

    _run(interrupted_dir, n_steps=4, head=_head(init_seed=0),
         c_t_batch=2 * OUTER_BATCH)
    (interrupted_dir / "checkpoints" / "final.pt").unlink()
    _run(interrupted_dir, n_steps=8, head=_head(init_seed=999),
         c_t_batch=2 * OUTER_BATCH)

    _assert_logs_bit_identical(uninterrupted_dir, interrupted_dir, n_rows=8)
