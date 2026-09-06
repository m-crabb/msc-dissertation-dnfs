"""Pin for outer_batch_size decoupled BELOW batch_size (2026-08-19).

Every archived run left `outer_batch_size=None`, so one number set the
gradient batch, the rollout/buffer width, and the c_t sample size at once.
The first cells to separate them set outer_batch_size < batch_size with
c_t_batch pinned at the old coupled width. The existing c_t_batch tests
only ever exercise outer_batch == batch_size, so this file pins the one
new mechanism before any such cell is trusted:

1. **The replay buffer's row count reflects outer_batch** — per cycle it
   gains n_euler_steps x outer_batch rows, NOT batch_size or c_t_batch
   rows. The buffer takes the first outer_batch rows of the rollout;
   c_t_batch only widens the no-grad c_t estimate, and batch_size only
   sizes the with-replacement inner draws FROM the buffer.
2. **c_t still sees the full c_t_batch rows** while the buffer is narrow —
   the decoupling cuts rollout retention without cutting the per-slot
   normaliser's sample size.
3. **Training runs to completion** with an inner batch larger than the
   per-cycle buffer contribution (draws are with replacement, so a narrow
   buffer must not starve the inner step).

Harness mirrors tests/test_c_t_batch.py: tiny 4x4 (d=16) mask-one head,
bare config bags, two outer cycles, spies on the buffer append and the
c_t grid call.
"""

import csv

import torch

from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_training
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# All three widths distinct, so a wrong wiring cannot pass by coincidence.
OUTER_BATCH = 4
INNER_BATCH = 8
C_T_BATCH = 16
N_EULER_STEPS = 8
REPLAY_CYCLES = 2


def test_buffer_rows_reflect_outer_batch_not_batch_size_or_c_t_batch(
    tmp_path, monkeypatch
):
    captured = {"append_widths": [], "c_t_widths": [], "buffer_rows": []}

    real_append = swap_training._append_replay_buffer

    def append_spy(x_chunks, t_idx_chunks, x_traj, t_idx_buffer, cycles):
        captured["append_widths"].append(x_traj.shape[1])
        x_buffer, t_idx = real_append(
            x_chunks, t_idx_chunks, x_traj, t_idx_buffer, cycles
        )
        captured["buffer_rows"].append(x_buffer.shape[0])
        assert t_idx.numel() == x_buffer.shape[0]
        return x_buffer, t_idx

    real_c_t = swap_training.compute_c_t_grid_swap

    def c_t_spy(t_grid, x_traj, target, head, *, mode, chunk_rows=None):
        captured["c_t_widths"].append(x_traj.shape[1])
        return real_c_t(t_grid, x_traj, target, head, mode=mode, chunk_rows=chunk_rows)

    monkeypatch.setattr(swap_training, "_append_replay_buffer", append_spy)
    monkeypatch.setattr(swap_training, "compute_c_t_grid_swap", c_t_spy)

    torch.manual_seed(0)
    head = LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg = _Cfg(
        n_steps=4,
        batch_size=INNER_BATCH,
        outer_batch_size=OUTER_BATCH,
        c_t_batch=C_T_BATCH,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=REPLAY_CYCLES,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
        resume_every_outer=1,
    )
    ctmc_cfg = _Cfg(n_euler_steps=N_EULER_STEPS)
    eval_cfg = _Cfg(eval_every=4, n_eval_samples=8)

    run_dir = tmp_path / "run"
    train_swap(
        head,
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        run_dir,
        use_wandb=False,
        estimator_mode="naive_mc",
    )

    # Two outer cycles (n_steps=4 at 2 inner steps per cycle).
    # The buffer receives outer_batch-wide chunks, never the gradient
    # batch's or the rollout's width...
    assert captured["append_widths"] == [OUTER_BATCH, OUTER_BATCH]
    # ...while the c_t grid saw the full enlarged rollout both cycles.
    assert captured["c_t_widths"] == [C_T_BATCH, C_T_BATCH]
    # Live buffer rows: one cycle's chunk, then two retained cycles.
    per_cycle_rows = N_EULER_STEPS * OUTER_BATCH
    assert captured["buffer_rows"] == [per_cycle_rows, 2 * per_cycle_rows]

    # The run completed: inner draws (batch 8, with replacement) from the
    # 32-row first-cycle buffer logged all 4 steps.
    with (run_dir / "training_log.csv").open() as log_file:
        assert len(list(csv.DictReader(log_file))) == 4
