"""Preemption-resume contract for `train_swap` (added after the 2026-07-23
Modal GPU recall restarted the MO 100k run from scratch).

The contract these tests pin:

1. **Bit-exact continuation.** A run interrupted at an outer-cycle boundary
   and resumed must produce the SAME training trajectory as an uninterrupted
   run — bit-for-bit on CPU fp32. This is only achievable because the resume
   checkpoint carries full boundary state: model + optimiser (AdamW moments),
   step counter, torch CPU/CUDA RNG states, and the replay buffer (with
   replay_buffer_cycles > 1 the buffer holds past outer trajectories that are
   not reconstructible). Curriculum stage / warmup / intended lr are NOT
   stored — they are derivable from the step counter because the sigma ladder
   uses absolute start_steps; the tests run a two-stage curriculum across the
   interruption point to pin that fast-forward.
2. **Idempotent completion.** Re-invoking training on a finished run dir
   (Modal retries re-run the function with identical inputs) must be a no-op.
3. **Orphan log rows.** A dead attempt may have flushed log rows PAST the
   last checkpoint; resume must truncate them so every step appears exactly
   once.
"""
import csv
from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
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


def _head(init_seed: int) -> DoublyHollowSwapHead:
    torch.manual_seed(init_seed)
    return DoublyHollowSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _cfgs(n_steps: int):
    train_cfg = _Cfg(n_steps=n_steps, batch_size=8, outer_batch_size=8,
                     inner_steps_per_outer=2, lr=1e-3, seed=0,
                     replay_buffer_cycles=2, grad_clip_max_norm=500.0,
                     warmup_steps=0, resume_every_outer=1)
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=16)
    return train_cfg, ctmc_cfg, eval_cfg


def _run(run_dir: Path, n_steps: int, head) -> None:
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(n_steps)
    train_swap(head, target, train_cfg, ctmc_cfg, eval_cfg, run_dir,
               use_wandb=False, estimator_mode="control_variate",
               sigma_curriculum=TWO_STAGE_CURRICULUM)


def _log_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def test_resumed_run_is_bit_exact_with_uninterrupted(tmp_path):
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"

    # Reference: 8 steps (4 outer cycles) in one go.
    _run(uninterrupted_dir, n_steps=8, head=_head(init_seed=0))

    # Interrupted twin: identical run dying at the step-4 boundary. Running
    # to n_steps=4 leaves exactly the state a preempted 8-step run would
    # have checkpointed there (same seed => same first 4 steps); drop the
    # artefacts only a COMPLETED run writes.
    _run(interrupted_dir, n_steps=4, head=_head(init_seed=0))
    (interrupted_dir / "checkpoints" / "final.pt").unlink()

    # Resume with a DIFFERENTLY-initialised head: if the checkpoint restore
    # were incomplete, the continuation could not match the reference.
    _run(interrupted_dir, n_steps=8, head=_head(init_seed=999))

    reference_rows = _log_rows(uninterrupted_dir)
    resumed_rows = _log_rows(interrupted_dir)
    assert len(reference_rows) == len(resumed_rows) == 8
    for reference, resumed in zip(reference_rows, resumed_rows):
        # String equality on the CSV cells = bit-exact floats. sigma/lr
        # columns also pin the curriculum fast-forward across the resume.
        # wall_clock_step_s is real elapsed time -- the one column that can
        # never reproduce -- so it is excluded.
        reference.pop("wall_clock_step_s")
        resumed.pop("wall_clock_step_s")
        assert reference == resumed

    reference_state = torch.load(
        uninterrupted_dir / "checkpoints" / "final.pt", weights_only=True
    )
    resumed_state = torch.load(
        interrupted_dir / "checkpoints" / "final.pt", weights_only=True
    )
    assert reference_state.keys() == resumed_state.keys()
    for key in reference_state:
        assert torch.equal(reference_state[key], resumed_state[key]), key


def test_retry_on_completed_run_is_noop(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, n_steps=4, head=_head(init_seed=0))
    rows_before = _log_rows(run_dir)
    final_before = torch.load(
        run_dir / "checkpoints" / "final.pt", weights_only=True
    )

    _run(run_dir, n_steps=4, head=_head(init_seed=999))

    assert _log_rows(run_dir) == rows_before
    final_after = torch.load(
        run_dir / "checkpoints" / "final.pt", weights_only=True
    )
    for key in final_before:
        assert torch.equal(final_before[key], final_after[key]), key


def test_resume_truncates_orphan_log_rows(tmp_path):
    run_dir = tmp_path / "run"
    _run(run_dir, n_steps=4, head=_head(init_seed=0))
    (run_dir / "checkpoints" / "final.pt").unlink()

    # A preempted attempt can flush rows past its last checkpoint before
    # dying; fake two such orphans beyond the step-4 boundary.
    with (run_dir / "training_log.csv").open("a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([4] + ["9.9"] * 12)
        writer.writerow([5] + ["9.9"] * 12)

    _run(run_dir, n_steps=8, head=_head(init_seed=999))

    rows = _log_rows(run_dir)
    assert [int(row["step"]) for row in rows] == list(range(8))
    assert all(row["loss"] != "9.9" for row in rows)
