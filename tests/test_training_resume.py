"""Preemption-resume contract for the flip trainer `samplers.training.train`.

The swap trainer has had this since a Modal GPU recall; the flip trainer
did not, and a matched-base family paid for it — eight runs killed at ~94%
of a 50k-step budget with nothing on disk but a
weights-only `latest.pt`, so every one of them had to start again from zero.

The contract these tests pin:

1. **Bit-exact continuation.** A run interrupted at an outer-cycle boundary
   and resumed produces the SAME trajectory as an uninterrupted run —
   bit-for-bit on CPU fp32. That requires the checkpoint to carry model +
   optimiser (AdamW moments), the step counter, the torch RNG states, and
   the replay buffer, which with `replay_buffer_cycles > 1` holds past outer
   trajectories that cannot be reconstructed from anything else.
2. **All four replay lists travel.** The flip trainer's amortised path
   retains per-state composition and per-state c_t baselines alongside the
   states and time indices. Dropping either would still train — on states
   paired with the WRONG baseline — which is the failure mode that produces
   a plausible number rather than a crash, so it gets its own test.
3. **Curriculum fast-forward.** Stage indices are derived from the step
   counter, not stored. The amortised test crosses a σ boundary BEFORE the
   interruption (so the fast-forward must land on it) and λ and composition
   boundaries AFTER it (so the ordinary transition loop must still fire).
4. **Arming resume does not perturb an uninterrupted run.** Every archived
   flip run must stay comparable to runs from the new trainer, so writing a
   checkpoint must not touch the RNG stream or anything else the trajectory
   depends on.
5. **Orphan log rows.** A dead attempt flushes a CSV row every inner step but
   checkpoints only every `resume_every_outer` cycles; resume must drop the
   rows past the checkpoint so every step appears exactly once.
"""
import csv
from types import SimpleNamespace

import pytest
import torch

from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import IsingTarget

from experiments.dnfs_baseline_01.configs import (
    CompositionCurriculumStageCfg,
    CurriculumStageCfg,
    LambdaCurriculumStageCfg,
)

# n_steps=12 at 2 inner steps per outer => outer cycles begin at steps
# 0, 2, 4, 6, 8, 10. The interruption lands at the step-6 boundary, which
# puts the σ stage before it and the λ / composition stages after it.
N_STEPS = 12
INNER_PER_OUTER = 2
INTERRUPT_STEP = 6

SIGMA_STAGES = (
    CurriculumStageCfg(start_step=0, sigma=0.1, lr=1e-3),
    CurriculumStageCfg(start_step=4, sigma=0.223, lr=3e-4),
)
LAMBDA_STAGES = (
    LambdaCurriculumStageCfg(start_step=0, composition_penalty_strength=5.0),
    LambdaCurriculumStageCfg(start_step=8, composition_penalty_strength=25.0),
)
COMPOSITION_STAGES = (
    CompositionCurriculumStageCfg(start_step=0, half_width=0.05),
    CompositionCurriculumStageCfg(start_step=10, half_width=0.2),
)


class _Preempted(Exception):
    """Stands in for the container dying with no chance to clean up."""


def _die_after(n_checkpoints: int):
    """`on_checkpoint` hook that raises once `n_checkpoints` have landed.

    Raising *from the hook* rather than truncating `n_steps` is what makes
    the twin a faithful preemption: resume.pt is already on disk (the hook
    fires after the write), the CSV holds whatever was flushed, and none of
    the end-of-run artefacts exist.
    """
    calls = {"n": 0}

    def hook():
        calls["n"] += 1
        if calls["n"] >= n_checkpoints:
            raise _Preempted
    return hook


def _cfgs(*, seed=0, resume_every_outer=1, n_steps=N_STEPS):
    train_cfg = SimpleNamespace(
        n_steps=n_steps, inner_steps_per_outer=INNER_PER_OUTER,
        batch_size=8, outer_batch_size=4, replay_buffer_cycles=2,
        lr=1e-3, seed=seed, grad_clip_max_norm=500.0, warmup_steps=0,
        resume_every_outer=resume_every_outer,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=3)
    # Fires at steps 0, 4 and 8 -- the last one after the resume, so the
    # eval draw's RNG consumption is pinned across the interruption too.
    eval_cfg = SimpleNamespace(eval_every=4, n_eval_samples=8)
    return train_cfg, ctmc_cfg, eval_cfg


def _model(target, *, conditioned, init_seed):
    torch.manual_seed(init_seed)
    return LeTFRateMatrix(
        d=target.d, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2,
        condition_on_composition=conditioned,
    )


def _specialist_target():
    return IsingTarget(D=2, sigma=0.1)


def _soft_target():
    return IsingTarget(
        D=2, sigma=0.1, target_composition=0.5,
        composition_penalty_strength=5.0,
    )


def _run_specialist(run_dir, *, init_seed, on_checkpoint=None, **cfg_kw):
    target = _specialist_target()
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(**cfg_kw)
    train(
        model=_model(target, conditioned=False, init_seed=init_seed),
        target=target, train_cfg=train_cfg, ctmc_cfg=ctmc_cfg,
        eval_cfg=eval_cfg, output_dir=run_dir, use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=SIGMA_STAGES,
        on_checkpoint=on_checkpoint,
    )


def _run_amortised(run_dir, *, init_seed, on_checkpoint=None, **cfg_kw):
    target = _soft_target()
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(**cfg_kw)
    train(
        model=_model(target, conditioned=True, init_seed=init_seed),
        target=target, train_cfg=train_cfg, ctmc_cfg=ctmc_cfg,
        eval_cfg=eval_cfg, output_dir=run_dir, use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=SIGMA_STAGES,
        lambda_curriculum=LAMBDA_STAGES,
        composition_curriculum=COMPOSITION_STAGES,
        composition_centre=0.5, composition_half_width=0.05,
        on_checkpoint=on_checkpoint,
    )


def _rows(run_dir):
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def _assert_trajectories_identical(reference_dir, resumed_dir):
    reference_rows, resumed_rows = _rows(reference_dir), _rows(resumed_dir)
    assert len(reference_rows) == len(resumed_rows) == N_STEPS
    for reference, resumed in zip(reference_rows, resumed_rows):
        # String equality on the CSV cells is bit-exactness on the floats.
        # The sigma / lr / composition columns double as the assertion that
        # the curriculum fast-forward landed on the right stage.
        # wall_clock_step_s is real elapsed time, the one column that can
        # never reproduce.
        reference.pop("wall_clock_step_s")
        resumed.pop("wall_clock_step_s")
        assert reference == resumed

    reference_state = torch.load(
        reference_dir / "checkpoints" / "final.pt", weights_only=True
    )
    resumed_state = torch.load(
        resumed_dir / "checkpoints" / "final.pt", weights_only=True
    )
    assert reference_state.keys() == resumed_state.keys()
    for key in reference_state:
        assert torch.equal(reference_state[key], resumed_state[key]), key


def test_resumed_specialist_run_is_bit_exact_with_uninterrupted(tmp_path):
    reference_dir, interrupted_dir = tmp_path / "ref", tmp_path / "int"

    _run_specialist(reference_dir, init_seed=0)

    # Checkpoints land at the end of outer cycles 0, 1, 2 -> the third one
    # records step 6; the hook then kills the process.
    with pytest.raises(_Preempted):
        _run_specialist(interrupted_dir, init_seed=0, on_checkpoint=_die_after(3))
    assert not (interrupted_dir / "checkpoints" / "final.pt").exists()
    resume_state = torch.load(
        interrupted_dir / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume_state["step"] == INTERRUPT_STEP

    # Resume from a DIFFERENTLY-initialised model: an incomplete restore
    # cannot then coincidentally reproduce the reference.
    _run_specialist(interrupted_dir, init_seed=999)

    _assert_trajectories_identical(reference_dir, interrupted_dir)


def test_resumed_amortised_run_is_bit_exact_with_uninterrupted(tmp_path):
    """The four-replay-list contract, plus a σ stage crossed before the
    interruption and λ / composition stages crossed after it."""
    reference_dir, interrupted_dir = tmp_path / "ref", tmp_path / "int"

    _run_amortised(reference_dir, init_seed=0)

    with pytest.raises(_Preempted):
        _run_amortised(interrupted_dir, init_seed=0, on_checkpoint=_die_after(3))

    resume_state = torch.load(
        interrupted_dir / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume_state["step"] == INTERRUPT_STEP
    # Both amortised-only lists must be present and populated, or the
    # continuation trains states against baselines they never came with.
    for key in ("composition_replay_chunks", "c_t_replay_chunks"):
        assert resume_state[key], f"{key} missing from resume.pt"

    _run_amortised(interrupted_dir, init_seed=999)

    _assert_trajectories_identical(reference_dir, interrupted_dir)


def test_arming_resume_does_not_perturb_an_uninterrupted_run(tmp_path):
    """Archived-run comparability guard.

    Writing a resume checkpoint must not consume randomness or otherwise
    move the trajectory: a run that checkpoints every outer cycle and one
    that never checkpoints at all have to agree step for step. Without this
    every flip run in the dissertation becomes incomparable to runs from
    the new trainer.
    """
    every_cycle_dir, rarely_dir = tmp_path / "every", tmp_path / "rarely"

    every_writes, rare_writes = [], []
    _run_amortised(
        every_cycle_dir, init_seed=0, resume_every_outer=1,
        on_checkpoint=lambda: every_writes.append(1),
    )
    _run_amortised(
        rarely_dir, init_seed=0, resume_every_outer=10_000,
        on_checkpoint=lambda: rare_writes.append(1),
    )

    # Cadence: one per outer cycle vs the end-of-run checkpoint alone (which
    # always fires, so a run killed before `final.pt` stays recoverable).
    assert len(every_writes) == N_STEPS // INNER_PER_OUTER
    assert len(rare_writes) == 1
    _assert_trajectories_identical(every_cycle_dir, rarely_dir)


def test_resume_truncates_orphan_log_rows(tmp_path):
    """Rows flushed after the last checkpoint describe steps the resumed run
    redoes; they must not survive to be counted twice."""
    run_dir = tmp_path / "run"

    with pytest.raises(_Preempted):
        _run_specialist(run_dir, init_seed=0, on_checkpoint=_die_after(3))

    # Forge the rows a death mid-cycle-3 would have flushed past step 6.
    log_path = run_dir / "training_log.csv"
    with log_path.open() as log_file:
        rows = list(csv.reader(log_file))
    header, body = rows[0], rows[1:]
    assert [int(row[0]) for row in body] == list(range(INTERRUPT_STEP))
    orphan = list(body[-1])
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)
        writer.writerows(body)
        for step in (INTERRUPT_STEP, INTERRUPT_STEP + 1):
            orphan[0] = str(step)
            writer.writerow(orphan)

    _run_specialist(run_dir, init_seed=0)

    steps = [int(row["step"]) for row in _rows(run_dir)]
    assert steps == list(range(N_STEPS)), "each step must appear exactly once"


def test_resume_past_n_steps_writes_final_without_training(tmp_path):
    """A run killed between its last inner step and `final.pt` is retried by
    Modal with identical inputs; it must finish, not train a second time."""
    run_dir = tmp_path / "run"

    # Kill on the last checkpoint (end of the final outer cycle), which
    # records step == n_steps.
    with pytest.raises(_Preempted):
        _run_specialist(run_dir, init_seed=0, on_checkpoint=_die_after(6))
    resume_state = torch.load(
        run_dir / "checkpoints" / "resume.pt", weights_only=True
    )
    assert resume_state["step"] == N_STEPS
    assert not (run_dir / "checkpoints" / "final.pt").exists()

    _run_specialist(run_dir, init_seed=999)

    assert [int(row["step"]) for row in _rows(run_dir)] == list(range(N_STEPS))
    final_state = torch.load(
        run_dir / "checkpoints" / "final.pt", weights_only=True
    )
    for key, tensor in resume_state["model"].items():
        assert torch.equal(final_state[key], tensor), key
