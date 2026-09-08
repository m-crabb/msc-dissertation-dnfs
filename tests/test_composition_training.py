"""Tests for amortised training — one model, many target compositions.

The correctness crux is the ∂_t log Z_t baseline. The annealing path makes
Z_t depend on the target composition, so a baseline averaged over a batch
spanning several compositions mixes incompatible normalisers and biases every
row's residual target. The loop avoids that by drawing one composition per
outer cycle — leaving the average over the full outer batch, exactly as
precise as a specialist run's — and then carrying each state's own
composition and baseline through the replay buffer, so inner batches may
still mix compositions drawn across cycles.

What is pinned here:
  1. The shared b-major alignment rule and the per-cycle draw.
  2. That the retention rule evicts every parallel per-state quantity in
     lockstep, so a state can never be paired with another cycle's baseline.
  3. That a specialist run (no composition conditioning) is untouched —
     same code path, same RNG consumption, reproducible run-to-run.
  4. That the draw window widens on a curriculum, and that compositions stay
     inside the window that was in force.
"""

import csv
from types import SimpleNamespace

import pytest
import torch

from discrete_flow_sampler.composition import draw_composition, expand_b_major
from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.training import _retain_chunks, train
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)


def _tiny_cfgs(*, n_steps=4, inner=2, seed=0):
    train_cfg = SimpleNamespace(
        n_steps=n_steps,
        inner_steps_per_outer=inner,
        batch_size=8,
        outer_batch_size=4,
        replay_buffer_cycles=2,
        lr=1e-3,
        seed=seed,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=3)
    eval_cfg = SimpleNamespace(eval_every=1_000_000, n_eval_samples=4)
    return train_cfg, ctmc_cfg, eval_cfg


def _soft_target():
    return IsingTarget(
        D=2,
        sigma=0.1,
        target_composition=0.5,
        composition_penalty_strength=5.0,
    )


def _model(target, *, conditioned):
    torch.manual_seed(0)
    return LeTFRateMatrix(
        d=target.d,
        vocab_size=2,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        condition_on_composition=conditioned,
    )


def _rows(output_dir):
    with (output_dir / "training_log.csv").open() as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Shared helpers — these serve both constraint routes
# --------------------------------------------------------------------------


def test_expand_b_major_aligns_and_rejects_ragged():
    values = torch.tensor([0.3, 0.8])
    torch.testing.assert_close(
        expand_b_major(values, 6), torch.tensor([0.3, 0.3, 0.3, 0.8, 0.8, 0.8])
    )
    torch.testing.assert_close(expand_b_major(values, 2), values)
    with pytest.raises(ValueError, match="b-major"):
        expand_b_major(values, 5)


def test_draw_composition_respects_window_and_value_set():
    generator = torch.Generator().manual_seed(0)
    draws = [draw_composition(0.5, 0.2, None, generator=generator) for _ in range(200)]
    assert all(0.3 - 1e-9 <= c <= 0.7 + 1e-9 for c in draws)
    assert max(draws) - min(draws) > 0.2, "window is not being explored"

    values = (0.3, 0.8)
    picked = {
        draw_composition(0.5, 0.2, values, generator=generator) for _ in range(50)
    }
    assert picked == {0.3, 0.8}


def test_draw_composition_quantises_onto_the_realisable_lattice():
    """A fixed-composition target has no slice unless c·d is an integer."""
    target = FixedCompositionIsingTarget(D=3, sigma=0.1, target_composition=1 / 3)
    assert target.composition_quantum == 9
    assert IsingTarget(D=3, sigma=0.1).composition_quantum is None

    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        composition = draw_composition(
            0.5,
            0.3,
            None,
            quantise_to=target.composition_quantum,
            generator=generator,
        )
        n_plus = composition * target.d
        assert abs(n_plus - round(n_plus)) < 1e-9


def test_adapter_expands_composition_and_forwards_attributes():
    target = _soft_target()
    model = _model(target, conditioned=True)
    adapter = CompositionConditioned(model, torch.tensor([0.3, 0.8]))

    assert adapter.is_locally_equivariant is True
    assert adapter.vocab_size == model.vocab_size
    assert adapter.d == model.d

    x = torch.randint(0, 2, (6, target.d)).float() * 2 - 1
    t = torch.rand(6)
    torch.testing.assert_close(
        adapter(x, t), model(x, t, expand_b_major(torch.tensor([0.3, 0.8]), 6))
    )


def test_retain_chunks_evicts_parallel_quantities_in_lockstep():
    """A state must never be paired with another cycle's baseline.

    Alignment is structural: each parallel quantity is retained by the same
    rule, so appends and evictions happen on identical schedules.
    """
    states, baselines = [], []
    for cycle in range(4):
        state_buffer = _retain_chunks(
            states, torch.full((3,), float(cycle)), max_cycles=2
        )
        baseline_buffer = _retain_chunks(
            baselines, torch.full((3,), float(cycle) * 10), max_cycles=2
        )
    assert state_buffer.shape == baseline_buffer.shape == (6,)
    # Only the last two cycles survive, and each state keeps its own baseline.
    torch.testing.assert_close(baseline_buffer, state_buffer * 10)
    torch.testing.assert_close(
        state_buffer, torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    )


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def test_amortised_run_draws_and_logs_a_composition_per_cycle(tmp_path):
    target = _soft_target()
    model = _model(target, conditioned=True)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(n_steps=8, inner=2)

    train(
        model,
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        tmp_path,
        use_wandb=False,
        composition_centre=0.5,
        composition_half_width=0.2,
    )

    rows = _rows(tmp_path)
    assert len(rows) == 8
    drawn = [float(row["composition_current"]) for row in rows]
    assert all(0.3 - 1e-9 <= c <= 0.7 + 1e-9 for c in drawn)
    # Constant within an outer cycle, and not constant across the run.
    assert drawn[0] == drawn[1] and drawn[2] == drawn[3]
    assert len(set(drawn)) > 1, "composition never changed across outer cycles"


def test_amortised_run_rejects_an_unconditioned_model(tmp_path):
    """Silently ignoring c would train an 'amortised' model that never saw it."""
    target = _soft_target()
    model = _model(target, conditioned=False)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()

    with pytest.raises(ValueError, match="composition conditioning"):
        train(
            model,
            target,
            train_cfg,
            ctmc_cfg,
            eval_cfg,
            tmp_path,
            use_wandb=False,
            composition_centre=0.5,
            composition_half_width=0.1,
        )


def test_specialist_run_is_unchanged_and_reproducible(tmp_path):
    """The amortisation machinery must not perturb a specialist run.

    Same seed twice must give an identical loss trace: any stray RNG draw on
    the non-amortised path (an extra composition sample, a different context)
    would show up here, and would invalidate comparisons against the
    archived per-composition runs.
    """
    losses = []
    for index in range(2):
        target = _soft_target()
        model = _model(target, conditioned=False)
        train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(n_steps=4, seed=3)
        output_dir = tmp_path / f"run{index}"
        train(
            model,
            target,
            train_cfg,
            ctmc_cfg,
            eval_cfg,
            output_dir,
            use_wandb=False,
        )
        rows = _rows(output_dir)
        assert all(row["composition_current"] == "nan" for row in rows)
        losses.append([float(row["loss"]) for row in rows])

    assert losses[0] == losses[1]


def test_composition_curriculum_widens_the_draw_window(tmp_path):
    """Widening mirrors the λ anneal: learn where it is easy, then generalise.

    c ≈ 0.5 is the easy end — base and target compositions already agree, so
    the penalty starts near its minimum — while the window edges carry the
    largest gap for the path to close.
    """
    target = _soft_target()
    model = _model(target, conditioned=True)
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(n_steps=40, inner=2)

    stages = (
        SimpleNamespace(start_step=0, half_width=0.02, lr=None),
        SimpleNamespace(start_step=20, half_width=0.3, lr=None),
    )
    train(
        model,
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        tmp_path,
        use_wandb=False,
        composition_centre=0.5,
        composition_half_width=0.02,
        composition_curriculum=stages,
    )

    rows = _rows(tmp_path)
    early = [float(row["composition_current"]) for row in rows[:20]]
    late = [float(row["composition_current"]) for row in rows[20:]]
    assert all(abs(c - 0.5) <= 0.02 + 1e-9 for c in early)
    assert all(abs(c - 0.5) <= 0.3 + 1e-9 for c in late)
    assert max(abs(c - 0.5) for c in late) > 0.02, "window never widened"
    assert [float(row["composition_half_width"]) for row in rows[:20]] == [0.02] * 20
