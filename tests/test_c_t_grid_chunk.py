"""Tests for the batched c_t grid computation.

Motivation: the c_t grid pays n_grid sequential no-grad head calls per
outer cycle; at d256 trajectory+c_t is ~75% of wall.

What correct looks like, independent of implementation:

1. **Off is byte-identical.** ``c_t_grid_chunk_rows = None`` (the default,
   and the value every archived run implicitly carries) must run the
   existing per-slot sequential loop — n_grid integrand calls at
   outer_batch rows each — so the falsification record of every archived
   cell stays valid.
2. **Parity is the gate.** The chunked path computes the SAME quantities
   (the c_t grid and the per-state integrand matrix) with the same fp32
   ops modulo batch-dim blocking, so it must match the sequential path
   within the established 1e-5 batch-blocking class (cf. the SDPA/chunk
   parities in test_swap_perf_refactors.py) in BOTH estimator modes. No
   quality change is permitted — the parity test IS the gate.
3. **The row cap is respected.** No integrand call may see more than
   chunk_rows rows; the call count is ceil(n_rows / chunk_rows) and the
   final call takes the remainder.
4. **Nonsense is rejected.** Non-positive, boolean, or non-integer row
   caps refuse loudly.
5. **The trainer wires the knob.** train_cfg.c_t_grid_chunk_rows reaches
   the compute call every outer cycle; explicit None and the unknobbed
   getattr default are bit-identical training runs. The knob is stateless
   and consumes no RNG, so there is no resume contract beyond wiring
   (contrast test_c_t_batch.py, where the knob enlarges a base DRAW).
"""

import csv
from pathlib import Path

import pytest
import torch
from experiments.dnfs_baseline_01.configs import CurriculumStageCfg

from discrete_flow_sampler.constraints.swap_readout import (
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_ctmc, swap_training
from discrete_flow_sampler.samplers.swap_ctmc import compute_c_t_grid_swap
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


TWO_STAGE_CURRICULUM = (
    CurriculumStageCfg(start_step=0, sigma=0.1, lr=1e-3),
    CurriculumStageCfg(start_step=2, sigma=0.223, lr=3e-4),
)

N_GRID = 8
OUTER_BATCH = 8
N_ROWS = N_GRID * OUTER_BATCH


def _head(init_seed: int) -> LeTFMaskOneSwapHead:
    # Mask-one at d=16: cheap on CPU and head-agnostic for these contracts
    # (same reasoning as tests/test_c_t_batch.py).
    torch.manual_seed(init_seed)
    return LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _fixture():
    """A small on-manifold "trajectory": c_t evaluation is per-state, so
    base draws stand in for trajectory rows (no rollout structure needed)."""
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    torch.manual_seed(0)
    x_traj = target.sample_base(N_ROWS, device="cpu").reshape(N_GRID, OUTER_BATCH, -1)
    t_grid = torch.linspace(0.0, 1.0, N_GRID)
    return t_grid, x_traj, target, _head(init_seed=0)


# --------------------------------------------------------------------------
# Function-level contracts


def test_default_none_runs_the_sequential_loop(monkeypatch):
    """None must keep the per-slot loop: n_grid integrand calls at
    outer_batch rows each, in both modes (the archived behaviour)."""
    t_grid, x_traj, target, head = _fixture()
    seen = {"xi": [], "naive": []}

    real_xi = swap_ctmc.compute_xi_t_swap
    real_naive = FixedCompositionIsingTarget.dt_log_p_tilde_t

    def xi_spy(x, t, head, target):
        seen["xi"].append(x.shape[0])
        return real_xi(x, t, head, target)

    def naive_spy(self, x, t):
        seen["naive"].append(x.shape[0])
        return real_naive(self, x, t)

    monkeypatch.setattr(swap_ctmc, "compute_xi_t_swap", xi_spy)
    monkeypatch.setattr(FixedCompositionIsingTarget, "dt_log_p_tilde_t", naive_spy)

    compute_c_t_grid_swap(t_grid, x_traj, target, head, mode="control_variate")
    assert seen["xi"] == [OUTER_BATCH] * N_GRID

    # xi_t_swap reads dt_log_p_tilde_t internally, so the CV pass also
    # touched the naive spy; reset to isolate the naive-mode pass.
    seen["naive"].clear()
    compute_c_t_grid_swap(t_grid, x_traj, target, head, mode="naive_mc")
    assert seen["naive"] == [OUTER_BATCH] * N_GRID


@pytest.mark.parametrize("mode", ["control_variate", "naive_mc"])
def test_chunked_matches_sequential_parity(mode):
    """THE GATE: chunked must reproduce the sequential c_t grid and the
    per-state integrand matrix within the 1e-5 batch-blocking class — same
    fp32 ops, only the call blocking differs."""
    t_grid, x_traj, target, head = _fixture()
    c_t_ref, integrand_ref = compute_c_t_grid_swap(
        t_grid, x_traj, target, head, mode=mode
    )
    c_t_chunk, integrand_chunk = compute_c_t_grid_swap(
        t_grid, x_traj, target, head, mode=mode, chunk_rows=5
    )
    assert c_t_chunk.shape == c_t_ref.shape
    assert integrand_chunk.shape == integrand_ref.shape
    assert torch.allclose(c_t_chunk, c_t_ref, atol=1e-5)
    assert torch.allclose(integrand_chunk, integrand_ref, atol=1e-5)
    # Internal consistency: the grid is the row-mean of the matrix.
    assert torch.allclose(c_t_chunk, integrand_chunk.mean(dim=-1), atol=1e-6)


def test_chunk_row_cap_respected_and_remainder(monkeypatch):
    """No call exceeds the cap; call count is ceil(n_rows / cap) with the
    remainder in the final call — checked in both modes because the two
    integrand paths chunk independently."""
    t_grid, x_traj, target, head = _fixture()
    seen = {"xi": [], "naive": []}

    real_xi = swap_ctmc.compute_xi_t_swap
    real_naive = FixedCompositionIsingTarget.dt_log_p_tilde_t

    def xi_spy(x, t, head, target):
        seen["xi"].append(x.shape[0])
        return real_xi(x, t, head, target)

    def naive_spy(self, x, t):
        seen["naive"].append(x.shape[0])
        return real_naive(self, x, t)

    monkeypatch.setattr(swap_ctmc, "compute_xi_t_swap", xi_spy)
    monkeypatch.setattr(FixedCompositionIsingTarget, "dt_log_p_tilde_t", naive_spy)

    compute_c_t_grid_swap(
        t_grid, x_traj, target, head, mode="control_variate", chunk_rows=5
    )
    # 64 rows: cap 5 -> 12 full chunks + 4.
    assert seen["xi"] == [5] * 12 + [4]

    # xi_t_swap reads dt_log_p_tilde_t internally, so the CV pass also
    # touched the naive spy; reset to isolate the naive-mode pass.
    seen["naive"].clear()
    compute_c_t_grid_swap(t_grid, x_traj, target, head, mode="naive_mc", chunk_rows=9)
    # cap 9 -> 7 full chunks + 1.
    assert seen["naive"] == [9] * 7 + [1]


def test_chunk_rows_validation(monkeypatch):
    t_grid, x_traj, target, head = _fixture()
    for bad in (0, -3, True, 2.5):
        with pytest.raises((ValueError, TypeError), match="chunk_rows"):
            compute_c_t_grid_swap(
                t_grid,
                x_traj,
                target,
                head,
                mode="naive_mc",
                chunk_rows=bad,
            )
    # Cap of 1 is the extreme of the semantics: one call per row.
    seen = []
    real_naive = FixedCompositionIsingTarget.dt_log_p_tilde_t

    def naive_spy(self, x, t):
        seen.append(x.shape[0])
        return real_naive(self, x, t)

    monkeypatch.setattr(FixedCompositionIsingTarget, "dt_log_p_tilde_t", naive_spy)
    compute_c_t_grid_swap(t_grid, x_traj, target, head, mode="naive_mc", chunk_rows=1)
    assert seen == [1] * N_ROWS


# --------------------------------------------------------------------------
# Trainer wiring


def _cfgs(n_steps: int, c_t_grid_chunk_rows, n_eval_samples: int = 16):
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
        resume_every_outer=1,
        c_t_grid_chunk_rows=c_t_grid_chunk_rows,
    )
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=n_eval_samples)
    return train_cfg, ctmc_cfg, eval_cfg


def _run(run_dir: Path, n_steps: int, head, c_t_grid_chunk_rows) -> None:
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(n_steps, c_t_grid_chunk_rows)
    train_swap(
        head,
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        run_dir,
        use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=TWO_STAGE_CURRICULUM,
    )


def _log_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def test_trainer_wires_the_knob_every_outer_cycle(tmp_path, monkeypatch):
    seen = []
    real_c_t = swap_training.compute_c_t_grid_swap

    def c_t_spy(t_grid, x_traj, target, head, *, mode, chunk_rows=None):
        seen.append(chunk_rows)
        return real_c_t(t_grid, x_traj, target, head, mode=mode, chunk_rows=chunk_rows)

    monkeypatch.setattr(swap_training, "compute_c_t_grid_swap", c_t_spy)
    _run(tmp_path / "run", n_steps=4, head=_head(init_seed=0), c_t_grid_chunk_rows=5)
    # n_steps=4 -> two outer cycles, and the knob must arrive at both.
    assert seen == [5, 5]


def test_explicit_none_is_bit_identical_to_unknobbed(tmp_path):
    none_dir = tmp_path / "explicit_none"
    default_dir = tmp_path / "unknobbed"

    _run(none_dir, n_steps=4, head=_head(init_seed=0), c_t_grid_chunk_rows=None)

    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg, ctmc_cfg, eval_cfg = _cfgs(4, None)
    del train_cfg.c_t_grid_chunk_rows  # the getattr-default path
    train_swap(
        _head(init_seed=0),
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        default_dir,
        use_wandb=False,
        estimator_mode="control_variate",
        sigma_curriculum=TWO_STAGE_CURRICULUM,
    )

    rows_a, rows_b = _log_rows(none_dir), _log_rows(default_dir)
    assert len(rows_a) == len(rows_b) == 4
    for a, b in zip(rows_a, rows_b):
        a.pop("wall_clock_step_s")
        b.pop("wall_clock_step_s")
        assert a == b
