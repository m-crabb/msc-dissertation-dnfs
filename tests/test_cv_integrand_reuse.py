"""c_t CV-grid integrand reuse from rollout forwards (swap and flip routes).

In control_variate mode the outer step re-runs the head/model on
(trajectory[k], t_k) for every grid slot k — but rollout step k already
computed that exact forward on those tensors and threw it away (the swap
step returns its pair scores; the flip LE step holds G_t before the relu).
`return_cv_integrand=True` accumulates ξ_t (Eq. 8) into a (T, B) buffer
DURING the rollout; only the final slot needs one fresh forward. At d256
this removes 127 of 128 c_t-grid head forwards per outer cycle (~7-8 h
eager per 16x16 CV run).

What correct looks like, independent of implementation:

1. **BIT-IDENTICAL, not close.** No RNG is touched and the arithmetic is
   unchanged, so (a) the trajectory equals a flag-off rollout under the
   same seed, and (b) the returned integrand equals the sequential
   (chunk_rows=None) grid recompute on that trajectory — `torch.equal`,
   for the swap sampler (both step kinds) and the flip sampler (LE and
   non-LE models).
2. **Guards refuse loudly.** The flag without `return_all_states`,
   without `target`, or with resampling enabled is a contract error —
   the reuse is only certified for the plain buffer rollout.
3. **The eval path stays bit-exact.** The flip LE refactor (Euler step
   returns G_t instead of relu(G_t); ξ_t reuses it) must leave
   `sample_ctmc(return_log_weights=True)` bit-equal to a fresh-forward
   reference loop under the same seed — the same reuse contract the swap
   sampler already pins in test_swap_perf_refactors.py.
4. **Trainer wiring, both trainers.** `train_cfg.c_t_from_rollout=True`
   (getattr default False: every archived config is untouched) makes a
   control-variate run skip the grid recompute entirely, with a training
   log bit-identical to the knob-off sequential path. Setting the knob
   together with rollout resampling refuses loudly.
5. **Free rider.** In naive_mc mode the variance bookkeeping's
   `dt_log_p_tilde_t` recompute is the integrand itself; with the knob on
   the trainer reuses it, so `cv_var_ratio` is exactly 1.0 and the two
   variance columns are exactly equal.
"""

import csv
from types import SimpleNamespace

import pytest
import torch
from experiments.dnfs_baseline_01.configs import CurriculumStageCfg

from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers import swap_training, training
from discrete_flow_sampler.samplers.ctmc import (
    _euler_step,
    compute_xi_t,
    sample_ctmc,
)
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid
from discrete_flow_sampler.samplers.resampling import ResamplingConfig
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_c_t_grid_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)

# --------------------------------------------------------------------------
# Fixtures


def _swap_fixture():
    torch.manual_seed(0)
    head = LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    return head, target


def _flip_le_fixture():
    torch.manual_seed(0)
    model = LeMLPRateMatrix(d=4, vocab_size=2, hidden_dim=16, n_summands=2)
    target = IsingTarget(D=2, sigma=0.1)
    return model, target


def _flip_general_fixture():
    torch.manual_seed(0)
    model = MLPRateMatrix(d=4, hidden_dim=16, n_layers=2)
    target = IsingTarget(D=2, sigma=0.1)
    return model, target


TS = torch.linspace(0.0, 1.0, 9)


# --------------------------------------------------------------------------
# 1. Bit-identity of the reused integrand — swap sampler


@torch.no_grad()
@pytest.mark.parametrize("multi_event", [False, True])
def test_swap_reuse_bit_identical_to_sequential_grid(multi_event):
    head, target = _swap_fixture()
    torch.manual_seed(7)
    x0 = target.sample_base(8, device="cpu")

    torch.manual_seed(11)
    trajectory_off = sample_swap_ctmc(
        head,
        x0,
        TS,
        return_all_states=True,
        target=target,
        multi_event=multi_event,
    )
    torch.manual_seed(11)
    trajectory_on, integrand = sample_swap_ctmc(
        head,
        x0,
        TS,
        return_all_states=True,
        target=target,
        multi_event=multi_event,
        return_cv_integrand=True,
    )

    # (a) Samples untouched: the flag consumes no RNG and changes no state.
    assert torch.equal(trajectory_on, trajectory_off)

    # (b) The integrand IS the sequential grid recompute, bit for bit.
    want_c_t, want_integrand = compute_c_t_grid_swap(
        TS,
        trajectory_off,
        target,
        head,
        mode="control_variate",
        chunk_rows=None,
    )
    assert integrand.shape == (len(TS), 8)
    assert torch.equal(integrand, want_integrand)
    assert torch.equal(integrand.mean(dim=-1), want_c_t)


# --------------------------------------------------------------------------
# 1b. Bit-identity — flip sampler, LE and non-LE models


@torch.no_grad()
@pytest.mark.parametrize("fixture", [_flip_le_fixture, _flip_general_fixture])
def test_flip_reuse_bit_identical_to_grid(fixture):
    model, target = fixture()
    torch.manual_seed(7)
    x0 = target.sample_base(8, device="cpu")

    torch.manual_seed(11)
    trajectory_off = sample_ctmc(model, x0, TS, return_all_states=True, target=target)
    torch.manual_seed(11)
    trajectory_on, integrand = sample_ctmc(
        model,
        x0,
        TS,
        return_all_states=True,
        target=target,
        return_cv_integrand=True,
    )

    assert torch.equal(trajectory_on, trajectory_off)

    want_c_t, want_integrand = compute_c_t_grid(
        TS, trajectory_off, target, model, mode="control_variate"
    )
    assert integrand.shape == (len(TS), 8)
    assert torch.equal(integrand, want_integrand)
    assert torch.equal(integrand.mean(dim=-1), want_c_t)


# --------------------------------------------------------------------------
# 2. Guards


@torch.no_grad()
def test_swap_reuse_guards():
    head, target = _swap_fixture()
    x0 = target.sample_base(4, device="cpu")
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_swap_ctmc(
            head, x0, TS, target=target, return_cv_integrand=True
        )  # no return_all_states
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_swap_ctmc(
            head, x0, TS, return_all_states=True, return_cv_integrand=True
        )  # no target
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_swap_ctmc(
            head,
            x0,
            TS,
            return_all_states=True,
            target=target,
            return_cv_integrand=True,
            resampling=ResamplingConfig(ess_threshold_fraction=0.5),
        )


@torch.no_grad()
def test_flip_reuse_guards():
    model, target = _flip_le_fixture()
    x0 = target.sample_base(4, device="cpu")
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_ctmc(model, x0, TS, target=target, return_cv_integrand=True)
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_ctmc(model, x0, TS, return_all_states=True, return_cv_integrand=True)
    with pytest.raises(ValueError, match="return_cv_integrand"):
        sample_ctmc(
            model,
            x0,
            TS,
            return_all_states=True,
            target=target,
            return_cv_integrand=True,
            resampling=ResamplingConfig(ess_threshold_fraction=0.5),
        )


# --------------------------------------------------------------------------
# 3. The flip eval path stays bit-exact after the LE G_t-reuse refactor


@torch.no_grad()
def _reference_sample_ctmc_log_weights(model, x0, ts, target):
    """The pre-refactor eval loop: a FRESH forward inside every ξ_t call.

    RNG consumption matches production (only the Euler step draws), so a
    shared seed makes the two paths comparable bit for bit.
    """
    state = x0.clone()
    batch_size = state.shape[0]
    log_weights = torch.zeros(batch_size, dtype=state.dtype)
    for step in range(len(ts) - 1):
        t_per_batch = ts[step].expand(batch_size)
        step_dt = ts[step + 1] - ts[step]
        new_state, _ = _euler_step(model, state, t_per_batch, step_dt)
        xi_t = compute_xi_t(state, t_per_batch, model, target)
        log_weights = log_weights + xi_t * step_dt
        state = new_state
    return state, log_weights


@torch.no_grad()
@pytest.mark.parametrize("fixture", [_flip_le_fixture, _flip_general_fixture])
def test_flip_eval_log_weights_bit_exact_vs_fresh_forward_reference(fixture):
    model, target = fixture()
    torch.manual_seed(7)
    x0 = target.sample_base(8, device="cpu")

    torch.manual_seed(13)
    want_x, want_w = _reference_sample_ctmc_log_weights(model, x0, TS, target)
    torch.manual_seed(13)
    got_x, got_w = sample_ctmc(model, x0, TS, return_log_weights=True, target=target)
    assert torch.equal(got_x, want_x)
    assert torch.equal(got_w, want_w)


# --------------------------------------------------------------------------
# 4. Trainer wiring — hard (train_swap)


TWO_STAGE_CURRICULUM = (
    CurriculumStageCfg(start_step=0, sigma=0.1, lr=1e-3),
    CurriculumStageCfg(start_step=2, sigma=0.223, lr=3e-4),
)


def _swap_head(init_seed: int) -> LeTFMaskOneSwapHead:
    torch.manual_seed(init_seed)
    return LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _swap_train_cfg(c_t_from_rollout: bool, **extra):
    return SimpleNamespace(
        n_steps=4,
        batch_size=8,
        outer_batch_size=8,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=2,
        grad_clip_max_norm=500.0,
        warmup_steps=0,
        resume_every_outer=1,
        c_t_grid_chunk_rows=None,
        c_t_from_rollout=c_t_from_rollout,
        **extra,
    )


def _run_swap(run_dir, c_t_from_rollout, estimator_mode="control_variate", **extra):
    target = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_swap(
        _swap_head(init_seed=0),
        target,
        _swap_train_cfg(c_t_from_rollout, **extra),
        SimpleNamespace(n_euler_steps=8),
        SimpleNamespace(eval_every=2, n_eval_samples=16),
        run_dir,
        use_wandb=False,
        estimator_mode=estimator_mode,
        sigma_curriculum=TWO_STAGE_CURRICULUM,
    )


def _log_rows(run_dir):
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def _assert_logs_bit_identical(rows_a, rows_b):
    assert len(rows_a) == len(rows_b) == 4
    for a, b in zip(rows_a, rows_b):
        a.pop("wall_clock_step_s")
        b.pop("wall_clock_step_s")
        assert a == b


def test_swap_trainer_reuse_is_bit_identical_to_sequential(tmp_path):
    _run_swap(tmp_path / "reuse", c_t_from_rollout=True)
    _run_swap(tmp_path / "sequential", c_t_from_rollout=False)
    _assert_logs_bit_identical(
        _log_rows(tmp_path / "reuse"), _log_rows(tmp_path / "sequential")
    )


def test_swap_trainer_reuse_skips_the_grid_recompute(tmp_path, monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "compute_c_t_grid_swap must not run when c_t_from_rollout is on "
            "in control_variate mode"
        )

    monkeypatch.setattr(swap_training, "compute_c_t_grid_swap", _must_not_be_called)
    _run_swap(tmp_path / "reuse", c_t_from_rollout=True)
    assert len(_log_rows(tmp_path / "reuse")) == 4


def test_swap_trainer_reuse_with_resampling_refuses(tmp_path):
    with pytest.raises(ValueError, match="c_t_from_rollout"):
        _run_swap(
            tmp_path / "clash",
            c_t_from_rollout=True,
            rollout_resample_ess_fraction=0.5,
        )


def test_swap_trainer_naive_mode_free_rider(tmp_path):
    """Naive mode with the knob: the grid still runs (its integrand is
    target-only), but the variance bookkeeping reuses it — the ratio must
    be EXACTLY 1.0, the two variance columns exactly equal."""
    _run_swap(tmp_path / "naive", c_t_from_rollout=True, estimator_mode="naive_mc")
    for row in _log_rows(tmp_path / "naive"):
        assert float(row["cv_var_ratio"]) == 1.0
        assert row["var_dt_log_p_tilde"] == row["var_estimator_integrand"]


# --------------------------------------------------------------------------
# 5. Trainer wiring — unconstrained/soft (train)


def _flip_train_cfg(c_t_from_rollout: bool, **extra):
    return SimpleNamespace(
        n_steps=4,
        batch_size=8,
        outer_batch_size=8,
        inner_steps_per_outer=2,
        lr=1e-3,
        seed=0,
        replay_buffer_cycles=1,
        c_t_from_rollout=c_t_from_rollout,
        **extra,
    )


def _run_flip(run_dir, c_t_from_rollout, estimator_mode="control_variate", **extra):
    target = IsingTarget(D=2, sigma=0.1)
    torch.manual_seed(0)
    model = LeMLPRateMatrix(d=4, vocab_size=2, hidden_dim=16, n_summands=2)
    train(
        model=model,
        target=target,
        train_cfg=_flip_train_cfg(c_t_from_rollout, **extra),
        ctmc_cfg=SimpleNamespace(n_euler_steps=8),
        eval_cfg=SimpleNamespace(eval_every=2, n_eval_samples=8),
        output_dir=run_dir,
        use_wandb=False,
        estimator_mode=estimator_mode,
    )


def test_flip_trainer_reuse_is_bit_identical_to_recompute(tmp_path):
    _run_flip(tmp_path / "reuse", c_t_from_rollout=True)
    _run_flip(tmp_path / "recompute", c_t_from_rollout=False)
    _assert_logs_bit_identical(
        _log_rows(tmp_path / "reuse"), _log_rows(tmp_path / "recompute")
    )


def test_flip_trainer_reuse_skips_the_grid_recompute(tmp_path, monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "compute_c_t_grid must not run when c_t_from_rollout is on in "
            "control_variate mode"
        )

    monkeypatch.setattr(training, "compute_c_t_grid", _must_not_be_called)
    _run_flip(tmp_path / "reuse", c_t_from_rollout=True)
    assert len(_log_rows(tmp_path / "reuse")) == 4


def test_flip_trainer_reuse_with_resampling_refuses(tmp_path):
    with pytest.raises(ValueError, match="c_t_from_rollout"):
        _run_flip(
            tmp_path / "clash",
            c_t_from_rollout=True,
            rollout_resample_ess_fraction=0.5,
        )


def test_flip_trainer_naive_mode_free_rider(tmp_path):
    # No cv_var_ratio column here — the CV-inversion observer is
    # swap-trainer-only; exact equality of the parents is the same pin.
    _run_flip(tmp_path / "naive", c_t_from_rollout=True, estimator_mode="naive_mc")
    for row in _log_rows(tmp_path / "naive"):
        assert row["var_dt_log_p_tilde"] == row["var_estimator_integrand"]


# --------------------------------------------------------------------------
# 6. The new-base-recipe transform (hard route)


def test_optimised_recipe_flips_only_the_declared_flags():
    """`optimised_recipe` is the transform that lands these
    optimisations: exactly compile_head and train.c_t_from_rollout flip,
    every other field is untouched (twin discipline — the transform must
    never smuggle a third change into a new cell)."""
    from dataclasses import fields

    from experiments.constrained_hard_03.configs import (
        CONFIGS as HARD_CONFIGS,
    )
    from experiments.constrained_hard_03.configs import (
        optimised_recipe,
    )

    base = HARD_CONFIGS["H2_d64_c50_s223_letf_mo"]
    optimised = optimised_recipe(base)
    assert base.compile_head is False
    assert optimised.compile_head is True
    assert optimised.train.c_t_from_rollout is True
    for field in fields(base):
        if field.name in ("compile_head", "train"):
            continue
        assert getattr(optimised, field.name) == getattr(base, field.name)
    for field in fields(base.train):
        if field.name == "c_t_from_rollout":
            continue
        assert getattr(optimised.train, field.name) == getattr(base.train, field.name)
