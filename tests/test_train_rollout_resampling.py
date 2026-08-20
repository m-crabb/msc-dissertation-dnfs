"""ESS-triggered SMC resampling inside the TRAINING rollout (both routes).

Written before the implementation — these encode "what correct looks like".
Sections 1-6 pinned the swap route (landed 49dc229); section 7 holds the
flip-route twins, written before `train()` was wired to the same flag.

Background. LEAPS (Algorithm 1, lines 11-14) resamples the walker
population whenever the interim ESS drops below a threshold and resets the
accumulated log-weights, and its training loop (Algorithm 2, line 5) draws
its batch from exactly that routine. Our trainer's buffer rebuild calls
`sample_swap_ctmc(..., return_all_states=True)`, which carries no weights
at all, so the training rollout has never resampled. This suite pins the
opt-in flag that closes that gap.

What correct looks like:

1. OFF IS OFF. Absent flag == flag None == the archived rollout, and the
   never-firing threshold tau = 0 is bit-identical too (the no-fire path of
   `resample_if_needed` consumes no RNG, so arming the machinery without
   firing it cannot move a single sample).
2. THE TRIGGER ACTUALLY FIRES, and the firing count reaches the training
   log so a run can be judged on whether the flag did anything.
3. SLICES ARE RECORDED AFTER THE CHECKPOINT. A trajectory slice must be
   the ensemble that CONTINUES from that time — the post-resample,
   equally-weighted one — because that is the ensemble whose plain batch
   mean estimates E_{p_t}[.]. Pinned with a stub that collapses every
   resample to a single ancestor: every recorded slice after the first
   event must then be all-clones. (This also exercises the degenerate
   collapse the real trigger is chosen to avoid.)
4. THE BUFFER SURVIVES IT: post-rollout buffer rows keep their shape,
   dtype, finiteness, and — the hard-constraint invariant — every row
   stays on the fixed-composition manifold, since resampling only ever
   duplicates whole on-manifold rows.
5. THE c_t ESTIMATOR STAYS VALID. c_t is a batch mean of xi_t over the
   rollout states (Eq. 8, swap form), and Eq. 8's identity
   E_{p_t}[xi_t] = d_t log Z_t holds under p_t, i.e. under the ACCUMULATED
   IMPORTANCE WEIGHT, not under the raw particle law. Two tests split the
   claim: (a) the weighting that makes the batch mean valid is the IS
   weight; (b) systematic resampling reproduces exactly that weighting in
   expectation, so the post-resample UNWEIGHTED mean is conditionally
   unbiased for the self-normalised weighted mean.
6. THE EVAL PATH IS UNTOUCHED. The in-training ESS column must stay a
   plain-IS draw or it stops being comparable with every archived cell, so
   no call that asks for log-weights may carry a resampling config.
"""
import csv
import math
from pathlib import Path

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers import swap_ctmc as swap_ctmc_module
from discrete_flow_sampler.samplers import swap_training
from discrete_flow_sampler.samplers import training as training_module
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.resampling import (
    ResamplingConfig,
    log_mean_exp,
    systematic_resample_indices,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_xi_t_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import (
    FixedCompositionIsingTarget,
    IsingTarget,
)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ConstantRateModel:
    """Constant flip rate per site — enough to drive the flip-CTMC wiring."""

    def __init__(self, flip_rate: float):
        self.flip_rate = flip_rate

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.flip_rate, dtype=torch.float)


_LATTICE_SIDE = 4          # d = 16 sites on the fixed-composition slice
_N_EULER_STEPS = 8
_OUTER_BATCH = 8
_INNER_PER_OUTER = 2
_N_STEPS = 4


def _tiny_head(n_sites=_LATTICE_SIDE**2):
    return DoublyHollowSwapHead(
        LeTFRateMatrix(
            d=n_sites, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
        )
    )


def _head_and_target(lattice_side=_LATTICE_SIDE, seed=42, sigma=0.3):
    torch.manual_seed(seed)
    target = FixedCompositionIsingTarget(
        D=lattice_side, sigma=sigma, target_composition=0.5
    )
    return _tiny_head(lattice_side**2), target


def _tiny_cfgs(**train_overrides):
    train_kwargs = dict(
        n_steps=_N_STEPS, batch_size=_OUTER_BATCH,
        outer_batch_size=_OUTER_BATCH, inner_steps_per_outer=_INNER_PER_OUTER,
        lr=1e-3, seed=0, replay_buffer_cycles=1, grad_clip_max_norm=500.0,
        warmup_steps=0,
    )
    train_kwargs.update(train_overrides)
    return (
        _Cfg(**train_kwargs),
        _Cfg(n_euler_steps=_N_EULER_STEPS),
        # eval_every > n_steps would skip the eval draw entirely; keep it on
        # so test 6 sees the eval call site.
        _Cfg(eval_every=2, n_eval_samples=8),
    )


def _run_tiny_training(output_dir, **train_overrides):
    """One tiny end-to-end train_swap, returning the run directory."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(
        D=_LATTICE_SIDE, sigma=0.1, target_composition=0.5
    )
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(**train_overrides)
    train_swap(
        _tiny_head(), target, train_cfg, ctmc_cfg, eval_cfg, Path(output_dir),
        use_wandb=False, estimator_mode="control_variate",
    )
    return Path(output_dir)


def _log_rows(run_dir):
    with (run_dir / "training_log.csv").open() as log_file:
        return list(csv.DictReader(log_file))


def _final_weights(run_dir):
    return torch.load(
        run_dir / "checkpoints" / "final.pt", weights_only=True
    )


# ------------------------------------------------------------------ 1. OFF


def test_flag_absent_and_flag_none_and_never_firing_are_all_identical(tmp_path):
    """Every existing recipe must be byte-identical unchanged.

    Three runs from the same seed: the archived call (no field at all), the
    explicit None, and tau = 0.0 (machinery armed, trigger can never fire —
    ESS >= 0 always). The last one is the sharp case: it walks the
    resampling branch, accumulates log-weights and evaluates the trigger
    every step, and must STILL land on identical weights, because the
    no-fire path returns its inputs unchanged and consumes no RNG.
    """
    baseline_dir = _run_tiny_training(tmp_path / "baseline")
    explicit_none_dir = _run_tiny_training(
        tmp_path / "explicit_none", rollout_resample_ess_fraction=None
    )
    never_fires_dir = _run_tiny_training(
        tmp_path / "never_fires", rollout_resample_ess_fraction=0.0
    )

    baseline_weights = _final_weights(baseline_dir)
    for other_dir in (explicit_none_dir, never_fires_dir):
        other_weights = _final_weights(other_dir)
        assert baseline_weights.keys() == other_weights.keys()
        for name, tensor in baseline_weights.items():
            assert torch.equal(tensor, other_weights[name]), name

    baseline_losses = [row["loss"] for row in _log_rows(baseline_dir)]
    for other_dir in (explicit_none_dir, never_fires_dir):
        assert [row["loss"] for row in _log_rows(other_dir)] == baseline_losses


@torch.no_grad()
def test_never_firing_trajectory_mode_is_bit_exact_parity():
    """Sampler level: the trajectory a never-firing config produces is the
    plain rollout, tensor-for-tensor, under the same seed."""
    head, target = _head_and_target()
    time_grid = torch.linspace(0.0, 1.0, 12)

    torch.manual_seed(7)
    x_initial = target.sample_base(16, device="cpu")
    plain_trajectory = sample_swap_ctmc(
        head, x_initial, time_grid, return_all_states=True
    )

    torch.manual_seed(7)
    x_initial = target.sample_base(16, device="cpu")
    smc_trajectory, stats = sample_swap_ctmc(
        head, x_initial, time_grid, return_all_states=True, target=target,
        resampling=ResamplingConfig(ess_threshold_fraction=0.0),
    )
    assert stats.n_events == 0 and stats.event_steps == []
    assert torch.equal(smc_trajectory, plain_trajectory)


# -------------------------------------------------------------- 2. IT FIRES


@torch.no_grad()
def test_trajectory_mode_fires_and_every_slice_stays_on_the_manifold():
    head, target = _head_and_target()
    torch.manual_seed(11)
    x_initial = target.sample_base(32, device="cpu")
    time_grid = torch.linspace(0.0, 1.0, 20)
    trajectory, stats = sample_swap_ctmc(
        head, x_initial, time_grid, return_all_states=True, target=target,
        # tau = 1.0 fires at every checkpoint where the weights are not
        # exactly uniform, i.e. from the first weight update onwards.
        resampling=ResamplingConfig(ess_threshold_fraction=1.0),
    )
    assert trajectory.shape == (20, 32, _LATTICE_SIDE**2)
    assert stats.n_events > 0
    assert len(stats.event_steps) == stats.n_events
    assert torch.isfinite(stats.log_z_increment)
    for slice_index in range(trajectory.shape[0]):
        target.assert_on_manifold(trajectory[slice_index])


@torch.no_grad()
def test_flip_sampler_takes_the_same_trajectory_mode():
    """Sampler-agnosticism: `sample_ctmc` carries the identical contract the
    swap sampler does, pinned here at the sampler level; section 7 pins the
    flip trainer's consumption of the same config flag."""
    torch.manual_seed(3)
    target = IsingTarget(D=2, sigma=0.1)
    x_initial = torch.randint(0, 2, (16, 4)).float() * 2 - 1
    time_grid = torch.linspace(0.0, 1.0, 15)

    plain_trajectory = sample_ctmc(
        _ConstantRateModel(flip_rate=0.5), x_initial, time_grid,
        return_all_states=True,
    )
    torch.manual_seed(3)
    x_initial = torch.randint(0, 2, (16, 4)).float() * 2 - 1
    trajectory, stats = sample_ctmc(
        _ConstantRateModel(flip_rate=0.5), x_initial, time_grid,
        return_all_states=True, target=target,
        resampling=ResamplingConfig(ess_threshold_fraction=1.0),
    )
    assert trajectory.shape == plain_trajectory.shape
    assert stats.n_events > 0
    assert torch.isfinite(trajectory).all()


def test_trajectory_mode_still_needs_a_target():
    """The trigger reads weights, and weights need xi_t — so a trajectory
    rollout asking to resample without a target is a configuration error,
    not a silently unresampled run."""
    head, target = _head_and_target()
    x_initial = target.sample_base(4, device="cpu")
    time_grid = torch.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        sample_swap_ctmc(
            head, x_initial, time_grid, return_all_states=True,
            resampling=ResamplingConfig(),
        )
    with pytest.raises(ValueError):
        sample_ctmc(
            _ConstantRateModel(0.1), x_initial, time_grid,
            return_all_states=True, resampling=ResamplingConfig(),
        )


def test_resample_event_count_reaches_the_training_log(tmp_path):
    """The flag is judged on whether it fired, so the count is logged: an
    integer per outer cycle when armed, NaN when off (the established idiom
    for a column a run cannot populate)."""
    fired_rows = _log_rows(
        _run_tiny_training(
            tmp_path / "fires", rollout_resample_ess_fraction=1.0
        )
    )
    assert all(
        float(row["rollout_resample_events"]) > 0 for row in fired_rows
    )

    off_rows = _log_rows(_run_tiny_training(tmp_path / "off"))
    assert all(
        math.isnan(float(row["rollout_resample_events"])) for row in off_rows
    )


# ---------------------------------------------- 3. SLICES RECORDED AFTER


@torch.no_grad()
def test_slices_are_recorded_after_the_resample_checkpoint(monkeypatch):
    """A recorded slice must be the ensemble that CONTINUES from that time.

    Rationale: the plain batch mean of xi_t over a slice only estimates
    E_{p_t}[xi_t] once the accumulated IS weights have been applied, and a
    resample is what applies them. Recording the pre-resample rows would
    store the still-weighted ensemble and hand c_t the wrong measure.

    Pinned with a stub that collapses every resample onto ancestor 0, so
    "recorded after" is observable as all-clone slices. The stub doubles as
    the degenerate-collapse rehearsal: total ancestry collapse must still
    produce a well-formed, on-manifold trajectory rather than a crash.
    """
    head, target = _head_and_target()

    def collapse_to_first_row(state, log_weights, ess_threshold_fraction):
        return (
            state[0].expand_as(state).clone(),
            torch.zeros_like(log_weights),
            log_mean_exp(log_weights),
            True,
        )

    monkeypatch.setattr(
        swap_ctmc_module, "resample_if_needed", collapse_to_first_row
    )
    torch.manual_seed(3)
    x_initial = target.sample_base(8, device="cpu")
    time_grid = torch.linspace(0.0, 1.0, 6)
    trajectory, stats = sample_swap_ctmc(
        head, x_initial, time_grid, return_all_states=True, target=target,
        resampling=ResamplingConfig(ess_threshold_fraction=0.5),
    )
    assert stats.n_events == 5
    for slice_index in range(1, trajectory.shape[0]):
        recorded = trajectory[slice_index]
        assert torch.equal(recorded, recorded[0].expand_as(recorded))
        target.assert_on_manifold(recorded)


# --------------------------------------------------------- 4. THE BUFFER


def test_buffer_after_a_resampled_rollout_is_well_formed(tmp_path, monkeypatch):
    """Shape, dtype, finiteness and the composition invariant, read off the
    buffer the inner loop actually trains on."""
    original_append = swap_training._append_replay_buffer
    captured_buffers = []

    def recording_append(*args, **kwargs):
        x_buffer, t_idx_buffer = original_append(*args, **kwargs)
        captured_buffers.append((x_buffer, t_idx_buffer))
        return x_buffer, t_idx_buffer

    monkeypatch.setattr(
        swap_training, "_append_replay_buffer", recording_append
    )
    _run_tiny_training(tmp_path, rollout_resample_ess_fraction=1.0)

    manifold_target = FixedCompositionIsingTarget(
        D=_LATTICE_SIDE, sigma=0.1, target_composition=0.5
    )
    assert captured_buffers
    for x_buffer, t_idx_buffer in captured_buffers:
        assert x_buffer.shape == (
            _N_EULER_STEPS * _OUTER_BATCH, _LATTICE_SIDE**2
        )
        assert x_buffer.dtype == torch.float32
        assert torch.isfinite(x_buffer).all()
        assert t_idx_buffer.shape == (_N_EULER_STEPS * _OUTER_BATCH,)
        manifold_target.assert_on_manifold(x_buffer)


# --------------------------------------------------- 5. c_t STAYS VALID


@torch.no_grad()
def test_the_weighting_that_validates_the_c_t_batch_mean_is_the_is_weight():
    """Design answer (1), first half.

    Eq. (8) reads E_{p_t}[xi_t] = d_t log Z_t: the identity holds under the
    ANNEALED TARGET, and a particle ensemble represents p_t only through
    its accumulated importance weights. On the enumerable d = 16 slice,
    take the whole state space as the ensemble with log-weights
    log p~_t (proposal uniform over the slice): the self-normalised
    weighted mean of xi_t hits d_t log Z_t exactly, while the unweighted
    mean of the same particles does not. That gap is precisely what the
    resample removes.
    """
    head, target = _head_and_target(lattice_side=2)
    states = enumerate_states(4).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]
    time_value = torch.full((slice_states.shape[0],), 0.5)

    log_weights = target.log_p_tilde_t(slice_states, time_value)
    normalised_weights = torch.softmax(log_weights, dim=0)
    dt_log_z = (
        normalised_weights * target.dt_log_p_tilde_t(slice_states, time_value)
    ).sum()

    xi_t = compute_xi_t_swap(slice_states, time_value, head, target)
    weighted_mean = (normalised_weights * xi_t).sum()
    unweighted_mean = xi_t.mean()

    assert torch.isclose(weighted_mean, dt_log_z, atol=1e-5)
    assert not torch.isclose(unweighted_mean, dt_log_z, atol=1e-3)


@torch.no_grad()
def test_resampling_reproduces_that_weighting_in_expectation():
    """Design answer (1), second half.

    Systematic resampling draws ancestor counts with
    E[counts_i] = B * w_i exactly, so conditional on the weights

        E[ (1/B) sum_m xi_t(x^{a(m)}) ] = sum_i w_i xi_t(x^i),

    the self-normalised weighted mean. The post-resample UNWEIGHTED batch
    mean is therefore an unbiased estimator of the weighted mean the
    previous test showed to be the valid one. Integrated over the single
    systematic jitter on a fine grid (counts are piecewise constant in the
    jitter, so the grid average converges to the exact expectation).
    """
    head, target = _head_and_target(lattice_side=2, seed=5)
    torch.manual_seed(17)
    states = target.sample_base(16, device="cpu")
    time_value = torch.full((16,), 0.5)
    xi_t = compute_xi_t_swap(states, time_value, head, target)
    log_weights = torch.linspace(-3.0, 3.0, 16)

    weighted_mean = (torch.softmax(log_weights, dim=0) * xi_t).sum()
    jitters = (torch.arange(4096) + 0.5) / 4096
    resampled_means = torch.stack([
        xi_t[systematic_resample_indices(log_weights, uniform=jitter)].mean()
        for jitter in jitters
    ])
    assert torch.isclose(resampled_means.mean(), weighted_mean, atol=1e-3)


# --------------------------------------------------- 6. EVAL PATH INTACT


def test_only_the_rollout_resamples_never_the_in_training_eval_draw(
    tmp_path, monkeypatch
):
    """The `ess` column must remain a plain-IS reading, comparable with
    every archived cell, so the resampling config may reach the buffer
    rollout and nothing else."""
    original_sampler = swap_training.sample_swap_ctmc
    recorded_calls = []

    def recording_sampler(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(
        swap_training, "sample_swap_ctmc", recording_sampler
    )
    _run_tiny_training(tmp_path, rollout_resample_ess_fraction=1.0)

    rollout_calls = [
        kwargs for kwargs in recorded_calls
        if kwargs.get("return_all_states")
    ]
    eval_calls = [
        kwargs for kwargs in recorded_calls
        if kwargs.get("return_log_weights")
    ]
    assert rollout_calls and eval_calls
    assert all(
        kwargs.get("resampling") is not None for kwargs in rollout_calls
    )
    assert all(kwargs.get("resampling") is None for kwargs in eval_calls)


# ------------------------------------- 7. THE FLIP TRAINER'S OWN CALL SITE


def _run_tiny_flip_training(output_dir, *, amortised=False, **train_overrides):
    """One tiny end-to-end `train()` on the flip route; returns the run dir.

    The flip loop serves both the unconstrained and soft chapters. The
    amortised variant is the soft production shape: the rollout executes
    inside the composition binding (`_bound`), so the ESS trigger's xi_t
    must read the BOUND target's annealed density, not the bare one.
    """
    torch.manual_seed(0)
    if amortised:
        target = IsingTarget(
            D=2, sigma=0.1, target_composition=0.5,
            composition_penalty_strength=5.0,
        )
    else:
        target = IsingTarget(D=2, sigma=0.1)
    model = LeTFRateMatrix(
        d=target.d, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2,
        condition_on_composition=amortised,
    )
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs(**train_overrides)
    amortised_kwargs = (
        {"composition_centre": 0.5, "composition_half_width": 0.25}
        if amortised else {}
    )
    train(
        model, target, train_cfg, ctmc_cfg, eval_cfg, Path(output_dir),
        use_wandb=False, **amortised_kwargs,
    )
    return Path(output_dir)


def test_flip_trainer_flag_absent_none_and_never_firing_are_identical(tmp_path):
    """Twin of the swap-loop OFF test: every archived unconstrained/soft
    recipe must be byte-identical unchanged — absent flag, explicit None,
    and the armed-but-never-firing tau = 0.0 all land on the same weights,
    because the no-fire path consumes no RNG."""
    baseline_dir = _run_tiny_flip_training(tmp_path / "baseline")
    explicit_none_dir = _run_tiny_flip_training(
        tmp_path / "explicit_none", rollout_resample_ess_fraction=None
    )
    never_fires_dir = _run_tiny_flip_training(
        tmp_path / "never_fires", rollout_resample_ess_fraction=0.0
    )

    baseline_weights = _final_weights(baseline_dir)
    for other_dir in (explicit_none_dir, never_fires_dir):
        other_weights = _final_weights(other_dir)
        assert baseline_weights.keys() == other_weights.keys()
        for name, tensor in baseline_weights.items():
            assert torch.equal(tensor, other_weights[name]), name

    baseline_losses = [row["loss"] for row in _log_rows(baseline_dir)]
    for other_dir in (explicit_none_dir, never_fires_dir):
        assert [row["loss"] for row in _log_rows(other_dir)] == baseline_losses


def test_flip_trainer_event_count_reaches_the_training_log(tmp_path):
    """Same judgement contract as the swap loop: an integer per outer cycle
    when armed, NaN when the flag is off."""
    fired_rows = _log_rows(
        _run_tiny_flip_training(
            tmp_path / "fires", rollout_resample_ess_fraction=1.0
        )
    )
    assert all(
        float(row["rollout_resample_events"]) > 0 for row in fired_rows
    )

    off_rows = _log_rows(_run_tiny_flip_training(tmp_path / "off"))
    assert all(
        math.isnan(float(row["rollout_resample_events"])) for row in off_rows
    )


def test_flip_trainer_eval_draw_never_resamples(tmp_path, monkeypatch):
    """The flip loop's `ess` column must stay a plain-IS reading too — the
    resampling config may reach the buffer rollout and nothing else."""
    original_sampler = training_module.sample_ctmc
    recorded_calls = []

    def recording_sampler(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(training_module, "sample_ctmc", recording_sampler)
    _run_tiny_flip_training(tmp_path, rollout_resample_ess_fraction=1.0)

    rollout_calls = [
        kwargs for kwargs in recorded_calls
        if kwargs.get("return_all_states")
    ]
    eval_calls = [
        kwargs for kwargs in recorded_calls
        if kwargs.get("return_log_weights")
    ]
    assert rollout_calls and eval_calls
    assert all(
        kwargs.get("resampling") is not None for kwargs in rollout_calls
    )
    assert all(kwargs.get("resampling") is None for kwargs in eval_calls)


def test_flip_trainer_amortised_soft_rollout_fires_inside_the_binding(tmp_path):
    """The soft production recipe amortises over compositions, so the rollout
    (and therefore the ESS trigger's xi_t) runs inside `_bound`'s composition
    binding. A firing amortised run that completes with finite losses pins
    that the resampling machinery composes with the binding — the trigger
    reads the annealed density of the composition the cycle actually drew."""
    rows = _log_rows(
        _run_tiny_flip_training(
            tmp_path, amortised=True, rollout_resample_ess_fraction=1.0
        )
    )
    assert all(float(row["rollout_resample_events"]) > 0 for row in rows)
    assert all(math.isfinite(float(row["loss"])) for row in rows)
