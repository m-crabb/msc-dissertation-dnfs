import csv
from pathlib import Path

import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_training import train_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _tiny_head():
    return DoublyHollowSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    )


def _tiny_cfgs():
    train_cfg = _Cfg(n_steps=4, batch_size=8, outer_batch_size=8,
                      inner_steps_per_outer=2, lr=1e-3, seed=0,
                      replay_buffer_cycles=1, grad_clip_max_norm=500.0,
                      warmup_steps=0)
    ctmc_cfg = _Cfg(n_euler_steps=8)
    eval_cfg = _Cfg(eval_every=2, n_eval_samples=16)
    return train_cfg, ctmc_cfg, eval_cfg


def _read_csv_rows(csv_path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def test_train_swap_smoke_runs_and_stays_on_manifold(tmp_path):
    torch.manual_seed(0)
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    head = _tiny_head()
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, Path(tmp_path),
               use_wandb=False, estimator_mode="control_variate")
    assert (Path(tmp_path) / "training_log.csv").exists()
    assert (Path(tmp_path) / "checkpoints" / "final.pt").exists()


def test_train_swap_logs_swap_rate_diagnostics_and_preserves_composition(tmp_path):
    """Pins the AMENDMENT: both the Λ·dt>1 clip fraction and the log-ratio
    clamp-hit fraction are logged at eval cadence, and the trained head's
    sampler never leaves the fixed-composition manifold."""
    torch.manual_seed(0)
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    head = _tiny_head()
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, Path(tmp_path),
               use_wandb=False, estimator_mode="control_variate")

    rows = _read_csv_rows(Path(tmp_path) / "training_log.csv")
    diagnostic_columns = {
        "rate_pair_mean", "rate_pair_p99",
        "lambda_dt_clipped_frac", "log_ratio_clamp_frac",
    }
    assert diagnostic_columns.issubset(rows[0].keys())

    # eval_every=2 over 4 steps -> rows 0 and 2 are eval rows, populated.
    for eval_row in (rows[0], rows[2]):
        for column in diagnostic_columns:
            value = float(eval_row[column])
            assert value == value, f"{column} is NaN on an eval row"
            assert 0.0 <= value <= 1.0 or column.startswith("rate_pair")

    from discrete_flow_sampler.samplers.swap_ctmc import sample_swap_ctmc

    with torch.no_grad():
        x0 = tgt.sample_base(16, device=tgt.device)
        t_grid = torch.linspace(0.0, 1.0, ctmc_cfg.n_euler_steps, device=tgt.device)
        x_final = sample_swap_ctmc(head, x0, t_grid)
    tgt.assert_on_manifold(x_final)


def test_train_swap_use_matching_step_threads_through_sampler(tmp_path, monkeypatch):
    """CTMCCfg.use_matching_step=True must reach EVERY trajectory simulation
    in the training loop — the buffer rebuild and the in-training eval draw
    alike. The 16x16 rung trains on the matching step at n_euler=128, which
    is only clip-safe because the matching step decouples trajectory length
    from Λ (one-event would need ~390 steps at d=256); a knob that silently
    left one call site one-event would void that clip-safety argument."""
    import discrete_flow_sampler.samplers.swap_training as swap_training_module

    torch.manual_seed(0)
    recorded_multi_event = []
    real_sample_swap_ctmc = swap_training_module.sample_swap_ctmc

    def recording_sample_swap_ctmc(*args, **kwargs):
        recorded_multi_event.append(kwargs.get("multi_event", False))
        return real_sample_swap_ctmc(*args, **kwargs)

    monkeypatch.setattr(
        swap_training_module, "sample_swap_ctmc", recording_sample_swap_ctmc
    )

    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    head = _tiny_head()
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    ctmc_cfg.use_matching_step = True
    train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, Path(tmp_path),
               use_wandb=False, estimator_mode="control_variate")

    assert recorded_multi_event, "training never simulated a trajectory"
    assert all(recorded_multi_event), (
        "a training-loop sample_swap_ctmc call ignored use_matching_step"
    )

    # Default (no field, as every archived config bag): stays one-event.
    recorded_multi_event.clear()
    tgt_default = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    train_cfg2, ctmc_cfg2, eval_cfg2 = _tiny_cfgs()
    train_swap(_tiny_head(), tgt_default, train_cfg2, ctmc_cfg2, eval_cfg2,
               Path(tmp_path) / "default", use_wandb=False,
               estimator_mode="control_variate")
    assert recorded_multi_event and not any(recorded_multi_event)


def test_train_swap_eval_sample_chunk_bounds_head_batch(tmp_path):
    """With eval_sample_chunk set, the ESS eval streams n_eval_samples through
    sample_swap_ctmc in slices, so the head never sees the full eval batch at
    once. This is the GPU-memory contract for the D>=8 rungs: the vectorised
    mask_one head rides d anchor copies per sample, so an unchunked
    5000-sample eval would build (d*5000)-row attention buffers and OOM the
    L4. The chunked eval must still log a usable (non-NaN) ESS."""
    torch.manual_seed(0)
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    head = _tiny_head()
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    eval_cfg.eval_sample_chunk = 8  # n_eval_samples=16 -> two slices of 8

    head_batches = []
    hook = head.register_forward_pre_hook(
        lambda module, args: head_batches.append(args[0].shape[0])
    )
    try:
        train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, Path(tmp_path),
                   use_wandb=False, estimator_mode="control_variate")
    finally:
        hook.remove()

    # Training batches are 8; unchunked eval would show 16 here.
    assert max(head_batches) <= 8, (
        f"eval fed the head {max(head_batches)} samples at once; "
        "eval_sample_chunk=8 not honoured"
    )
    rows = _read_csv_rows(Path(tmp_path) / "training_log.csv")
    for eval_row in (rows[0], rows[2]):
        ess = float(eval_row["ess"])
        assert ess == ess, "ess is NaN on an eval row under chunked eval"


def test_train_swap_in_training_eval_draw_size(tmp_path):
    """`n_eval_samples_training` shrinks ONLY the in-training diagnostic ESS
    draws (the objective is the one final n_eval_samples eval in run.py, which
    train_swap does not perform); the default None preserves the current
    behaviour of drawing the full n_eval_samples every eval."""
    from experiments.dnfs_baseline_01.configs import EvalCfg

    assert EvalCfg().n_eval_samples_training is None

    for training_draw, expected_eval_draw in ((4, 4), (None, 16)):
        torch.manual_seed(0)
        tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
        head = _tiny_head()
        train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
        eval_cfg.n_eval_samples_training = training_draw

        draw_sizes = []
        original_sample_base = tgt.sample_base

        def recording_sample_base(n, device=None, _orig=original_sample_base):
            draw_sizes.append(n)
            return _orig(n, device=device)

        tgt.sample_base = recording_sample_base
        run_dir = Path(tmp_path) / f"draw_{training_draw}"
        run_dir.mkdir()
        train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, run_dir,
                   use_wandb=False, estimator_mode="control_variate")

        assert expected_eval_draw in draw_sizes, (
            f"n_eval_samples_training={training_draw}: no eval draw of "
            f"{expected_eval_draw} observed (draws: {sorted(set(draw_sizes))})"
        )
        if training_draw is not None:
            assert 16 not in draw_sizes, (
                "in-training eval still drew the full n_eval_samples"
            )


def test_train_swap_eval_autocast_bf16_flag(tmp_path):
    """`eval_autocast_bf16` must default False (fp32 eval, behaviour
    unchanged) and, when set, run ONLY the in-training eval head calls under
    bf16 autocast -- training forward/backward stays fp32 (gradients are
    Tier-3, untouchable)."""
    from experiments.dnfs_baseline_01.configs import EvalCfg

    assert EvalCfg().eval_autocast_bf16 is False

    for flag_on in (False, True):
        torch.manual_seed(0)
        tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
        head = _tiny_head()
        train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
        eval_cfg.eval_autocast_bf16 = flag_on

        # The head's OUTPUT dtype is fp32 even under autocast (the readout's
        # final op promotes), so detect autocast state, not tensor dtype.
        autocast_states = []
        hook = head.register_forward_pre_hook(
            lambda module, args: autocast_states.append(
                torch.is_autocast_enabled("cpu")
            )
        )
        run_dir = Path(tmp_path) / f"autocast_{flag_on}"
        run_dir.mkdir()
        try:
            train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, run_dir,
                       use_wandb=False, estimator_mode="control_variate")
        finally:
            hook.remove()

        assert (True in autocast_states) == flag_on, (
            f"eval_autocast_bf16={flag_on} but autocast head calls "
            f"{'missing' if flag_on else 'present'}"
        )
        # Training forward/backward must stay fp32 under either flag value.
        assert False in autocast_states


def test_train_swap_logs_c_t_offset_rms(tmp_path):
    """Δ = E_buffer[ξ_t] − c_t, per time slot and RMS'd, is logged every step.

    c_t is estimated on the fresh rollout while the loss averages over the
    replay buffer, so this column is the only measurement of the resulting
    mismatch — the term that enters the gradient as 2·Δ·E[∇ξ]."""
    torch.manual_seed(0)
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    head = _tiny_head()
    train_cfg, ctmc_cfg, eval_cfg = _tiny_cfgs()
    train_swap(head, tgt, train_cfg, ctmc_cfg, eval_cfg, Path(tmp_path),
               use_wandb=False, estimator_mode="control_variate")

    rows = _read_csv_rows(Path(tmp_path) / "training_log.csv")
    assert "c_t_offset_rms" in rows[0]
    values = [float(row["c_t_offset_rms"]) for row in rows]
    # Finite and non-negative on every step: it is an RMS, and every step
    # follows at least one inner draw, so no row can be the empty-slot NaN.
    assert all(value == value and value >= 0.0 for value in values)
