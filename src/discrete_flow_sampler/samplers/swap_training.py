"""Outer/inner training loop for the swap-move CTMC (hard-constraint route).

Mirror of `training.train` (paper Algorithm 1, App. C.1) with the single-site
sampler/loss replaced by the swap-move counterparts: `sample_swap_ctmc`,
`compute_c_t_grid_swap` and `loss_swap` (Eq. 10, swap form). The swap move
set enforces n_plus == N_A exactly, so there is no composition penalty and
no `lambda_curriculum`.

Pair diagnostics replace the flip trainer's per-site columns with
`rate_pair_mean`, `rate_pair_p99` and `lambda_dt_clipped_frac` (see
`_swap_rate_diagnostics`). `log_ratio_clamp_frac` measures saturation at
`SWAP_LOG_RATIO_CLAMP`; `log_ratio_p99` is omitted. `rollout_resample_events`
counts outer-cycle ESS-triggered SMC events, NaN when disabled
(`TrainCfg.rollout_resample_ess_fraction`).
"""

import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.ema import CTGridEMA, ExponentialMovingAverage
from discrete_flow_sampler.samplers._swap_neighbours import (
    SWAP_LOG_RATIO_CLAMP,
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.optim import StableAdamW
from discrete_flow_sampler.samplers.resampling import ResamplingConfig
from discrete_flow_sampler.samplers.resume import (
    capture_rng_state,
    load_resume_state,
    restore_rng_state,
    save_resume_state,
    truncate_log_to_step,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_c_t_grid_swap,
    n_slices,
    reduce_c_t_grid,
    sample_swap_ctmc,
    slice_index_of,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    c_t_offset_rms,
    loss_swap_backward_microbatched,
)
from discrete_flow_sampler.samplers.training import (
    _append_replay_buffer,
    _clear_replay,
    _normalise_curriculum,
    _set_optimizer_lr,
)
from discrete_flow_sampler.seeding import seed_everything

# torch.quantile refuses inputs above 2**24 elements. The pair-rate slab
# (outer_batch, d(d-1)/2) is 16,711,680 at d=256 with outer batch 512 (99.6%
# of the cap); at d=400 it is 40,857,600, and the step-0 init diagnostic
# raised "quantile() input tensor is too large" on the first 20x20 run.
_QUANTILE_MAX_ELEMENTS = 2**24


def _p99(values: torch.Tensor) -> float:
    """p99 of a flat tensor, on torch.quantile's own 'linear' convention.

    Below the cap this calls torch.quantile unchanged, so every archived d256
    number (all logged through it) stays bit-identical. Above the cap it
    sorts and interpolates by hand on the convention torch documents --
    position q*(n-1), linear between the neighbouring order statistics -- so
    a d400 row stays comparable with a d256 one.
    """
    n = values.numel()
    if n <= _QUANTILE_MAX_ELEMENTS:
        return torch.quantile(values, 0.99).item()
    ordered = values.sort().values
    position = 0.99 * (n - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, n - 1)
    weight = position - lower
    return (ordered[lower] * (1.0 - weight) + ordered[upper] * weight).item()


def _swap_rate_diagnostics(head, x, t, step_dt: float, *, target) -> dict[str, float]:
    """Eval-time diagnostics for the swap-CTMC rate scale.

    Mirrors `training._rate_diagnostics` for the pair-rate matrix, gathering
    forward rates at the same i<j pairs `_euler_step_swap` uses so the logged
    clip fraction matches the sampler:

        Lambda * dt = sum_{i<j} [G_swap(i,j|x)]_+ * dt

    `lambda_dt_clipped_frac` is the fraction of states with Lambda*dt > 1,
    where the one-event step's stay slot clamps to 0 and exactly one swap is
    forced. Under `use_matching_step=True` it is a one-event clip-safety
    diagnostic only; the matching step's fidelity is `proposal_drop_frac` and
    `events_per_site_per_step`. `lambda_dt_p99` records the tail of the
    per-state rate load directly: the one-event Euler budget rule reads the
    tail, and Lambda is too fat-tailed to reconstruct it from mean and
    exceedance fraction. `log_ratio_clamp_frac` is the fraction of pair
    log-ratios log p_tilde_t(swap) - log p_tilde_t(x) reaching
    `SWAP_LOG_RATIO_CLAMP` (Eq. 8 / Eq. 10, swap form); expected ~0, nonzero
    flags residual/xi_t bias.
    """
    pairs = upper_tri_pairs(x.shape[1], x.device)
    forward_rates = F.relu(gather_pair_scores(head(x, t), pairs))  # (B, n_pairs)
    lambda_dt = (forward_rates * step_dt).sum(dim=-1)  # (B,)

    log_ratio = target.swap_log_ratio(x, t, pairs)  # unclamped: measures saturation

    return {
        "rate_pair_mean": forward_rates.mean().item(),
        # .float(): quantile is fp32/64-only; heads may emit reduced
        # precision under the eval autocast block.
        "rate_pair_p99": _p99(forward_rates.reshape(-1).float()),
        "lambda_dt_clipped_frac": (lambda_dt > 1.0).float().mean().item(),
        # tail of the per-state rate load (see docstring)
        "lambda_dt_p99": torch.quantile(lambda_dt.float(), 0.99).item(),
        "log_ratio_clamp_frac": (
            (log_ratio > SWAP_LOG_RATIO_CLAMP).float().mean().item()
        ),
    }


def _save_resume_state(
    ckpt_dir: Path,
    *,
    step: int,
    head,
    optimiser,
    x_replay_chunks,
    t_idx_replay_chunks,
    replay_sigma: float,
    ema=None,
    c_t_ema=None,
) -> None:
    """Assemble the swap trainer's outer-boundary state for preemption resume.

    The write is `resume.save_resume_state` (atomic, shared with the flip
    trainer); the payload is swap-specific. Replay chunks move to CPU so the
    checkpoint is device-portable; RNG states make the continuation
    bit-identical (exactly on CPU fp32, modulo kernel nondeterminism on
    CUDA). Curriculum stage, warmup and lr are not stored: all derive from
    `step` (absolute start_steps), and lr travels in the optimiser state dict.
    """
    state = {
        "step": step,
        "model": head.state_dict(),
        "optimiser": optimiser.state_dict(),
        **capture_rng_state(),
        "x_replay_chunks": [chunk.cpu() for chunk in x_replay_chunks],
        "t_idx_replay_chunks": [chunk.cpu() for chunk in t_idx_replay_chunks],
        "replay_sigma": replay_sigma,
        # Shadow and update counter: a re-seeded shadow re-creates the
        # init-contamination failure; a reset counter restarts the warmup.
        "ema": ema.state_dict() if ema is not None else None,
        # The smoothed grid depends on every past cycle's raw estimate, so
        # it cannot be rebuilt at resume; it must travel.
        "c_t_ema": c_t_ema.state_dict() if c_t_ema is not None else None,
    }
    save_resume_state(ckpt_dir, state)


def _cv_inversion_sustained(ratio_history: list[float], window: int) -> bool:
    """True iff the trailing `window` outer-cycle var-ratios are all > 1.0.

    Strictly-greater and windowed by design: the healthy warm-start pattern
    opens ~30x against and crosses below 1 within ~1000 steps, so a single
    healthy cycle inside the window must reset the case; and exactly 1.0
    (the naive mode's wiring value) is not an inversion.
    """
    if len(ratio_history) < window:
        return False
    return all(ratio > 1.0 for ratio in ratio_history[-window:])


def train_swap(
    head,
    target,
    train_cfg,
    ctmc_cfg,
    eval_cfg,
    output_dir: Path,
    *,
    use_wandb: bool = True,
    estimator_mode: str = "control_variate",
    sigma_curriculum=None,
    on_checkpoint=None,
    ema_decay: float = 0.0,
):
    """Run paper Algorithm 1 for `train_cfg.n_steps` total inner steps.

    ema_decay > 0 keeps a warmup-corrected parameter shadow (see
    discrete_flow_sampler.ema), updated after every optimiser step and saved
    as checkpoints/final_ema.pt alongside final.pt. The shadow never feeds
    the loss; the raw eval stays the primary number, the EMA eval is
    recorded alongside.

    Args:
        head: a swap-readout head (e.g. `DoublyHollowSwapHead`) wrapping a
            `LeTFRateMatrix` backbone, instantiated on `target.device`.
        target: a `FixedCompositionIsingTarget`; swaps preserve its
            n_plus == N_A manifold, so `sample_base` is already on-manifold.
        train_cfg, ctmc_cfg, eval_cfg, output_dir, use_wandb, estimator_mode,
            sigma_curriculum: same contract as `training.train`. There is no
            `lambda_curriculum`: no soft penalty to anneal on this route.
        on_checkpoint: optional zero-arg callable invoked after each resume
            checkpoint lands (Modal passes `volume.commit`).

    Preemption resume: every `train_cfg.resume_every_outer` outer cycles
    (default 10) the boundary state is checkpointed to `checkpoints/resume.pt`;
    if it exists on entry, training restores and continues (see
    `_save_resume_state`). A resume at step >= n_steps is a completed run
    being retried: return immediately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    resume_state = load_resume_state(ckpt_dir, map_location=target.device)

    if resume_state is None:
        seed_everything(train_cfg.seed)
    optimiser_kind = getattr(train_cfg, "optimiser", "adamw")
    if optimiser_kind == "adamw":
        optimiser = torch.optim.AdamW(
            head.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
    elif optimiser_kind == "stable_adamw":
        optimiser = StableAdamW(head.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
    else:
        raise ValueError(f"unknown optimiser {optimiser_kind!r}")
    ema = (
        ExponentialMovingAverage(head.parameters(), ema_decay, warmup=True)
        if ema_decay > 0
        else None
    )
    start_step = 0
    if resume_state is not None:
        head.load_state_dict(resume_state["model"])
        optimiser.load_state_dict(resume_state["optimiser"])
        start_step = int(resume_state["step"])
        if ema is not None:
            saved_ema = resume_state.get("ema")
            if saved_ema is not None:
                ema.load_state_dict(saved_ema)
            else:
                # The historical average cannot be reconstructed: start a
                # new warmup-corrected shadow at the restored model.
                ema = ExponentialMovingAverage(
                    head.parameters(), ema_decay, warmup=True
                )
                print(
                    "[train_swap] WARNING: resume.pt has no EMA state; "
                    "shadow re-seeded at resume weights; EMA history and "
                    "warmup restart",
                    flush=True,
                )

    if start_step >= train_cfg.n_steps:
        # Completed run re-invoked (e.g. a Modal retry): ensure the terminal
        # artefacts exist, touch nothing else. final_ema.pt can be missing
        # alone (preempted between the two saves); backfill from the shadow.
        if not (ckpt_dir / "final.pt").exists():
            torch.save(head.state_dict(), ckpt_dir / "final.pt")
        if ema is not None and not (ckpt_dir / "final_ema.pt").exists():
            ema.swap_in()
            torch.save(head.state_dict(), ckpt_dir / "final_ema.pt")
            ema.swap_out()
        return

    if use_wandb:
        import wandb

    n_dims = target.d
    device = target.device
    inner_batch = train_cfg.batch_size
    train_autocast_bf16 = getattr(train_cfg, "train_autocast_bf16", False)
    outer_batch = train_cfg.outer_batch_size or train_cfg.batch_size
    n_grid = ctmc_cfg.n_euler_steps
    n_composition_slices = n_slices(
        target
    )  # 1 for a specialist, K on a composition mixture
    # Trajectory step for every simulation in this loop: the matching step
    # is required from d=256 up, where one-event clip-safety would need ~3x
    # the Euler grid. getattr: test call sites pass bare config bags.
    multi_event = getattr(ctmc_cfg, "use_matching_step", False)
    inner_steps_per_outer = train_cfg.inner_steps_per_outer
    replay_buffer_cycles = getattr(train_cfg, "replay_buffer_cycles", 1)
    halt_on_cv_inversion_after = getattr(train_cfg, "halt_on_cv_inversion_after", None)
    halt_cv_inversion_window = int(getattr(train_cfg, "halt_cv_inversion_window", 10))
    cv_var_ratio_history: list[float] = []
    loss_microbatch_size = getattr(train_cfg, "loss_microbatch_size", None)
    # Per-slot c_t EMA across outer cycles. Default 0.0 preserves the
    # archived path; see TrainCfg.c_t_ema_halflife_cycles and ema.CTGridEMA.
    c_t_ema_halflife = float(getattr(train_cfg, "c_t_ema_halflife_cycles", 0.0))
    c_t_ema = (
        CTGridEMA(n_grid, c_t_ema_halflife)
        if CTGridEMA.is_enabled(c_t_ema_halflife)
        else None
    )
    # c_t rollout batch, decoupled from the buffer batch: c_t = mean_m xi_t
    # over the rollout states (Eq. 8), so its standard error falls with the
    # row count. None = outer_batch = the archived setting, byte-identical.
    c_t_batch = getattr(train_cfg, "c_t_batch", None)
    if c_t_batch is not None:
        if isinstance(c_t_batch, bool) or int(c_t_batch) != c_t_batch:
            raise TypeError(f"c_t_batch must be an int or None, got {c_t_batch!r}")
        c_t_batch = int(c_t_batch)
        if c_t_batch < outer_batch:
            raise ValueError(
                f"c_t_batch must be >= outer_batch ({outer_batch}) when "
                f"set, got {c_t_batch}: the replay buffer takes the first "
                f"outer_batch rollout rows, so a smaller c_t_batch would "
                f"starve it."
            )
    # Chunk flattened (n_grid x n_rollout) integrand rows; None preserves
    # the archived per-slot loop. compute_c_t_grid_swap validates the cap;
    # tests/test_c_t_grid_chunk.py pins parity with the sequential path.
    c_t_grid_chunk_rows = getattr(train_cfg, "c_t_grid_chunk_rows", None)
    # ESS-triggered SMC resampling inside the buffer-rebuild rollout (LEAPS
    # Alg. 1 lines 11-14, whose trajectories Alg. 2 line 5 trains on). None
    # = off = every archived run, bit-identical; argument on the config field.
    rollout_resample_ess_fraction = getattr(
        train_cfg, "rollout_resample_ess_fraction", None
    )
    if rollout_resample_ess_fraction is not None and not (
        0.0 <= float(rollout_resample_ess_fraction) <= 1.0
    ):
        raise ValueError(
            f"rollout_resample_ess_fraction must lie in [0, 1] when set, "
            f"got {rollout_resample_ess_fraction}: it is a fraction of the "
            f"rollout batch, and 1.0 already fires at every checkpoint."
        )
    rollout_resampling = (
        ResamplingConfig(ess_threshold_fraction=float(rollout_resample_ess_fraction))
        if rollout_resample_ess_fraction is not None
        else None
    )
    # Build the CV c_t grid from the rollout's own head forwards instead of
    # re-running them: bit-identical to the sequential grid (same tensors,
    # no RNG; tests/test_cv_integrand_reuse.py), removing 127 of 128 c_t-grid
    # head forwards per outer at d256. False = archived, byte-identical.
    c_t_from_rollout = bool(getattr(train_cfg, "c_t_from_rollout", False))
    if c_t_from_rollout and rollout_resampling is not None:
        raise ValueError(
            "c_t_from_rollout requires rollout resampling OFF: resampled "
            "trajectories are certified only through the recompute path."
        )
    if replay_buffer_cycles < 1:
        raise ValueError(
            f"replay_buffer_cycles must be >= 1, got {replay_buffer_cycles}"
        )

    if train_cfg.n_steps % inner_steps_per_outer != 0:
        raise ValueError(
            f"n_steps={train_cfg.n_steps} not divisible by "
            f"inner_steps_per_outer={inner_steps_per_outer}; the outer/"
            f"inner contract requires whole outer cycles."
        )
    n_outer = train_cfg.n_steps // inner_steps_per_outer
    curriculum = _normalise_curriculum(
        sigma_curriculum,
        n_steps=train_cfg.n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
    )

    if start_step % inner_steps_per_outer != 0:
        raise ValueError(
            f"resume.pt records step {start_step}, not an outer-cycle "
            f"boundary (inner_steps_per_outer={inner_steps_per_outer}) -- "
            f"checkpoints are only ever written at boundaries, so this file "
            f"was not produced by this loop."
        )
    start_outer = start_step // inner_steps_per_outer
    resume_every_outer = int(getattr(train_cfg, "resume_every_outer", 10))

    log_path = output_dir / "training_log.csv"
    log_mode = (
        "a"
        if resume_state is not None and truncate_log_to_step(log_path, start_step)
        else "w"
    )
    with log_path.open(log_mode, newline="") as log_file:
        writer = csv.writer(log_file)
        if log_mode == "w":
            writer.writerow(
                [
                    "step",
                    "loss",
                    "ess",
                    "var_dt_log_p_tilde",
                    "var_estimator_integrand",
                    "cv_var_ratio",
                    "grad_norm",
                    "rate_pair_mean",
                    "rate_pair_p99",
                    "lambda_dt_clipped_frac",
                    "lambda_dt_p99",
                    "log_ratio_clamp_frac",
                    "proposal_drop_frac",
                    "events_per_site_per_step",
                    "rollout_resample_events",
                    "sigma_current",
                    "lr_current",
                    "c_t_ema_rms_delta",
                    "c_t_offset_rms",
                    "grad_sqnorm_slice_mean",
                    "wall_clock_step_s",
                ]
            )

        step = start_step
        curriculum_idx = -1
        x_replay_chunks: list[torch.Tensor] = []
        t_idx_replay_chunks: list[torch.Tensor] = []
        replay_sigma = float(target.sigma)
        current_intended_lr = float(train_cfg.lr)
        warmup_steps = int(getattr(train_cfg, "warmup_steps", 0))
        # rewarmup_on_stage re-runs the warmup ramp from each sigma
        # transition; off => anchor stays 0, the archived step-0-only ramp.
        rewarmup_on_stage = bool(getattr(train_cfg, "rewarmup_on_stage", False))
        warmup_anchor = 0
        # flush_replay_on_stage empties the buffer at each sigma transition
        # (True = every archived run). False keeps the window: the loss
        # recomputes both target terms at the live sigma, so retained states
        # are evaluation points under the new target, not stale labels. The
        # c_t EMA reset below is unconditional either way.
        flush_replay_on_stage = bool(getattr(train_cfg, "flush_replay_on_stage", True))
        # Per-stage best checkpoint: save the head whenever the trailing
        # median (window 3) of the train-eval ESS makes a new stage best. A
        # median, because single-step train-ESS peaks on the archived 16x16
        # record are noise excursions. Raw weights only: the EMA shadow lags
        # mid-stage. Pure IO; the flag is off in every archived config.
        stage_best_enabled = bool(getattr(train_cfg, "stage_best_checkpoints", False))
        stage_best_json_path = ckpt_dir / "stage_best.json"
        # Reload on resume so best-so-far survives preemption; the trailing
        # window itself restarts, which can only delay a re-save, not fake one.
        stage_best_records: dict = (
            json.loads(stage_best_json_path.read_text())
            if stage_best_enabled and stage_best_json_path.exists()
            else {}
        )
        stage_ess_recent: list[float] = []
        stage_ess_stage = -1

        if resume_state is not None:
            # Fast-forward without replaying sigma transitions that clear
            # restored chunks. Keep the loaded optimiser LR, including
            # warmup scaling at the boundary.
            while (
                curriculum
                and curriculum_idx + 1 < len(curriculum)
                and step >= curriculum[curriculum_idx + 1][0]
            ):
                curriculum_idx += 1
                _start, sigma_now, lr_now = curriculum[curriculum_idx]
                if lr_now is not None:
                    current_intended_lr = float(lr_now)
            if curriculum_idx >= 0:
                target.set_sigma(curriculum[curriculum_idx][1])
                if rewarmup_on_stage:
                    # Anchor at the resumed stage's start so a mid-ramp
                    # resume continues the same ramp it left.
                    warmup_anchor = int(curriculum[curriculum_idx][0])
            x_replay_chunks = [
                chunk.to(device) for chunk in resume_state["x_replay_chunks"]
            ]
            t_idx_replay_chunks = [
                chunk.to(device) for chunk in resume_state["t_idx_replay_chunks"]
            ]
            replay_sigma = float(resume_state["replay_sigma"])
            if c_t_ema is not None:
                saved_c_t_ema = resume_state.get("c_t_ema")
                if saved_c_t_ema is not None:
                    c_t_ema.load_state_dict(saved_c_t_ema)
                else:
                    # Checkpoint predating the c_t EMA: the next cycle passes
                    # through raw, a one-cycle lag vs the archived trajectory.
                    print(
                        "[train_swap] WARNING: resume.pt has no c_t_ema "
                        "state; grid EMA re-seeded at the next cycle",
                        flush=True,
                    )
            # RNG restore comes last so nothing above can perturb the
            # stream the continuation will consume.
            restore_rng_state(resume_state)

        # Init-basin diagnostic at t=0 before the first optimiser step; RNG
        # state is saved and restored so it does not perturb training.
        # Skipped on resume: it belongs to step 0 and is already on disk.
        if resume_state is None:
            rng_state_cpu = torch.get_rng_state()
            rng_state_cuda = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
            with torch.no_grad():
                x_diag = target.sample_base(outer_batch, device=device)
                t_diag = torch.zeros(outer_batch, device=device)
                init_diag = _swap_rate_diagnostics(
                    head,
                    x_diag,
                    t_diag,
                    step_dt=1.0 / max(n_grid - 1, 1),
                    target=target,
                )
            torch.set_rng_state(rng_state_cpu)
            if rng_state_cuda is not None:
                torch.cuda.set_rng_state(rng_state_cuda)
            (output_dir / "init_diagnostics.json").write_text(
                json.dumps(init_diag, indent=2)
            )
            if use_wandb:
                wandb.log({f"init/{k}": v for k, v in init_diag.items()}, step=0)

        for outer in range(start_outer, n_outer):
            # Update σ before rebuilding the buffer so inner-step samples are
            # consistent with the σ they will be trained against. Curriculum
            # runs are piecewise-constant plateaus.
            if curriculum:
                while (
                    curriculum_idx + 1 < len(curriculum)
                    and step >= curriculum[curriculum_idx + 1][0]
                ):
                    curriculum_idx += 1
                    _start, sigma_now, lr_now = curriculum[curriculum_idx]
                    target.set_sigma(sigma_now)
                    if lr_now is not None:
                        _set_optimizer_lr(optimiser, lr_now)
                        current_intended_lr = float(lr_now)
                    if sigma_now != replay_sigma:
                        if flush_replay_on_stage:
                            _clear_replay(x_replay_chunks, t_idx_replay_chunks)
                        replay_sigma = sigma_now
                        if c_t_ema is not None:
                            # c_t is a function of sigma: smoothing must
                            # never mix estimates across the boundary.
                            c_t_ema.reset()
                        if rewarmup_on_stage:
                            warmup_anchor = step
                    if use_wandb:
                        wandb.log(
                            {
                                "train/sigma_current": sigma_now,
                                "train/lr_current": optimiser.param_groups[0]["lr"],
                                "train/curriculum_stage": curriculum_idx,
                            },
                            step=step,
                        )

            # OUTER STEP -- rebuild buffer + c_t. Trajectory and c_t are
            # both detached from autograd by the no_grad block; this is
            # the paper's R_t^{θ_sg} (stop-gradient) treatment.
            t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)
            # c_t's standard error falls 1/sqrt(M) in the rollout row
            # count; the rollout is the c_t estimator's sample size.
            # n_rollout = outer_batch when the knob is off (byte-identical).
            n_rollout = outer_batch if c_t_batch is None else c_t_batch
            x_initial = target.sample_base(n_rollout, device=device)
            outer_matching_stats: dict | None = {} if multi_event else None
            # In CV mode the rollout hands back the per-slot ξ_t from its own
            # head forwards and the grid recompute is skipped. Naive mode has
            # no head in the integrand, so the grid path stays.
            reuse_rollout_integrand = (
                c_t_from_rollout and estimator_mode == "control_variate"
            )
            with torch.no_grad():
                rollout_result = sample_swap_ctmc(
                    head,
                    x_initial,
                    t_grid,
                    return_all_states=True,
                    multi_event=multi_event,
                    matching_stats=outer_matching_stats,
                    target=target,
                    resampling=rollout_resampling,
                    return_cv_integrand=reuse_rollout_integrand,
                )  # (T, n_rollout, D)
                if reuse_rollout_integrand:
                    x_traj_full, integrand_per_t = rollout_result
                    rollout_resample_events = float("nan")
                elif rollout_resampling is None:
                    x_traj_full = rollout_result
                    rollout_resample_events = float("nan")
                else:
                    # No log-weights come back: after the resets they are a
                    # per-segment residue (the `ess` column is its own plain-IS
                    # draw). Every slice is the post-resample equally-weighted
                    # ensemble, so the c_t mean below is a mean over p_t.
                    x_traj_full, rollout_smc_stats = rollout_result
                    rollout_resample_events = float(rollout_smc_stats.n_events)
                if reuse_rollout_integrand:
                    # c_t = mean_m ξ_t (Eq. 8), the reduction
                    # compute_c_t_grid_swap applies: a plain mean for a
                    # specialist, a within-slice (T, K) mean on a mixture.
                    c_t_grid = reduce_c_t_grid(integrand_per_t, x_traj_full[0], target)
                else:
                    c_t_grid, integrand_per_t = compute_c_t_grid_swap(
                        t_grid,
                        x_traj_full,
                        target,
                        head,
                        mode=estimator_mode,
                        chunk_rows=c_t_grid_chunk_rows,
                    )  # (T,), (T, n_rollout)
                # Smooth the grid across cycles (first cycle after reset
                # passes through raw). The rms delta logs how much correction
                # the EMA applies: 0.0 on passthrough cycles.
                c_t_ema_rms_delta = float("nan")
                if c_t_ema is not None:
                    raw_c_t_grid = c_t_grid
                    c_t_grid = c_t_ema.update(c_t_grid)
                    c_t_ema_rms_delta = (
                        (raw_c_t_grid - c_t_grid).pow(2).mean().sqrt().item()
                    )

                # Mean within-slot variance over the full rollout set: each
                # slot has its own ∂_t log p̃ baseline, so this is estimator
                # noise per time slot on the c_t estimator's own rows. In
                # naive mode the integrand is ∂_t log p̃_t on these rows, so
                # the reuse knob skips the (T·M) recompute (cv_var_ratio = 1).
                if c_t_from_rollout and estimator_mode == "naive_mc":
                    naive_per_t = integrand_per_t
                else:
                    t_grid_per_state = t_grid.repeat_interleave(n_rollout)
                    x_traj_flat = x_traj_full.reshape(n_grid * n_rollout, n_dims)
                    naive_per_t = target.dt_log_p_tilde_t(
                        x_traj_flat,
                        t_grid_per_state,
                    ).reshape(n_grid, n_rollout)
                var_dt_log_p_tilde = naive_per_t.var(dim=-1).mean().item()
                var_estimator_integrand = integrand_per_t.var(dim=-1).mean().item()

            # CV-inversion observer: controlled/naive integrand variance
            # ratio on the same rollout rows (exactly 1.0 in naive mode, a
            # wiring self-check). Sustained ratio > 1 is a 5/5 in-run
            # classifier of the d256 cold-CV inversion (cold: never < 1.5
            # across 5k steps; healthy warm: below 1 within ~1000 steps).
            cv_var_ratio = (
                var_estimator_integrand / var_dt_log_p_tilde
                if var_dt_log_p_tilde > 0
                else float("nan")
            )
            cv_var_ratio_history.append(cv_var_ratio)
            if (
                halt_on_cv_inversion_after is not None
                and estimator_mode == "control_variate"
                and step >= halt_on_cv_inversion_after
                and _cv_inversion_sustained(
                    cv_var_ratio_history, halt_cv_inversion_window
                )
            ):
                # Tripwire for CV continuation cells (None = off, every
                # archived cell untouched): stop a run the classifier has
                # called. Marker for the judge; loop exits; final.pt saves.
                (output_dir / "cv_inversion_halt.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "window": halt_cv_inversion_window,
                            "trailing_ratios": cv_var_ratio_history[
                                -halt_cv_inversion_window:
                            ],
                        },
                        indent=2,
                    )
                )
                break

            # The buffer takes the first outer_batch rows of the rollout;
            # c_t above used all n_rollout. Base draws are iid, so a prefix
            # is a uniform subset. .contiguous() releases the enlarged
            # storage (replay chunks are views; a c_t_batch=512 chunk is
            # ~400 MB at d256); a no-op view when n_rollout == outer_batch.
            #
            # After a resample the rows are sorted by ancestor
            # (`systematic_resample_indices` returns CDF order, pinned in
            # tests/test_resampling.py), so a prefix over-represents the
            # low-CDF end and clusters duplicate lineages: shuffle first.
            # Skipped otherwise so the flag-off path consumes no RNG.
            if rollout_resampling is not None and n_rollout > outer_batch:
                shuffled_rows = torch.randperm(n_rollout, device=device)
                x_traj_full = x_traj_full[:, shuffled_rows]
            x_traj = x_traj_full[:, :outer_batch].contiguous()
            del x_traj_full

            # Matching-step fidelity for this cycle's buffer states (constant
            # across its inner rows). The Luby matching silently drops
            # proposals still contested after its round budget; these two
            # columns show the run stayed in the validated regime
            # (drop_frac ~< 1%, events/site/step <= 0.1).
            if multi_event and outer_matching_stats.get("state_steps"):
                proposed_total = float(outer_matching_stats["proposed"])
                accepted_total = float(outer_matching_stats["accepted"])
                proposal_drop_frac = (
                    1.0 - accepted_total / proposed_total if proposed_total > 0 else 0.0
                )
                events_per_site_per_step = accepted_total / (
                    float(outer_matching_stats["state_steps"]) * n_dims
                )
            else:
                proposal_drop_frac = float("nan")
                events_per_site_per_step = float("nan")

            t_idx_buffer = torch.arange(n_grid, device=device).repeat_interleave(
                outer_batch
            )
            # Flatten and retain the most recent outer trajectory batches for
            # uniform inner-step sampling. `c_t_grid` intentionally remains
            # the latest outer-step estimate, matching the public DNFS code's
            # OnlineData(update_dt_log_Zt=False) behaviour.
            x_buffer, t_idx_buffer = _append_replay_buffer(
                x_replay_chunks,
                t_idx_replay_chunks,
                x_traj,
                t_idx_buffer,
                replay_buffer_cycles,
            )
            buffer_size = x_buffer.shape[0]

            # Δ accumulators reset per outer cycle: c_t is fixed across the
            # cycle's inner steps, so that is the window over which a per-slot
            # residual mean estimates one Δ_t. Indexed by (slot, slice) so
            # that on a mixture the per-slice offsets, equal and opposite
            # around the pooled mean, cannot cancel in the signed slot mean.
            delta_residual_sum = torch.zeros(
                n_grid * n_composition_slices, device=device
            )
            delta_residual_count = torch.zeros(
                n_grid * n_composition_slices, device=device
            )

            for _inner in range(inner_steps_per_outer):
                step_start = time.time()

                # LR warmup: linear ramp from 0 to current_intended_lr over
                # `warmup_steps` inner updates, multiplicative so it composes
                # with curriculum LR transitions.
                if warmup_steps > 0:
                    warmup_rel_step = step - warmup_anchor
                    if warmup_rel_step < warmup_steps:
                        warmup_scale = (warmup_rel_step + 1) / warmup_steps
                        _set_optimizer_lr(optimiser, current_intended_lr * warmup_scale)
                    elif warmup_rel_step == warmup_steps:
                        _set_optimizer_lr(optimiser, current_intended_lr)

                # INNER STEP -- N uniform draws from buffer (paper line 7).
                sample_idx = torch.randint(
                    buffer_size,
                    (inner_batch,),
                    device=device,
                )
                x_sample = x_buffer[sample_idx]  # (N, D)
                t_idx_sample = t_idx_buffer[sample_idx]  # (N,)
                t_sample = t_grid[t_idx_sample]  # (N,)
                # Each row's baseline is its slice's ∂_t log Z_t: the grid
                # is (T,) for a specialist and (T, K) on a mixture, and
                # the slice is read off the state (swaps conserve it).
                slice_sample = slice_index_of(target, x_sample)  # (N,)
                c_t_sample = c_t_grid.reshape(n_grid, -1)[
                    t_idx_sample, slice_sample
                ]  # (N,)

                # loss_microbatch_size slices the backward over batch rows
                # (gradient-identical; None = the archived single backward).
                # The per-slice gradient squared-norms pair with the pre-clip
                # grad_norm for gradient_noise_scale_components.
                optimiser.zero_grad()
                slice_grad_sqnorms: list | None = (
                    [] if loss_microbatch_size is not None else None
                )
                # bf16 on the loss update only: the head keeps G fp32 in its
                # autocast-disabled readout, and the swap log-ratio is
                # bit-identical (small-integer field sum). Eval stays fp32.
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=train_autocast_bf16,
                ):
                    loss_value, residual_sample = loss_swap_backward_microbatched(
                        x_sample,
                        t_sample,
                        c_t_sample,
                        head,
                        target,
                        microbatch_size=loss_microbatch_size,
                        slice_grad_sqnorms_out=slice_grad_sqnorms,
                    )
                # Mean over full slices only: a ragged tail is a different
                # batch size b and would bias E|g_b|^2.
                full_slice_sqnorms = [
                    sqnorm
                    for rows, sqnorm in (slice_grad_sqnorms or [])
                    if rows == loss_microbatch_size
                ]
                grad_sqnorm_slice_mean = (
                    sum(full_slice_sqnorms) / len(full_slice_sqnorms)
                    if full_slice_sqnorms
                    else float("nan")
                )
                # Δ diagnostic: −E[residual] per slot is the offset between
                # c_t and the buffer mean of ξ_t, the only channel by which
                # c_t reaches the gradient. Accumulated over the cycle (one
                # inner batch gives ~1 sample per slot); sign flipped on readout.
                delta_slot = t_idx_sample * n_composition_slices + slice_sample
                delta_residual_sum.index_add_(0, delta_slot, residual_sample)
                delta_residual_count.index_add_(
                    0, delta_slot, torch.ones_like(residual_sample)
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    getattr(train_cfg, "grad_clip_max_norm", 500.0),
                )
                optimiser.step()
                if ema is not None:
                    ema.update()

                wall_clock_step_s = time.time() - step_start

                ess_value = float("nan")
                rate_diag = {
                    "rate_pair_mean": float("nan"),
                    "rate_pair_p99": float("nan"),
                    "lambda_dt_clipped_frac": float("nan"),
                    "lambda_dt_p99": float("nan"),
                    "log_ratio_clamp_frac": float("nan"),
                }
                if step % eval_cfg.eval_every == 0:
                    eval_autocast = torch.autocast(
                        device.type,
                        dtype=torch.bfloat16,
                        enabled=getattr(eval_cfg, "eval_autocast_bf16", False),
                    )
                    with torch.no_grad(), eval_autocast:
                        eval_grid = torch.linspace(
                            0.0,
                            1.0,
                            n_grid,
                            device=device,
                        )
                        # Stream the eval draw in slices: the swap head builds
                        # (d*B)-row attention buffers and OOMs at large d; IS
                        # weights are independent per sample, so slicing is
                        # exact. run.py's final eval draws the full n_eval_samples.
                        n_train_eval = (
                            getattr(eval_cfg, "n_eval_samples_training", None)
                            or eval_cfg.n_eval_samples
                        )
                        eval_chunk = (
                            getattr(eval_cfg, "eval_sample_chunk", None) or n_train_eval
                        )
                        log_weight_slices = []
                        remaining = n_train_eval
                        while remaining > 0:
                            n_slice = min(eval_chunk, remaining)
                            x_eval_initial = target.sample_base(n_slice, device=device)
                            _, slice_log_weights = sample_swap_ctmc(
                                head,
                                x_eval_initial,
                                eval_grid,
                                return_log_weights=True,
                                target=target,
                                multi_event=multi_event,
                            )
                            log_weight_slices.append(slice_log_weights)
                            remaining -= n_slice
                        log_weights = torch.cat(log_weight_slices)
                        ess_value = ess_from_log_weights(log_weights).item()
                        rate_diag = _swap_rate_diagnostics(
                            head,
                            x_sample,
                            t_sample,
                            step_dt=1.0 / max(n_grid - 1, 1),
                            target=target,
                        )
                    torch.save(head.state_dict(), ckpt_dir / "latest.pt")

                    if stage_best_enabled and not math.isnan(ess_value):
                        current_stage = max(curriculum_idx, 0)
                        if current_stage != stage_ess_stage:
                            # New stage: the window must not mix ESS values
                            # across a sigma boundary (the target changed).
                            stage_ess_recent.clear()
                            stage_ess_stage = current_stage
                        stage_ess_recent.append(ess_value)
                        if len(stage_ess_recent) > 3:
                            del stage_ess_recent[0]
                        ess_trailing_median = statistics.median(stage_ess_recent)
                        stage_key = str(current_stage)
                        stage_best = stage_best_records.get(stage_key)
                        if (
                            stage_best is None
                            or ess_trailing_median > stage_best["ess_trailing_median"]
                        ):
                            torch.save(
                                head.state_dict(),
                                ckpt_dir / f"best_stage{current_stage}.pt",
                            )
                            stage_best_records[stage_key] = {
                                "step": step,
                                "ess_trailing_median": ess_trailing_median,
                            }
                            stage_best_json_path.write_text(
                                json.dumps(stage_best_records, indent=2)
                            )

                c_t_offset_value = c_t_offset_rms(
                    delta_residual_sum, delta_residual_count
                )
                writer.writerow(
                    [
                        step,
                        loss_value.item(),
                        ess_value,
                        var_dt_log_p_tilde,
                        var_estimator_integrand,
                        cv_var_ratio,
                        grad_norm.item(),
                        rate_diag["rate_pair_mean"],
                        rate_diag["rate_pair_p99"],
                        rate_diag["lambda_dt_clipped_frac"],
                        rate_diag["lambda_dt_p99"],
                        rate_diag["log_ratio_clamp_frac"],
                        proposal_drop_frac,
                        events_per_site_per_step,
                        rollout_resample_events,
                        float(target.sigma),
                        optimiser.param_groups[0]["lr"],
                        c_t_ema_rms_delta,
                        c_t_offset_value,
                        grad_sqnorm_slice_mean,
                        wall_clock_step_s,
                    ]
                )
                log_file.flush()

                if use_wandb:
                    log_dict = {
                        "train/loss": loss_value.item(),
                        "train/var_dt_log_p_tilde": var_dt_log_p_tilde,
                        "train/var_estimator_integrand": var_estimator_integrand,
                        "train/cv_var_ratio": cv_var_ratio,
                        "train/grad_norm": grad_norm.item(),
                        "train/sigma_current": float(target.sigma),
                        "train/lr_current": optimiser.param_groups[0]["lr"],
                        "train/c_t_ema_rms_delta": c_t_ema_rms_delta,
                        "train/c_t_offset_rms": c_t_offset_value,
                        "train/grad_sqnorm_slice_mean": grad_sqnorm_slice_mean,
                        "train/wall_clock_step_s": wall_clock_step_s,
                    }
                    if step % eval_cfg.eval_every == 0:
                        log_dict["train/ess"] = ess_value
                        log_dict.update(
                            {f"train/{key}": value for key, value in rate_diag.items()}
                        )
                    wandb.log(log_dict, step=step)

                step += 1

            # End-of-cycle boundary: full state is closed here (the next
            # cycle rebuilds trajectory + c_t from scratch), so this is the
            # only place a resume checkpoint is valid.
            if (outer + 1) % resume_every_outer == 0 or outer == n_outer - 1:
                _save_resume_state(
                    ckpt_dir,
                    step=step,
                    head=head,
                    optimiser=optimiser,
                    x_replay_chunks=x_replay_chunks,
                    t_idx_replay_chunks=t_idx_replay_chunks,
                    replay_sigma=replay_sigma,
                    ema=ema,
                    c_t_ema=c_t_ema,
                )
                if on_checkpoint is not None:
                    on_checkpoint()

    torch.save(head.state_dict(), ckpt_dir / "final.pt")
    if ema is not None:
        # Full loadable state dict with EMA parameters and the original
        # buffers: swap the shadow in, save, swap back.
        ema.swap_in()
        torch.save(head.state_dict(), ckpt_dir / "final_ema.pt")
        ema.swap_out()
