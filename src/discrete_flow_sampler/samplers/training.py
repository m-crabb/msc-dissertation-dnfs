"""Outer/inner training loop per paper Algorithm 1 (App. C.1).

Outer step (1 per `inner_steps_per_outer` gradient updates):
    1. Build time grid t_grid = linspace(0, 1, T) where T = n_euler_steps.
    2. Sample M trajectories from t=0 to t=1 under torch.no_grad with the
       *current* model (paper's R_t^{θ_sg} -- stop-gradient by construction
       since we're inside no_grad).
    3. Cache the (T, M, D) trajectory plus a (T,) c_t_grid -- the per-time-
       slot scalar c_t computed by averaging an integrand over the M outer-
       batch samples (paper Algorithm 1 line 4). c_t is also detached.

Inner step (× inner_steps_per_outer, one gradient update each):
    1. Draw N entries uniformly from the (T*M)-entry buffer (paper line 7
       -- mixed t per inner mini-batch by uniform sampling).
    2. Look up c_t per-sample by t-index. Compute kolmogorov_loss
       (= squared Eq. 7 residual) at the current θ -- ξ_θ flows gradient,
       c_t is constant.
    3. Backward + Adam step.

The outer-step amortises the trajectory simulation over `inner_steps_per_outer`
gradient updates (paper default 100), reducing the per-gradient-step cost
from O(T) to O(2T/inner_steps + 1) forward-equivalents -- the 25x speedup
identified in the training-loop investigation.

Eval: existing full t=0->1 IS-trajectory + ESS, gated on the inner-step
counter so `eval_every` keeps its meaning.
"""
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from discrete_flow_sampler.composition import draw_composition
from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.ema import ExponentialMovingAverage
from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned,
)
from discrete_flow_sampler.samplers._neighbours import (
    _log_p_tilde_at_neighbours,
    log_ratio_clamp,
)
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid
from discrete_flow_sampler.samplers.resampling import ResamplingConfig
from discrete_flow_sampler.samplers.resume import (
    capture_rng_state,
    load_resume_state,
    restore_rng_state,
    save_resume_state,
    truncate_log_to_step,
)
from discrete_flow_sampler.seeding import seed_everything


def _retain_chunks(
    chunks: list[torch.Tensor],
    new_chunk: torch.Tensor,
    max_cycles: int,
) -> torch.Tensor:
    """Append one outer batch to a replay chunk list; return the retained view.

    The single expression of the replay retention rule. Any per-state
    quantity the trainer wants to keep alongside the states — time index,
    and for an amortised run the composition and its ∂_t log Z_t baseline —
    is retained by calling this with its own chunk list, so every parallel
    quantity is appended and evicted on exactly the same schedule and cannot
    drift out of alignment with the states it describes.

    Replication-discipline consult log: the quarantined DNFS reference repo
    was consulted 2026-08-19, under explicit authorisation and for
    comparison only, to resolve a paper-vs-code discrepancy in this rule.
    Algorithm 1 line 5 prints an unbounded ``B <- B U {...}`` with no
    eviction; the released code bounds retention to ~1024 trajectories
    (``DataBuffer.max_size = 1024 // outer_batch``, FIFO pop-oldest). The
    bounded FIFO here matches the reference code in kind, though our
    invariant is a constant cycle count (``max_cycles``) rather than a
    constant trajectory count. No reference source was copied.
    """
    chunks.append(new_chunk.detach())
    if len(chunks) > max_cycles:
        chunks.pop(0)
    return torch.cat(chunks, dim=0)


def _append_replay_buffer(
    x_chunks: list[torch.Tensor],
    t_idx_chunks: list[torch.Tensor],
    x_traj: torch.Tensor,
    t_idx_buffer: torch.Tensor,
    max_cycles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one outer batch and return the retained replay-buffer view."""
    return (
        _retain_chunks(
            x_chunks, x_traj.reshape(-1, x_traj.shape[-1]), max_cycles
        ),
        _retain_chunks(t_idx_chunks, t_idx_buffer, max_cycles),
    )


def _normalise_curriculum(
    curriculum,
    *,
    n_steps: int,
    inner_steps_per_outer: int,
    value_attr: str = "sigma",
) -> list[tuple[int, float, float | None]]:
    name = f"{value_attr}_curriculum"
    if curriculum is None:
        return []

    stages = []
    for stage in curriculum:
        start_step = int(getattr(stage, "start_step"))
        value = float(getattr(stage, value_attr))
        lr = getattr(stage, "lr", None)
        stages.append((start_step, value, None if lr is None else float(lr)))

    if not stages:
        raise ValueError(f"{name} must contain at least one stage")
    if stages[0][0] != 0:
        raise ValueError(f"{name} first stage must start at step 0")

    prev_step = -1
    for start_step, _value, _lr in stages:
        if start_step <= prev_step:
            raise ValueError(f"{name} stages must be strictly increasing")
        if start_step >= n_steps:
            raise ValueError(
                f"curriculum start_step={start_step} must be < n_steps={n_steps}"
            )
        if start_step % inner_steps_per_outer != 0:
            raise ValueError(
                f"curriculum start_step={start_step} must align with "
                f"inner_steps_per_outer={inner_steps_per_outer}"
            )
        prev_step = start_step
    return stages


def _set_optimizer_lr(optimiser: torch.optim.Optimizer, lr: float) -> None:
    for group in optimiser.param_groups:
        group["lr"] = lr


def _clear_replay(*chunk_lists: list[torch.Tensor]) -> None:
    """Drop every retained chunk. Called when the *target* moves.

    σ and λ stage boundaries invalidate retained states and their baselines,
    so the buffer is emptied. A change to the *distribution compositions are
    drawn from* is not such a boundary: every retained state still carries
    the composition and baseline it was generated under, so it stays valid
    training data and must be kept.
    """
    for chunks in chunk_lists:
        chunks.clear()


_GRADIENT_GROUPS = (
    "gains",
    "omega",
    "composition_embedder",
    "trunk",
)


def _gradient_group_norms(model) -> dict[str, float]:
    """Pre-clip L2 norms partitioned by the soft-collapse mechanism groups.

    Every parameter with a gradient lands in exactly one group. The foreach
    norm keeps the opt-in diagnostic to one multi-tensor reduction per group
    rather than launching a reduction for every transformer tensor.
    """
    grouped: dict[str, list[torch.Tensor]] = {
        name: [] for name in _GRADIENT_GROUPS
    }
    reference = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        reference = gradient if reference is None else reference
        components = name.split(".")
        if name in {
            "gain_constant",
            "gain_slope",
            "composition_gain_constant",
            "composition_gain_slope",
        }:
            group = "gains"
        elif "omega" in components:
            group = "omega"
        elif "comp_embedder" in components:
            group = "composition_embedder"
        else:
            group = "trunk"
        grouped[group].append(gradient)

    if reference is None:
        return {f"grad_norm_{name}": 0.0 for name in _GRADIENT_GROUPS}

    norm_tensors = []
    for name in _GRADIENT_GROUPS:
        gradients = grouped[name]
        if gradients:
            parameter_norms = torch._foreach_norm(gradients, 2.0)
            group_norm = torch.linalg.vector_norm(torch.stack(parameter_norms))
        else:
            group_norm = reference.new_zeros(())
        norm_tensors.append(group_norm)
    # The exact-field wrapper historically registers its scalar gains after
    # the inner model has moved to CUDA, so those gains remain CPU scalars
    # while omega/embedder/trunk gradients live on the accelerator. PyTorch's
    # optimizer and clip_grad_norm_ support that mixed-device parameter list;
    # collect each already-reduced scalar independently rather than stacking
    # group norms across devices.
    values = [float(group_norm.cpu()) for group_norm in norm_tensors]
    return {
        f"grad_norm_{name}": float(value)
        for name, value in zip(_GRADIENT_GROUPS, values)
    }


def _rate_diagnostics(
    model, x, t, step_dt: float, *, target=None
) -> dict[str, float]:
    """Cheap eval-time diagnostics for CTMC rate scale.

    ESS alone cannot distinguish a no-op sampler (rates near zero) from a
    stiff sampler (rates so large Euler probabilities clip). Logging per-site
    outflow rates at eval cadence makes those failure modes visible without
    changing the training objective.

    When `target` is supplied AND the model is locally equivariant, also
    logs the saturation fraction of the log-target ratio against whatever
    ceiling is actually in force at `kolmogorov.residual_lenet` and
    `ctmc._compute_xi_t_lenet` — `_neighbours.log_ratio_clamp(target)`,
    the paper's 5.0 unless the target overrides it. Reading the live value
    rather than a literal is what keeps the column comparable across cells
    that vary the ceiling: it always means "saturating the clamp this run
    is using". `log_ratio_clamp_frac` is the share of (B, d, S) entries
    that exceed the ceiling; `log_ratio_p99` is the unclipped 99th
    percentile so the magnitude of the saturated tail is visible
    (saturation alone is ambiguous between "just above" and "an order of
    magnitude above").
    """
    if getattr(model, "is_locally_equivariant", False):
        rates = F.relu(model(x, t)).sum(dim=-1)  # (B, d), per-site outflow
    else:
        rates = model(x, t)                      # (B, d), per-site outflow

    flip_prob = rates * step_dt
    metrics = {
        "rate_site_mean": rates.mean().item(),
        "rate_site_p99": torch.quantile(rates.reshape(-1), 0.99).item(),
        "flip_prob_site_p99": torch.quantile(flip_prob.reshape(-1), 0.99).item(),
        "flip_prob_clipped_frac": (flip_prob > 1.0).float().mean().item(),
        "log_ratio_clamp_frac": float("nan"),
        "log_ratio_p99": float("nan"),
    }

    if target is not None and getattr(model, "is_locally_equivariant", False):
        log_p_neighbours = _log_p_tilde_at_neighbours(
            x, t, target, model.vocab_size
        )
        log_p_x = target.log_p_tilde_t(x, t)
        log_ratio = log_p_neighbours - log_p_x[:, None, None]
        metrics["log_ratio_clamp_frac"] = (
            (log_ratio > log_ratio_clamp(target)).float().mean().item()
        )
        metrics["log_ratio_p99"] = torch.quantile(
            log_ratio.reshape(-1), 0.99
        ).item()

    return metrics


def train(
    model,
    target,
    train_cfg,
    ctmc_cfg,
    eval_cfg,
    output_dir: Path,
    *,
    use_wandb: bool = True,
    estimator_mode: str = "control_variate",
    sigma_curriculum=None,
    lambda_curriculum=None,
    composition_centre=None,
    composition_half_width: float = 0.0,
    composition_values=None,
    composition_curriculum=None,
    on_checkpoint=None,
    ema_decay: float = 0.0,
):
    """Run paper Algorithm 1 for `train_cfg.n_steps` total inner steps.

    Args:
        model: a `RateMatrix` -- already instantiated, on `target.device`.
        target: an `IsingTarget`.
        train_cfg: object with
            .n_steps              -- total inner gradient updates.
            .batch_size           -- inner mini-batch N (paper line 7).
            .outer_batch_size     -- trajectories per outer step M
                                     (paper line 3); falls back to
                                     batch_size if None.
            .inner_steps_per_outer -- paper default 100.
            .replay_buffer_cycles -- number of recent outer batches retained
                                     in the replay buffer; default 1 preserves
                                     the pre-recovery behaviour. The public
                                     DNFS reference retains four N=256 outer
                                     batches via DataBuffer(max_size=1024/N).
            .lr, .seed.
        ctmc_cfg: object with .n_euler_steps -- length T of outer-step
            time grid (paper's K+1).
        eval_cfg: object with .eval_every (inner-step cadence) and
            .n_eval_samples.
        output_dir: per-run artefact directory.
        use_wandb: skip wandb.log when False (handy for tests).
        estimator_mode: "naive_mc" | "control_variate". Selects which
            integrand `compute_c_t_grid` averages per time slot.
        sigma_curriculum: optional piecewise-constant schedule of objects
            with `.start_step`, `.sigma`, and optional `.lr`. Stage
            boundaries clear the replay buffer so retained states are always
            drawn under the current target temperature.
        lambda_curriculum: optional piecewise-constant schedule of objects
            with `.start_step`, `.composition_penalty_strength`, and
            optional `.lr`. Anneals the soft-composition penalty (typically
            upward, so the physics is learned before the constraint
            tightens). Same boundary rules and replay-buffer clearing as
            `sigma_curriculum`.
        composition_centre: enables **amortised** training when set (with
            `composition_values`, either suffices). One target composition is
            drawn per outer cycle and the model is conditioned on it, so a
            single network serves many compositions. `None` leaves every code
            path here byte-identical to a specialist run.
        composition_half_width: initial half-width of the draw window
            [centre − w, centre + w]. Ignored when `composition_values` is set.
        composition_values: finite set to draw compositions from, instead of
            the continuous window.
        composition_curriculum: optional piecewise-constant schedule of
            objects with `.start_step`, `.half_width` and optional `.lr`,
            widening the draw window during training. Same boundary rules as
            the other two curricula but — deliberately — it does NOT clear the
            replay buffer; see `_clear_replay`.
        on_checkpoint: optional zero-arg callable invoked after each resume
            checkpoint lands on disk (Modal passes `volume.commit` so the
            checkpoint survives a preemption that skips the death-flush).

    Preemption resume: every `train_cfg.resume_every_outer` outer cycles
    (default 10) the full outer-boundary state is checkpointed to
    `checkpoints/resume.pt`; if that file exists on entry, training restores
    it and continues instead of starting over. The outer boundary is the only
    valid checkpoint instant — mid-cycle the replay buffer and c_t grid are
    half-rebuilt. What travels, and why each piece has to: model weights and
    AdamW moments (a fresh optimiser would re-warm its second-moment estimate
    and take a different-sized first step), the step counter (every curriculum
    is keyed on it), the torch RNG states (or the continuation diverges at the
    first draw despite identical weights), and all four replay-chunk lists
    (with `replay_buffer_cycles > 1` the buffer holds past outer trajectories
    reconstructible from nothing else — and for an amortised run each retained
    state carries its own composition and its own c_t baseline, so dropping
    those two lists would train states against baselines they never came
    with: a wrong number, not a crash). Curriculum stage indices are NOT
    stored — they are derivable from the step counter because every ladder
    uses absolute `start_step`s — and the optimiser LR travels inside the
    optimiser state dict, warmup scaling included.

    Amortisation and ∂_t log Z_t: the annealing path makes Z_t a function of
    the target composition, so the `c_t` baseline is only valid for the
    composition it was averaged over. Drawing one composition per outer cycle
    keeps that average over the full outer batch — no loss of precision
    relative to a specialist run — and each state then carries its own
    baseline through the replay buffer, so inner batches may freely mix
    compositions drawn across cycles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    seed_everything(train_cfg.seed)
    optimiser_kind = getattr(train_cfg, "optimiser", "adamw")
    if optimiser_kind == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
    elif optimiser_kind == "stable_adamw":
        # The normalised/trust-region update the amortisation forensics
        # recommend over a raw-gradient clip (see samplers/optim.py for the
        # algorithm and why the update-RMS threshold is the size-invariant
        # object the clip threshold is not).
        from discrete_flow_sampler.samplers.optim import StableAdamW

        optimiser = StableAdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
    else:
        raise ValueError(f"unknown optimiser {optimiser_kind!r}")

    # Restore before anything else reads model or optimiser state. The weights
    # loaded here are also what the RNG restore further down is paired with:
    # together they are the run, and half of either is a different run.
    resume_state = load_resume_state(ckpt_dir, map_location=target.device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimiser.load_state_dict(resume_state["optimiser"])
    start_step = int(resume_state["step"]) if resume_state is not None else 0

    # EMA dual-eval instrument (ported from the swap trainer for the
    # soft-chapter revamp, s95): a warmup-corrected parameter shadow over
    # the TOP-LEVEL module, so a channel-wrapped model contributes its gain
    # parameters too (tests/test_flip_trainer_ema.py pins the coverage).
    # Passive observer: updated after each optimiser step, never read by
    # training, so runs differing only in ema_decay train bit-identically.
    # Built after the resume restore above — the shadow's own state then
    # overwrites the fresh clone, carrying counter AND shadow across
    # preemption (re-initialising either re-creates the init-contamination
    # failure the warmup schedule exists to kill; see ema.py).
    ema = (
        ExponentialMovingAverage(model.parameters(), ema_decay, warmup=True)
        if ema_decay > 0 else None
    )
    if ema is not None and resume_state is not None:
        saved_ema = resume_state.get("ema")
        if saved_ema is not None:
            ema.load_state_dict(saved_ema)

    if use_wandb:
        import wandb

    n_dims = target.d
    device = target.device
    inner_batch = train_cfg.batch_size
    outer_batch = train_cfg.outer_batch_size or train_cfg.batch_size
    n_grid = ctmc_cfg.n_euler_steps
    inner_steps_per_outer = train_cfg.inner_steps_per_outer
    replay_buffer_cycles = getattr(train_cfg, "replay_buffer_cycles", 1)
    if replay_buffer_cycles < 1:
        raise ValueError(
            f"replay_buffer_cycles must be >= 1, got {replay_buffer_cycles}"
        )
    # ESS-triggered SMC resampling inside the buffer-rebuild rollout, the
    # flip-route twin of the swap trainer's wiring (LEAPS Alg. 1 lines
    # 11-14, whose trajectories Alg. 2 line 5 trains on). None = OFF =
    # every archived run, bit-identical. The full argument for why the c_t
    # batch mean survives it — and why the resample supplies the weighting
    # Eq. 8 asks for — is on the config field.
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
        ResamplingConfig(
            ess_threshold_fraction=float(rollout_resample_ess_fraction)
        )
        if rollout_resample_ess_fraction is not None
        else None
    )
    # B1-flip (optimisation decision, 2026-08-24): build the CV c_t grid
    # from the rollout's own model forwards instead of re-running them —
    # bit-identical to compute_c_t_grid (same tensors, same arithmetic,
    # no RNG; tests/test_cv_integrand_reuse.py); at T grid slots this
    # removes T-1 of the T model forwards the grid pass would pay.
    # Default False = archived behaviour, byte-identical.
    c_t_from_rollout = bool(getattr(train_cfg, "c_t_from_rollout", False))
    if c_t_from_rollout and rollout_resampling is not None:
        raise ValueError(
            "c_t_from_rollout requires rollout resampling OFF: resampled "
            "trajectories are certified only through the recompute path."
        )

    if train_cfg.n_steps % inner_steps_per_outer != 0:
        raise ValueError(
            f"n_steps={train_cfg.n_steps} not divisible by "
            f"inner_steps_per_outer={inner_steps_per_outer}; the outer/"
            f"inner contract requires whole outer cycles."
        )
    n_outer = train_cfg.n_steps // inner_steps_per_outer
    if start_step % inner_steps_per_outer != 0:
        raise ValueError(
            f"resume.pt records step {start_step}, not an outer-cycle "
            f"boundary (inner_steps_per_outer={inner_steps_per_outer}). "
            f"Mid-cycle the replay buffer and c_t grid are half-rebuilt, so "
            f"there is no consistent state to continue from."
        )
    start_outer = start_step // inner_steps_per_outer
    resume_every_outer = int(getattr(train_cfg, "resume_every_outer", 10))
    curriculum = _normalise_curriculum(
        sigma_curriculum,
        n_steps=train_cfg.n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
    )
    lambda_stages = _normalise_curriculum(
        lambda_curriculum,
        n_steps=train_cfg.n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
        value_attr="composition_penalty_strength",
    )
    composition_stages = _normalise_curriculum(
        composition_curriculum,
        n_steps=train_cfg.n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
        value_attr="half_width",
    )

    amortised = composition_centre is not None or composition_values is not None
    if amortised and not getattr(model, "condition_on_composition", False):
        raise ValueError(
            "amortised training was requested but the model was not built "
            "with composition conditioning; it would never see the "
            "composition its states were drawn under."
        )
    if amortised and composition_centre is None:
        composition_centre = float(sum(composition_values) / len(composition_values))
    half_width_now = float(composition_half_width)

    log_path = output_dir / "training_log.csv"
    # Append on resume, but only after dropping the rows a dead attempt
    # flushed past its last checkpoint -- those steps are about to be redone
    # and must not appear twice. A missing log falls back to a fresh header.
    log_mode = (
        "a"
        if resume_state is not None and truncate_log_to_step(log_path, start_step)
        else "w"
    )
    log_gradient_groups = bool(
        getattr(train_cfg, "log_gradient_group_norms", False)
    )
    gradient_log_path = output_dir / "gradient_group_log.csv"
    gradient_log_mode = None
    if log_gradient_groups:
        gradient_log_mode = (
            "a"
            if resume_state is not None
            and truncate_log_to_step(gradient_log_path, start_step)
            else "w"
        )
    gradient_log_context = (
        gradient_log_path.open(gradient_log_mode, newline="")
        if log_gradient_groups else nullcontext(None)
    )
    with (
        log_path.open(log_mode, newline="") as log_file,
        gradient_log_context as gradient_log_file,
    ):
        writer = csv.writer(log_file)
        if log_mode == "w":
            writer.writerow(
                ["step", "loss", "ess", "var_dt_log_p_tilde",
                 "var_estimator_integrand", "grad_norm",
                 "rate_site_mean", "rate_site_p99", "flip_prob_site_p99",
                 "flip_prob_clipped_frac", "log_ratio_clamp_frac",
                 "log_ratio_p99", "sigma_current", "lr_current",
                 "wall_clock_step_s", "composition_current",
                 "composition_half_width", "rollout_resample_events"]
            )
        gradient_writer = None
        if gradient_log_file is not None:
            gradient_writer = csv.writer(gradient_log_file)
            if gradient_log_mode == "w":
                gradient_writer.writerow(
                    [
                        "step",
                        "grad_norm_gains",
                        "grad_norm_omega",
                        "grad_norm_composition_embedder",
                        "grad_norm_trunk",
                        "grad_norm_reconstructed",
                        "grad_clip_scale",
                    ]
                )

        step = start_step
        curriculum_idx = -1
        lambda_idx = -1
        composition_idx = -1
        composition_now = float("nan")
        x_replay_chunks: list[torch.Tensor] = []
        t_idx_replay_chunks: list[torch.Tensor] = []
        # Parallel to the two above, retained on the same schedule: each
        # state's own composition and its own ∂_t log Z_t baseline. Populated
        # only for amortised runs.
        composition_replay_chunks: list[torch.Tensor] = []
        c_t_replay_chunks: list[torch.Tensor] = []
        replay_sigma = float(target.sigma)
        replay_lambda = float(
            getattr(target, "composition_penalty_strength", 0.0)
        )
        current_intended_lr = float(train_cfg.lr)
        warmup_steps = int(getattr(train_cfg, "warmup_steps", 0))

        centre_composition = (
            torch.full((1,), float(composition_centre), device=device)
            if amortised else None
        )

        def _bound(composition):
            """(model, context) pair for one composition binding.

            Returns the bare model and a no-op context when not amortising, so
            a specialist run executes exactly the code it always did.
            """
            if composition is None:
                return model, nullcontext()
            return (
                CompositionConditioned(model, composition),
                target.composition_batch(composition),
            )

        if resume_state is not None:
            # Fast-forward all three ladders EXPLICITLY rather than letting
            # the transition loops below replay them: those loops call
            # `_clear_replay` on a σ or λ change, which would destroy the
            # very chunks being restored two blocks down. The optimiser LR is
            # deliberately not set here -- `load_state_dict` above already
            # carries the exact value, warmup scaling included -- but
            # `current_intended_lr` still has to advance, because it is what
            # the warmup ramp multiplies and what a LATER stage boundary
            # would otherwise be measured against.
            while (
                curriculum
                and curriculum_idx + 1 < len(curriculum)
                and step >= curriculum[curriculum_idx + 1][0]
            ):
                curriculum_idx += 1
                _start, _sigma_now, lr_now = curriculum[curriculum_idx]
                if lr_now is not None:
                    current_intended_lr = float(lr_now)
            if curriculum_idx >= 0:
                target.set_sigma(curriculum[curriculum_idx][1])
            while (
                lambda_stages
                and lambda_idx + 1 < len(lambda_stages)
                and step >= lambda_stages[lambda_idx + 1][0]
            ):
                lambda_idx += 1
                _start, _lambda_now, lr_now = lambda_stages[lambda_idx]
                if lr_now is not None:
                    current_intended_lr = float(lr_now)
            if lambda_idx >= 0:
                target.set_composition_penalty_strength(
                    lambda_stages[lambda_idx][1]
                )
            while (
                composition_stages
                and composition_idx + 1 < len(composition_stages)
                and step >= composition_stages[composition_idx + 1][0]
            ):
                composition_idx += 1
                _start, half_width_now, lr_now = (
                    composition_stages[composition_idx]
                )
                if lr_now is not None:
                    current_intended_lr = float(lr_now)

            # The replay tags come from the checkpoint, not from the target:
            # they record the σ and λ the RETAINED states were drawn under,
            # which is what the next boundary compares against to decide
            # whether those states are still valid training data.
            x_replay_chunks = [
                chunk.to(device) for chunk in resume_state["x_replay_chunks"]
            ]
            t_idx_replay_chunks = [
                chunk.to(device)
                for chunk in resume_state["t_idx_replay_chunks"]
            ]
            composition_replay_chunks = [
                chunk.to(device)
                for chunk in resume_state["composition_replay_chunks"]
            ]
            c_t_replay_chunks = [
                chunk.to(device) for chunk in resume_state["c_t_replay_chunks"]
            ]
            replay_sigma = float(resume_state["replay_sigma"])
            replay_lambda = float(resume_state["replay_lambda"])
            # RNG restore comes LAST so nothing above can perturb the stream
            # the continuation is about to consume.
            restore_rng_state(resume_state)

        # Pre-training stiff-sampler / init-basin diagnostic. Computed at t=0
        # before the first optimiser step; RNG state is saved and restored so
        # the diagnostic does not perturb training-trajectory randomness.
        # The logged `flip_prob_clipped_frac` here is the only place the
        # init-time stiffness signal is captured -- the in-loop diagnostic
        # only ever sees the post-first-update model.
        # Skipped on resume: it describes step 0, it is already on disk, and
        # recomputing it against the resumed weights would both overwrite the
        # real init record and answer a different question than the file name
        # claims.
        if resume_state is None:
            rng_state_cpu = torch.get_rng_state()
            rng_state_cuda = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
            model_diag, bind_diag = _bound(centre_composition)
            with torch.no_grad(), bind_diag:
                x_diag = target.sample_base(outer_batch, device=device)
                t_diag = torch.zeros(outer_batch, device=device)
                init_diag = _rate_diagnostics(
                    model_diag, x_diag, t_diag,
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
                wandb.log(
                    {f"init/{k}": v for k, v in init_diag.items()}, step=0
                )

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
                        _clear_replay(
                            x_replay_chunks, t_idx_replay_chunks,
                            composition_replay_chunks, c_t_replay_chunks,
                        )
                        replay_sigma = sigma_now
                    if use_wandb:
                        wandb.log(
                            {
                                "train/sigma_current": sigma_now,
                                "train/lr_current": optimiser.param_groups[0]["lr"],
                                "train/curriculum_stage": curriculum_idx,
                            },
                            step=step,
                        )

            # λ annealing mirrors the σ curriculum: tighten the penalty on
            # outer-cycle boundaries and clear the replay buffer so retained
            # states are always drawn under the current soft target.
            if lambda_stages:
                while (
                    lambda_idx + 1 < len(lambda_stages)
                    and step >= lambda_stages[lambda_idx + 1][0]
                ):
                    lambda_idx += 1
                    _start, lambda_now, lr_now = lambda_stages[lambda_idx]
                    target.set_composition_penalty_strength(lambda_now)
                    if lr_now is not None:
                        _set_optimizer_lr(optimiser, lr_now)
                        current_intended_lr = float(lr_now)
                    if lambda_now != replay_lambda:
                        _clear_replay(
                            x_replay_chunks, t_idx_replay_chunks,
                            composition_replay_chunks, c_t_replay_chunks,
                        )
                        replay_lambda = lambda_now
                    if use_wandb:
                        wandb.log(
                            {
                                "train/lambda_current": lambda_now,
                                "train/lr_current": optimiser.param_groups[0]["lr"],
                                "train/lambda_stage": lambda_idx,
                            },
                            step=step,
                        )

            # Widen the composition draw window. Unlike the σ and λ stages
            # above this does NOT clear the replay buffer: the target has not
            # moved, so retained states remain valid training data, and
            # clearing would discard exactly what the widening accumulates.
            if composition_stages:
                while (
                    composition_idx + 1 < len(composition_stages)
                    and step >= composition_stages[composition_idx + 1][0]
                ):
                    composition_idx += 1
                    _start, half_width_now, lr_now = (
                        composition_stages[composition_idx]
                    )
                    if lr_now is not None:
                        _set_optimizer_lr(optimiser, lr_now)
                        current_intended_lr = float(lr_now)
                    if use_wandb:
                        wandb.log(
                            {
                                "train/composition_half_width": half_width_now,
                                "train/lr_current": optimiser.param_groups[0]["lr"],
                                "train/composition_stage": composition_idx,
                            },
                            step=step,
                        )

            # One composition per outer cycle, so the c_t below is averaged
            # over the full outer batch at a single composition.
            if amortised:
                composition_now = draw_composition(
                    composition_centre,
                    half_width_now,
                    composition_values,
                    quantise_to=target.composition_quantum,
                )
                cycle_composition = torch.full(
                    (1,), composition_now, device=device
                )
            else:
                cycle_composition = None
            model_cycle, bind_cycle = _bound(cycle_composition)

            # OUTER STEP -- rebuild buffer + c_t. Trajectory and c_t are
            # both detached from autograd by the no_grad block; this is
            # the paper's R_t^{θ_sg} (stop-gradient) treatment.
            t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)
            # B1-flip: in CV mode with the knob on, the rollout hands back
            # the per-slot ξ_t from its own forwards and the grid recompute
            # below is skipped. Naive mode is target-only, so the grid
            # path stays.
            reuse_rollout_integrand = (
                c_t_from_rollout and estimator_mode == "control_variate"
            )
            with bind_cycle:
                x_initial = target.sample_base(outer_batch, device=device)
                with torch.no_grad():
                    rollout_result = sample_ctmc(
                        model_cycle, x_initial, t_grid, return_all_states=True,
                        target=target, resampling=rollout_resampling,
                        return_cv_integrand=reuse_rollout_integrand,
                    )                                          # (T, M, D)
                    if reuse_rollout_integrand:
                        x_traj, integrand_per_t = rollout_result
                        rollout_resample_events = float("nan")
                    elif rollout_resampling is None:
                        x_traj = rollout_result
                        rollout_resample_events = float("nan")
                    else:
                        # No log-weights come back by design: after the
                        # resets they are a per-segment residue, and nothing
                        # in training may read them (the `ess` column is its
                        # own plain-IS draw below). Every slice is already
                        # the post-resample, equally-weighted ensemble,
                        # which is what makes the c_t mean below a mean over
                        # p_t rather than over the raw rollout law. Inside
                        # bind_cycle, so the trigger's xi_t reads the
                        # annealed density of the composition this cycle
                        # actually drew.
                        x_traj, rollout_smc_stats = rollout_result
                        rollout_resample_events = float(
                            rollout_smc_stats.n_events
                        )
                    if reuse_rollout_integrand:
                        # c_t = mean_m ξ_t (Eq. 8) — the same reduction
                        # compute_c_t_grid applies, on the same values.
                        c_t_grid = integrand_per_t.mean(dim=-1)
                    else:
                        c_t_grid, integrand_per_t = compute_c_t_grid(
                            t_grid, x_traj, target, model_cycle,
                            mode=estimator_mode,
                        )                                      # (T,), (T, M)

                    # Per-outer variance bookkeeping. Average within-slot
                    # variance: keeps the column comparable across t (each
                    # slot has its own ∂_t log p̃ baseline) and meaningful as
                    # "estimator noise per time slot".
                    # B1 free rider: in naive mode the integrand IS
                    # ∂_t log p̃_t on these rows, so the knob skips the
                    # (T·M) recompute.
                    if c_t_from_rollout and estimator_mode == "naive_mc":
                        naive_per_t = integrand_per_t
                    else:
                        t_grid_per_state = t_grid.repeat_interleave(
                            outer_batch
                        )
                        x_traj_flat = x_traj.reshape(
                            n_grid * outer_batch, n_dims
                        )
                        naive_per_t = target.dt_log_p_tilde_t(
                            x_traj_flat, t_grid_per_state,
                        ).reshape(n_grid, outer_batch)
                    var_dt_log_p_tilde = naive_per_t.var(dim=-1).mean().item()
                    var_estimator_integrand = (
                        integrand_per_t.var(dim=-1).mean().item()
                    )

            t_idx_buffer = (
                torch.arange(n_grid, device=device)
                .repeat_interleave(outer_batch)
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
            if amortised:
                composition_buffer = _retain_chunks(
                    composition_replay_chunks,
                    torch.full(
                        (n_grid * outer_batch,), composition_now, device=device
                    ),
                    replay_buffer_cycles,
                )
                # x_traj flattens t-major, matching t_idx_buffer above, so the
                # per-slot c_t repeats M times to land on its own states.
                c_t_buffer = _retain_chunks(
                    c_t_replay_chunks,
                    c_t_grid.repeat_interleave(outer_batch),
                    replay_buffer_cycles,
                )
            buffer_size = x_buffer.shape[0]

            for _inner in range(inner_steps_per_outer):
                step_start = time.time()

                # LR warmup: linearly ramp from 0 to current_intended_lr over
                # the first `warmup_steps` inner updates. Applied multiplicatively
                # so it composes with curriculum LR transitions. Targets the
                # early-training regime where random init can emit high-magnitude
                # rates that destabilise the first few optimiser steps.
                if warmup_steps > 0:
                    if step < warmup_steps:
                        warmup_scale = (step + 1) / warmup_steps
                        _set_optimizer_lr(
                            optimiser, current_intended_lr * warmup_scale
                        )
                    elif step == warmup_steps:
                        _set_optimizer_lr(optimiser, current_intended_lr)

                # INNER STEP -- N uniform draws from buffer (paper line 7).
                sample_idx = torch.randint(
                    buffer_size, (inner_batch,), device=device,
                )
                x_sample = x_buffer[sample_idx]                    # (N, D)
                t_idx_sample = t_idx_buffer[sample_idx]            # (N,)
                t_sample = t_grid[t_idx_sample]                    # (N,)
                if amortised:
                    # Each state's own baseline. A specialist run keeps the
                    # latest-grid lookup so archived runs still reproduce.
                    c_t_sample = c_t_buffer[sample_idx]            # (N,)
                    batch_composition = composition_buffer[sample_idx]
                else:
                    c_t_sample = c_t_grid[t_idx_sample]            # (N,)
                    batch_composition = None
                model_batch, bind_batch = _bound(batch_composition)

                with bind_batch:
                    loss_value = kolmogorov_loss(
                        x_sample, t_sample, c_t_sample, model_batch, target,
                    )
                optimiser.zero_grad()
                loss_value.backward()
                gradient_group_norms = (
                    _gradient_group_norms(model)
                    if log_gradient_groups else None
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    getattr(train_cfg, "grad_clip_max_norm", 500.0),
                )
                optimiser.step()
                if ema is not None:
                    ema.update()

                wall_clock_step_s = time.time() - step_start

                ess_value = float("nan")
                rate_diag = {
                    "rate_site_mean": float("nan"),
                    "rate_site_p99": float("nan"),
                    "flip_prob_site_p99": float("nan"),
                    "flip_prob_clipped_frac": float("nan"),
                    "log_ratio_clamp_frac": float("nan"),
                    "log_ratio_p99": float("nan"),
                }
                if step % eval_cfg.eval_every == 0:
                    # The in-loop ESS probe is held at one composition (the
                    # window centre) for the whole run, so the curve tracks
                    # training health rather than which compositions happened
                    # to be drawn. Per-composition eval is a separate sweep.
                    model_eval, bind_eval = _bound(centre_composition)
                    with torch.no_grad():
                        eval_grid = torch.linspace(
                            0.0, 1.0, n_grid, device=device,
                        )
                        with bind_eval:
                            x_eval_initial = target.sample_base(
                                eval_cfg.n_eval_samples, device=device
                            )
                            _, log_weights = sample_ctmc(
                                model_eval, x_eval_initial, eval_grid,
                                return_log_weights=True, target=target,
                            )
                            ess_value = ess_from_log_weights(log_weights).item()
                        with _bound(batch_composition)[1]:
                            rate_diag = _rate_diagnostics(
                                model_batch,
                                x_sample,
                                t_sample,
                                step_dt=1.0 / max(n_grid - 1, 1),
                                target=target,
                            )
                    torch.save(model.state_dict(), ckpt_dir / "latest.pt")

                writer.writerow(
                    [step, loss_value.item(), ess_value,
                     var_dt_log_p_tilde, var_estimator_integrand,
                     grad_norm.item(), rate_diag["rate_site_mean"],
                     rate_diag["rate_site_p99"],
                     rate_diag["flip_prob_site_p99"],
                     rate_diag["flip_prob_clipped_frac"],
                     rate_diag["log_ratio_clamp_frac"],
                     rate_diag["log_ratio_p99"],
                     float(target.sigma), optimiser.param_groups[0]["lr"],
                     wall_clock_step_s, composition_now, half_width_now,
                     rollout_resample_events]
                )
                log_file.flush()
                if gradient_writer is not None:
                    reconstructed = sum(
                        value * value
                        for value in gradient_group_norms.values()
                    ) ** 0.5
                    grad_norm_value = grad_norm.item()
                    clip_max_norm = float(
                        getattr(train_cfg, "grad_clip_max_norm", 500.0)
                    )
                    clip_scale = min(
                        1.0, clip_max_norm / (grad_norm_value + 1e-6)
                    )
                    gradient_writer.writerow(
                        [
                            step,
                            gradient_group_norms["grad_norm_gains"],
                            gradient_group_norms["grad_norm_omega"],
                            gradient_group_norms[
                                "grad_norm_composition_embedder"
                            ],
                            gradient_group_norms["grad_norm_trunk"],
                            reconstructed,
                            clip_scale,
                        ]
                    )
                    gradient_log_file.flush()

                if use_wandb:
                    log_dict = {
                        "train/loss": loss_value.item(),
                        "train/var_dt_log_p_tilde": var_dt_log_p_tilde,
                        "train/var_estimator_integrand": var_estimator_integrand,
                        "train/grad_norm": grad_norm.item(),
                        "train/sigma_current": float(target.sigma),
                        "train/lr_current": optimiser.param_groups[0]["lr"],
                        "train/wall_clock_step_s": wall_clock_step_s,
                    }
                    if amortised:
                        log_dict["train/composition_current"] = composition_now
                        log_dict["train/composition_half_width"] = half_width_now
                    if gradient_group_norms is not None:
                        log_dict.update(
                            {
                                f"train/{key}": value
                                for key, value in gradient_group_norms.items()
                            }
                        )
                    if step % eval_cfg.eval_every == 0:
                        log_dict["train/ess"] = ess_value
                        log_dict.update(
                            {
                                f"train/{key}": value
                                for key, value in rate_diag.items()
                            }
                        )
                        # Exact-field gains are the live mechanism under test
                        # in the soft amortisation arm. Keep them in W&B at
                        # the existing eval cadence (not every step, avoiding
                        # extra device synchronisation in the hot path). The
                        # getattr route is inert for every unwrapped model and
                        # archived global-gain channel.
                        for gain_name in (
                            "gain_constant",
                            "gain_slope",
                            "composition_gain_constant",
                            "composition_gain_slope",
                        ):
                            gain_param = getattr(model, gain_name, None)
                            if gain_param is not None:
                                log_dict[f"train/{gain_name}"] = (
                                    gain_param.detach().item()
                                )
                    wandb.log(log_dict, step=step)

                # Step-tagged checkpoints (opt-in): `final.pt` alone cannot
                # serve runs whose late loss cycles through excursions, since
                # it samples an arbitrary phase of the cycle. Zero-padded so
                # lexicographic order is step order; step 0 (random init)
                # is excluded.
                checkpoint_every = getattr(train_cfg, "checkpoint_every", None)
                if (
                    checkpoint_every is not None
                    and step > 0
                    and step % checkpoint_every == 0
                ):
                    torch.save(
                        model.state_dict(), ckpt_dir / f"step_{step:06d}.pt"
                    )

                step += 1

            # END OF OUTER CYCLE -- the only instant at which the training
            # state is self-consistent (the buffer and c_t grid are whole,
            # and the next cycle rebuilds both from scratch), so the only
            # instant at which a resume checkpoint is valid. Also fires on
            # the last cycle so a run killed between its final inner step
            # and `final.pt` is recoverable without redoing anything.
            if (outer + 1) % resume_every_outer == 0 or outer == n_outer - 1:
                save_resume_state(
                    ckpt_dir,
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "x_replay_chunks": [
                            chunk.cpu() for chunk in x_replay_chunks
                        ],
                        "t_idx_replay_chunks": [
                            chunk.cpu() for chunk in t_idx_replay_chunks
                        ],
                        "composition_replay_chunks": [
                            chunk.cpu() for chunk in composition_replay_chunks
                        ],
                        "c_t_replay_chunks": [
                            chunk.cpu() for chunk in c_t_replay_chunks
                        ],
                        "replay_sigma": replay_sigma,
                        "replay_lambda": replay_lambda,
                        "ema": ema.state_dict() if ema is not None else None,
                        **capture_rng_state(),
                    },
                )
                if on_checkpoint is not None:
                    on_checkpoint()

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    if ema is not None:
        ema.swap_in()
        torch.save(model.state_dict(), ckpt_dir / "final_ema.pt")
        ema.swap_out()
