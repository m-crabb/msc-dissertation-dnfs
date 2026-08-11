"""Outer/inner training loop for the swap-move CTMC (hard-constraint route).

Focused mirror of `training.train` (paper Algorithm 1, App. C.1) with the
single-site sampler/loss swapped for the swap-move counterparts throughout:
`sample_swap_ctmc` replaces `sample_ctmc`, `compute_c_t_grid_swap` replaces
`compute_c_t_grid`, and `loss_swap` (Eq. 10, swap form) replaces the
single-site Kolmogorov loss. There is no soft composition penalty on this
route -- the swap move set enforces n_plus == N_A exactly, so `lambda_curriculum`
and its stage handling are dropped entirely.

Diagnostic-columns decision (mirrors `training._rate_diagnostics`): the
per-site columns (`rate_site_mean`, `rate_site_p99`, `flip_prob_site_p99`,
`flip_prob_clipped_frac`) have no clean one-to-one swap analogue -- a swap
event is a joint choice over i<j pairs, not an independent per-site flip --
so they are renamed to `rate_pair_mean`/`rate_pair_p99` and the clip signal
becomes `lambda_dt_clipped_frac` (see `_swap_rate_diagnostics`). The
`log_ratio_clamp_frac` column is reused unchanged: same meaning (fraction of
neighbour log-ratios saturating `SWAP_LOG_RATIO_CLAMP`), evaluated at the
swap neighbour set instead of the single-flip one. `log_ratio_p99` is
dropped since neither the base instructions nor the amendment ask for it.
"""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.samplers._swap_neighbours import (
    SWAP_LOG_RATIO_CLAMP,
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    compute_c_t_grid_swap,
    sample_swap_ctmc,
)
from discrete_flow_sampler.samplers.optim import StableAdamW
from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap
from discrete_flow_sampler.samplers.training import (
    _append_replay_buffer,
    _clear_replay,
    _normalise_curriculum,
    _set_optimizer_lr,
)
from discrete_flow_sampler.seeding import seed_everything


def _swap_rate_diagnostics(head, x, t, step_dt: float, *, target) -> dict[str, float]:
    """Eval-time diagnostics for the swap-CTMC rate scale (AMENDMENT).

    Mirrors `training._rate_diagnostics` for the pair-rate matrix. Gathers
    forward rates at the SAME i<j pairs `_euler_step_swap` uses, so the
    logged clip fraction matches the sampler's actual clip behaviour:

        Lambda * dt = sum_{i<j} [G_swap(i,j|x)]_+ * dt

    `lambda_dt_clipped_frac` is the fraction of states with Lambda*dt > 1 --
    when this clips, `_euler_step_swap`'s stay slot clamps to 0 and
    `torch.multinomial` renormalises the pair probabilities, forcing exactly
    one swap that step (see that function's docstring). NOTE: that clipping
    exists in the ONE-EVENT step only. Under `use_matching_step=True` this
    column is a diagnostic of one-event clip-safety, not of the running
    step; the matching step's own fidelity is the `proposal_drop_frac` and
    `events_per_site_per_step` columns (accumulated at the buffer rebuild),
    which is what the d256 divergence review (2026-08-11) found missing.
    `lambda_dt_p99` records the tail of the per-state rate load directly,
    since the one-event Euler budget rule reads the tail and the state
    distribution of Lambda is too fat-tailed to reconstruct it from the
    mean and an exceedance fraction.

    `log_ratio_clamp_frac` is the fraction of pair log-ratios
    log p_tilde_t(swap) - log p_tilde_t(x) reaching `SWAP_LOG_RATIO_CLAMP`
    (Eq. 8 / Eq. 10, swap form) -- expected ~0 across the gate sigma-ladder;
    a nonzero value flags residual/xi_t bias.
    """
    pairs = upper_tri_pairs(x.shape[1], x.device)
    forward_rates = F.relu(gather_pair_scores(head(x, t), pairs))  # (B, n_pairs)
    lambda_dt = (forward_rates * step_dt).sum(dim=-1)              # (B,)

    log_ratio = target.swap_log_ratio(x, t, pairs)   # unclamped: measures saturation

    return {
        "rate_pair_mean": forward_rates.mean().item(),
        # .float(): quantile is fp32/64-only; heads may emit reduced
        # precision under the eval autocast block.
        "rate_pair_p99": torch.quantile(
            forward_rates.reshape(-1).float(), 0.99
        ).item(),
        "lambda_dt_clipped_frac": (lambda_dt > 1.0).float().mean().item(),
        # p99 of the per-state total-rate load: the one-event budget rule
        # (n_euler from the tail of Lambda) needs the tail directly — the
        # d256 review showed it is NOT recoverable from mean + exceedance
        # because the state distribution of Lambda is fat-tailed.
        "lambda_dt_p99": torch.quantile(
            lambda_dt.float(), 0.99
        ).item(),
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
) -> None:
    """Checkpoint full outer-boundary training state for preemption resume.

    Saved atomically (tmp file + rename) so a preemption mid-write can never
    leave a truncated resume.pt behind. Replay chunks move to CPU so the
    checkpoint is device-portable; RNG states make the continuation
    bit-identical to an uninterrupted run (exactly on CPU fp32, modulo
    kernel nondeterminism on CUDA). Curriculum stage / warmup / intended lr
    are NOT stored — all are derivable from `step` because the sigma ladder
    uses absolute start_steps, and the optimiser lr travels inside the
    optimiser state dict.
    """
    state = {
        "step": step,
        "model": head.state_dict(),
        "optimiser": optimiser.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "x_replay_chunks": [chunk.cpu() for chunk in x_replay_chunks],
        "t_idx_replay_chunks": [chunk.cpu() for chunk in t_idx_replay_chunks],
        "replay_sigma": replay_sigma,
    }
    tmp_path = ckpt_dir / "resume.pt.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_dir / "resume.pt")


def _truncate_log_to_step(log_path: Path, resume_step: int) -> bool:
    """Drop training-log rows at/after `resume_step` (a dead attempt may have
    flushed rows past its last checkpoint). Returns True if the log survives
    to be appended to, False if it is missing and needs a fresh header."""
    if not log_path.exists():
        return False
    with log_path.open(newline="") as log_file:
        rows = list(csv.reader(log_file))
    header, body = rows[0], rows[1:]
    kept = [row for row in body if int(row[0]) < resume_step]
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)
        writer.writerows(kept)
    return True


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
):
    """Run paper Algorithm 1 for `train_cfg.n_steps` total inner steps.

    Args:
        head: a swap-readout head (e.g. `DoublyHollowSwapHead` /
            `LeTFMaskOneSwapHead`) wrapping a `LeTFRateMatrix` backbone --
            already instantiated, on `target.device`. `head.parameters()`
            covers the backbone too since it is a registered submodule.
        target: a `FixedCompositionIsingTarget` -- swaps preserve its
            n_plus == N_A manifold by construction, so `sample_base` already
            returns on-manifold states with no extra handling needed here.
        train_cfg, ctmc_cfg, eval_cfg, output_dir, use_wandb, estimator_mode:
            same contract as `training.train`.
        sigma_curriculum: optional piecewise-constant schedule, same
            contract as `training.train`. There is no `lambda_curriculum`
            counterpart: the hard-constraint route has no soft composition
            penalty to anneal.
        on_checkpoint: optional zero-arg callable invoked after each resume
            checkpoint lands on disk (Modal passes `volume.commit` so the
            checkpoint survives a preemption that skips the death-flush).

    Preemption resume: every `train_cfg.resume_every_outer` outer cycles
    (default 10) the full boundary state is checkpointed to
    `checkpoints/resume.pt`; if that file exists on entry, training restores
    it and continues instead of starting over (see `_save_resume_state` for
    what "full state" means and why the continuation is bit-exact). A resume
    at step >= n_steps is a completed run being retried: return immediately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    resume_path = ckpt_dir / "resume.pt"
    resume_state = None
    if resume_path.exists():
        resume_state = torch.load(
            resume_path, map_location=target.device, weights_only=True
        )

    if resume_state is None:
        seed_everything(train_cfg.seed)
    optimiser_kind = getattr(train_cfg, "optimiser", "adamw")
    if optimiser_kind == "adamw":
        optimiser = torch.optim.AdamW(
            head.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
    elif optimiser_kind == "stable_adamw":
        optimiser = StableAdamW(
            head.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
    else:
        raise ValueError(f"unknown optimiser {optimiser_kind!r}")
    start_step = 0
    if resume_state is not None:
        head.load_state_dict(resume_state["model"])
        optimiser.load_state_dict(resume_state["optimiser"])
        start_step = int(resume_state["step"])

    if start_step >= train_cfg.n_steps:
        # Completed run being re-invoked (e.g. a Modal retry after success):
        # make sure the terminal artefact exists, touch nothing else.
        if not (ckpt_dir / "final.pt").exists():
            torch.save(head.state_dict(), ckpt_dir / "final.pt")
        return

    if use_wandb:
        import wandb

    n_dims = target.d
    device = target.device
    inner_batch = train_cfg.batch_size
    outer_batch = train_cfg.outer_batch_size or train_cfg.batch_size
    n_grid = ctmc_cfg.n_euler_steps
    # Trajectory step for every simulation in this loop (buffer rebuild and
    # in-training eval draw): the matching step is required from d=256 up,
    # where one-event clip-safety would need ~3x the Euler grid. getattr
    # because test call sites pass bare config bags without the field.
    multi_event = getattr(ctmc_cfg, "use_matching_step", False)
    inner_steps_per_outer = train_cfg.inner_steps_per_outer
    replay_buffer_cycles = getattr(train_cfg, "replay_buffer_cycles", 1)
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
        if resume_state is not None and _truncate_log_to_step(log_path, start_step)
        else "w"
    )
    with log_path.open(log_mode, newline="") as log_file:
        writer = csv.writer(log_file)
        if log_mode == "w":
            writer.writerow(
                ["step", "loss", "ess", "var_dt_log_p_tilde",
                 "var_estimator_integrand", "grad_norm",
                 "rate_pair_mean", "rate_pair_p99",
                 "lambda_dt_clipped_frac", "lambda_dt_p99",
                 "log_ratio_clamp_frac",
                 "proposal_drop_frac", "events_per_site_per_step",
                 "sigma_current", "lr_current", "wall_clock_step_s"]
            )

        step = start_step
        curriculum_idx = -1
        x_replay_chunks: list[torch.Tensor] = []
        t_idx_replay_chunks: list[torch.Tensor] = []
        replay_sigma = float(target.sigma)
        current_intended_lr = float(train_cfg.lr)
        warmup_steps = int(getattr(train_cfg, "warmup_steps", 0))
        # rewarmup_on_stage: re-run the warmup ramp from each sigma
        # transition (anchor moves); off => anchor stays 0 and the ramp is
        # the historical step-0-only behaviour, bit-identical.
        rewarmup_on_stage = bool(getattr(train_cfg, "rewarmup_on_stage", False))
        warmup_anchor = 0

        if resume_state is not None:
            # Fast-forward the curriculum EXPLICITLY rather than letting the
            # stage loop below replay every transition: its transition code
            # clears the replay buffer on sigma changes, which would destroy
            # the restored chunks. The optimiser lr is deliberately not
            # touched -- load_state_dict above already carries the exact lr
            # (including warmup scaling at the boundary).
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
            # RNG restore comes LAST in the restore sequence so nothing
            # above can perturb the stream the continuation will consume.
            torch.set_rng_state(resume_state["rng_cpu"].cpu())
            if resume_state["rng_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in resume_state["rng_cuda"]]
                )

        # Pre-training stiff-sampler / init-basin diagnostic. Computed at t=0
        # before the first optimiser step; RNG state is saved and restored so
        # the diagnostic does not perturb training-trajectory randomness.
        # Skipped on resume: it belongs to step 0 and already exists on disk.
        if resume_state is None:
            rng_state_cpu = torch.get_rng_state()
            rng_state_cuda = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
            with torch.no_grad():
                x_diag = target.sample_base(outer_batch, device=device)
                t_diag = torch.zeros(outer_batch, device=device)
                init_diag = _swap_rate_diagnostics(
                    head, x_diag, t_diag,
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
                        _clear_replay(x_replay_chunks, t_idx_replay_chunks)
                        replay_sigma = sigma_now
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
            x_initial = target.sample_base(outer_batch, device=device)
            outer_matching_stats: dict | None = {} if multi_event else None
            with torch.no_grad():
                x_traj = sample_swap_ctmc(
                    head, x_initial, t_grid, return_all_states=True,
                    multi_event=multi_event,
                    matching_stats=outer_matching_stats,
                )                                              # (T, M, D)
                c_t_grid, integrand_per_t = compute_c_t_grid_swap(
                    t_grid, x_traj, target, head, mode=estimator_mode,
                )                                              # (T,), (T, M)

                # Per-outer variance bookkeeping. Average within-slot
                # variance: keeps the column comparable across t (each
                # slot has its own ∂_t log p̃ baseline) and meaningful as
                # "estimator noise per time slot".
                t_grid_per_state = t_grid.repeat_interleave(outer_batch)
                x_traj_flat = x_traj.reshape(n_grid * outer_batch, n_dims)
                naive_per_t = target.dt_log_p_tilde_t(
                    x_traj_flat, t_grid_per_state,
                ).reshape(n_grid, outer_batch)
                var_dt_log_p_tilde = naive_per_t.var(dim=-1).mean().item()
                var_estimator_integrand = (
                    integrand_per_t.var(dim=-1).mean().item()
                )

            # Matching-native fidelity for THIS outer cycle's buffer states
            # (constant across the cycle's inner rows). This is the running
            # step's own certificate: `lambda_dt_clipped_frac` gates the
            # dormant one-event path, and the Luby matching silently drops
            # proposals still contested after its round budget — without
            # these two columns a multi-event run has no logged evidence it
            # stayed in the regime the matching step was validated for
            # (drop_frac ~< 1%, events/site/step <= 0.1).
            if multi_event and outer_matching_stats.get("state_steps"):
                proposed_total = float(outer_matching_stats["proposed"])
                accepted_total = float(outer_matching_stats["accepted"])
                proposal_drop_frac = (
                    1.0 - accepted_total / proposed_total
                    if proposed_total > 0 else 0.0
                )
                events_per_site_per_step = accepted_total / (
                    float(outer_matching_stats["state_steps"]) * n_dims
                )
            else:
                proposal_drop_frac = float("nan")
                events_per_site_per_step = float("nan")

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
            buffer_size = x_buffer.shape[0]

            for _inner in range(inner_steps_per_outer):
                step_start = time.time()

                # LR warmup: linearly ramp from 0 to current_intended_lr over
                # the first `warmup_steps` inner updates. Applied multiplicatively
                # so it composes with curriculum LR transitions. Targets the
                # early-training regime where random init can emit high-magnitude
                # rates that destabilise the first few optimiser steps.
                if warmup_steps > 0:
                    warmup_rel_step = step - warmup_anchor
                    if warmup_rel_step < warmup_steps:
                        warmup_scale = (warmup_rel_step + 1) / warmup_steps
                        _set_optimizer_lr(
                            optimiser, current_intended_lr * warmup_scale
                        )
                    elif warmup_rel_step == warmup_steps:
                        _set_optimizer_lr(optimiser, current_intended_lr)

                # INNER STEP -- N uniform draws from buffer (paper line 7).
                sample_idx = torch.randint(
                    buffer_size, (inner_batch,), device=device,
                )
                x_sample = x_buffer[sample_idx]                    # (N, D)
                t_idx_sample = t_idx_buffer[sample_idx]            # (N,)
                t_sample = t_grid[t_idx_sample]                    # (N,)
                c_t_sample = c_t_grid[t_idx_sample]                # (N,)

                loss_value = loss_swap(
                    x_sample, t_sample, c_t_sample, head, target,
                )
                optimiser.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    getattr(train_cfg, "grad_clip_max_norm", 500.0),
                )
                optimiser.step()

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
                        device.type, dtype=torch.bfloat16,
                        enabled=getattr(eval_cfg, "eval_autocast_bf16", False),
                    )
                    with torch.no_grad(), eval_autocast:
                        eval_grid = torch.linspace(
                            0.0, 1.0, n_grid, device=device,
                        )
                        # Stream the eval draw in slices: the vectorised swap
                        # head rides d anchor copies per sample, so feeding
                        # all n_eval_samples at once builds (d*B)-row
                        # attention buffers and OOMs at large d. IS weights
                        # are independent per sample, so slicing changes
                        # nothing statistically.
                        # In-training evals are a diagnostic; run.py's final
                        # eval always draws the full n_eval_samples.
                        n_train_eval = (
                            getattr(eval_cfg, "n_eval_samples_training", None)
                            or eval_cfg.n_eval_samples
                        )
                        eval_chunk = (
                            getattr(eval_cfg, "eval_sample_chunk", None)
                            or n_train_eval
                        )
                        log_weight_slices = []
                        remaining = n_train_eval
                        while remaining > 0:
                            n_slice = min(eval_chunk, remaining)
                            x_eval_initial = target.sample_base(
                                n_slice, device=device
                            )
                            _, slice_log_weights = sample_swap_ctmc(
                                head, x_eval_initial, eval_grid,
                                return_log_weights=True, target=target,
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

                writer.writerow(
                    [step, loss_value.item(), ess_value,
                     var_dt_log_p_tilde, var_estimator_integrand,
                     grad_norm.item(), rate_diag["rate_pair_mean"],
                     rate_diag["rate_pair_p99"],
                     rate_diag["lambda_dt_clipped_frac"],
                     rate_diag["lambda_dt_p99"],
                     rate_diag["log_ratio_clamp_frac"],
                     proposal_drop_frac, events_per_site_per_step,
                     float(target.sigma), optimiser.param_groups[0]["lr"],
                     wall_clock_step_s]
                )
                log_file.flush()

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
                    if step % eval_cfg.eval_every == 0:
                        log_dict["train/ess"] = ess_value
                        log_dict.update(
                            {
                                f"train/{key}": value
                                for key, value in rate_diag.items()
                            }
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
                )
                if on_checkpoint is not None:
                    on_checkpoint()

    torch.save(head.state_dict(), ckpt_dir / "final.pt")
