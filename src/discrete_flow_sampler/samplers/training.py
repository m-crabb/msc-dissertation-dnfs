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
from pathlib import Path

import torch
import torch.nn.functional as F

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.samplers._neighbours import _log_p_tilde_at_neighbours
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid
from discrete_flow_sampler.seeding import seed_everything


def _append_replay_buffer(
    x_chunks: list[torch.Tensor],
    t_idx_chunks: list[torch.Tensor],
    x_traj: torch.Tensor,
    t_idx_buffer: torch.Tensor,
    max_cycles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one outer batch and return the retained replay-buffer view."""
    x_chunks.append(x_traj.reshape(-1, x_traj.shape[-1]).detach())
    t_idx_chunks.append(t_idx_buffer.detach())
    if len(x_chunks) > max_cycles:
        x_chunks.pop(0)
        t_idx_chunks.pop(0)
    return torch.cat(x_chunks, dim=0), torch.cat(t_idx_chunks, dim=0)


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


def _clear_replay(
    x_replay_chunks: list[torch.Tensor],
    t_idx_replay_chunks: list[torch.Tensor],
) -> None:
    x_replay_chunks.clear()
    t_idx_replay_chunks.clear()


def _rate_diagnostics(
    model, x, t, step_dt: float, *, target=None
) -> dict[str, float]:
    """Cheap eval-time diagnostics for CTMC rate scale.

    ESS alone cannot distinguish a no-op sampler (rates near zero) from a
    stiff sampler (rates so large Euler probabilities clip). Logging per-site
    outflow rates at eval cadence makes those failure modes visible without
    changing the training objective.

    When `target` is supplied AND the model is locally equivariant, also
    logs the saturation fraction of the log-target ratio against the 5.0
    clamp at `kolmogorov.residual_lenet` and `ctmc._compute_xi_t_lenet`.
    `log_ratio_clamp_frac` is the share of (B, d, S) entries that exceed
    the ceiling; `log_ratio_p99` is the unclipped 99th percentile so the
    magnitude of the saturated tail is visible (saturation alone is
    ambiguous between "just above 5" and "an order of magnitude above").
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
            (log_ratio > 5.0).float().mean().item()
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    seed_everything(train_cfg.seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
    )

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
    lambda_stages = _normalise_curriculum(
        lambda_curriculum,
        n_steps=train_cfg.n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
        value_attr="composition_penalty_strength",
    )

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            ["step", "loss", "ess", "var_dt_log_p_tilde",
             "var_estimator_integrand", "grad_norm",
             "rate_site_mean", "rate_site_p99", "flip_prob_site_p99",
             "flip_prob_clipped_frac", "log_ratio_clamp_frac",
             "log_ratio_p99", "sigma_current", "lr_current",
             "wall_clock_step_s"]
        )

        step = 0
        curriculum_idx = -1
        lambda_idx = -1
        x_replay_chunks: list[torch.Tensor] = []
        t_idx_replay_chunks: list[torch.Tensor] = []
        replay_sigma = float(target.sigma)
        replay_lambda = float(
            getattr(target, "composition_penalty_strength", 0.0)
        )
        current_intended_lr = float(train_cfg.lr)
        warmup_steps = int(getattr(train_cfg, "warmup_steps", 0))

        # Pre-training stiff-sampler / init-basin diagnostic. Computed at t=0
        # before the first optimiser step; RNG state is saved and restored so
        # the diagnostic does not perturb training-trajectory randomness.
        # The logged `flip_prob_clipped_frac` here is the only place the
        # init-time stiffness signal is captured -- the in-loop diagnostic
        # only ever sees the post-first-update model.
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        with torch.no_grad():
            x_diag = (
                torch.randint(0, 2, (outer_batch, n_dims), device=device)
                .float() * 2 - 1
            )
            t_diag = torch.zeros(outer_batch, device=device)
            init_diag = _rate_diagnostics(
                model, x_diag, t_diag,
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

        for outer in range(n_outer):
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
                        _clear_replay(x_replay_chunks, t_idx_replay_chunks)
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

            # OUTER STEP -- rebuild buffer + c_t. Trajectory and c_t are
            # both detached from autograd by the no_grad block; this is
            # the paper's R_t^{θ_sg} (stop-gradient) treatment.
            t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)
            x_initial = (
                torch.randint(
                    0, 2, (outer_batch, n_dims), device=device,
                ).float()
                * 2 - 1
            )
            with torch.no_grad():
                x_traj = sample_ctmc(
                    model, x_initial, t_grid, return_all_states=True,
                )                                              # (T, M, D)
                c_t_grid, integrand_per_t = compute_c_t_grid(
                    t_grid, x_traj, target, model, mode=estimator_mode,
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
                c_t_sample = c_t_grid[t_idx_sample]                # (N,)

                loss_value = kolmogorov_loss(
                    x_sample, t_sample, c_t_sample, model, target,
                )
                optimiser.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    getattr(train_cfg, "grad_clip_max_norm", 500.0),
                )
                optimiser.step()

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
                    with torch.no_grad():
                        eval_grid = torch.linspace(
                            0.0, 1.0, n_grid, device=device,
                        )
                        x_eval_initial = (
                            torch.randint(
                                0, 2, (eval_cfg.n_eval_samples, n_dims),
                                device=device,
                            ).float()
                            * 2 - 1
                        )
                        _, log_weights = sample_ctmc(
                            model, x_eval_initial, eval_grid,
                            return_log_weights=True, target=target,
                        )
                        ess_value = ess_from_log_weights(log_weights).item()
                        rate_diag = _rate_diagnostics(
                            model,
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

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
