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
    one swap that step (see that function's docstring).

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
        "log_ratio_clamp_frac": (
            (log_ratio > SWAP_LOG_RATIO_CLAMP).float().mean().item()
        ),
    }


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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    seed_everything(train_cfg.seed)
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=train_cfg.lr, weight_decay=1e-4
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

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            ["step", "loss", "ess", "var_dt_log_p_tilde",
             "var_estimator_integrand", "grad_norm",
             "rate_pair_mean", "rate_pair_p99",
             "lambda_dt_clipped_frac", "log_ratio_clamp_frac",
             "sigma_current", "lr_current", "wall_clock_step_s"]
        )

        step = 0
        curriculum_idx = -1
        x_replay_chunks: list[torch.Tensor] = []
        t_idx_replay_chunks: list[torch.Tensor] = []
        replay_sigma = float(target.sigma)
        current_intended_lr = float(train_cfg.lr)
        warmup_steps = int(getattr(train_cfg, "warmup_steps", 0))

        # Pre-training stiff-sampler / init-basin diagnostic. Computed at t=0
        # before the first optimiser step; RNG state is saved and restored so
        # the diagnostic does not perturb training-trajectory randomness.
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

            # OUTER STEP -- rebuild buffer + c_t. Trajectory and c_t are
            # both detached from autograd by the no_grad block; this is
            # the paper's R_t^{θ_sg} (stop-gradient) treatment.
            t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)
            x_initial = target.sample_base(outer_batch, device=device)
            with torch.no_grad():
                x_traj = sample_swap_ctmc(
                    head, x_initial, t_grid, return_all_states=True,
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
                     rate_diag["log_ratio_clamp_frac"],
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

    torch.save(head.state_dict(), ckpt_dir / "final.pt")
