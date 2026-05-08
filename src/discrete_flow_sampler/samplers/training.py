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
import time
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid


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
            .lr, .seed.
        ctmc_cfg: object with .n_euler_steps -- length T of outer-step
            time grid (paper's K+1).
        eval_cfg: object with .eval_every (inner-step cadence) and
            .n_eval_samples.
        output_dir: per-run artefact directory.
        use_wandb: skip wandb.log when False (handy for tests).
        estimator_mode: "naive_mc" | "control_variate". Selects which
            integrand `compute_c_t_grid` averages per time slot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(train_cfg.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    if use_wandb:
        import wandb

    n_dims = target.d
    device = target.device
    inner_batch = train_cfg.batch_size
    outer_batch = train_cfg.outer_batch_size or train_cfg.batch_size
    n_grid = ctmc_cfg.n_euler_steps
    inner_steps_per_outer = train_cfg.inner_steps_per_outer

    if train_cfg.n_steps % inner_steps_per_outer != 0:
        raise ValueError(
            f"n_steps={train_cfg.n_steps} not divisible by "
            f"inner_steps_per_outer={inner_steps_per_outer}; the outer/"
            f"inner contract requires whole outer cycles."
        )
    n_outer = train_cfg.n_steps // inner_steps_per_outer

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            ["step", "loss", "ess", "var_dt_log_p_tilde",
             "var_estimator_integrand", "wall_clock_step_s"]
        )

        step = 0
        for outer in range(n_outer):
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

            # Flatten buffer for uniform inner-step sampling.
            x_buffer = x_traj.reshape(n_grid * outer_batch, n_dims)
            t_idx_buffer = (
                torch.arange(n_grid, device=device)
                .repeat_interleave(outer_batch)
            )
            buffer_size = n_grid * outer_batch

            for _inner in range(inner_steps_per_outer):
                step_start = time.time()

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
                optimiser.step()

                wall_clock_step_s = time.time() - step_start

                ess_value = float("nan")
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
                    torch.save(model.state_dict(), ckpt_dir / "latest.pt")

                writer.writerow(
                    [step, loss_value.item(), ess_value,
                     var_dt_log_p_tilde, var_estimator_integrand,
                     wall_clock_step_s]
                )
                log_file.flush()

                if use_wandb:
                    log_dict = {
                        "train/loss": loss_value.item(),
                        "train/var_dt_log_p_tilde": var_dt_log_p_tilde,
                        "train/var_estimator_integrand": var_estimator_integrand,
                        "train/wall_clock_step_s": wall_clock_step_s,
                    }
                    if step % eval_cfg.eval_every == 0:
                        log_dict["train/ess"] = ess_value
                    wandb.log(log_dict, step=step)

                step += 1

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
