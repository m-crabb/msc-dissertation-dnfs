"""Generic optimiser loop for DNFS -- stage-agnostic.

One training step:
    1. Sample a single t ~ Uniform(0, 1) for the whole batch (matches the
       paper's protocol; the outer expectation in Eq. (5) is approximated
       by averaging over training steps, not within a step).
    2. Sample x_batch ~ p_t by simulating the model's CTMC from t = 0 to
       the chosen t, starting from uniform random {-1, +1} states.
    3. Estimate ∂_t log Z_t with the configured estimator.
    4. Compute the mean-squared Kolmogorov residual loss (Eq. 7).
    5. Backprop + Adam step.
    6. Periodically: simulate the full t = 0 -> 1 trajectory with IS
       log-weights, compute ESS, log to wandb + CSV.

Stage 1 / 2 / 3 differ only in which `model` and which `estimator` are
passed in -- this module never branches on stage.
"""
import csv
import time
from pathlib import Path

import torch

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.samplers.ctmc import sample_ctmc
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss


def train(
    model,
    target,
    estimator,
    train_cfg,
    ctmc_cfg,
    eval_cfg,
    output_dir: Path,
    *,
    use_wandb: bool = True,
):
    """Run training for `train_cfg.n_steps` steps and persist artefacts.

    Args:
        model: a `RateMatrix` -- already instantiated, on `target.device`.
        target: an `IsingTarget`.
        estimator: a `LogZEstimator` -- callable
            (t_batch, x_batch, target, model) -> 0-dim Tensor.
        train_cfg: object with .n_steps, .batch_size, .lr, .seed.
        ctmc_cfg:  object with .n_euler_steps.
        eval_cfg:  object with .eval_every, .n_eval_samples.
        output_dir: directory for per-run artefacts (CSV log + checkpoint).
        use_wandb: when False, skip wandb.log -- handy for tests / local
            debugging without the wandb sidecar. wandb is imported lazily
            so it isn't a hard dependency for callers who set this False.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train_cfg.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    if use_wandb:
        # Lazy import: keeps wandb optional for offline / unit-test paths.
        import wandb

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            ["step", "loss", "ess", "var_dt_log_p_tilde", "wall_clock_step_s"]
        )

        for step in range(train_cfg.n_steps):
            step_start = time.time()

            # Single t per step. Shared across the whole batch -- mixing t
            # within a batch would change the meaning of the residual mean.
            t_value = torch.rand(1, device=target.device).item()
            t_batch = torch.full(
                (train_cfg.batch_size,), t_value, device=target.device
            )

            # Sample x ~ p_t by running the CTMC from t = 0 to t_value.
            # No grad through the trajectory: gradients only flow through
            # the rate matrix at the *evaluated* states inside the loss,
            # not through the discrete sampling decisions.
            time_grid_to_t = torch.linspace(
                0.0, t_value, ctmc_cfg.n_euler_steps, device=target.device
            )
            x_initial = (
                torch.randint(
                    0, 2, (train_cfg.batch_size, target.d), device=target.device
                ).float()
                * 2 - 1
            )
            with torch.no_grad():
                x_batch = sample_ctmc(model, x_initial, time_grid_to_t)

            # Per-step scalar estimate of ∂_t log Z_t. Stage 1 = naive MC,
            # Stage 2 = control variate -- swappable behind LogZEstimator.
            dt_log_Zt = estimator(t_batch, x_batch, target, model)

            # Variance of the per-state integrand -- a cheap proxy for the
            # naive estimator's variance (which scales as Var/K). Tracking
            # this surfaces the high-D failure mode early, before ESS
            # collapses.
            with torch.no_grad():
                var_integrand = (
                    target.dt_log_p_tilde_t(x_batch, t_batch).var().item()
                )

            loss_value = kolmogorov_loss(
                x_batch, t_batch, dt_log_Zt, model, target
            )
            optimiser.zero_grad()
            loss_value.backward()
            optimiser.step()

            wall_clock_step_s = time.time() - step_start

            # Eval ESS occasionally: a full t = 0 -> 1 trajectory with IS
            # weights (Eq. 13). Expensive vs. a training step, so gated.
            ess_value = float("nan")
            if step % eval_cfg.eval_every == 0:
                with torch.no_grad():
                    full_time_grid = torch.linspace(
                        0.0, 1.0, ctmc_cfg.n_euler_steps, device=target.device
                    )
                    x_eval_initial = (
                        torch.randint(
                            0, 2, (eval_cfg.n_eval_samples, target.d),
                            device=target.device,
                        ).float()
                        * 2 - 1
                    )
                    _, log_weights = sample_ctmc(
                        model,
                        x_eval_initial,
                        full_time_grid,
                        return_log_weights=True,
                        target=target,
                    )
                    ess_value = ess_from_log_weights(log_weights).item()

            writer.writerow(
                [step, loss_value.item(), ess_value, var_integrand,
                 wall_clock_step_s]
            )
            log_file.flush()

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss_value.item(),
                        "train/ess": ess_value,
                        "train/var_dt_log_p_tilde": var_integrand,
                        "train/wall_clock_step_s": wall_clock_step_s,
                    },
                    step=step,
                )

    # Persist the final model. Not snapshotting intermediate checkpoints by
    # default -- Stage 1's whole point is to fail at D = 10, so per-step
    # snapshots aren't worth the disk.
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
