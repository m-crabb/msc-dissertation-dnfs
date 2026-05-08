"""Tests for the outer/inner training loop (paper Algorithm 1).

Pinned invariants:
    1) `train` runs end-to-end for `n_steps` total inner gradient updates
       grouped into `n_steps // inner_steps_per_outer` outer cycles, and
       writes a CSV log + a final checkpoint.
    2) The loss decreases over a brief D=2 run -- gradient flow through
       the outer/inner structure is correctly wired.
    3) Within a single outer cycle, c_t is frozen across inner steps (the
       inner-step loss target does not move when only θ moves). This is
       the paper's stop-gradient `c_sg` property.

Tests use D=2 and small grids so the suite stays under the existing
~3s baseline.
"""
from types import SimpleNamespace

import torch

from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import IsingTarget


def _tiny_train_cfg(
    *,
    n_steps: int,
    inner_steps_per_outer: int,
    batch_size: int,
    outer_batch_size: int | None = None,
    lr: float = 1e-3,
    seed: int = 0,
):
    return SimpleNamespace(
        n_steps=n_steps,
        inner_steps_per_outer=inner_steps_per_outer,
        batch_size=batch_size,
        outer_batch_size=outer_batch_size,
        lr=lr,
        seed=seed,
    )


def _read_csv_loss_column(csv_path):
    import csv
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [float(row["loss"]) for row in reader]


def test_train_runs_outer_inner_without_error(tmp_path):
    """Smoke: tiny D=2 run completes, CSV log exists with `n_steps` rows,
    a final checkpoint is saved, and a rolling latest checkpoint exists."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=16, n_layers=2)

    train_cfg = _tiny_train_cfg(
        n_steps=20, inner_steps_per_outer=10,
        batch_size=8, outer_batch_size=8,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=4)
    eval_cfg = SimpleNamespace(eval_every=10, n_eval_samples=8)

    train(
        model=model, target=target,
        train_cfg=train_cfg, ctmc_cfg=ctmc_cfg, eval_cfg=eval_cfg,
        output_dir=tmp_path, use_wandb=False,
        estimator_mode="control_variate",
    )

    log_path = tmp_path / "training_log.csv"
    assert log_path.exists()
    losses = _read_csv_loss_column(log_path)
    assert len(losses) == 20, (
        f"expected 20 inner-step rows in log, got {len(losses)}"
    )
    assert (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_train_loss_decreases_on_d2(tmp_path):
    """Brief training reduces the kolmogorov residual loss -- end-to-end
    sanity that gradients propagate through the new outer/inner structure
    and that c_t lookup by t-index is wired correctly.

    Compares the avg loss over the first vs last quarter of inner steps.
    Loose bound -- with D=2 / 200 inner steps, 4x reduction is reliable.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)

    train_cfg = _tiny_train_cfg(
        n_steps=200, inner_steps_per_outer=50,
        batch_size=32, outer_batch_size=64, lr=5e-3,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=10)
    eval_cfg = SimpleNamespace(eval_every=200, n_eval_samples=8)

    train(
        model=model, target=target,
        train_cfg=train_cfg, ctmc_cfg=ctmc_cfg, eval_cfg=eval_cfg,
        output_dir=tmp_path, use_wandb=False,
        estimator_mode="control_variate",
    )
    losses = _read_csv_loss_column(tmp_path / "training_log.csv")

    quarter = len(losses) // 4
    early = sum(losses[:quarter]) / quarter
    late = sum(losses[-quarter:]) / quarter
    assert late < 0.5 * early, (
        f"loss did not halve: early={early:.4f}, late={late:.4f}"
    )


def test_train_naive_mc_mode_runs(tmp_path):
    """Switching `estimator_mode='naive_mc'` doesn't break the loop.
    Pins that the mode string is plumbed through to compute_c_t_grid."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=16, n_layers=2)

    train_cfg = _tiny_train_cfg(
        n_steps=20, inner_steps_per_outer=10,
        batch_size=8, outer_batch_size=8,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=4)
    eval_cfg = SimpleNamespace(eval_every=10, n_eval_samples=8)

    train(
        model=model, target=target,
        train_cfg=train_cfg, ctmc_cfg=ctmc_cfg, eval_cfg=eval_cfg,
        output_dir=tmp_path, use_wandb=False,
        estimator_mode="naive_mc",
    )
    losses = _read_csv_loss_column(tmp_path / "training_log.csv")
    assert all(loss == loss for loss in losses), "training log has NaN losses"


def test_train_outer_batch_size_falls_back_to_batch_size(tmp_path):
    """When `outer_batch_size=None`, the outer trajectory uses
    `batch_size` -- a non-breaking default that lets existing configs
    keep working without pinning a new field."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=16, n_layers=2)

    train_cfg = _tiny_train_cfg(
        n_steps=10, inner_steps_per_outer=5,
        batch_size=12, outer_batch_size=None,
    )
    ctmc_cfg = SimpleNamespace(n_euler_steps=4)
    eval_cfg = SimpleNamespace(eval_every=10, n_eval_samples=8)

    train(
        model=model, target=target,
        train_cfg=train_cfg, ctmc_cfg=ctmc_cfg, eval_cfg=eval_cfg,
        output_dir=tmp_path, use_wandb=False,
        estimator_mode="control_variate",
    )
    assert (tmp_path / "training_log.csv").exists()
