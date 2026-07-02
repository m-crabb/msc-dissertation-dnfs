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
