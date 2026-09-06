"""Tests for LR warmup and pre-training init diagnostics."""

import csv
import json

import pytest
import torch
from experiments.dnfs_baseline_01.configs import CTMCCfg, EvalCfg, TrainCfg

from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers.training import train
from discrete_flow_sampler.targets.ising import IsingTarget


def _tiny_train(tmp_path, warmup_steps: int):
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1, bias=0.0, device="cpu")
    model = MLPRateMatrix(d=target.d, hidden_dim=8, n_layers=2)
    train_cfg = TrainCfg(
        n_steps=20,
        batch_size=4,
        lr=1e-2,
        seed=0,
        inner_steps_per_outer=5,
        replay_buffer_cycles=1,
        grad_clip_max_norm=500.0,
        warmup_steps=warmup_steps,
    )
    ctmc_cfg = CTMCCfg(n_euler_steps=4)
    eval_cfg = EvalCfg(eval_every=1000, n_eval_samples=4)
    train(
        model,
        target,
        train_cfg,
        ctmc_cfg,
        eval_cfg,
        tmp_path,
        use_wandb=False,
        estimator_mode="control_variate",
    )
    with (tmp_path / "training_log.csv").open() as f:
        return [float(row["lr_current"]) for row in csv.DictReader(f)]


def test_warmup_scales_lr_linearly(tmp_path):
    lrs = _tiny_train(tmp_path, warmup_steps=10)
    assert lrs[0] == pytest.approx(1e-2 * 1 / 10)
    assert lrs[4] == pytest.approx(1e-2 * 5 / 10)
    assert lrs[8] == pytest.approx(1e-2 * 9 / 10)
    assert lrs[9] == pytest.approx(1e-2)
    assert lrs[10] == pytest.approx(1e-2)
    assert lrs[19] == pytest.approx(1e-2)


def test_warmup_zero_is_no_op(tmp_path):
    lrs = _tiny_train(tmp_path, warmup_steps=0)
    assert all(lr == pytest.approx(1e-2) for lr in lrs)


def test_init_diagnostics_written(tmp_path):
    _tiny_train(tmp_path, warmup_steps=0)
    init_diag = json.loads((tmp_path / "init_diagnostics.json").read_text())
    for key in (
        "rate_site_mean",
        "rate_site_p99",
        "flip_prob_site_p99",
        "flip_prob_clipped_frac",
    ):
        assert key in init_diag
        assert isinstance(init_diag[key], float)
