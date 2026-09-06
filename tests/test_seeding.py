import random

import experiments.dnfs_baseline_01.run as run_module
import numpy as np
import torch
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)

from discrete_flow_sampler.seeding import seed_everything


def test_seed_everything_resets_python_numpy_and_torch_rngs():
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(3))

    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_top_level_train_seeds_let_model_initialisation(monkeypatch, tmp_path):
    captured_states = []

    def fake_train_loop(model, target, train_cfg, ctmc_cfg, eval_cfg, output_dir, **_):
        captured_states.append(
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        )

    def fake_sample_ctmc(_model, x_initial, _time_grid, *, return_log_weights, **_):
        log_weights = torch.zeros(x_initial.shape[0], device=x_initial.device)
        return x_initial, log_weights

    monkeypatch.setattr(run_module, "train_loop", fake_train_loop)
    monkeypatch.setattr(run_module, "sample_ctmc", fake_sample_ctmc)
    monkeypatch.setattr(
        run_module,
        "_compute_eval_metrics",
        lambda _samples, log_weights, _target, **_: {
            "n_eval_samples": int(log_weights.numel()),
            "ess": float(log_weights.numel()),
        },
    )
    monkeypatch.setattr(run_module, "_trailing_ess_metrics", lambda _run_dir: {})
    test_cfg = StageCfg(
        name="seed_test_let",
        ising=IsingCfg(D=2, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=1, batch_size=2, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=2),
        eval=EvalCfg(eval_every=1, n_eval_samples=2),
        model=ModelCfg(
            kind="let",
            hidden_dim=8,
            n_layers=1,
            n_heads=2,
            vocab_size=2,
        ),
        estimator="control_variate",
    )

    run_module.train(test_cfg, seed=123, output_dir=tmp_path / "run_a", use_wandb=False)
    run_module.train(test_cfg, seed=123, output_dir=tmp_path / "run_b", use_wandb=False)
    run_module.train(test_cfg, seed=124, output_dir=tmp_path / "run_c", use_wandb=False)

    same_seed_a, same_seed_b, different_seed = captured_states
    assert same_seed_a.keys() == same_seed_b.keys()
    assert all(
        torch.equal(same_seed_a[name], same_seed_b[name]) for name in same_seed_a
    )
    assert any(
        not torch.equal(same_seed_a[name], different_seed[name]) for name in same_seed_a
    )
