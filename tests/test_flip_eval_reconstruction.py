"""Finished flip evaluations must use the target at the end of training."""
import json
from dataclasses import asdict

import pytest
import torch

from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    CurriculumCfg,
    CurriculumStageCfg,
    EvalCfg,
    IsingCfg,
    LambdaCurriculumCfg,
    LambdaCurriculumStageCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)
from experiments.dnfs_baseline_01.run import _rebuild_from_run_dir, eval_only, train
from discrete_flow_sampler.targets.ising import IsingTarget


def _curriculum_cfg():
    return StageCfg(
        name="tiny_terminal_target",
        ising=IsingCfg(
            D=2, sigma=0.1, target_composition=0.5,
            composition_penalty_strength=1.0,
        ),
        model=ModelCfg(kind="mlp", hidden_dim=8, n_layers=1),
        estimator="naive_mc",
        train=TrainCfg(
            n_steps=4, inner_steps_per_outer=2, batch_size=4,
            outer_batch_size=4, warmup_steps=0,
        ),
        ctmc=CTMCCfg(n_euler_steps=4),
        eval=EvalCfg(n_eval_samples=16, eval_every=2),
        curriculum=CurriculumCfg(stages=(
            CurriculumStageCfg(start_step=0, sigma=0.1),
            CurriculumStageCfg(start_step=2, sigma=0.22),
        )),
        lambda_curriculum=LambdaCurriculumCfg(stages=(
            LambdaCurriculumStageCfg(0, 1.0),
            LambdaCurriculumStageCfg(2, 5.0),
        )),
    )


def test_rebuild_uses_terminal_density_and_preserves_saved_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(_curriculum_cfg())))
    original = path.read_bytes()

    cfg, target, device = _rebuild_from_run_dir(tmp_path)

    assert cfg.ising.sigma == target.sigma == 0.22
    assert cfg.ising.composition_penalty_strength == 5.0
    reference = IsingTarget(
        D=2, sigma=0.22, target_composition=0.5,
        composition_penalty_strength=5.0, device=device,
    )
    states = torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1]], dtype=torch.float32, device=device,
    )
    torch.testing.assert_close(target.log_prob(states), reference.log_prob(states))
    assert path.read_bytes() == original


@pytest.mark.parametrize("omit_curricula", [False, True])
def test_legacy_fixed_targets_keep_saved_parameters(tmp_path, omit_curricula):
    saved = asdict(_curriculum_cfg())
    for key in ("curriculum", "lambda_curriculum"):
        if omit_curricula:
            saved.pop(key)
        else:
            saved[key] = None
    (tmp_path / "config.json").write_text(json.dumps(saved))

    cfg, target, _ = _rebuild_from_run_dir(tmp_path)

    assert cfg.ising.sigma == target.sigma == 0.1
    assert cfg.ising.composition_penalty_strength == 1.0
    assert target.composition_penalty_strength == 1.0


def test_alloy_rebuild_uses_terminal_temperature(tmp_path):
    from experiments.constrained_soft_02.configs import CONFIGS

    saved = asdict(CONFIGS["A1_cuau16_T500_letf_10k_curr"])
    (tmp_path / "config.json").write_text(json.dumps(saved))

    cfg, target, _ = _rebuild_from_run_dir(tmp_path)

    expected_beta = 1.0 / (8.617333262e-5 * 500.0)
    assert target.beta == pytest.approx(expected_beta)
    assert cfg.ising.sigma == pytest.approx(expected_beta / 2.0)


def test_curriculum_rescore_matches_training_final_eval(tmp_path):
    run_dir = train(
        _curriculum_cfg(), seed=0, output_dir=tmp_path, use_wandb=False,
    )
    original = json.loads((run_dir / "eval" / "metrics.json").read_text())
    tensor_bytes = {
        name: (run_dir / "eval" / name).read_bytes()
        for name in ("samples.pt", "log_weights.pt")
    }

    rescored = eval_only(run_dir)

    for key in (
        "free_energy_per_site", "internal_energy_per_site",
        "entropy_per_site", "free_energy_per_site_exact",
        "internal_energy_per_site_exact", "entropy_per_site_exact",
    ):
        assert rescored[key] == pytest.approx(original[key], abs=1e-7)
    for name, original_bytes in tensor_bytes.items():
        assert (run_dir / "eval" / name).read_bytes() == original_bytes
