"""Sigma-curriculum wiring for the hard-constraint cells: `train_swap` has
accepted `sigma_curriculum` since the baseline, but `run.py` never passed it —
the d=64 sigma_c cells trained cold at 0.223 (the 25k budget probe reached ESS
frac 0.12 with the loss still descending, 2026-07-06). These tests pin the
one-argument wire end-to-end and the smoke-mode shrink of the 50k ladder."""
import csv

import torch
from experiments.constrained_hard_03.configs import CONFIGS, HardStageCfg
from experiments.constrained_hard_03.run import smoke_config, train
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    CurriculumCfg,
    CurriculumStageCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    TrainCfg,
)


def _tiny_curriculum_cfg():
    return HardStageCfg(
        name="tiny_hard_curriculum",
        ising=IsingCfg(D=4, sigma=0.223, bias=0.0, target_composition=0.5),
        train=TrainCfg(n_steps=4, batch_size=4, inner_steps_per_outer=2, seed=0),
        ctmc=CTMCCfg(n_euler_steps=8),
        eval=EvalCfg(eval_every=4, n_eval_samples=8, eval_sample_chunk=4),
        model=ModelCfg(kind="letf", hidden_dim=16, n_layers=2, n_heads=2,
                       vocab_size=2),
        estimator="control_variate",
        head_kind="mask_one",
        wandb_project="test",
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.1, lr=1e-3),
                CurriculumStageCfg(start_step=2, sigma=0.223, lr=3e-4),
            )
        ),
    )


def test_train_passes_sigma_curriculum_through(tmp_path):
    """Two plateaus over four steps: the training log must show sigma moving
    0.1 -> 0.223 at the stage boundary (cold runs log 0.223 throughout)."""
    torch.manual_seed(0)
    run_dir = train(_tiny_curriculum_cfg(), seed=0, output_dir=tmp_path,
                    use_wandb=False)

    with (run_dir / "training_log.csv").open() as log_file:
        rows = list(csv.DictReader(log_file))
    sigmas = [float(row["sigma_current"]) for row in rows]
    assert sigmas[:2] == [0.1, 0.1]
    assert sigmas[2:4] == [0.223, 0.223]


def test_smoke_config_shrinks_curriculum_to_runnable_two_stages():
    """The 50k ladder's start_steps exceed the 4-step smoke budget; smoke_config
    must keep a transition to exercise (stage-0 sigma -> final sigma at the
    outer-cycle boundary) rather than crash `_normalise_curriculum`."""
    smoke = smoke_config(CONFIGS["H2_d64_c50_s223_letf_mo_50k_curr"])

    stages = smoke.curriculum.stages
    assert len(stages) == 2
    assert (stages[0].start_step, stages[0].sigma) == (0, 0.100)
    assert (stages[1].start_step, stages[1].sigma) == (2, 0.223)
    assert all(
        stage.start_step < smoke.train.n_steps
        and stage.start_step % smoke.train.inner_steps_per_outer == 0
        for stage in stages
    )
