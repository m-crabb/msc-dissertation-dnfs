"""Frozen, named configurations for DNFS Ising baseline runs.

Configs are dataclasses keyed by name in `CONFIGS`; new entries show up
automatically in `run.py` via `--cfg <name>`. The dataclasses are frozen
so a single config object cannot be mutated mid-run (any per-seed override
goes through `dataclasses.replace`).

Stage layout:
    stage_1_d4   -- D = 4 sanity gate (TVD < 0.05 target).
    stage_1_d10  -- D = 10 main run; expected to expose the high-variance
                    failure mode of the naive estimator.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IsingCfg:
    D: int = 10
    sigma: float = 0.1
    bias: float = 0.0


@dataclass(frozen=True)
class TrainCfg:
    n_steps: int = 50_000
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class CTMCCfg:
    n_euler_steps: int = 100
    time_grid: Literal["uniform"] = "uniform"


@dataclass(frozen=True)
class EvalCfg:
    eval_every: int = 500
    n_eval_samples: int = 5_000


@dataclass(frozen=True)
class ModelCfg:
    kind: Literal["mlp", "let"] = "mlp"
    hidden_dim: int = 256
    n_layers: int = 3


@dataclass(frozen=True)
class StageCfg:
    name: str
    ising: IsingCfg
    train: TrainCfg
    ctmc: CTMCCfg
    eval: EvalCfg
    model: ModelCfg
    estimator: Literal["naive_mc", "control_variate"]


CONFIGS: dict[str, StageCfg] = {
    # Stage 1 sanity gate. Tiny lattice (16 states, 65k joint states), short
    # training, generous eval budget. The whole point is exact-TVD < 0.05.
    "stage_1_d4": StageCfg(
        name="stage_1_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=2_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2),
        estimator="naive_mc",
    ),
    # Stage 1 main run. D = 10 puts the joint state space at 2**100 -- way
    # past exact enumeration; we judge by ESS, energy histogram and loss.
    # The naive estimator is *expected* to struggle here; that's the
    # baseline failure that motivates Stage 2 and Stage 3.
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3),
        estimator="naive_mc",
    ),
}
