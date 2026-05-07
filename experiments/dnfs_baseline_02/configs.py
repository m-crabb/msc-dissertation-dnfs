"""Frozen, named configurations for DNFS Ising baseline runs.

Configs are dataclasses keyed by name in `CONFIGS`; new entries show up
automatically in `run.py` via `--cfg <name>`. The dataclasses are frozen
so a single config object cannot be mutated mid-run (any per-seed override
goes through `dataclasses.replace`).

Eval sample budget: `n_eval_samples = 5000` matches the paper's Figure 13
energy-histogram pass (Appendix D.1) and is strictly ≥ the N = 2,048 used
for the Table 2 numerical pass, so a single eval feeds both downstream
artefacts. Across-seed std (paper Table 2 reports mean ± std over 10
independent runs) is a multi-seed sweep planned for a later step.

Stage layout:
    stage_0_d4    -- D = 4 broken-baseline (vanilla MLP + Eq. 7 + naive MC).
                     Not locally equivariant; rates are not one-way. Exists
                     to motivate leMLP empirically by direct comparison to
                     stage_1_d4.
    stage_0_d10   -- D = 10 broken baseline at paper scale.
    stage_1_d4    -- D = 4 leMLP + naive MC. First architecture-correct
                     run; small-lattice sanity vs. stage_2_d4.
    stage_1_d10   -- D = 10 leMLP + naive MC. Paper-scale; expected high
                     estimator variance motivates the control variate.
    stage_2_d4/10 -- leMLP + control variate (paper Eq. 8). Identical
                     to stage_1_* except for the estimator, so any
                     improvement attributes to the control variate alone.
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


# Default changed from "mlp" to "lemlp" at stage_1+; legacy stage_0 configs
# explicitly pass kind="mlp" so there are no silent behaviour changes.
@dataclass(frozen=True)
class ModelCfg:
    kind: Literal["mlp", "lemlp", "let"] = "lemlp"
    hidden_dim: int = 256
    n_layers: int = 3        # interpreted as n_summands K when kind="lemlp"
    vocab_size: int = 2


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
    # Stage 0: BROKEN BASELINE. Vanilla MLP + Eq. (7) residual + naive MC.
    # Architecture is NOT locally equivariant; rates are not one-way.
    # This run exists to motivate leMLP empirically — comparing stage_0
    # against stage_1 isolates "what does LE buy us" from "what does the
    # control variate buy us".
    "stage_0_d4": StageCfg(
        name="stage_0_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_0_d10": StageCfg(
        name="stage_0_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="naive_mc",
    ),
    # Stage 1: leMLP + naive MC. First architecture-correct run; failure
    # of naive MC at D=10 (high estimator variance) motivates Stage 2.
    "stage_1_d4": StageCfg(
        name="stage_1_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="naive_mc",
    ),
    # Stage 2: leMLP + control variate. Same architecture as stage 1;
    # apples-to-apples per-config attribution of the variance reduction.
    "stage_2_d4": StageCfg(
        name="stage_2_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_2_d10": StageCfg(
        name="stage_2_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="control_variate",
    ),
}
