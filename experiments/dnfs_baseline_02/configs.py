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
    stage_1_d4     -- D = 4 small-lattice sanity (16 spins, 65k joint states);
                      enumeration-based exact F/E/S references available.
    stage_1_d4_xl  -- capacity / training-budget probe over the same target.
    stage_1_d10    -- D = 10 paper-scale run, MLP-256x3.
    stage_1_d10_xl -- capacity probe at paper scale; mirrors d4_xl's 512x4
                      MLP so the d4_small/d4_xl/d10_small/d10_xl 2x2 grid
                      cleanly disentangles capacity-scale interactions.
    stage_2_*      -- Stage 1 ladder repeated with `estimator="control_variate"`
                      (paper Eq. 8) and otherwise-identical training settings,
                      so any improvement attributes to the estimator alone.
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
    # Stage 1 small-lattice probe. D = 4 → 16 spins, 65k joint states;
    # exact F/E/S references available via enumeration.
    "stage_1_d4": StageCfg(
        name="stage_1_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2),
        estimator="naive_mc",
    ),
    # Capacity / training-budget probe over the same D = 4 target. The
    # 4×-wider, 2×-deeper MLP at half the learning rate over 5× the steps
    # rules out underfitting / budget exhaustion as the dominant failure
    # mode for the naive MC estimator on this target.
    "stage_1_d4_xl": StageCfg(
        name="stage_1_d4_xl",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=128, lr=5e-4, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=512, n_layers=4),
        estimator="naive_mc",
    ),
    # Paper-scale main run. D = 10 → 100 spins, 2^100 joint states; far
    # past enumeration. Eval reports ESS plus IS estimates of F/E/S; the
    # exact-reference comparison uses Ferdinand & Fisher (1969) at this
    # scale (reference helper TBD; not in this module yet). The naive
    # estimator is expected to struggle here -- that's the baseline
    # failure motivating Stage 2's control variate (paper Eq. 8).
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3),
        estimator="naive_mc",
    ),
    # Capacity probe at paper scale: same MLP shape (512 x 4) as stage_1_d4_xl,
    # so "same architecture, four data points" gives a clean 2x2 grid of
    # capacity x lattice-size that disambiguates "naive MC fails because of
    # size" vs "naive MC fails because of capacity".
    "stage_1_d10_xl": StageCfg(
        name="stage_1_d10_xl",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=512, n_layers=4),
        estimator="naive_mc",
    ),
    # Stage 2 ladder: identical to Stage 1 in every parameter except
    # estimator="control_variate". Apples-to-apples per-config attribution.
    "stage_2_d4": StageCfg(
        name="stage_2_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2),
        estimator="control_variate",
    ),
    "stage_2_d4_xl": StageCfg(
        name="stage_2_d4_xl",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=128, lr=5e-4, seed=0),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=512, n_layers=4),
        estimator="control_variate",
    ),
    "stage_2_d10": StageCfg(
        name="stage_2_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3),
        estimator="control_variate",
    ),
    "stage_2_d10_xl": StageCfg(
        name="stage_2_d10_xl",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=0),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=512, n_layers=4),
        estimator="control_variate",
    ),
}
