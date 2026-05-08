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

Stage layout (framing clarified by Zijing 2026-05-08):
    stage_0_d{4,10}     -- vanilla MLP + Eq. 7 + naive MC. Eq. 7 is a valid
                           loss for any single-site-flip parameterisation,
                           but has high empirical variance; naive_mc
                           amplifies that — expected R≡0 collapse at d=10.
    stage_0_d{4,10}_cv  -- vanilla MLP + Eq. 7 + control variate. Tests
                           Zijing's claim that Eq. 7 doesn't scale "even
                           with variance reduction." Apples-to-apples
                           against stage_2_d* (only the architecture
                           differs, isolating the LE contribution).
    stage_1_d{4,10}     -- leMLP + Eq. 10 + naive MC. Eq. 10 is the LE
                           specialisation (Prop. 1 + Eq. 20 fold the
                           reverse rate into the forward tensor). First
                           architecture-correct run.
    stage_2_d{4,10}     -- leMLP + Eq. 10 + control variate. Stacks
                           estimator-side variance reduction on top of
                           the architectural one.
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
    seed: int = 42
    # Paper Algorithm 1 (App. C.1) outer/inner replay-buffer training:
    # one outer step generates `outer_batch_size` trajectories under
    # stop-gradient; `inner_steps_per_outer` gradient updates draw N=
    # batch_size mixed-t entries from that buffer. n_steps must be a
    # multiple of inner_steps_per_outer.
    #
    # Reference: J-zin/DNFS main.py Ising config uses M=256, N=128,
    # steps_per_epoch=100 (consulted 2026-05-08).
    inner_steps_per_outer: int = 100
    outer_batch_size: int | None = None  # None -> falls back to batch_size


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
    # Stage 0: HIGH-VARIANCE BASELINE. Vanilla MLP + Eq. (7) residual.
    # Eq. 7 does not require local equivariance — it's a valid loss for
    # any single-site-flip parameterisation. Per Zijing (2026-05-08), its
    # empirical variance is intractable at scale even with control variates.
    # stage_0_d* uses naive_mc; stage_0_d*_cv adds the paper's control-variate
    # estimator to test Zijing's strong claim. Comparing stage_0_d*_cv vs.
    # stage_2_d* isolates "what does LE buy us" (architecture differs;
    # estimator the same).
    "stage_0_d4": StageCfg(
        name="stage_0_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_0_d10": StageCfg(
        name="stage_0_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="naive_mc",
    ),
    # Stage 0_cv: vanilla MLP + Eq. (7) + control variate. Same architecture
    # and loss as stage_0_d*, only the estimator changes. Tests Zijing's
    # "Eq. 7 doesn't scale even with variance reduction" claim within the
    # post-redo result set; compares apples-to-apples against stage_2_d*.
    "stage_0_d4_cv": StageCfg(
        name="stage_0_d4_cv",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_0_d10_cv": StageCfg(
        name="stage_0_d10_cv",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="control_variate",
    ),
    # Stage 1: leMLP + naive MC. First architecture-correct run; failure
    # of naive MC at D=10 (high estimator variance) motivates Stage 2.
    "stage_1_d4": StageCfg(
        name="stage_1_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
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
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_2_d10": StageCfg(
        name="stage_2_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="control_variate",
    ),
}
