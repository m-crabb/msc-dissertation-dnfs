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
    kind: Literal["mlp", "lemlp", "leconv", "leconv_deep", "let"] = "lemlp"
    hidden_dim: int = 256
    n_layers: int = 3        # n_summands K for lemlp/leconv; Linear blocks for mlp
    kernel_size: int = 3     # leconv only; ignored elsewhere
    kernel_schedule: tuple[int, ...] = ()  # leconv_deep only; per-layer kernels
    n_heads: int = 4         # leTF only; ignored elsewhere
    vocab_size: int = 2


@dataclass(frozen=True)
class WarmupCfg:
    """MDNS-style temperature warm-up (App D.2.4 of `papers/mdns.pdf`).

    Train at `sigma` for `n_steps` first, then swap target.σ to
    `IsingCfg.sigma` (the final, harder σ) for the remaining steps. The
    swap is pinned to a replay-buffer rebuild boundary in `samplers.training`.
    """
    n_steps: int                              # multiple of inner_steps_per_outer
    sigma: float                              # easier σ for the warm-up phase


@dataclass(frozen=True)
class StageCfg:
    name: str
    ising: IsingCfg
    train: TrainCfg
    ctmc: CTMCCfg
    eval: EvalCfg
    model: ModelCfg
    estimator: Literal["naive_mc", "control_variate"]
    warmup: WarmupCfg | None = None


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
    # Stage 3: leConv (locally equivariant 2D conv with hollow zero-centre
    # kernel, K parallel summands) + control variate. Translation symmetry
    # of the Ising lattice is encoded structurally via circular-padded
    # convolution; combined with the hollow constraint and Prop. 2 readout,
    # G is both LE and translation-equivariant. Hidden dim mirrors the
    # paper's leTF Ising experiment (Sec. E.1) at 64. Smaller capacity than
    # stage_2's leMLP (h=128/256); a comparable result would demonstrate
    # the parameter efficiency of translation-equivariant weight sharing.
    # See docs/design/2026-05-08-leconv-stage-3-design.md.
    "stage_3_d4": StageCfg(
        name="stage_3_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="leconv", hidden_dim=64, n_layers=3, kernel_size=3, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_3_d10": StageCfg(
        name="stage_3_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="leconv", hidden_dim=64, n_layers=3, kernel_size=3, vocab_size=2),
        estimator="control_variate",
    ),
    # MARS V submission cell: 10x10 Ising at the critical temperature
    # sigma = 0.22305 (paper Table 2). Sampling at T_c is the regime where
    # naive mean-field-style approximations fail; a working sampler here
    # is doing real work that mean field cannot.
    "stage_3_d10_critical": StageCfg(
        name="stage_3_d10_critical",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="leconv", hidden_dim=64, n_layers=3, kernel_size=3, vocab_size=2),
        estimator="control_variate",
    ),
    # First d10_critical attempt collapsed in ESS (~5/256) at 14k steps despite
    # loss decreasing — diagnosed as receptive-field starvation: the 3x3 kernel
    # is structurally too local for the long-range critical fluctuations on
    # a 10x10 torus. _big bumps the kernel to 7x7 (covers 49 sites, ~half the
    # lattice diameter) and the hidden dim to 128. If this also fails ESS, the
    # fallback is the d4_critical / d6_critical pair (smaller N matched to
    # the architecture's reach).
    "stage_3_d10_critical_big": StageCfg(
        name="stage_3_d10_critical_big",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="leconv", hidden_dim=128, n_layers=3, kernel_size=7, vocab_size=2),
        estimator="control_variate",
    ),
    # LEAPS-style deep LEC at critical sigma. Reference: Holderrieth/Albergo/
    # Jaakkola, papers/leaps.pdf, Section 9 + Figure 7. Their depth-5 LEC
    # with kernels [3,5,7,9,15] hit ESS ~68% on a 15x15 critical Ising at
    # ~100k params. We trim to [3,5,7,9] (no lattice-spanning kernel since
    # D=10) and let the user pick d_l (per-layer channel dim) at impl time.
    "stage_3_d10_critical_deep": StageCfg(
        name="stage_3_d10_critical_deep",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9),
            vocab_size=2,
        ),
        estimator="control_variate",
    ),
    # MDNS-style temperature warm-up applied to the deep LEC config.
    # Reference: Zhu et al. 2025, `papers/mdns.pdf` §4.1 + App D.2.4.
    # MDNS reports LEAPS gets ESS=0.384 at L=16 β_critical without warm-up;
    # MDNS itself reaches 0.933 with a warm-up at β_high. We test whether
    # DNFS Algorithm 1 + deep LEC can lift its 5.41% σ_critical ESS by
    # warming up at the paper's leTF-Fig.3 setting (σ=0.1) for 20k steps
    # before continuing at σ_critical for the remaining 30k.
    "stage_3_d10_critical_deep_warmup": StageCfg(
        name="stage_3_d10_critical_deep_warmup",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9),
            vocab_size=2,
        ),
        estimator="control_variate",
        warmup=WarmupCfg(n_steps=20_000, sigma=0.1),
    ),
}
