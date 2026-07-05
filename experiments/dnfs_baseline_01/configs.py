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
    stage_3_d{4,10}     -- TODO fill in
    stage_4_d{4,10}     -- TODO fill in                      
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IsingCfg:
    D: int = 10
    sigma: float = 0.1
    bias: float = 0.0
    target_composition: float | None = None
    composition_penalty_strength: float = 0.0
    base_composition: float = 0.5


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
    # steps_per_epoch=100, and a four-outer-batch replay buffer via
    # DataBuffer(max_size=1024 // N) (consulted 2026-05-09).
    inner_steps_per_outer: int = 100
    outer_batch_size: int | None = None  # None -> falls back to batch_size
    replay_buffer_cycles: int = 1        # number of retained outer batches
    grad_clip_max_norm: float = 500.0  # some transformer runs override this
    # LR warmup over the first N inner steps. **Paper deviation:** the DNFS
    # paper doesn't specify warmup; reference repo has none. Added 2026-05-13
    # after a 4-seed probe on stage_4_d10_paper showed 1/4 seeds healthy
    # without warmup (init-basin sensitivity exposed by adding seed_everything).
    # With warmup_steps=500, all 4 seeds recover to ESS frac 0.95-0.97; cross-
    # seed std drops ~50x. Step-0 only — does NOT re-fire at curriculum
    # σ-transitions, because that failure mode is structurally different (not
    # random-init) and the curriculum's LR drops already play the warmup role
    # at sensitive transitions. See docs/findings/2026-05-13-letf-init-basin.md.
    warmup_steps: int = 500              # 0 to disable; e.g. paper-faithful runs


@dataclass(frozen=True)
class CTMCCfg:
    n_euler_steps: int = 100
    time_grid: Literal["uniform"] = "uniform"


@dataclass(frozen=True)
class EvalCfg:
    eval_every: int = 500
    n_eval_samples: int = 5_000
    # Stream the eval draw in slices of this many samples (None = all at
    # once). Needed by the swap-head route at large d, where the vectorised
    # head rides d anchor copies per sample; the single-site path ignores it.
    eval_sample_chunk: int | None = None


# Default changed from "mlp" to "lemlp" at stage_1+; legacy stage_0 configs
# explicitly pass kind="mlp" so there are no silent behaviour changes.
@dataclass(frozen=True)
class ModelCfg:
    kind: Literal["mlp", "lemlp", "leconv_deep", "let"] = "lemlp"
    hidden_dim: int = 256
    n_layers: int = 3        # n_summands K for lemlp; Linear blocks for mlp
    kernel_schedule: tuple[int, ...] = ()  # leconv_deep only; per-layer kernels
    hollow_global_context: bool = False    # leconv_deep only
    n_heads: int = 4         # leTF only; ignored elsewhere
    vocab_size: int = 2


@dataclass(frozen=True)
class CurriculumStageCfg:
    """Piecewise-constant training stage for near-critical curricula.

    `start_step` must align with an outer-cycle boundary. `lr=None` leaves the
    optimiser LR unchanged at that stage; otherwise all AdamW param groups are
    updated before rebuilding the replay buffer.
    """

    start_step: int
    sigma: float
    lr: float | None = None


@dataclass(frozen=True)
class CurriculumCfg:
    stages: tuple[CurriculumStageCfg, ...]


@dataclass(frozen=True)
class LambdaCurriculumStageCfg:
    """Piecewise-constant λ-annealing stage for soft-composition cells.

    Same boundary rules as `CurriculumStageCfg`: `start_step` must align
    with an outer-cycle boundary, and `lr=None` leaves the optimiser LR
    unchanged at that stage.
    """

    start_step: int
    composition_penalty_strength: float
    lr: float | None = None


@dataclass(frozen=True)
class LambdaCurriculumCfg:
    stages: tuple[LambdaCurriculumStageCfg, ...]


@dataclass(frozen=True)
class StageCfg:
    name: str
    ising: IsingCfg
    train: TrainCfg
    ctmc: CTMCCfg
    eval: EvalCfg
    model: ModelCfg
    estimator: Literal["naive_mc", "control_variate"]
    curriculum: CurriculumCfg | None = None
    lambda_curriculum: LambdaCurriculumCfg | None = None
    wandb_project: str = "dnfs-baseline"


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
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_0_d10": StageCfg(
        name="stage_0_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
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
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_0_d10_cv": StageCfg(
        name="stage_0_d10_cv",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
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
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
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
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_2_d10": StageCfg(
        name="stage_2_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
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
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
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
    # Same deep LEC recurrence, but with the full LEAPS Figure-7 depth-5
    # schedule [3,5,7,9,15]. On a D=10 torus the k=15 layer is deliberately
    # lattice-spanning. Future runs under these names also include the
    # time-conditioned kernel state in LeConvDeepRateMatrix.compute_body.
    "stage_3_d10_critical_deep_k15": StageCfg(
        name="stage_3_d10_critical_deep_k15",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9, 15),
            vocab_size=2,
        ),
        estimator="control_variate",
    ),
    "stage_3_d10_critical_deep_k15_replay4": StageCfg(
        name="stage_3_d10_critical_deep_k15_replay4",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9, 15),
            vocab_size=2,
        ),
        estimator="control_variate",
    ),
    # Plateau curriculum for the deep conv critical run. A linear ramp keeps
    # changing σ every outer cycle, which means the replay buffer is cleared
    # every cycle and replay4 cannot help near the hard σ≈0.20-0.223 regime.
    # This schedule gives each intermediate distribution a fixed plateau,
    # then lowers LR once the near-critical variance spike begins.
    "stage_3_d10_critical_deep_k15_curriculum_replay4": StageCfg(
        name="stage_3_d10_critical_deep_k15_curriculum_replay4",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9, 15),
            vocab_size=2,
        ),
        estimator="control_variate",
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
                CurriculumStageCfg(start_step=5_000, sigma=0.140, lr=1e-3),
                CurriculumStageCfg(start_step=10_000, sigma=0.170, lr=1e-3),
                CurriculumStageCfg(start_step=15_000, sigma=0.190, lr=1e-3),
                CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
                CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
                CurriculumStageCfg(start_step=30_000, sigma=0.22305, lr=3e-4),
            )
        ),
    ),
    # Same plateau schedule as above, but kernel generation receives a
    # leave-one-out global token summary at each site. The summary excludes
    # x_i, so the Prop. 2 readout remains locally equivariant while giving
    # the conv path direct access to critical-scale magnetisation context.
    "stage_3_d10_critical_deep_k15_curriculum_global_replay4": StageCfg(
        name="stage_3_d10_critical_deep_k15_curriculum_global_replay4",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="leconv_deep",
            hidden_dim=64,
            kernel_schedule=(3, 5, 7, 9, 15),
            hollow_global_context=True,
            vocab_size=2,
        ),
        estimator="control_variate",
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
                CurriculumStageCfg(start_step=5_000, sigma=0.140, lr=1e-3),
                CurriculumStageCfg(start_step=10_000, sigma=0.170, lr=1e-3),
                CurriculumStageCfg(start_step=15_000, sigma=0.190, lr=1e-3),
                CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
                CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
                CurriculumStageCfg(start_step=30_000, sigma=0.22305, lr=3e-4),
            )
        ),
    ),
    # Stage 4: leTF (Locally Equivariant Transformer, DNFS Sec 3.3 + App B.3).
    # Paper-faithful per App. E.1.1: 3 bidirectional causal layers, 4 heads,
    # AdamW + lr 1e-3 + batch size 128. d4 cell runs at 10k steps for the
    # cross-architecture comparison plot at small d. d10 cell mirrors
    # Fig 3 / Fig 14 setup at 64 hidden / 50k steps; d10_critical mirrors
    # Table 2 row at sigma=0.22305 with 128 hidden / 100k steps.
    "stage_4_d4": StageCfg(
        name="stage_4_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    # 4x4 row at the critical coupling for the Stage-4 report table: gives the
    # critical operating point an exact-enumeration reference (2^16 states),
    # which the 10x10 critical run cannot have. Trains direct at sigma_c with
    # no curriculum: the sigma-transition collapse that motivated the d10
    # curriculum was a D=10 finding, and the finite 4x4 lattice has no phase
    # transition to fight. Fall back to a curriculum only if this fails.
    "stage_4_d4_critical": StageCfg(
        name="stage_4_d4_critical",
        ising=IsingCfg(D=4, sigma=0.22305, bias=0.0),
        train=TrainCfg(n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    # Legacy/debug subcritical leTF config. Kept as a diagnostic point for the
    # earlier cautious stack (smaller h, replay1, tight clip), not as the final
    # Stage 4 comparison row; use `stage_4_d10_budget` for that.
    #
    # Revised 2026-05-09 (second pass) after second-launch logs showed clip 1.0
    # was choking learning: pre-clip grad norms ran 100-200 → effective LR
    # ≈ 2e-6, ESS plateaued at ~2%. Stack now: (1) grad clip 10.0 (still
    # tight enough to catch the 30k-norm spikes that motivated the original
    # clip; ~16x looser effective step), (2) LR 3e-4, (3) per-block raw-input
    # skip in CausalStack. d10_critical now uses a linear sigma ramp 0.1 →
    # 0.22305 over 20k steps instead of the hard swap (training.py); the hard
    # swap collapsed ESS 13.6% → 0.09% at the boundary.
    "stage_4_d10": StageCfg(
        name="stage_4_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=3e-4,
            seed=42,
            grad_clip_max_norm=10.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    # Canonical subcritical leTF comparison run. This keeps the paper-aligned
    # architecture/training stack from `stage_4_d10_paper`, but caps the
    # budget at 50k steps for budget-matched cross-architecture comparison.
    "stage_4_d10_budget": StageCfg(
        name="stage_4_d10_budget",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=500,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    "stage_4_d10_critical": StageCfg(
        name="stage_4_d10_critical",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=100_000,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=3e-4,
            seed=42,
            grad_clip_max_norm=10.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.100, lr=3e-4),
                CurriculumStageCfg(start_step=20_000, sigma=0.170, lr=3e-4),
                CurriculumStageCfg(start_step=40_000, sigma=0.205, lr=3e-4),
                CurriculumStageCfg(start_step=60_000, sigma=0.22305, lr=3e-4),
            )
        ),
    ),
    # Paper-aligned leTF Ising cells for Table 2 / App E.1.1 debugging.
    # The earlier `stage_4_d10` run intentionally used the smaller Fig. 3
    # h=64 / 50k-step setting. The Table 2 Ising setup is larger and faster:
    # h=128, 64 time steps, lr=1e-3, 200k steps, batch 128, outer M=256,
    # and a four-outer-batch replay buffer. We keep a wide clip (500) rather
    # than the current stage_4 clip=10 because W&B shows median pre-clip
    # transformer norms well above 10; clip=10 turns lr=3e-4 into an
    # effective ~5e-5 step on typical batches.
    "stage_4_d10_paper": StageCfg(
        name="stage_4_d10_paper",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=200_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=500,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    # 10k-step probe of stage_4_d10_paper with LR warmup enabled. Used for
    # the 4-seed gate (42/43/44/45) measuring whether warmup recovers the
    # init-basin sensitivity exposed by adding seed_everything. See
    # docs/findings/2026-05-13-letf-init-basin.md.
    "stage_4_d10_paper_probe_warmup": StageCfg(
        name="stage_4_d10_paper_probe_warmup",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
            warmup_steps=500,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    "stage_4_d10_critical_paper": StageCfg(
        name="stage_4_d10_critical_paper",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=200_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
    ),
    "stage_4_d10_critical_paper_curriculum": StageCfg(
        name="stage_4_d10_critical_paper_curriculum",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=200_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
        ),
        estimator="control_variate",
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
                CurriculumStageCfg(start_step=10_000, sigma=0.140, lr=1e-3),
                CurriculumStageCfg(start_step=20_000, sigma=0.170, lr=1e-3),
                CurriculumStageCfg(start_step=30_000, sigma=0.190, lr=1e-3),
                CurriculumStageCfg(start_step=40_000, sigma=0.205, lr=1e-3),
                CurriculumStageCfg(start_step=55_000, sigma=0.215, lr=5e-4),
                CurriculumStageCfg(start_step=70_000, sigma=0.220, lr=5e-4),
                CurriculumStageCfg(start_step=85_000, sigma=0.22305, lr=3e-4),
            )
        ),
    ),
}
