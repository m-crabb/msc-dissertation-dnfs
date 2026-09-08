"""Frozen, named configurations for DNFS Ising baseline runs.

Frozen dataclasses keyed by name in `CONFIGS` (`run.py --cfg <name>`);
per-seed overrides go through `dataclasses.replace`. The critical coupling is
targets/ising.py SIGMA_C = ln(1+sqrt(2))/4 = 0.220343 (exact). Cells carrying
sigma=0.22305 (or a 0.223 curriculum stage) are archived runs at the legacy
value; those literals are records and must not be edited. New sigma_c cells
import SIGMA_C; archived ones get `_sc` twins via `sigma_c_twin`.
`n_eval_samples = 5000` matches the paper's Figure 13 pass (App. D.1) and
exceeds the N = 2,048 of the Table 2 pass (mean ± std over 10 runs).
"""

from dataclasses import dataclass, replace
from typing import Literal

from discrete_flow_sampler.targets.ising import SIGMA_C


@dataclass(frozen=True)
class IsingCfg:
    D: int = 10
    sigma: float = 0.1
    bias: float = 0.0
    target_composition: float | None = None
    composition_penalty_strength: float = 0.0
    base_composition: float = 0.5
    # Base reads the composition the run is conditioned on (bound per-cycle
    # vector when amortised, else target_composition) instead of base_composition.
    base_matches_composition: bool = False
    # Ceiling on log p̃_t(y)/p̃_t(x) at single-flip neighbours (paper: 5). A
    # penalty of strength λ adds ∓2λ·(c(x)−c_target), so 5 binds past 5/(2λ).
    log_ratio_clamp: float = 5.0
    # Alloy in place of the torus: path to a binary spin-product expansion
    # (ClusterExpansionTarget); `sigma` = 1/(2 k_B T) in 1/eV, d read from the file.
    expansion_json: str | None = None


@dataclass(frozen=True)
class TrainCfg:
    n_steps: int = 50_000
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    # bf16 autocast on the inner gradient step only (eval stays fp32). Default
    # fp32 keeps every archived cell byte-identical.
    train_autocast_bf16: bool = False
    # Paper Algorithm 1 (App. C.1): each outer step rolls out `outer_batch_size`
    # trajectories, then `inner_steps_per_outer` updates (must divide n_steps).
    inner_steps_per_outer: int = 100
    outer_batch_size: int | None = None  # None -> falls back to batch_size
    replay_buffer_cycles: int = 1  # number of retained outer batches
    grad_clip_max_norm: float = 500.0  # some transformer runs override this
    # Diagnostic: pre-clip grad norms per parameter group every update, written
    # to gradient_group_log.csv; training_log.csv schema untouched.
    log_gradient_group_norms: bool = False
    # Rows per backward slice (swap trainer; None = one full-batch backward,
    # every archived run). Gradient-exact up to summation order; a memory lever.
    loss_microbatch_size: int | None = None
    # LR warmup over N inner steps, step 0 only (paper has none; 0 disables). On
    # stage_4_d10_paper 1/4 seeds trained without it, 4/4 with 500 (ESS 0.95-0.97).
    warmup_steps: int = 500
    # "adamw" for every archived run. "stable_adamw" adds StableAdamW (Wortsman
    # et al. 2023) per-tensor update clipping, whose threshold is size-invariant.
    optimiser: str = "adamw"
    # Re-apply the warmup ramp at every curriculum sigma-transition: at d=256
    # each step clears the buffer and jumps the target with no ramp. Off = archived.
    rewarmup_on_stage: bool = False
    # Empty the replay buffer at every curriculum sigma-transition. True is every
    # archived run's behaviour; retaining is unmeasured either way at every size.
    flush_replay_on_stage: bool = True
    # CV-inversion tripwire: with the control variate, `cv_var_ratio` > 1.0 for a
    # full trailing window after this step halts (cv_inversion_halt.json). None = off.
    halt_on_cv_inversion_after: int | None = None
    halt_cv_inversion_window: int = 10
    # Save `checkpoints/best_stage<k>.pt` when the trailing median (window 3) of
    # train-eval ESS makes a new best in stage k. Pure IO; off = every archived cfg.
    stage_best_checkpoints: bool = False
    # Step-tagged checkpoint every N inner steps (None = latest.pt + final.pt only),
    # so eval can pick a state by a pre-fixed rule rather than final.pt's phase.
    checkpoint_every: int | None = None
    # Outer cycles between full-state resume checkpoints (optimiser, RNG, buffer);
    # 10 = 1000 inner steps, ~2% of a 50k run. Arming it does not move the trajectory.
    resume_every_outer: int = 10
    # Per-slot EMA of the c_t grid across outer cycles (variance reduction at the
    # Eq.-8 fixed point); resets at sigma transitions. 0.0 = off (archived), try 4.0.
    c_t_ema_halflife_cycles: float = 0.0
    # Rollout rows for the c_t estimate (SE ~ 1/sqrt(M)); the inner batch and
    # buffer stay at outer_batch. None = outer_batch = off (archived); d256 uses 512.
    c_t_batch: int | None = None
    # Run the (n_grid x n_rollout) c_t integrand calls flattened in row-chunks of
    # this size (1e-5-class parity, GPU-memory cap). None = sequential (archived).
    c_t_grid_chunk_rows: int | None = None
    # Accumulate xi_t during the rollout instead of re-running T-1 grid forwards
    # (bit-identical; needs resampling off; supersedes chunk_rows). False = archived.
    c_t_from_rollout: bool = False
    # ESS-triggered SMC resampling inside the training rollout (both trainers;
    # mechanism in `samplers.resampling`). None = off = every archived run.
    rollout_resample_ess_fraction: float | None = None


@dataclass(frozen=True)
class CTMCCfg:
    n_euler_steps: int = 100
    time_grid: Literal["uniform"] = "uniform"
    # Vertex-disjoint matching (multi-event) swap step wherever the cell samples;
    # needed from 16x16 up (one-event needs ~390 Euler steps vs 128). False = archived.
    use_matching_step: bool = False


@dataclass(frozen=True)
class EvalCfg:
    eval_every: int = 500
    n_eval_samples: int = 5_000
    # Stream the eval draw in slices of this many samples (None = all at once);
    # the swap head at large d needs it, the single-site path ignores it.
    eval_sample_chunk: int | None = None
    # In-training eval draw size (None -> n_eval_samples); the end-of-run eval
    # always draws n_eval_samples, so this diagnostic may run smaller (e.g. 512).
    n_eval_samples_training: int | None = None
    # bf16 autocast for the in-training eval only (default fp32). No gradients;
    # swap-antisymmetry and composition preservation hold exactly at bf16.
    eval_autocast_bf16: bool = False


# Default changed from "mlp" to "lemlp" at stage_1+; legacy stage_0 configs
# pass kind="mlp" explicitly, so no archived behaviour moved.
@dataclass(frozen=True)
class ModelCfg:
    kind: Literal["mlp", "lemlp", "leconv_deep", "let"] = "lemlp"
    hidden_dim: int = 256
    n_layers: int = 3  # n_summands K for lemlp; Linear blocks for mlp
    kernel_schedule: tuple[int, ...] = ()  # leconv_deep only; per-layer kernels
    hollow_global_context: bool = False  # leconv_deep only
    n_heads: int = 4  # leTF only; ignored elsewhere
    # leTF only: fused-kernel readout attention, never materialising the
    # (B, n_heads, d, 2d) score buffer. Same math, no state_dict change.
    use_sdpa_readout: bool = False
    vocab_size: int = 2
    # leTF only: target composition as a second conditioning input, so one model
    # serves many compositions. Off leaves archived checkpoints loadable.
    condition_on_composition: bool = False
    # Soft route only: exact flip log-ratio as a fixed score under a zero-init
    # gain (ExactFieldFlipModel; ~95% of a lambda=50 specialist). Off = archived.
    exact_field_channel: bool = False
    # Soft amortised only: keep the channel's g0 + g1*t gain and add a centred
    # (c-c0)*(h0+h1*t) correction; zero-init, opt-in so archived state-dicts load.
    exact_field_composition_gain: bool = False
    # rope_vit only (hard route): side of the p x p patches whose pooled keys carry
    # the far field. 1 = dense causal attention; leTF cells never read it.
    patch_size: int = 1
    # torch.compile the rate model only (the Euler loop would graph-break);
    # 1e-5-class against eager, never bit-parity. False = every archived cell.
    compile_model: bool = False


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
class CompositionCurriculumStageCfg:
    """Piecewise-constant widening of the composition draw window.

    Same boundary rules as the σ and λ stages, except that crossing one does
    not clear the replay buffer: the target has not moved, only the
    distribution compositions are drawn from, so retained states stay valid
    and discarding them would throw away the wide-window samples the widening
    exists to accumulate.
    """

    start_step: int
    half_width: float
    lr: float | None = None


@dataclass(frozen=True)
class CompositionCfg:
    """Amortise one model over a range of target compositions.

    The draw window is [centre − half_width, centre + half_width], one
    composition per outer cycle. Centred rather than (lo, hi) because
    zero-bias Ising is invariant under the joint map (x → −x, c → 1−c), so a
    symmetric window respects a symmetry the target has and the schedule
    anneals through the single scalar half_width.

    Set `values` instead to draw from a finite set — the control that
    amortises only over compositions that already have specialists.
    """

    centre: float = 0.5
    half_width: float = 0.0
    values: tuple[float, ...] | None = None
    curriculum: tuple[CompositionCurriculumStageCfg, ...] | None = None


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
    composition: CompositionCfg | None = None
    wandb_project: str = "dnfs-baseline"
    # > 0 arms a warmup-corrected parameter shadow (discrete_flow_sampler.ema),
    # saved as final_ema.pt and evaluated alongside raw; training never reads it.
    ema_decay: float = 0.0


def optimised_recipe(cell: StageCfg) -> StageCfg:
    """Flip-route optimisation bundle: model.compile_model=True (1.58x updates /
    2.01x rollout on stage_4_d10 leTF) and train.c_t_from_rollout=True. Every
    new cell goes through it; archived cells and eager twins keep both off,
    since compiled runs are 1e-5-class against eager, never bit-parity.
    """
    return replace(
        cell,
        model=replace(cell.model, compile_model=True),
        train=replace(cell.train, c_t_from_rollout=True),
    )


def sigma_c_twin(parent: StageCfg) -> StageCfg:
    """Wave-1 retrain twin: sigma 0.22305 -> exact SIGMA_C in `ising.sigma` and
    the final curriculum stage, then `optimised_recipe`. The ladder below stays
    verbatim (its 0.220 stage now sits 0.000343 under the endpoint) so the
    coupling change is not confounded with a schedule change.
    """
    curriculum = parent.curriculum
    if curriculum is not None:
        *ladder, final_stage = curriculum.stages
        curriculum = replace(
            curriculum, stages=(*ladder, replace(final_stage, sigma=SIGMA_C))
        )
    return optimised_recipe(
        replace(
            parent,
            name=parent.name + "_sc",
            ising=replace(parent.ising, sigma=SIGMA_C),
            curriculum=curriculum,
        )
    )


CONFIGS: dict[str, StageCfg] = {
    # Stage 0: vanilla MLP + Eq. (7) residual + naive MC. Eq. 7 needs no local
    # equivariance but its variance is reported intractable at scale even with
    # control variates; stage_0_d*_cv tests that, and against stage_2_d* isolates
    # what LE buys (architecture differs, estimator the same).
    "stage_0_d4": StageCfg(
        name="stage_0_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_0_d10": StageCfg(
        name="stage_0_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="naive_mc",
    ),
    # Stage 0_cv: vanilla MLP + Eq. (7) + control variate; only the estimator
    # changes from stage_0_d*, so it compares directly against stage_2_d*.
    "stage_0_d4_cv": StageCfg(
        name="stage_0_d4_cv",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_0_d10_cv": StageCfg(
        name="stage_0_d10_cv",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="mlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="control_variate",
    ),
    # Stage 1: leMLP + Eq. 10 (Prop. 1 + Eq. 20 fold the reverse rate into the
    # forward tensor) + naive MC; naive MC's failure at D=10 motivates Stage 2.
    "stage_1_d4": StageCfg(
        name="stage_1_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="naive_mc",
    ),
    "stage_1_d10": StageCfg(
        name="stage_1_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="naive_mc",
    ),
    # Stage 2: leMLP + control variate. Same architecture as stage 1, so the
    # difference is the variance reduction alone.
    "stage_2_d4": StageCfg(
        name="stage_2_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=128, n_layers=2, vocab_size=2),
        estimator="control_variate",
    ),
    "stage_2_d10": StageCfg(
        name="stage_2_d10",
        ising=IsingCfg(D=10, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=100),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="lemlp", hidden_dim=256, n_layers=3, vocab_size=2),
        estimator="control_variate",
    ),
    # LEAPS-style deep LEC at critical sigma (Holderrieth, Albergo & Jaakkola,
    # Sec. 9 + Fig. 7: depth-5 kernels [3,5,7,9,15], ESS ~68% on a 15x15 critical
    # Ising at ~100k params). Trimmed to [3,5,7,9], no lattice-spanning kernel at
    # D=10; hidden_dim is the per-layer channel dim d_l.
    "stage_3_d10_critical_deep": StageCfg(
        name="stage_3_d10_critical_deep",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
        ),
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
    # Full LEAPS Fig.-7 depth-5 schedule [3,5,7,9,15]; on a D=10 torus k=15 is
    # lattice-spanning. These names also carry the time-conditioned kernel
    # state in LeConvDeepRateMatrix.compute_body.
    "stage_3_d10_critical_deep_k15": StageCfg(
        name="stage_3_d10_critical_deep_k15",
        ising=IsingCfg(D=10, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000, batch_size=256, replay_buffer_cycles=4, lr=1e-3, seed=42
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
    # Plateau curriculum for the deep conv critical run: a linear ramp moves σ
    # every outer cycle and clears the buffer each time, so replay4 cannot help
    # near σ≈0.20-0.223. Plateaus hold each σ; LR drops at the near-critical spike.
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
    # Same plateau schedule, but kernel generation receives a leave-one-out
    # global token summary at each site. The summary excludes x_i, so the
    # Prop. 2 readout stays locally equivariant while the conv path sees
    # critical-scale magnetisation context.
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
    # Stage 4: leTF (DNFS Sec 3.3 + App B.3), paper-faithful per App. E.1.1: 3
    # bidirectional causal layers, 4 heads, AdamW, lr 1e-3, batch 128. d4 = 10k
    # steps for the small-d plot; d10 mirrors Fig 3 / Fig 14 (64 hidden, 50k);
    # d10_critical mirrors the Table 2 row at sigma=0.22305 (128 hidden, 100k).
    "stage_4_d4": StageCfg(
        name="stage_4_d4",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
    ),
    # 4x4 row at the critical coupling: an exact-enumeration reference (2^16
    # states) the 10x10 critical run cannot have. No curriculum: the finite 4x4
    # lattice has no phase transition, so the d10 sigma-transition collapse is moot.
    "stage_4_d4_critical": StageCfg(
        name="stage_4_d4_critical",
        ising=IsingCfg(D=4, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=10_000, batch_size=128, replay_buffer_cycles=8, lr=1e-3, seed=42
        ),
        ctmc=CTMCCfg(n_euler_steps=50),
        eval=EvalCfg(eval_every=200, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
    ),
    # Legacy subcritical leTF diagnostic (use `stage_4_d10_budget` for the Stage
    # 4 row). Clip 1.0 choked learning (pre-clip norms 100-200, effective LR
    # ~2e-6, ESS ~2%), so clip 10.0 + LR 3e-4 + per-block raw-input skip;
    # d10_critical's ramp replaced a hard sigma swap that dropped ESS 13.6% -> 0.09%.
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
        model=ModelCfg(kind="let", hidden_dim=64, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
    ),
    # Canonical subcritical leTF comparison run: the paper-aligned stack of
    # `stage_4_d10_paper` capped at 50k steps for budget-matched
    # cross-architecture comparison.
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
    # Paper-aligned leTF cells for Table 2 / App E.1.1: h=128, 64 time steps,
    # lr=1e-3, 200k steps, batch 128, outer M=256, four-outer-batch replay. Clip
    # stays 500 rather than stage_4's 10: median pre-clip transformer norms sit
    # well above 10, where clip=10 turns lr=3e-4 into an effective ~5e-5 step.
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
        estimator="control_variate",
    ),
    # 10k-step, 4-seed (42-45) probe of stage_4_d10_paper with LR warmup; outcome
    # in the TrainCfg.warmup_steps comment.
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
    "stage_4_d8_critical_paper_curriculum": StageCfg(
        name="stage_4_d8_critical_paper_curriculum",
        ising=IsingCfg(D=8, sigma=0.22305, bias=0.0),
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
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
    "stage_4_d16_critical_50k_ladder": StageCfg(
        name="stage_4_d16_critical_50k_ladder",
        ising=IsingCfg(D=16, sigma=0.22305, bias=0.0),
        train=TrainCfg(
            n_steps=50_000,
            batch_size=128,
            outer_batch_size=256,
            replay_buffer_cycles=4,
            lr=1e-3,
            seed=42,
            grad_clip_max_norm=500.0,
        ),
        ctmc=CTMCCfg(n_euler_steps=64),
        eval=EvalCfg(eval_every=500, n_eval_samples=5_000),
        model=ModelCfg(kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2),
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
}

_SMC_FLIP_BASELINE_BASE = CONFIGS["stage_4_d8_critical_paper_curriculum"]
for _tau, _tau_tag in ((0.3, "smc03"), (0.6, "smc06")):
    _arm_name = f"{_SMC_FLIP_BASELINE_BASE.name}_{_tau_tag}"
    CONFIGS[_arm_name] = replace(
        _SMC_FLIP_BASELINE_BASE,
        name=_arm_name,
        train=replace(
            _SMC_FLIP_BASELINE_BASE.train,
            rollout_resample_ess_fraction=_tau,
        ),
    )

for _wave1_parent_name in (
    "stage_4_d4_critical",
    "stage_4_d10_critical_paper_curriculum",
    "stage_4_d8_critical_paper_curriculum",
):
    _wave1_twin = sigma_c_twin(CONFIGS[_wave1_parent_name])
    CONFIGS[_wave1_twin.name] = _wave1_twin


# Unconstrained 8x8 / 16x16 cells on the hard chapter's house recipes, with
# exact-field-channel (`_efc`) twins; on the Cu-Au alloy the channel was the
# largest single lever (soft c=0.25: 0.66 -> 0.86). 8x8 = 50k steps, batch 128;
# 16x16 = 100k, batch 512 microbatched 128; both ne128, EMA 0.9999, replay 8,
# the seven-stage ladder to exact sigma_c, compiled, flip model `let` 128x3. They
# replace the archived `stage_4_d16_critical_50k_ladder` (legacy 0.22305, ESS 0.0).
_HARD_HOUSE_LADDER = CurriculumCfg(
    stages=(
        CurriculumStageCfg(start_step=0, sigma=0.10, lr=1e-3),
        CurriculumStageCfg(start_step=5_000, sigma=0.14, lr=1e-3),
        CurriculumStageCfg(start_step=10_000, sigma=0.17, lr=1e-3),
        CurriculumStageCfg(start_step=15_000, sigma=0.19, lr=1e-3),
        CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
        CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
        CurriculumStageCfg(start_step=30_000, sigma=SIGMA_C, lr=3e-4),
    )
)

# 10x10: the 16x16 100k recipe at the paper's own size, four seeds, testing
# whether house recipe + channel reaches the paper's 10x10 result at 100k where
# the printed sigma_c row (0.902) used 200k. No microbatching at d=100; the
# in-training eval draws 512 (end-of-run still 5000) to save GPU time.
for _side, _n_steps, _batch, _microbatch, _n_eval_training in (
    (8, 50_000, 128, None, None),
    (16, 100_000, 512, 128, None),
    (10, 100_000, 512, None, 512),
):
    _parent = CONFIGS["stage_4_d8_critical_paper_curriculum_sc"]
    _name = f"stage_4_d{_side}_sc_hardrecipe"
    _cell = replace(
        _parent,
        name=_name,
        ising=replace(_parent.ising, D=_side, sigma=0.10),
        train=replace(
            _parent.train,
            n_steps=_n_steps,
            batch_size=_batch,
            outer_batch_size=None,
            replay_buffer_cycles=8,
            loss_microbatch_size=_microbatch,
        ),
        ctmc=replace(_parent.ctmc, n_euler_steps=128),
        eval=replace(_parent.eval, n_eval_samples_training=_n_eval_training),
        curriculum=_HARD_HOUSE_LADDER,
        ema_decay=0.9999,
    )
    CONFIGS[_name] = _cell
    CONFIGS[f"{_name}_efc"] = replace(
        _cell,
        name=f"{_name}_efc",
        model=replace(_cell.model, exact_field_channel=True),
    )
