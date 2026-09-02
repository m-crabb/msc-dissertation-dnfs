"""Frozen, named configurations for DNFS Ising baseline runs.

SIGMA_C MIGRATION (s58, 2026-08-24): the project's critical coupling is
targets/ising.py SIGMA_C = ln(1+sqrt(2))/4 = 0.220343 (exact). Every cell
below that carries sigma=0.22305 (or a 0.223 curriculum stage) describes an
ARCHIVED run trained at the legacy value; those literals are records and must
not be edited (stored run configs and the eval config-drift guard are pinned
to them). Any NEW sigma_c cell imports SIGMA_C; the archived sigma_c cells
are replaced by retrain waves, unconstrained chapter first, hard chapter
once the head family is finalised.

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
    # Matched base (s101, plan 2026-08-31-soft-camort-matched-base): the
    # base reads the composition the run is conditioned on — the bound
    # per-cycle vector during amortised training, else target_composition —
    # instead of the static base_composition. One lever; motivated by the
    # 8x8 mb twins (off-centre collapse = base reachability).
    base_matches_composition: bool = False
    # Ceiling on log p̃_t(y)/p̃_t(x) at single-flip neighbours. The paper's 5
    # suits an unpenalised Ising target; a composition penalty of strength λ
    # adds ∓2λ·(c(x)−c_target) to the same ratio, so 5 binds once obedience
    # error exceeds 5/(2λ). Raise it only in cells that mean to test that.
    log_ratio_clamp: float = 5.0
    # A real alloy in place of the Ising torus: path (repo-relative) to a
    # binary spin-product expansion exported by
    # experiments/alloy_ce/export_binary_expansion.py. When set, the target is
    # ClusterExpansionTarget on that cell, `sigma` means beta/2 = 1/(2 k_B T)
    # in 1/eV (11.60 at 500 K), and D is a label: d comes from the file.
    expansion_json: str | None = None


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
    # Run the LOSS UPDATE under bf16 autocast (opt-in; default fp32, so every
    # archived cell is byte-identical). Scoped to the inner gradient step: the
    # end-of-run eval stays fp32, which is the chapter's standing precision
    # discipline, so a bf16-TRAINED cell is still JUDGED in fp32.
    #
    # It is a BYTE lever, which is the kind that works here. The d400 swap
    # step is bandwidth-bound -- 36% GEMM, and those GEMMs at 8.0 FLOP/byte
    # against the A100's fp32 ridge point of 10.1 -- so halving the width of
    # the (B, d, d, f) pair slabs buys what halving the arithmetic does not.
    # Measured d=400 R=3, batch 512, compiled, A100-80GB: fp32 0.400 s /
    # 63.41 GB, TF32 0.359 / 63.41, bf16 0.271 / 41.63, bf16+TF32 0.271 /
    # 41.60 -- TF32 adds nothing on top of bf16, which is the mechanism
    # confirmed rather than the outcome observed.
    #
    # The importance weights do NOT move: the closed-form swap log-ratio runs
    # through h = x @ A, a sum over four torus neighbours, so h is in
    # {-4,-2,0,2,4} at every lattice size and bf16 holds those exactly
    # (measured drift 0.000e+00 at D=16 and D=20, pinned in
    # tests/test_train_autocast_bf16.py). What DOES move is the head's G,
    # i.e. the proposal -- and self-normalised importance sampling is exact
    # for whatever proposal actually ran, so this is a variance question and
    # never a bias one.
    train_autocast_bf16: bool = False
    inner_steps_per_outer: int = 100
    outer_batch_size: int | None = None  # None -> falls back to batch_size
    replay_buffer_cycles: int = 1        # number of retained outer batches
    grad_clip_max_norm: float = 500.0  # some transformer runs override this
    # Opt-in mechanism diagnostic: record pre-clip gradient L2 norms split
    # into exact-field gains, omega readout, composition embedder, and the
    # remaining trunk on every update. Written to gradient_group_log.csv so
    # the stable training_log.csv schema and archived resume path do not move.
    log_gradient_group_norms: bool = False
    # Rows per inner-step backward slice (swap trainer only; None = one
    # backward over the full batch, every archived run). A memory schedule,
    # not a recipe variable: the swap loss is a per-row mean, so slicing the
    # backward accumulates the IDENTICAL total gradient up to float summation
    # order (loss_swap_backward_microbatched's docstring has the algebra;
    # tests/test_loss_microbatch_parity.py pins it). Set on the two 16x16
    # screen arms whose single-backward retained graph is measured to exceed
    # an A100-80GB (masked-attention h128; mask_one at d=256).
    loss_microbatch_size: int | None = None
    # LR warmup over the first N inner steps. **Paper deviation:** the DNFS
    # paper doesn't specify warmup; reference repo has none. Added 2026-05-13
    # after a 4-seed probe on stage_4_d10_paper showed 1/4 seeds healthy
    # without warmup (init-basin sensitivity exposed by adding seed_everything).
    # With warmup_steps=500, all 4 seeds recover to ESS frac 0.95-0.97; cross-
    # seed std drops ~50x. Step-0 only — does NOT re-fire at curriculum
    # σ-transitions, because that failure mode is structurally different (not
    # random-init) and the curriculum's LR drops already play the warmup role
    # at sensitive transitions.
    warmup_steps: int = 500              # 0 to disable; e.g. paper-faithful runs
    # Optimiser selection. "adamw" is the recipe of record for every archived
    # run. "stable_adamw" adds Adafactor-style per-tensor UPDATE clipping
    # (StableAdamW, Wortsman et al. 2023): the trust-region alternative to
    # raw-gradient-norm clipping, whose threshold is unit-free and therefore
    # size-invariant — the 16x16 rung showed grad_clip_max_norm's units are
    # extensive (an unnormalised pair-sum loss), so one clip value means
    # "spike guard" at d=64 and "permanent normalisation" at d=256.
    optimiser: str = "adamw"
    # Re-apply the lr warmup ramp at every curriculum sigma-transition, not
    # only at step 0. The comment above warmup_steps records the d=10-era
    # decision NOT to do this ("the curriculum's LR drops already play the
    # warmup role"); the d=256 log refutes that at scale — each sigma step
    # clears the replay buffer and jumps the target, launching an
    # initialisation-scale transient with no ramp. Off by default so every
    # archived run's semantics are unchanged.
    rewarmup_on_stage: bool = False
    # Empty the replay buffer at every curriculum sigma-transition. True =
    # every archived run's behaviour, so the default changes nothing.
    #
    # WHY THIS IS A REAL CHOICE AND NOT AN OBVIOUS ONE. The swap loss
    # (samplers/swap_kolmogorov.loss_swap) squares the Kolmogorov residual
    # delta_t(x) = dt_log_p_tilde_t(x) - c_t + sum_{i<j}(...), and both
    # target terms are recomputed LIVE from `target` at the sigma in force
    # when the gradient step runs. A retained state is therefore an
    # EVALUATION POINT, never a stale label: nothing in the buffer encodes
    # the old target. The fixed point delta == 0 holds pointwise, so it is
    # unchanged by which distribution the states are drawn from, as long as
    # that distribution has support. What retention changes is the WEIGHTING
    # of a weighted least-squares — at finite capacity, residual accuracy
    # gets spent where the previous stage's model law put mass rather than
    # where the new target does — plus the offset between c_t (estimated on
    # the CURRENT cycle's fresh rollout) and the measure the loss actually
    # averages over, which the `c_t_offset_rms` column already witnesses.
    #
    # The failure mode flushing guards against is that weighting mismatch;
    # the failure mode flushing CAUSES is that the buffer collapses to a
    # single cycle exactly at the boundary, multiplying per-state re-fit
    # density by `replay_buffer_cycles` for the ~800 steps it takes to
    # refill, at the moment the target has just moved. Which dominates has
    # never been measured at any size, so it gets a flag and an arm.
    #
    # NOT evidence either way: the DNFS reference never flushes because it
    # has no curriculum, so it never faced this choice.
    flush_replay_on_stage: bool = True
    # CV-inversion tripwire (adversarial panel, 2026-08-18). The swap
    # trainer logs `cv_var_ratio` — controlled/naive integrand variance
    # over the same rollout rows, a validated 5/5 in-run classifier of the
    # d256 cold-CV inversion — every step, unconditionally. When
    # `halt_on_cv_inversion_after` is set AND the estimator is the control
    # variate, a ratio above 1.0 for a full trailing window of outer
    # cycles at/after that step stops the run gracefully
    # (cv_inversion_halt.json marker; final.pt still saved). None = off =
    # every archived cell's behaviour. Intended consumer: the registered
    # CV CONTINUATION cell, armed past the warm-heal horizon (~1000
    # steps) so a re-inversion cannot burn walltime unnoticed.
    halt_on_cv_inversion_after: int | None = None
    halt_cv_inversion_window: int = 10
    # Per-stage best checkpoints (boundary-shock arm, 2026-08-19). When on,
    # the swap trainer saves `checkpoints/best_stage<k>.pt` whenever the
    # TRAILING MEDIAN (window 3) of the periodic train-eval ESS makes a new
    # best within curriculum stage k, with the step and value recorded in
    # `checkpoints/stage_best.json`. Median, never the single-step value:
    # the 16x16 record shows single-step train-ESS peaks are noise
    # excursions over a stationary series, so best-by-peak would checkpoint
    # noise. Pure IO — dynamics, CSV schema and final.pt untouched; the
    # frozen eval still reads final.pt, and a stage-best eval is a separate
    # eval-only pass declared at judging. Off = every archived config.
    stage_best_checkpoints: bool = False
    # Save a step-tagged checkpoint every N inner steps (None = only the
    # rolling `latest.pt` + end-of-run `final.pt`). Motivation: a run whose
    # late-training loss enters an excursion/recovery cycle ends with a
    # `final.pt` sampling an arbitrary phase of that cycle, so eval on it can
    # understate the model the run actually reached by an order of magnitude
    # (observed at D=10: in-run states at loss ~5 while final.pt scored
    # obedience slope 0.079). Step tags let eval select the healthiest state
    # by a rule fixed before the run.
    checkpoint_every: int | None = None
    # Outer cycles between full-state preemption checkpoints
    # (`checkpoints/resume.pt`). Distinct from `checkpoint_every` above,
    # which writes weights only for eval selection: this one carries the
    # optimiser moments, step counter, RNG streams and replay buffer, i.e.
    # everything needed to CONTINUE rather than to score. Motivating cost:
    # the 2026-08-21 N11 matched-base family was killed at ~94% of a 50k
    # budget and, with only a weights-only `latest.pt` on disk, all eight
    # runs had to restart from step 0. Cadence is a trade between rewritten
    # work after a kill (up to one interval) and IO; 10 cycles = 1000 inner
    # steps at the standard 100-step cycle, ~2% of a 50k run. Arming it does
    # not move the trajectory (pinned by tests/test_training_resume.py), so
    # runs from this trainer stay comparable to every archived one.
    resume_every_outer: int = 10
    # Per-slot EMA of the c_t (dt log Z_t) grid across outer cycles
    # (M2, 2026-08-14). c_t noise enters the loss gradient multiplicatively
    # through (xi - c)·grad(xi); at d256-naive the per-slot SE is ~0.93 nats.
    # The Eq.-8 identity E[xi] = dt log Z_t holds for the model's own law, so
    # smoothing across recent cycles is pure variance reduction at an
    # unchanged fixed point — the across-cycle complement of the Stein
    # CV's within-cycle reduction. The EMA resets at every curriculum
    # sigma transition (c_t is a function of sigma). 0.0 = OFF, the
    # byte-identical archived behaviour; the candidate value is 4.0 cycles.
    c_t_ema_halflife_cycles: float = 0.0
    # Decoupled c_t rollout batch (M3, 2026-08-14; plan Task 3). c_t =
    # mean_m xi_t over the outer cycle's rollout states, so its standard
    # error falls 1/sqrt(M) in the rollout row count while only the no-grad
    # trajectory phase pays for the extra rows (the run-D rollout lever is
    # priced at 95 h because older cells scaled EVERYTHING; here the inner
    # batch and replay buffer stay at outer_batch). The buffer takes the
    # first outer_batch rows of the enlarged rollout (iid base draws make
    # the prefix a uniform subset). None = outer_batch = OFF, the
    # byte-identical archived behaviour; candidate d256 value 512.
    c_t_batch: int | None = None
    # Batched c_t grid calls (M7a, 2026-08-14; plan Task 7). The c_t grid
    # costs n_grid sequential no-grad integrand calls per outer cycle; at
    # d256 trajectory+c_t is ~75% of wall. When set, the (n_grid x
    # n_rollout) integrand evaluations run flattened in row-chunks of at
    # most this many rows — the same fp32 ops modulo batch-dim blocking,
    # parity-pinned at the established 1e-5 class (the parity test IS the
    # M7a gate: no quality change permitted). The cap keeps the flattened
    # batch inside GPU memory (~40 MB/row no-grad at d256-MA, so 512-2048
    # rows is the in-cap class on an 80 GB a100). None = OFF, the
    # byte-identical per-slot sequential loop every archived run used.
    c_t_grid_chunk_rows: int | None = None
    # c_t grid built from the rollout's own forwards (B1/B1-flip,
    # optimisation decision 2026-08-24). In control_variate mode the outer
    # step re-runs the head/model on (trajectory[k], t_k) for every grid
    # slot, but rollout step k already computed that exact forward — with
    # this knob the sampler accumulates xi_t during the rollout and only
    # the final slot pays a fresh call, removing T-1 of T grid forwards
    # (~7-8 h eager per 16x16 CV run). BIT-IDENTICAL to the sequential
    # (chunk_rows=None) grid — same tensors, same arithmetic, no RNG
    # (tests/test_cv_integrand_reuse.py); vs a chunked grid the difference
    # is the established 1e-5 batch-blocking class. Requires rollout
    # resampling OFF. Supersedes c_t_grid_chunk_rows when on (the grid
    # pass it chunked no longer runs). False = OFF, the byte-identical
    # archived behaviour; True in the post-s60 base recipe.
    c_t_from_rollout: bool = False
    # ESS-triggered SMC resampling INSIDE the training rollout (both
    # trainers consume it: `swap_training.train_swap` and the flip-route
    # `training.train`; mechanism in `samplers.resampling`). None =
    # OFF = every archived run,
    rollout_resample_ess_fraction: float | None = None


@dataclass(frozen=True)
class CTMCCfg:
    n_euler_steps: int = 100
    time_grid: Literal["uniform"] = "uniform"
    # Simulate swap-CTMC trajectories with the vertex-disjoint matching
    # (multi-event) step everywhere the cell samples: training buffer
    # rebuilds, in-training eval draws, and the canonical final eval. Needed
    # from the 16x16 hard rung up, where clip-safe one-event simulation would
    # take ~390 Euler steps while the matching step holds n_euler_steps at
    # 128 by decoupling trajectory length from the total rate. The default
    # False reproduces every archived run bit-for-bit, which is what lets
    # `_backfill_missing_defaults` treat absence as "ran with the default".
    use_matching_step: bool = False


@dataclass(frozen=True)
class EvalCfg:
    eval_every: int = 500
    n_eval_samples: int = 5_000
    # Stream the eval draw in slices of this many samples (None = all at
    # once). Needed by the swap-head route at large d, where the vectorised
    # head rides d anchor copies per sample; the single-site path ignores it.
    eval_sample_chunk: int | None = None
    # In-training eval draw size; None -> n_eval_samples (behaviour
    # unchanged). The end-of-run eval in run.py always uses n_eval_samples:
    # the in-training ESS cadence is a diagnostic, not the objective, so it
    # may run smaller (e.g. 512) where the eval dominates GPU time.
    n_eval_samples_training: int | None = None
    # Run the in-training eval under bf16 autocast (opt-in; default fp32).
    # Eval-only: gradients never flow here, and the structural invariants
    # (swap-antisymmetry, composition preservation) hold exactly at bf16 --
    # only G's values move, within bf16 tolerance (test_swap_perf_refactors).
    eval_autocast_bf16: bool = False


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
    # leTF only: fused-kernel readout attention (never materialises the
    # (B, n_heads, d, 2d) score buffer). Tier-2 opt-in: same math, different
    # reduction order; no state_dict change.
    use_sdpa_readout: bool = False
    vocab_size: int = 2
    # leTF only: take the target composition as a second conditioning input,
    # so one trained model serves many compositions. OFF leaves parameter
    # construction identical to an unconditioned model, so archived
    # checkpoints stay loadable.
    condition_on_composition: bool = False
    # Soft route only (s90, 2026-08-29): wrap the rate model with the exact
    # flip log-ratio as a fixed additive score under a zero-init learned gain
    # (constraints/exact_field_channel.ExactFieldFlipModel, the flip twin of
    # the hard chapter's swap channel). The s90 4x4 regression put the closed
    # form at ~95% of every trained lambda=50 specialist, so the channel
    # supplies what training currently learns under lambda^2 penalty-variance
    # fire. OFF = every archived cell byte-identical.
    exact_field_channel: bool = False
    # Soft amortised route only: retain the exact-field channel's global
    # g0 + g1*t gain and add a centred composition correction
    # (c-c0)*(h0+h1*t). Opt-in so archived checkpoints keep their exact
    # state-dict schema; the two new scalars are zero-init and consume no RNG.
    exact_field_composition_gain: bool = False
    # rope_vit only (hard route, 2026-08-23): side of the p x p patches whose
    # pooled keys carry the far field in the causal stacks. 1 = dense causal
    # attention with periodic rotary positions; the leTF cells never read it.
    patch_size: int = 1
    # Opt-in torch.compile of the built rate model (optimisation board
    # section C, decided s60 2026-08-24; mirrors the hard route's
    # compile_head). Model only — the Euler loop's data-dependent sampling
    # would graph-break. Compiled runs are 1e-5-class vs eager, never
    # bit-parity. False = every archived cell byte-identical.
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

    Same boundary rules as the σ and λ stages, with one deliberate
    difference: crossing one of these does NOT clear the replay buffer. The
    target has not moved — only the distribution compositions are drawn from
    — so retained states remain valid, and discarding them would throw away
    exactly the wide-window samples the widening exists to accumulate.
    """

    start_step: int
    half_width: float
    lr: float | None = None


@dataclass(frozen=True)
class CompositionCfg:
    """Amortise one model over a range of target compositions.

    The draw window is [centre − half_width, centre + half_width], one
    composition per outer cycle. It is centred rather than given as (lo, hi)
    because zero-bias Ising is invariant under the joint map (x → −x,
    c → 1−c), so a symmetric window respects a symmetry the target has and
    the whole schedule anneals through the single scalar half_width.

    Set `values` instead to draw from a finite set — the control that
    amortises only over compositions we already have specialists for.
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
    # EMA dual-eval (ported from the hard chassis for the soft-chapter
    # revamp, s95): > 0 arms a warmup-corrected parameter shadow updated
    # after every optimiser step (discrete_flow_sampler.ema), saved as
    # checkpoints/final_ema.pt and evaluated alongside the raw weights
    # (eval/ + eval_ema/). The shadow is a passive observer — never read
    # by training — so runs differing only in this field train
    # bit-identically, and 0.0 leaves every archived cell's artefact
    # inventory unchanged.
    ema_decay: float = 0.0


def optimised_recipe(cell: StageCfg) -> StageCfg:
    """Optimisation bundle as a recipe transform  for the flip route 
    — the mirror of the hard route's `optimised_recipe`.

    Two declared changes, nothing else: model.compile_model=True (measured
    1.58x updates / 2.01x rollout on stage_4_d10 leTF, same-container
    Modal A100; GPU-stack gate passed) and train.c_t_from_rollout=True
    (bit-identical c_t grid from the rollout's own forwards). Every NEW cell
    — Wave-1 retrains included — goes through this transform; archived cells
    and eager twins of eager parents keep both flags off, because compiled
    runs are 1e-5-class vs eager, never bit-parity.
    """
    return replace(
        cell,
        model=replace(cell.model, compile_model=True),
        train=replace(cell.train, c_t_from_rollout=True),
    )


def sigma_c_twin(parent: StageCfg) -> StageCfg:
    """Wave-1 sigma_c retrain twin.

    The archived cell with the coupling — and ONLY the coupling — moved
    0.22305 -> the exact SIGMA_C = ln(1+sqrt(2))/4, in `ising.sigma` and the
    final curriculum stage. The sigma ladder below the endpoint stays
    verbatim: its 0.220 stage now sits just 0.000343 below the new endpoint,
    kept because reshaping the ladder would confound the coupling change
    with a schedule change (and a gentler final step is the safe direction —
    hard sigma-boundary shocks are the documented seed-killer, not soft
    ones). The s60 optimised recipe is then applied, the two declared
    exceptions to the twin discipline for every new cell.
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
    # D=10) and leave d_l (per-layer channel dim) to pick at impl time.
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
    # init-basin sensitivity exposed by adding seed_everything (finding
    # recorded 2026-05-13; see the TrainCfg.warmup_steps comment for the
    # outcome).
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
        model=ModelCfg(
            kind="let", hidden_dim=128, n_layers=3, n_heads=4, vocab_size=2
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


# ---------------------------------------------------------------------------
# Unconstrained 8x8 / 16x16 cells on the HARD chapter's house recipes, with
# exact-field-channel twins (s117, 2026-09-02). The channel had never been
# run on the unconstrained rung; on the Cu-Au alloy it was the largest
# single lever (soft c=0.25: 0.66 -> 0.86). Recipe = the hard house cells
# at the same size (`H2_d64_c50_s220_letf_mo_50k_curr_w2`, `H2_d256_..._100k_curr_b512_ne128_cv2_w3`):
# 8x8 = 50k steps, batch 128, ne128; 16x16 = 100k steps, batch 512 with
# microbatch 128, ne128; both EMA 0.9999, the seven-stage sigma ladder to
# the EXACT sigma_c, replay 8, compiled. The flip MODEL stays the baseline
# `let` 128x3, so the twin differs from its parent by the channel alone.
# The archived `stage_4_d16_critical_50k_ladder` ended at the legacy 0.22305
# and died (ESS 0.0); these replace it as the 16x16 unconstrained record.
_HARD_HOUSE_LADDER = CurriculumCfg(stages=(
    CurriculumStageCfg(start_step=0, sigma=0.10, lr=1e-3),
    CurriculumStageCfg(start_step=5_000, sigma=0.14, lr=1e-3),
    CurriculumStageCfg(start_step=10_000, sigma=0.17, lr=1e-3),
    CurriculumStageCfg(start_step=15_000, sigma=0.19, lr=1e-3),
    CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
    CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
    CurriculumStageCfg(start_step=30_000, sigma=SIGMA_C, lr=3e-4),
))

for _side, _n_steps, _batch, _microbatch in ((8, 50_000, 128, None), (16, 100_000, 512, 128)):
    _parent = CONFIGS["stage_4_d8_critical_paper_curriculum_sc"]
    _name = f"stage_4_d{_side}_sc_hardrecipe"
    _cell = replace(
        _parent, name=_name,
        ising=replace(_parent.ising, D=_side, sigma=0.10),
        train=replace(_parent.train, n_steps=_n_steps, batch_size=_batch,
                      outer_batch_size=None, replay_buffer_cycles=8,
                      loss_microbatch_size=_microbatch),
        ctmc=replace(_parent.ctmc, n_euler_steps=128),
        curriculum=_HARD_HOUSE_LADDER, ema_decay=0.9999,
    )
    CONFIGS[_name] = _cell
    CONFIGS[f"{_name}_efc"] = replace(
        _cell, name=f"{_name}_efc",
        model=replace(_cell.model, exact_field_channel=True),
    )
