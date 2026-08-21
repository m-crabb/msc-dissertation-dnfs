"""Hard-constraint configs: swap-move CTMC on the fixed-composition manifold.

Imports the shared schema dataclasses from the baseline experiment (same
pattern as `constrained_soft_02/configs.py`) and adds `HardStageCfg`, which
carries a `head_kind` selecting the swap-readout head. Unlike the soft-02
cells, the target here is `FixedCompositionIsingTarget`: composition is
enforced exactly by the swap move set (n_plus == N_A always), so there is no
`composition_penalty_strength` or `lambda_curriculum` to anneal.

Cell-name format: `H<S>_d<dim>_c<c_target_x100>_s<sigma label>_letf_<head tag>`,
where S is the species count — `H2_*` are the binary Ising cells, `H3_*` the
first Potts ones (composition then means the per-species share, c33 = thirds).
The sigma-ladder cells (`_dh` suffix) probe the swap-CTMC across the Ising
phase transition (σ_c ≈ 0.22305) with the correctness-gate
`DoublyHollowSwapHead`; the `_na` cell pairs the subcritical floor rung with
`NonAntisymSwapHead`, a deliberately antisymmetry-breaking negative control
for the antisymmetry ablation.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    CurriculumCfg,
    CurriculumStageCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    StageCfg,
    TrainCfg,
)
from torch import Tensor

from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.constraints.grouped_anchor_swap_head import (
    GroupedAnchorSwapHead,
)
from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
    _masked_body,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix


@dataclass(frozen=True)
class HardStageCfg(StageCfg):
    """`StageCfg` plus the swap-head selector for the hard-constraint route.

    `head_kind` picks which swap-readout head `build_swap_head` instantiates:
    "doubly_hollow" (DoublyHollowSwapHead, O(d^2), the architecture-agnostic
    correctness gate), "mask_one" (LeTFMaskOneSwapHead, O(d), the efficient
    head that agrees with doubly_hollow numerically), "non_antisym"
    (NonAntisymSwapHead, a negative control that must never be used for a
    real run -- see that class's docstring), "interval"
    (IntervalSwapHead, the one-pass three-interval spike head -- property-
    equivalent to doubly_hollow, not numerically equal: different H), or
    "masked_attention" (MaskedAttentionSwapHead, the reported one-pass head:
    same three-interval structure, band aggregated by exclusion-mask
    attention -- bit-exact blindness, decision 2026-07-07), or "factorised"
    (FactorisedSwapHead, one-pass low-rank bilinear causal factors plus
    hole-subtracted global context -- no per-pair network, 2026-08-13).
    """

    head_kind: Literal[
        "doubly_hollow", "mask_one", "non_antisym", "interval", "masked_attention",
        "grouped_anchor", "factorised",
    ] = "doubly_hollow"
    # Anchor-batch chunk for the mask_one head's vectorised forward; None =
    # unchunked. d=256 needs this: the stacked d-anchor-copies pass would
    # otherwise build a (d*B)-row buffer that OOMs the L4.
    anchor_chunk_size: int | None = None
    # Opt-in torch.compile of the built head (Tier 2, default OFF). Head
    # only — the Euler loop's data-dependent sampling would graph-break.
    compile_head: bool = False
    # Band-capacity knobs for the interval / masked_attention heads
    # (band-capacity push, 2026-07-08). None = the constructions every prior
    # run used:
    # band_feature_dim 16, attention_dim 32, pair_offsets (1, D) — row and
    # column adjacency of the flattened D x D lattice. attention_dim is
    # masked_attention-only (the interval head has no attention).
    band_feature_dim: int | None = None
    attention_dim: int | None = None
    pair_offsets: tuple[int, ...] | None = None
    # Round-2 stencil family: add the 5-point lattice-stencil
    # band-feature family to the masked_attention head. False = the depth-1
    # unary+offset band every prior run used. lattice_side is cfg.ising.D
    # (the flattened D x D grid), so no separate field is needed.
    use_stencil: bool = False
    # muP readout compensation for the interval/masked_attention pair
    # score (muP-init arm, 2026-08-18): fixed multiplier on G cancelling
    # the sqrt(hidden) init growth of <LayerNorm'd H, omega_diff> — the
    # dot has no fan-in compensation, so score scale IS the initial rate
    # scale (verified 2x at h32 -> h128). The muP value for width h
    # against the h32 reference is 32/h. 1.0 = every archived cell,
    # byte-identical (the multiply is skipped in the head's forward).
    readout_score_scale: float = 1.0
    # Grouped-anchor head knobs (2026-07-22). n_groups is k,
    # the number of masked body passes: k = d reproduces mask_one bit-exactly,
    # smaller k trades masked-site count for passes. Only read when head_kind
    # is "grouped_anchor", so every other cell stays byte-identical.
    n_groups: int | None = None
    grouping: str = "diagonal"
    group_chunk_size: int | None = None
    # Factorised-head knobs (2026-08-13). None = the head's own defaults
    # (bilinear_rank 8, factor_dim 32, global_feature_dim 16); the two use_*
    # switches are the head's ablation arms (bilinear-only has no interior
    # visibility, global-only no deep exterior). Only read when head_kind is
    # "factorised", so every other cell stays byte-identical.
    bilinear_rank: int | None = None
    factor_dim: int | None = None
    global_feature_dim: int | None = None
    use_bilinear: bool = True
    use_global: bool = True
    # Causal-sweep directions for the factorised bilinear term (A-prime,
    # 2026-08-14): extras from {"col", "diag"} give every pair a second
    # blind (prefix, suffix) split, so deep coverage grows to the
    # complement of the intersection of the per-ordering intervals. The
    # default ("row",) is byte-identical to the pre-extension head.
    site_orderings: tuple[str, ...] = ("row",)
    # Dual-eval EMA instrument (2026-08-13). 0.0 = off (every archived
    # cell). > 0 arms a warmup-corrected parameter shadow
    # (discrete_flow_sampler.ema) updated after each optimiser step:
    # training dynamics are untouched, eval/ stays the raw-parameter
    # primary, and an eval_ema/ reading is recorded alongside from
    # checkpoints/final_ema.pt. An instrument, not a recipe change — cells
    # differing only in ema_decay train bit-identically.
    ema_decay: float = 0.0
    # Target family on the fixed-composition manifold (Potts extension,
    # 2026-07-31).
    # "ising" = FixedCompositionIsingTarget, composition a scalar n_plus held
    # in `ising.target_composition`; "potts" = FixedCompositionPottsTarget,
    # composition an S-vector held in `potts_composition` below. Both fields
    # default to the Ising route so every run dir written before Potts existed
    # backfills to what it actually ran under `eval_only`'s drift guard (89363b3).
    target_kind: Literal["ising", "potts"] = "ising"
    # Per-species fractions, length S, summing to 1; each entry times d must be
    # integral or no exact slice exists. This is the SINGLE source of S: the
    # backbone's vocab_size is derived from its length in `_hard_cell`, so the
    # embedding tables and the target's species count cannot disagree (they
    # would otherwise fail as an index error inside nn.Embedding). Read only
    # when target_kind is "potts", mirroring n_groups / "grouped_anchor".
    potts_composition: tuple[float, ...] | None = None


class NonAntisymSwapHead(nn.Module):
    """Negative control: same readout as DoublyHollowSwapHead, UNMASKED body.

    G[:, i, j] = <H[:, j, :], omega_{x_i} - omega_{x_j}>, exactly as in
    `DoublyHollowSwapHead`, but `H` comes from a SINGLE unmasked leTF body
    pass (`_masked_body(backbone, x, t, ())`) instead of a fresh doubly-
    masked pass per (i, j). Same-spin pairs still read zero (the token
    difference vanishes), so the swap CTMC still conserves composition and
    the run is comparable to the `_dh` cells -- but `H[:, j, :]` now sees the
    live value of `x_i` too (nothing masks it), so the exact swap-
    antisymmetry `G(i, j | x) = -G(i, j | Swap2(x, i, j))` does NOT hold.
    This is exactly the property the antisymmetry-ablation control must
    falsify. Gate/ablation only -- never use this head for a real run.
    """

    def __init__(self, backbone: LeTFRateMatrix):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        omega = m.omega(x_idx)  # (B, d, h)
        H = _masked_body(m, x, t, ())  # (B, d, h); ONE unmasked pass, shared
        # diff[:, i, j, :] = omega_{x_i} - omega_{x_j}
        diff = omega.unsqueeze(2) - omega.unsqueeze(1)  # (B, d, d, h)
        return torch.einsum("bijh,bjh->bij", diff, H)  # G[:, i, j]


def build_swap_head(cfg: HardStageCfg, backbone: LeTFRateMatrix) -> nn.Module:
    """Map `cfg.head_kind` to an instantiated swap-readout head."""
    if cfg.head_kind == "doubly_hollow":
        head = DoublyHollowSwapHead(backbone)
    elif cfg.head_kind == "mask_one":
        head = LeTFMaskOneSwapHead(backbone, anchor_chunk_size=cfg.anchor_chunk_size)
    elif cfg.head_kind == "non_antisym":
        head = NonAntisymSwapHead(backbone)
    elif cfg.head_kind == "interval":
        # Default offsets (1, D): row and column adjacency of the flattened
        # D x D lattice -- the interactions the Ising energy is built from.
        head = IntervalSwapHead(
            backbone,
            pair_offsets=cfg.pair_offsets or (1, cfg.ising.D),
            band_feature_dim=cfg.band_feature_dim or 16,
            readout_score_scale=cfg.readout_score_scale,
        )
    elif cfg.head_kind == "masked_attention":
        # Same offsets rationale as "interval"; only the band aggregator
        # differs (exclusion-mask attention, see the head's module docstring).
        head = MaskedAttentionSwapHead(
            backbone,
            pair_offsets=cfg.pair_offsets or (1, cfg.ising.D),
            band_feature_dim=cfg.band_feature_dim or 16,
            attention_dim=cfg.attention_dim or 32,
            use_stencil=cfg.use_stencil,
            lattice_side=cfg.ising.D,
            readout_score_scale=cfg.readout_score_scale,
        )
    elif cfg.head_kind == "factorised":
        head = FactorisedSwapHead(
            backbone,
            bilinear_rank=cfg.bilinear_rank or 8,
            factor_dim=cfg.factor_dim or 32,
            global_feature_dim=cfg.global_feature_dim or 16,
            use_bilinear=cfg.use_bilinear,
            use_global=cfg.use_global,
            site_orderings=cfg.site_orderings,
            lattice_side=cfg.ising.D,
        )
    elif cfg.head_kind == "grouped_anchor":
        # k masked passes instead of mask_one's d; lattice_side is cfg.ising.D
        # so the "diagonal" grouping can disperse across the D x D raster.
        if cfg.n_groups is None:
            raise ValueError("head_kind 'grouped_anchor' requires n_groups")
        head = GroupedAnchorSwapHead(
            backbone,
            n_groups=cfg.n_groups,
            grouping=cfg.grouping,
            lattice_side=cfg.ising.D,
            group_chunk_size=cfg.group_chunk_size,
        )
    else:
        raise ValueError(f"Unknown head_kind: {cfg.head_kind!r}")
    if cfg.compile_head:
        # In-place nn.Module.compile: state_dict keys stay unprefixed
        # (torch.compile(module) wrapping would add `_orig_mod.`).
        head.compile()
    return head


def _hard_cell(
    name: str,
    sigma: float,
    head_kind: str,
    *,
    D: int = 4,
    n_steps: int = 2_000,
    n_euler_steps: int = 100,
    n_eval_samples: int = 512,
    eval_sample_chunk: int | None = None,
    n_eval_samples_training: int | None = None,
    eval_every: int = 200,
    use_sdpa_readout: bool = False,
    eval_autocast_bf16: bool = False,
    use_matching_step: bool = False,
    curriculum: CurriculumCfg | None = None,
    potts_composition: tuple[float, ...] | None = None,
    grad_clip_max_norm: float = 500.0,
    estimator: str = "control_variate",
    optimiser: str = "adamw",
    rewarmup_on_stage: bool = False,
) -> HardStageCfg:
    """Shared shape for the sigma-ladder + control cells: the D=4 gate cells fix
    only sigma and head_kind (all otherwise identical). The keyword knobs open
    the same shape to the non-enumerable scaling rungs (§7 mixing probe): D sets
    the lattice, n_euler_steps must be clip-safe for the one-event step at that D
    (scout: ~2d at d=64), n_eval_samples sizes the IS-ESS eval drawn on the GPU
    job itself (evals-ride-the-gpu-job), and eval_sample_chunk streams that eval
    in slices so the vectorised head's d-anchor-copies batch fits GPU memory.

    `potts_composition` switches the cell to the S-species route: S and the
    manifold both come from that one tuple, so `vocab_size` follows its length
    and the binary `ising.target_composition` is dropped to None (a scalar
    n_plus has no meaning for S > 2, and leaving 0.5 there would be a false
    record). `sigma` is then the POTTS coupling — an Ising run at σ corresponds
    to S=2 Potts at 2σ (targets/potts.py module docstring)."""
    is_potts = potts_composition is not None
    return HardStageCfg(
        name=name,
        ising=IsingCfg(
            D=D, sigma=sigma, bias=0.0,
            target_composition=None if is_potts else 0.5,
            composition_penalty_strength=0.0,
        ),
        train=TrainCfg(
            n_steps=n_steps, batch_size=128, replay_buffer_cycles=8,
            lr=1e-3, seed=42, warmup_steps=500,
            grad_clip_max_norm=grad_clip_max_norm,
            optimiser=optimiser, rewarmup_on_stage=rewarmup_on_stage,
        ),
        ctmc=CTMCCfg(
            n_euler_steps=n_euler_steps, use_matching_step=use_matching_step,
        ),
        eval=EvalCfg(
            eval_every=eval_every, n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
            n_eval_samples_training=n_eval_samples_training,
            eval_autocast_bf16=eval_autocast_bf16,
        ),
        model=ModelCfg(
            kind="letf", hidden_dim=32, n_layers=2, n_heads=4,
            vocab_size=len(potts_composition) if is_potts else 2,
            use_sdpa_readout=use_sdpa_readout,
        ),
        estimator=estimator,
        head_kind=head_kind,
        wandb_project="dnfs-constraints",
        curriculum=curriculum,
        target_kind="potts" if is_potts else "ising",
        potts_composition=potts_composition,
    )


# The proven sigma-plateau ladder for the d=64 sigma_c rung (the baseline
# stage_3 conv-critical recipe, final stage at the hard cells' 0.223; LR
# drops when the near-critical variance spike begins at sigma=0.205).
_D64_SIGMA_LADDER = CurriculumCfg(
    stages=(
        CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
        CurriculumStageCfg(start_step=5_000, sigma=0.140, lr=1e-3),
        CurriculumStageCfg(start_step=10_000, sigma=0.170, lr=1e-3),
        CurriculumStageCfg(start_step=15_000, sigma=0.190, lr=1e-3),
        CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
        CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
        CurriculumStageCfg(start_step=30_000, sigma=0.223, lr=3e-4),
    )
)

# The 12k smoke arms only ever reach the ladder's first three stages, and the
# curriculum validator (correctly) rejects stages starting beyond n_steps, so
# the smokes share this truncated view of the SAME ladder rather than a copy.
_SMOKE12K_SIGMA_LADDER = CurriculumCfg(stages=_D64_SIGMA_LADDER.stages[:3])


def _d64_curriculum_cell(
    name: str, head_kind: str, n_steps: int = 50_000, **head_knobs
) -> HardStageCfg:
    """The d=64 sigma_c curriculum recipe — the shape of the PASSED D=8 rung.
    Every band-capacity-push cell shares it verbatim and differs only in
    head_kind and the declared head knobs, so outcome differences are
    attributable to the declared change (2026-07-08; twin-ness is
    pinned by test_band_push_cells_mirror_ma_twin_except_declared_fields).

    `n_steps` defaults to the 50k budget every batch-1 / round-2 cell used.
    The horizon-extension cells pass 100_000: the sigma
    ladder is a tuple of absolute start_steps, so it does NOT stretch with the
    budget — the extra steps all land on the final sigma=0.223 plateau, which
    is the phase the training logs show still descending at 50k. n_steps is
    also schedule-inert here (lr is the curriculum's per-stage value times a
    fixed-step warmup ramp, never normalised by the total), so a longer cell's
    first 50k steps are schedule-identical to the 50k cell's."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind=head_kind,
        D=8, n_steps=n_steps, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=256, n_eval_samples_training=512,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        curriculum=_D64_SIGMA_LADDER,
    )
    return replace(cell, **head_knobs)


def _d144_curriculum_cell(
    name: str, head_kind: str, n_steps: int = 50_000, **head_knobs
) -> HardStageCfg:
    """The 12x12 (d=144) rung — the volume between the healthy 8x8 rung and
    the 16x16 rung that does not train.

    Why this size exists at all: 8x8 reaches Var[log w]/site 0.0040 and
    16x16 sits at 0.0707, an 18x per-site gap across a single 4x volume
    step, with no rung in between ever run. A working 144 turns two points
    and a failure into a scaling curve; a failing 144 brackets the wall to a
    2.25x volume window. Either outcome is read on Var[log w]/site, not ESS:
    the self-normalised estimator cannot report below 1/N, so once
    Var[log w] passes ~log N the ESS column measures the largest weight in
    the draw rather than the sampler (measured — at 64 sites
    ESS/N = exp(-Var) holds to 1.1%, at 256 sites it is wrong by 2.3e5).

    Three fields differ from the 8x8 rung, each forced by the volume rather
    than chosen:

    `use_matching_step=True` (multi-event). The one-event Euler step clips
    when Lambda*dt > 1, and Lambda is a sum over ~d^2/2 pairs, so it is ~5x
    the 8x8 value here. The 8x8 rung's one-event step would clip on most
    states; this is the step the 16x16 cells already run, so the protocol
    matches the larger rung.

    `n_euler_steps=288` = 2d, the clip-safe budget the shared builder states
    ("~2d at d=64", which the 8x8 rung honours at 128). Every 16x16 cell has
    run 128 = 0.5d, a factor of 4 under that rule, and a resolution sweep on
    a frozen 16x16 checkpoint found Var[log w] still falling monotonically
    at the largest grid tested — the signature of an under-resolved grid.
    Matching the 8x8 rung's STEP COUNT instead would give this rung a
    coarser grid per site than the rung it is compared against, making a
    failure unattributable between volume and discretisation. Separating
    those two is the one thing this cell exists to do.

    `eval_sample_chunk=512`: the factorised head peaks at 0.87 GB at 256
    sites and batch 32, against masked attention's 5.00 GB, and its
    throughput saturates by batch 512. The 64-row chunk every 16x16 cell
    inherited was sized for masked attention and wastes most of the card.
    Chunking is eval-only and cannot move the trained model.

    Everything else — sigma ladder, batch, lr, replay depth, warmup, seed —
    is the 8x8 recipe verbatim, so head and volume stay the declared
    changes. The ladder is reused rather than rescaled because sigma_c in
    this convention is the thermodynamic-limit value (ln(1+sqrt 2)/2) and is
    already shared by the 4x4, 8x8 and 16x16 rungs."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind=head_kind,
        D=12, n_steps=n_steps, n_euler_steps=288, n_eval_samples=5000,
        eval_sample_chunk=512, n_eval_samples_training=512,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
    )
    return replace(cell, **head_knobs)


def _d64_fmo2_h128_cell(name: str) -> HardStageCfg:
    """The 8x8 factorised rung with hidden_dim 32 -> 128 the ONLY change.

    Why capacity is suspected. `hidden_dim=32, n_layers=2, n_heads=4` is set
    once in the shared cell builder and every cell at every volume has used
    it, so it has never appeared in a cell diff. The object the head must
    produce is the d x d pair field, and parameters per pair-field entry
    fall 663 -> 42.7 -> 3.4 across the 16 / 64 / 256-site rungs while total
    capacity grows only 1.4x. Independently, the factorised-head forensics
    measured the REQUIRED effective rank of that field growing roughly d/8
    at sigma_c: the function gets structurally richer with volume while the
    network does not.

    NOT proven binding, and one prior argument for it has been withdrawn.
    The relative-fit statistic `loss / var_estimator_integrand` was read as
    showing the 16x16 model underfitting its own training states; it does
    not survive recomputation — it is dominated by which c_t estimator is
    switched on (two 8x8 runs of near-identical quality read 0.446 with the
    control variate and 0.028 without), and the 16x16 value quoted came from
    the run that was killed for divergence rather than the health-clean one.
    The rank argument above stands on its own and is untouched by that.

    Why 128, and why at 8x8 first. Rank headroom at the healthy rung is
    32/(64/8) = 4x; holding that ratio at 256 sites needs hidden 128.
    Capacity cannot be bundled with a cross-volume warm start in one step,
    because the transfer resamples positional tables across volume but every
    weight matrix changes shape when hidden_dim moves. So this cell runs at
    the volume whose answer is already known and serves twice: as the
    capacity NULL (4x headroom is already present at 8x8, so neutral is
    expected and is the informative outcome — a gain HERE would mean the
    rank argument mis-locates the constraint), and as the transfer source
    for a matched-capacity larger rung.

    `n_heads` stays 4, so head_dim rides 8 -> 32 as a consequence of
    widening rather than as a second knob; `n_layers` stays 2. Depth is a
    separate cell and is deliberately not bundled."""
    cell = replace(
        _d64_curriculum_cell(name, head_kind="factorised"),
        ema_decay=0.9999,
        site_orderings=("row", "col"),
    )
    return replace(cell, model=replace(cell.model, hidden_dim=128))


def _d64_fmo2_loop_cell(
    name: str,
    *,
    n_euler_steps: int = 128,
    use_matching_step: bool = False,
    **train_overrides,
) -> HardStageCfg:
    """The archived 8x8 factorised rung, opened on its outer/inner-loop knobs.

    Every archived run at every size has carried the same outer/inner loop
    shape — 100 gradient steps per outer cycle, rollout width tied to the
    gradient batch, 8 retained buffer cycles — inherited from the shared
    builders and never varied in a cell diff. The 8x8 battery built on this
    helper measures each loop knob as a single declared variable at the size
    where training is known-healthy, on the factorised head family the
    larger rungs actually use.

    With no overrides this reproduces `H2_d64_c50_s223_letf_fmo2_50k_curr`
    exactly except the name (pinned in tests/test_screen_pins.py), so every
    battery arm is a true twin: factorised head with the dual row+col causal
    orderings, EMA shadow riding as pure instrumentation, control-variate
    estimator, batch 128, 128-step Euler grid, one-event stepping, the
    sigma-plateau ladder to 0.223, 50k steps, seed 42, 5000-draw frozen
    eval. Comparator (from that twin's archived run): raw eval ESS/N 0.7452
    / EMA 0.8104. Band convention shared by every arm, frozen before
    launch: INSENSITIVE = within +-0.03 EMA ESS/N of 0.8104, the noise
    scale borrowed from the d64 masked-attention cross-seed spread (~0.026;
    the fmo2 comparator is single-seed).

    `n_euler_steps` / `use_matching_step` open the trajectory-simulation
    knobs (CTMCCfg); `train_overrides` opens the TrainCfg loop knobs
    (inner_steps_per_outer, replay_buffer_cycles, outer_batch_size,
    c_t_batch) and the boundary knobs the 2026-08-19 arms declare
    (rewarmup_on_stage, stage_best_checkpoints, flush_replay_on_stage) —
    in fact anything TrainCfg carries, since the overrides go straight
    into `dataclasses.replace`. Defaults are byte-identical to the
    archived rung.

    Second band convention, for arms whose read sits at the ESS ceiling:
    the PRIMARY statistic is then the EMA eval Var[log w], rung reference
    0.2068 with bootstrap 95% CI (0.1983, 0.2152) — the population
    (ddof=0) variance of the frozen 5000-draw EMA eval log-weights with a
    2000-resample percentile bootstrap — UPLIFT means a CI disjoint
    BELOW that interval, REGRESSION a CI disjoint above, and the +-0.03
    EMA ESS/N band above becomes the endpoint read alongside rather than
    the verdict."""
    cell = replace(
        _d64_curriculum_cell(name, head_kind="factorised"),
        ema_decay=0.9999,
        site_orderings=("row", "col"),
    )
    cell = replace(
        cell,
        ctmc=replace(
            cell.ctmc,
            n_euler_steps=n_euler_steps,
            use_matching_step=use_matching_step,
        ),
    )
    if train_overrides:
        cell = replace(cell, train=replace(cell.train, **train_overrides))
    return cell


def _d256_fmo2_warm_cell(
    name: str, n_euler_steps: int = 128
) -> HardStageCfg:
    """16x16 factorised rung continued from a trained 8x8 factorised model
    (`--init-from` a cross-volume transfer built by the warm-start script,
    which resamples the positional tables bicubically on the torus and keeps
    the live readout).

    The warm-start arm that was never actually run. The archived phase-2
    continuation restarted a 16x16 model from a 16x16 checkpoint, so it
    tested an estimator switch and not transfer; and the head that makes a
    16x16 run affordable at all — 18.6 ms / 0.87 GB against masked
    attention's 87.8 ms / 5.00 GB at this volume — has never been trained
    here. Both gaps close in one cell.

    No curriculum, flat sigma_c, lr pinned to the ladder's final 3e-4: the
    source model has already climbed the sigma ladder at 8x8, so the
    optimiser regime continues rather than restarts. That is the archived
    continuation cell's shape exactly, which keeps the schedule from
    becoming a second variable.

    `n_euler_steps` is the arm variable, and the two registered values are
    the point of the pair:

    - **128** is what every archived 16x16 cell has run. It is also 0.5d,
      a factor of 4 under the clip-safe budget this module's own builder
      states ("~2d at d=64", which the 8x8 rung honours at 128). It is the
      CONTROL, kept so the transfer read stays comparable to the archive.
    - **512** is 2d, the rule's own value at this volume. A sweep across
      grids on a frozen 16x16 checkpoint found Var[log w] falling
      monotonically all the way to the largest grid tested — the signature
      of a grid that is still too coarse — but that sweep can only ask how
      an ALREADY-TRAINED model behaves when re-rolled finer. It cannot
      separate "the grid is coarse" from "the model was fitted to a coarse
      grid", because a model trained under a biased discretisation learns to
      compensate that bias. Training at the finer grid is the only test of
      the second reading, and it is the live hope of the pair.

    Run as two arms one variable apart rather than as a single choice: if
    only the fine arm ran and improved, the gain would not be separable from
    the transfer itself, and if only the coarse arm ran the resolution
    question would stay where the sweep left it."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind="factorised",
        D=16, n_steps=20_000, n_euler_steps=n_euler_steps,
        n_eval_samples=5000,
        eval_sample_chunk=512, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
    )
    return replace(
        cell, ema_decay=0.9999, site_orderings=("row", "col"),
        train=replace(cell.train, lr=3e-4),
    )


def _d64_smoke12k_replay2_cell(name: str) -> HardStageCfg:
    """M6 (plan Task 6): the MA curriculum recipe at the 12k smoke horizon,
    replay_buffer_cycles 8 -> 2 the ONLY declared change versus the archived
    d64 MA twin recipe (the ladder truncation to the shared 12k view is
    forced by the validator, as on every smoke arm)."""
    cell = _d64_curriculum_cell(name, "masked_attention", n_steps=12_000)
    return replace(
        cell,
        train=replace(cell.train, replay_buffer_cycles=2),
        curriculum=_SMOKE12K_SIGMA_LADDER,
    )


def _d64_m2_ctema4_cell(name: str) -> HardStageCfg:
    """M2 (plan Task 2): the archived d64 MA 50k curriculum recipe verbatim
    with c_t_ema_halflife_cycles 0.0 -> 4.0 the ONLY declared change
    (twin-ness pinned by
    test_m2_ctema4_gate_mirrors_ma_twin_except_declared_fields). The
    dual-eval EMA instrument is deliberately NOT ridden — the no-regression
    gate is judged raw-vs-archived-twin, so the twin stays pure."""
    cell = _d64_curriculum_cell(name, "masked_attention")
    return replace(
        cell, train=replace(cell.train, c_t_ema_halflife_cycles=4.0)
    )


def _d64_naive_twin_cell(name: str) -> HardStageCfg:
    """c_t transfer-function cell (2026-08-14): the archived MA 50k
    curriculum twin with estimator control_variate -> naive_mc the ONLY
    declared change.

    Why it exists: every variance lever in flight (the Stein CV, the c_t
    EMA, the enlarged c_t rollout) reduces c_t NOISE, but the quantity
    that has to fall ~8x for a usable d256 is per-site Var[log w]. The
    slope between them has never been measured — all 21 archived d64
    cells run control_variate, so there is no naive arm at any healthy
    size in either direction. This cell measures it where the control
    variate is known to WORK (a healthy run), rather than where it
    inverted (the diverged d256), which is the confound that makes the
    d256 naive-vs-CV comparison uninterpretable.

    Predictions are pre-registered in
    test_ctv_naive_twin_mirrors_ma_twin_except_the_estimator; the naive
    integrand variance 26.0 is already logged as the CV runs' own
    Var[dt log p tilde] column, so they are arithmetic, not guesses.
    A fourth outcome is live: if training destabilises, the CV is a
    STABILITY crutch and not only a variance reducer."""
    cell = _d64_curriculum_cell(name, "masked_attention")
    return replace(cell, estimator="naive_mc")


def _d64_smoke12k_ctb512_cell(name: str) -> HardStageCfg:
    """M3 plumbing fallback (plan Task 3): the MA curriculum recipe at the
    12k smoke horizon with c_t_batch=512 the ONLY mechanism change — the
    25-min a30 validation of the enlarged-rollout plumbing (buffer prefix,
    c_t over the full set, resume carriage) if a100 is blocked for the
    d256 mechanism cell."""
    cell = _d64_curriculum_cell(name, "masked_attention", n_steps=12_000)
    return replace(
        cell,
        train=replace(cell.train, c_t_batch=512),
        curriculum=_SMOKE12K_SIGMA_LADDER,
    )


def _d256_smoke12k_naive_ctb512_cell(name: str) -> HardStageCfg:
    """M3 mechanism cell (plan Task 3): the smoke12k Arm-B naive recipe
    verbatim with c_t_batch=512 the ONLY declared change (twin-ness pinned
    by test_m3_ctb512_smoke_mirrors_naive_arm_except_declared_fields; the
    arg-for-arg copy of the naive arm is guarded by that pin)."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        estimator="naive_mc",
    )
    return replace(cell, train=replace(cell.train, c_t_batch=512))


def _d256_cv2_cell(name: str) -> HardStageCfg:
    """Phase-2 twin of the d=256 naive rescue: same shape, estimator back to
    the control variate, NO curriculum (training continues from the naive
    run's final checkpoint via --init-from, so the ladder has already been
    climbed and sigma holds flat at sigma_c), lr pinned to the ladder's
    final 3e-4 so the optimiser regime continues rather than restarts, and
    the dual-eval EMA instrument riding as on every new hard run."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind="masked_attention",
        D=16, n_steps=20_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
    )
    return replace(
        cell, ema_decay=0.9999, train=replace(cell.train, lr=3e-4)
    )


def _d256_scr5k_cell(
    name: str, head_kind: str, *, eval_sample_chunk: int,
    n_euler_steps: int = 128,
) -> HardStageCfg:
    """One arm of the 16x16 stage-1 screen: flat sigma=0.10, 5,000 steps,
    naive c_t, the rescue recipe's shape otherwise verbatim.

    Why a screen at sigma=0.10 exists. The archived 16x16 failure does not
    appear at criticality — it is fully expressed in the first 5,000 steps
    of the rescue run, at the ladder's easiest rung: stage-1-tail
    loss/var_dt_log_p_tilde (FVU) reads 0.119 against the 8x8 naive twin's
    0.033 at the matched stage, and the 256-draw train eval reads ESS/N
    0.041 against 0.85. A flat-sigma 5k cell therefore reproduces the
    failure for ~1/10 the cost of a ladder run, and — because the h128 8x8
    arm lost ground specifically ACROSS sigma transitions under lr 1e-3
    while matching its twin at fixed sigma — it also removes the
    lr-x-moving-target interaction that made that capacity null
    unattributable. One variable per arm; every arm reads against the
    scr5k base of its own head family.

    Pre-registered read, frozen before any arm ran: stage-tail FVU = mean
    loss / mean var_dt_log_p_tilde over steps 3,000-4,999 (both columns in
    training_log.csv; subtract the naive c_t noise floor 1/128 = 0.0078
    for cross-estimator comparisons). Bands against the MA base's expected
    0.119 (the archived rescue's own first 5k, seed 42, byte-comparable
    recipe): LIVE if <= 0.08 (>1.5x, outside the +-0.005 within-run
    window noise); PARITY if <= 0.04 (the 8x8 naive yardstick); NULL if
    >= 0.10. Corroboration: final 1000-draw eval ESS/N (base expectation
    ~0.04; floor 1/1000 is far below any band). Scope stated honestly: a
    NULL kills an axis for the stage-1 deficit — the bulk of the failure
    (3.6x of the final 5.1x) — not necessarily for the sigma_c increment
    on top; a LIVE fires everywhere it fires.

    Fixed fields and why: naive_mc in EVERY arm (the estimator of the
    failure under investigation, and the c_t noise floor then cancels in
    same-estimator comparisons); n_eval_samples=1000 not 5000 (screen
    precision: SE(Var[log w]) ~ Var*sqrt(2/999) ~ 0.8 at the current 18,
    ample against bands 2x apart); n_euler=128 and batch 128 as archived
    (the ne512 arm varies both together — n_grid also sets c_t slots,
    buffer size and the loss's t-support, and inner_batch/n_grid is the
    per-slot gradient density, so the pair moves as one to hold density at
    1.0); no curriculum (sigma constant means no buffer clears, no EMA
    resets, no lr steps)."""
    return _hard_cell(
        name, sigma=0.10, head_kind=head_kind,
        D=16, n_steps=5_000, n_euler_steps=n_euler_steps,
        n_eval_samples=1000,
        eval_sample_chunk=eval_sample_chunk, n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        estimator="naive_mc",
    )


def _scr5k_fmo2_cell(name: str) -> HardStageCfg:
    """Factorised-multi-order screen arm: the scr5k shape with the fmo2
    family's own conventions riding (EMA shadow and the two causal
    orderings), and the eval chunk sized to the head that runs it — fmo2
    peaks 0.87 GB at batch 32 where masked attention peaks 5.00 GB, and its
    measured throughput saturates by batch 512, so the 64-row chunk every
    archived 16x16 cell inherited wastes most of an 80 GB card. Chunking is
    eval-only and cannot move the trained model."""
    cell = _d256_scr5k_cell(name, "factorised", eval_sample_chunk=512)
    return replace(cell, ema_decay=0.9999, site_orderings=("row", "col"))


def _scr5k_fmo2_with(
    name: str, *, n_euler_steps: int = 128, **overrides
) -> HardStageCfg:
    """One fmo2 screen arm = the fmo2 base with the named overrides the
    ONLY changes (twin-ness pinned in tests). Overrides route to the model
    (hidden_dim, n_layers) or training (lr, grad_clip_max_norm, batch_size)
    dataclass by field name; anything else is a typo and must fail loudly
    rather than silently produce an arm that is not the declared twin."""
    model_field_names = {"hidden_dim", "n_layers"}
    train_field_names = {
        "lr", "grad_clip_max_norm", "batch_size", "loss_microbatch_size",
        "c_t_batch",
    }
    unknown = set(overrides) - model_field_names - train_field_names
    if unknown:
        raise ValueError(f"unknown screen-arm override(s): {sorted(unknown)}")
    cell = _scr5k_fmo2_cell(name)
    model_overrides = {
        k: v for k, v in overrides.items() if k in model_field_names
    }
    train_overrides = {
        k: v for k, v in overrides.items() if k in train_field_names
    }
    if n_euler_steps != 128:
        cell = replace(cell, ctmc=replace(cell.ctmc, n_euler_steps=n_euler_steps))
    if model_overrides:
        cell = replace(cell, model=replace(cell.model, **model_overrides))
    if train_overrides:
        cell = replace(cell, train=replace(cell.train, **train_overrides))
    return cell


def _scr5k_ma_h128_lr03_cell(name: str) -> HardStageCfg:
    """The masked-attention bridge capacity arm: hidden 32 -> 128 WITH lr
    1e-3 -> 3e-4 co-varied, deliberately bundled. The only capacity
    variation ever run (8x8 fmo2 h128) came back negative with the fit
    statistic worsening on a superset function class — which optimisation,
    not expressivity, explains — and its per-stage trace shows the damage
    accruing under lr 1e-3. A frozen-lr h128 arm here would be positioned
    to reproduce that false negative, so the bridge arm spends its one
    slot on the (h128, lr03) corner; the fmo2 family carries the full
    2x2 (base / h128 / lr03 / h128+lr03) that de-confounds the pair.

    Memory schedule (2026-08-16): the single-backward smoke OOMed an
    A100-80GB by a whisker (77.0 GiB in use, 4.0 GiB further requested,
    backward pass) at this cell's exact frame (batch 128, d=256, hidden
    128), where the archived h32 twin trains inside the same card. Moving
    batch or precision would un-twin the arm, so the fit lever is
    loss_microbatch_size=64: the inner step's one backward runs as two
    64-row slices, which halves the retained graph (linear in rows —
    ample against a 4 GiB shortfall) while accumulating the IDENTICAL
    update at the full batch 128 — a gradient-exact memory schedule, not
    a recipe variable (loss_swap_backward_microbatched's docstring has
    the algebra; tests/test_loss_microbatch_parity.py pins it). The
    measured record otherwise stands: single-backward capacity work at
    16x16 fits only the factorised head (fmo2: 0.87 GB at batch 32 where
    masked attention reads 5.00 GB, the dominant (B, heads, d, 2d) score
    tensor being hidden-independent)."""
    cell = _d256_scr5k_cell(name, "masked_attention", eval_sample_chunk=128)
    return replace(
        cell,
        model=replace(cell.model, hidden_dim=128),
        train=replace(cell.train, lr=3e-4, loss_microbatch_size=64),
    )


def _scr5k_ma_clip60k_cell(name: str) -> HardStageCfg:
    """The MA screen base with the grad-clip threshold rescaled to d=256
    gradient units — the only change, so the read is chargeable to clip
    semantics alone. The threshold derivation, the pre-launch refutation
    of the pair-count heuristic, and the frozen verdict bands live at the
    registry entry, where the launch decision was made."""
    cell = _d256_scr5k_cell(name, "masked_attention", eval_sample_chunk=128)
    return replace(
        cell, train=replace(cell.train, grad_clip_max_norm=60_000.0)
    )


def _scr5k_mo_cell(name: str) -> HardStageCfg:
    """The mask_one screen arm: the best 8x8 head (0.00129 Var/site at
    100k, 2.2x under the MA twin), never before run at 16x16 — the A6
    question's d256 read.

    Memory schedule (2026-08-16): the single-backward smoke OOMed an
    A100-80GB (75.9 GiB in use, 73.4 GiB torch-allocated) at this frame
    (batch 128, d=256) — the head's d stacked anchor passes retain ~d
    trunk graphs per row for the backward, so its training memory scales
    with the lattice in a way neither other family's does. The fit lever
    is loss_microbatch_size=16: eight 16-row backward slices bound the
    retained graph to ~1/8 of the measured 73.4 GiB (~9 GiB, linear in
    rows) while accumulating the IDENTICAL update at the full batch 128 —
    gradient-exact, so not a recipe variable (algebra in
    loss_swap_backward_microbatched; pinned by
    tests/test_loss_microbatch_parity.py). Each slice still stacks all
    256 anchors (4096 trunk rows per forward), so per-pass GPU
    utilisation stays dense and total FLOPs are unchanged. This
    supersedes the briefly-registered batch-32 mini-family route, which
    would have moved the batch — a second variable — where slicing moves
    none."""
    cell = _d256_scr5k_cell(name, "mask_one", eval_sample_chunk=64)
    return replace(cell, train=replace(cell.train, loss_microbatch_size=16))


def _d256_fmo2_ladder_cell(
    name: str, *, estimator: str, n_euler_steps: int = 128,
    batch_size: int = 128, loss_microbatch_size: int | None = None,
) -> HardStageCfg:
    """The 16x16 fmo2 sigma-ladder shape: the archived MA naive-rescue
    recipe with the factorised family's conventions riding (EMA shadow,
    dual site orderings, the head's own eval chunk). The bare call builds
    the anchor cell; the batch/grid/microbatch knobs build the composed
    recipe cells (their licences and frozen bands live at the registry
    entries, where the launch decision is made)."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind="factorised",
        D=16, n_steps=50_000, n_euler_steps=n_euler_steps,
        n_eval_samples=5000, eval_sample_chunk=512,
        n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        estimator=estimator,
    )
    return replace(
        cell,
        ema_decay=0.9999,
        site_orderings=("row", "col"),
        train=replace(
            cell.train,
            batch_size=batch_size,
            loss_microbatch_size=loss_microbatch_size,
        ),
    )


def _d144_ma_bracket_cell(name: str) -> HardStageCfg:
    """The 12x12 volume bracket of the ARCHIVED failure: the 16x16 naive
    rescue recipe with the lattice side the only mechanism change.

    This supersedes the earlier fmo2 12x12 rung as the wall-bracketing
    cell, which sat three variables from the failure it was meant to
    bracket: factorised head (the failing archive is masked attention),
    control-variate c_t (the from-scratch 16x16 launch diverged under the
    CV and was rescued by naive), and n_euler=288 (every archived 16x16
    number is 128, and 288 also moves per-slot gradient density
    128/288 = 0.44 — the same slots/buffer/t-support coupling that made
    the cancelled ne512 transfer arm unattributable to resolution). A
    12x12 result from that cell could not have been charged to volume.
    That cell stays registered for the fmo2 family's own ladder later;
    THIS cell brackets the wall.

    The read is a single-estimator, single-head, single-grid scaling curve:
    8x8 naive 0.00472 Var[log w]/site (archived naive twin) -> 12x12 (this
    cell) -> 16x16 naive 0.0707 (archived rescue). Read on Var/site, never
    ESS: at Var ~ 18 the self-normalised estimator is floor-bound at 1/N.
    En route it self-serves the stage-1 screen read at zero extra cost —
    the ladder's first 5,000 steps ARE flat sigma=0.10, so stage-tail FVU
    at 12x12 lands ~day one and places this rung on the healthy (0.033) or
    failing (0.119) side before the ladder finishes.

    Eval sizing is the only departure, both fields eval-only: chunk 128
    (masked attention's (B, heads, d, 2d) score buffer at d=144 is ~1/3
    the d=256 row cost, so 128 rows sit well inside the card that ran
    chunk 64 at d=256) and 512-draw in-training evals (the 8x8 convention;
    a 256-draw eval cannot read below 1/256 and this rung is expected to
    land between 0.85 and 0.01)."""
    return _hard_cell(
        name, sigma=0.223, head_kind="masked_attention",
        D=12, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=128, n_eval_samples_training=512, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        estimator="naive_mc",
    )


def _d256_clip2000_cont_cell(name: str) -> HardStageCfg:
    """Clip-threshold continuation: the completed naive rescue continued
    (via --init-from its final checkpoint) for 10k further steps at flat
    sigma_c with grad_clip_max_norm 500 -> 2000 the only mechanism change.

    Why a continuation and not a cold twin. The clip fires on ~25% of the
    rescue's final-plateau steps (final-stage p50 grad norm 384, p99
    ~1050), removing ~9-10% of gradient magnitude — a permanent brake
    exactly where this arm operates. A cold unclipped 16x16 already exists
    and failed: the clip=20000 smoke arm never escaped rung 0 (85% of its
    final-rung steps at/above even that ceiling, final loss 328 vs the
    naive arm's 3.16) — confounded with the then-broken CV recipe, but it
    prices the early-phase divergence risk (the rescue's own early grad
    max was 6.1e6) that a continuation never takes. At threshold 2000 the
    final-plateau tail sits almost entirely unclipped (p99 ~1050), so this
    isolates the brake's steady-state cost; the early-phase question is
    the flat-sigma screen's clip arm.

    Read against the rescue's own final plateau, which is flat (FVU
    0.137 +- 0.005 across deciles; eval Var/site 0.0707 at N=5000): the
    null band is that flatness continuing. lr pinned to the ladder's
    final 3e-4 and estimator kept naive so the optimiser regime continues
    rather than restarts — the archived cv2 continuation changed the
    estimator at this same juncture, which is why it cannot serve as this
    arm's control. Eval chunk 64 -> 128, eval-only."""
    cell = _hard_cell(
        name, sigma=0.223, head_kind="masked_attention",
        D=16, n_steps=10_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=128, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        estimator="naive_mc",
        grad_clip_max_norm=2_000.0,
    )
    return replace(
        cell, ema_decay=0.9999, train=replace(cell.train, lr=3e-4)
    )


CONFIGS: dict[str, HardStageCfg] = {
    # First Potts cell (2026-07-31):
    # the training path on S=3, kept small enough to smoke end-to-end on CPU.
    # D=3 (d=9) is the smallest lattice that is BOTH non-degenerate (on the L=2
    # torus a site's two neighbours coincide) and divisible by 3, so the equal
    # three-way slice exists exactly at 3 sites per species.
    # sigma = 0.5025 is the 3-state Potts critical coupling: beta_c = ln(1+sqrt 3)
    # = 1.00505 per BOND, halved because our A double-counts each edge — the
    # same convention that puts Ising's sigma_c at ln(1+sqrt 2)/2 = 0.223.
    # A 3x3 lattice has no phase transition to sit at; the value is chosen so
    # the cell is the right shape to grow into the hardness-ladder rung rather
    # than needing a re-pick later. doubly_hollow because at d=9 the O(d^2)
    # correctness gate is free, and nothing has ever been trained on Potts.
    "H3_d9_c33_s503_letf_dh": _hard_cell(
        "H3_d9_c33_s503_letf_dh", sigma=0.5025, head_kind="doubly_hollow",
        D=3, potts_composition=(1 / 3, 1 / 3, 1 / 3),
    ),
    "H2_d16_c50_s010_letf_dh": _hard_cell(
        "H2_d16_c50_s010_letf_dh", sigma=0.10, head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s223_letf_dh": _hard_cell(
        "H2_d16_c50_s223_letf_dh", sigma=0.223, head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s040_letf_dh": _hard_cell(
        "H2_d16_c50_s040_letf_dh", sigma=0.40, head_kind="doubly_hollow",
    ),
    # Antisymmetry negative control: same sigma as the floor rung,
    # NonAntisymSwapHead instead of DoublyHollowSwapHead, same loss.
    "H2_d16_c50_s010_letf_na": _hard_cell(
        "H2_d16_c50_s010_letf_na", sigma=0.10, head_kind="non_antisym",
    ),
    # 4x4 supervisor-demo cells (2026-07-08): 10k-step MA/MO twins of the 2k
    # dh ladder at the floor and critical rungs. Only head_kind and n_steps
    # differ from the corresponding _dh cells (pinned by
    # test_demo_4x4_cells_mirror_dh_ladder_except_declared_fields); no
    # curriculum -- the 2k sigma_c dh cell already passed the gate cold.
    "H2_d16_c50_s010_letf_ma_10k": _hard_cell(
        "H2_d16_c50_s010_letf_ma_10k", sigma=0.10,
        head_kind="masked_attention", n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_ma_10k": _hard_cell(
        "H2_d16_c50_s223_letf_ma_10k", sigma=0.223,
        head_kind="masked_attention", n_steps=10_000,
    ),
    "H2_d16_c50_s010_letf_mo_10k": _hard_cell(
        "H2_d16_c50_s010_letf_mo_10k", sigma=0.10,
        head_kind="mask_one", n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_mo_10k": _hard_cell(
        "H2_d16_c50_s223_letf_mo_10k", sigma=0.223,
        head_kind="mask_one", n_steps=10_000,
    ),
    # Factorised-head 4x4 gate cells (2026-08-13): 10k twins of the MA demo
    # cells above -- only head_kind and the declared factorised knobs differ
    # (pinned by test_factorised_gate_cells_mirror_ma_twin_except_declared_
    # fields), so head effects stay attributable. Four arms: fab8 / fab16 =
    # bilinear+global at rank 8 / 16 (rank-sensitivity read), fbil =
    # bilinear-only (no interval-interior coverage), fglo = global-only (no
    # deep exterior).
    "H2_d16_c50_s010_letf_fab8_10k": _hard_cell(
        "H2_d16_c50_s010_letf_fab8_10k", sigma=0.10,
        head_kind="factorised", n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_fab8_10k": _hard_cell(
        "H2_d16_c50_s223_letf_fab8_10k", sigma=0.223,
        head_kind="factorised", n_steps=10_000,
    ),
    "H2_d16_c50_s010_letf_fab16_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fab16_10k", sigma=0.10,
            head_kind="factorised", n_steps=10_000,
        ),
        bilinear_rank=16,
    ),
    "H2_d16_c50_s223_letf_fab16_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fab16_10k", sigma=0.223,
            head_kind="factorised", n_steps=10_000,
        ),
        bilinear_rank=16,
    ),
    "H2_d16_c50_s010_letf_fbil_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fbil_10k", sigma=0.10,
            head_kind="factorised", n_steps=10_000,
        ),
        use_global=False,
    ),
    "H2_d16_c50_s223_letf_fbil_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fbil_10k", sigma=0.223,
            head_kind="factorised", n_steps=10_000,
        ),
        use_global=False,
    ),
    "H2_d16_c50_s010_letf_fglo_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fglo_10k", sigma=0.10,
            head_kind="factorised", n_steps=10_000,
        ),
        use_bilinear=False,
    ),
    "H2_d16_c50_s223_letf_fglo_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fglo_10k", sigma=0.223,
            head_kind="factorised", n_steps=10_000,
        ),
        use_bilinear=False,
    ),
    # Matched-param arm (fab8 PARTIAL protocol): factor_dim 40 raises the
    # head-owned count to 36,384 ~ the MA head's measured 34,272
    # readout-work params (21,440 head-owned + the 12,832-param backbone
    # attention_readout that only MA uses; the factorised head replaces
    # it). Rules out head-parameter deficit as the PARTIAL's cause.
    "H2_d16_c50_s010_letf_fmp40_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fmp40_10k", sigma=0.10,
            head_kind="factorised", n_steps=10_000,
        ),
        factor_dim=40,
    ),
    "H2_d16_c50_s223_letf_fmp40_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fmp40_10k", sigma=0.223,
            head_kind="factorised", n_steps=10_000,
        ),
        factor_dim=40,
    ),
    # A-prime gate arm (2026-08-14, user GO): fab8 + the column-major causal
    # stream (site_orderings=("row","col")) -- the interior-coverage repair,
    # adding INFORMATION where the rank/width arms only added capacity. The
    # 4x4 cells gate the d64 spend; bands FROZEN BEFORE LAUNCH: NO-REGRESSION
    # (>= fab8's 0.911 at s223 on >= 2/3 seeds) plus clean floor licenses the
    # d64 arm on the mechanism case; >= 0.94 on >= 2/3 seeds (the original
    # PASS bar, closing >= half the 0.057 gap) upgrades it to strong support;
    # < fab8's band means the extra stream hurts and the d64 arm is off.
    # 4x4 interiors are <= 14 sites, so a small gain here is expected even
    # if the mechanism is right; the bands are ordered accordingly.
    "H2_d16_c50_s010_letf_fmo2_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fmo2_10k", sigma=0.10,
            head_kind="factorised", n_steps=10_000,
        ),
        site_orderings=("row", "col"),
    ),
    "H2_d16_c50_s223_letf_fmo2_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fmo2_10k", sigma=0.223,
            head_kind="factorised", n_steps=10_000,
        ),
        site_orderings=("row", "col"),
    ),
    # First non-enumerable scaling rung for the §7 mixing probe: D=8 (d=64) at
    # sigma_c. mask_one head (O(d), bit-exact == doubly_hollow) since correctness
    # here rides the probe's reference chain, not exact enumeration. One-event
    # step sized clip-safe (scout: n~116 at d=64; 128 gives margin, verified via
    # lambda_dt_clipped_frac). 5000-sample eval on the GPU job. This is a
    # VALIDATION run: confirms b-scaling + matching-step fidelity before D=16.
    # Budget probe (2026-07-06): the 2000-step cell trained healthily but was
    # cut off mid-descent (loss 22->14 over the last 250 steps; final ESS
    # 0.0005-0.001 vs the D=4 sigma_c reference 0.68-0.80). One seed at 12.5x
    # the budget, cold at sigma_c, isolates the budget variable before
    # committing to the 3-seed curriculum protocol.
    "H2_d64_c50_s223_letf_mo_25k": _hard_cell(
        "H2_d64_c50_s223_letf_mo_25k", sigma=0.223, head_kind="mask_one",
        D=8, n_steps=25_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=256, n_eval_samples_training=512,
        use_sdpa_readout=True, eval_autocast_bf16=True,
    ),
    # Curriculum + budget rung (2026-07-06 eve): the 25k cold probe lifted the
    # final ESS frac 0.001 -> 0.12 with the loss STILL descending, so budget
    # dominates the collapse but had not saturated. This cell adds the other
    # pocketed lever: the sigma-plateau ladder the unconstrained baseline
    # needed at sigma_c (stage_3 conv-critical recipe, itself a 50k-step run:
    # 5k plateaus, LR drop when the near-critical variance spike begins at
    # sigma=0.205, 40% of the budget on the final plateau). Final sigma is the
    # hard cells' 0.223, matching the D=4 reference chain and the 25k probe.
    "H2_d64_c50_s223_letf_mo_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_mo_50k_curr", head_kind="mask_one",
    ),
    # Masked-attention twin of the PASSED 50k curriculum rung: every knob
    # identical, ONLY head_kind differs. Serves three purposes at once
    # (2026-07-07 head-switch follow-up): (i) true end-to-end wall-clock A/B
    # vs the 6.8 h mask_one record (head-level bench says 5.4x on head fwd;
    # the workload is eval-dominated, so the run measures what that buys);
    # (ii) re-validates the new head at a non-enumerable size before D=16;
    # (iii) retrains the 8x8 mixing-probe trend point so the probe's
    # network-pass currency is single-architecture across sizes.
    "H2_d64_c50_s223_letf_ma_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr", head_kind="masked_attention",
    ),
    # Factorised-head d64 scaling rung (2026-08-13, user GO): does the 4x4
    # verdict transfer — the ~0.057 expressivity price AND the one-pass
    # speed win — at the first non-enumerable size? Single-variable twin of
    # the archived MA curriculum rung above: only head_kind and the
    # declared ema_decay instrument differ (EMA never touches training, so
    # the trajectory comparison vs the MA twin stays single-variable on
    # the raw eval). Seed 42 (the d64 ladder's standing seed).
    "H2_d64_c50_s223_letf_fab8_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_fab8_50k_curr", head_kind="factorised",
        ),
        ema_decay=0.9999,
    ),
    # Rank-at-scale arm (2026-08-14, user GO): rank-16 twin of the fab8 d64
    # rung. At 4x4 the rank axis was refuted NEAR CEILING (0.057 deficit,
    # nothing for rank to buy); at the fab8 rung's measured 0.27 deficit it
    # has something to buy, and the trained MA field's spectrum leaves
    # measurable structure between rank 8 and 16 at this size. Bands FROZEN
    # BEFORE LAUNCH: raw ess_frac >= 0.60 = rank meaningfully binds (closes
    # >= a third of the 0.27 gap to the MA twin's 0.781); <= 0.55 = rank
    # refuted at scale too, interior coverage becomes the only live repair.
    # VERDICT (2026-08-19, recovered by the orphaned-results audit — the run
    # completed 2026-08-14 as DoC job 274188 and sat unadjudicated): raw
    # 0.5615, bootstrap 95% CI (0.5333, 0.5871), EMA 0.5920. BETWEEN-BANDS,
    # binds-EXCLUDED (CI upper < 0.60); refuted band straddled. Rank 16 buys
    # +0.05 raw over fab8 (~19% of the 0.27 gap), under the meaningful bar.
        "H2_d64_c50_s223_letf_fab16_50k_curr": replace(
            _d64_curriculum_cell(
                "H2_d64_c50_s223_letf_fab16_50k_curr", head_kind="factorised",
            ),
            ema_decay=0.9999,
            bilinear_rank=16,
        ),
        # A-prime at scale (2026-08-14, user GO — LOGGED AMENDMENT): fab8 d64
        # rung + the column-major causal stream (site_orderings=("row","col")),
        # seed 42, single-variable twin of the fab8 rung (pinned by
        # test_fmo2_d64_rung_mirrors_fab8_rung_except_orderings). The 4x4
        # no-regression band FIRED (0.883/0.938/0.879 vs the 0.911 bar) — the
        # amendment declares that read INCONCLUSIVE for size-dependent
        # interior mechanisms rather than refuting: 4x4 interiors are <= 14
        # sites (the gate was set at no-regression precisely because it is an
        # insensitive read there), the motivating forensic (field correlation
        # decaying with pair gap) was measured at d64, and the mean is -0.012
        # with one seed at the 0.938 strong bar. Decided BEFORE any d64 A-prime
        # data exists. Bands FROZEN at the amendment: MEANINGFUL raw >= 0.60
        # (closes >= 1/3 of the 0.27 gap to MA 0.781 — the same bar fab16
        # faced); STRONG raw >= 0.70 (the assessment's "good enough for run
        # D"); NEGATIVE raw <= 0.55 -> A-prime refuted at scale too and R1
        # (fint) is the only live interior repair.
        "H2_d64_c50_s223_letf_fmo2_50k_curr": replace(
            _d64_curriculum_cell(
                "H2_d64_c50_s223_letf_fmo2_50k_curr", head_kind="factorised",
            ),
            ema_decay=0.9999,
            site_orderings=("row", "col"),
        ),
        # Scaling slate (2026-08-15). The fmo2 rung above cleared its band at
        # 8x8 (raw 0.745 / EMA 0.810, per-site variance BELOW the masked-
        # attention twin) and the cost bench priced it at 4.7x faster and
        # 5.7x smaller than that twin at 16x16, so the head is no longer what
        # limits volume. What limits it is unlocated: every c_t lever is
        # measured at or under 1.19x, a 2.9x cut in estimator-integrand
        # variance moved 16x16 eval variance by 3%, and 20k further steps
        # moved it by 3% more — against the 7.9x a usable 16x16 needs. These
        # four cells attack the three axes that remain untested, one variable
        # each.
        #
        # 1. VOLUME. 8x8 works, 16x16 does not, nothing between has run.
        "H2_d144_c50_s223_letf_fmo2_50k_curr": replace(
            _d144_curriculum_cell(
                "H2_d144_c50_s223_letf_fmo2_50k_curr",
                head_kind="factorised",
            ),
            ema_decay=0.9999,
            site_orderings=("row", "col"),
        ),
        # 2. CAPACITY. hidden_dim has been 32 at every volume ever run.
        "H2_d64_c50_s223_letf_fmo2_h128_50k_curr": _d64_fmo2_h128_cell(
            "H2_d64_c50_s223_letf_fmo2_h128_50k_curr",
        ),
        # 3. TRANSFER x RESOLUTION, as a two-arm pair one variable apart.
        "H2_d256_c50_s223_letf_fmo2_20k_sc_warm": _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_20k_sc_warm",
        ),
        # CANCELLED (2026-08-15, back-annotated 2026-08-19): never ran — do
        # not relaunch. Its question ("does training at the finer grid beat
        # re-rolling finer?") was answered by the b512+ne512 recipe pair
        # (trained at ne512: endpoint NULL) and the eval-only grid sweep;
        # the builder docstring's "live hope" framing above predates both.
        "H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512": _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512",
            n_euler_steps=512,
        ),
    # --- 16x16 stage-1 screen (2026-08-15, post-review): flat sigma=0.10,
    # 5k steps, naive c_t, one variable per arm, read on stage-tail FVU
    # (bands frozen in the _d256_scr5k_cell docstring BEFORE launch).
    # The MA pair anchors the archived failure (base expectation 0.119,
    # the rescue's own first 5k); the fmo2 arms screen the head any scaled
    # run would use (4.7x faster / 5.7x smaller, measured at this volume).
    # lr arms exist because lr has been varied exactly as often as
    # hidden_dim across the whole archive: never. Every capacity arm
    # (h128, L3) ships with an lr=3e-4 sibling because the one capacity
    # variation ever run produced a negative attributable to the un-retuned
    # lr rather than to capacity.
    "H2_d256_scr5k_ma": _d256_scr5k_cell(
        "H2_d256_scr5k_ma", "masked_attention", eval_sample_chunk=128,
    ),
    "H2_d256_scr5k_ma_h128_lr03": _scr5k_ma_h128_lr03_cell(
        "H2_d256_scr5k_ma_h128_lr03",
    ),
    # muP-init arm (2026-08-18, init-review stream): the bridge arm with
    # readout_score_scale = 32/128 the ONLY change — the causal test of
    # the printed muP diagnosis (readout dot has no fan-in compensation;
    # G ~ sqrt(h) at init). Comparators measured from the archived
    # training logs BEFORE launch (2026-08-18): bridge arm (seed 42) step-0
    # rate_pair_mean 0.00335 / lambda_dt_clip 0.352 / grad_norm 3.8e7,
    # steps-to-FVU<1 = 825, tail FVU(3-5k) 0.0810; h32 base 0.00231 /
    # 0.047 / 6.1e6, 65-74 steps, tail 0.1203-0.1219. FROZEN BANDS, seed
    # 42: plumbing check — step-0 rate_pair_mean must read EXACTLY
    # 0.25 x 0.00335 = 0.00084 (same seed, deterministic init; any other
    # value means the knob missed the head).
    # [CORRECTION 2026-08-19: this plumbing check was MIS-SPECIFIED — the
    # first CSV row sits on buffer states rolled under scaled rates and
    # reads 0.00164; the honest instrument is init_diagnostics, which read
    # 0.0023753 = exactly 0.25x the bridge's 0.0095. Do not fail the knob
    # on the CSV row.] MECHANISM CONFIRMED iff
    # step-0 lambda_dt_clip <= 0.05 AND step-0 grad_norm <= 1e7
    # (h32-scale) AND steps-to-FVU<1 <= 150 (2x the h32 base's, vs the
    # bridge's 825). TRANSIENT-PRICED iff additionally tail FVU <= 0.075
    # (the freed ~16% of budget shows up); tail 0.078-0.084 = transient
    # was free at this horizon (mechanism can still confirm); tail >
    # 0.09 = the scale change damaged the trained model — report as-is.
    # Scope frozen: does NOT reopen the MA family — the ~70x sigma_c ESS
    # penalty at matched FVU is orthogonal to any transient fix.
    "H2_d256_scr5k_ma_h128_lr03_mup": replace(
        _scr5k_ma_h128_lr03_cell("H2_d256_scr5k_ma_h128_lr03_mup"),
        readout_score_scale=32 / 128,
    ),
    # d-scaled clip arm (2026-08-18, init-review stream): the MA screen
    # base with grad_clip_max_norm 500 -> 60,000 the ONLY change — the
    # causal test of the printed "clip changes meaning with lattice
    # size" diagnosis. Scale set by MEASUREMENT, not the pair-count
    # heuristic: archived early(0-500) median grad_norm is 925-1108 at
    # d64 vs 1.0-1.2e5 at d256 (ratio 91-131x), so 60,000 = 500 x ~120
    # restores the d64 ratio p50(grad)/clip ~ 2. The pair-count ratio
    # 16.2x is REFUTED pre-launch by the same logs (an 8,095 threshold
    # would still bind on ~100% of early steps) — the printed clause
    # needs that amendment regardless of this arm's outcome.
    # Comparators (seed 42): d256 base early(0-500) clip% 100, first-5k
    # 39.9, tail FVU 0.1203; d64 early 72-81%, first-5k 7.3-9.0. FROZEN
    # BANDS, seed 42: SEMANTICS RESTORED iff early(0-500) clip% <= 85
    # AND first-5k clip% <= 15 (the d64 profile); BRAKE-PRICED iff
    # additionally tail FVU <= 0.11 (below both base seeds); NULL if
    # tail in (0.11, 0.13); DAMAGE if tail > 0.13 or FVU >= 1 at any
    # step past 1000 (divergence risk priced: step-0 grads 6.1e6 still
    # clip 100x at this threshold, unlike the unguarded clip20000
    # smoke).
    "H2_d256_scr5k_ma_clip60k": _scr5k_ma_clip60k_cell(
        "H2_d256_scr5k_ma_clip60k",
    ),
    "H2_d256_scr5k_fmo2": _scr5k_fmo2_cell("H2_d256_scr5k_fmo2"),
    "H2_d256_scr5k_fmo2_h128": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_h128", hidden_dim=128,
    ),
    "H2_d256_scr5k_fmo2_h128_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_h128_lr03", hidden_dim=128, lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_lr03", lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_L3": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_L3", n_layers=3,
    ),
    "H2_d256_scr5k_fmo2_L3_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_L3_lr03", n_layers=3, lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_clip2000": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_clip2000", grad_clip_max_norm=2_000.0,
    ),
    "H2_d256_scr5k_fmo2_ne512_b512": _scr5k_fmo2_with(
        # n_grid also sets c_t slots, buffer size and the loss's t-support,
        # and inner_batch/n_grid is the per-slot gradient density — so grid
        # and batch move together to hold density at 1.0, or the arm tests
        # starvation rather than resolution.
        "H2_d256_scr5k_fmo2_ne512_b512", n_euler_steps=512, batch_size=512,
    ),
    # --- screen phase 2 (2026-08-18, user GO after judging): two arms, one
    # variable each against the SAME judged base pair (0.0364/0.0442).
    #
    # b512: the batch-only decomposition of the ne512_b512 bundle. That arm
    # moved (honest FVU 0.0118 vs base 0.0286 after floors of 1/512 vs
    # 1/128) but varies grid and batch together BY DESIGN; every
    # grid-stress instrument on it read unstressed at ne128
    # (lambda_dt_clipped_frac 0 post-warmup, proposal drops 0.3%,
    # lambda_dt_p99 pure dt scaling), predicting the gain is mostly batch.
    # This arm tests that prediction: batch 512 at the base grid, density
    # 4.0 accepted as the point under test (the bundle held it at 1.0).
    # loss_microbatch_size=128 rides NOT for memory (fmo2 fits easily) but
    # for the gradient-noise-scale instrumentation: four 128-row slices
    # log grad_sqnorm_slice_mean, and with the full-batch grad_norm they
    # invert the McCandlish two-batch identity, so this run MEASURES the
    # critical batch size instead of guessing it — gradient-exact by
    # test_loss_microbatch_parity, so the arm stays the declared twin.
    # FROZEN READ: honest (floor-subtracted, 1/512) stage-tail FVU —
    # lands at the bundle's ~0.012 => batch explains the bundle and the
    # 50k batch lever is licensed at the base grid; lands at base's
    # ~0.029 => the grid was the mover and the instruments misled;
    # in between => split, both levers real. Plus the B_crit readout.
    "H2_d256_scr5k_fmo2_b512": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_b512", batch_size=512, loss_microbatch_size=128,
    ),
    # cv: the published control-variate estimator, cold, on the factorised
    # head — never run at this size (the historical "CV inverts cold at
    # d256" was measured on MA). The 18-Aug warm-CV anatomy showed the
    # inversion tracks RATE MIS-SCALING, not size: var-ratio 30x-worse at
    # the shock's step 0, crossing 1 at ~step 914 as the rates healed, and
    # 0.121 (an 8.3x cut) once healthy. fmo2's COLD init is clean
    # (lambda_dt_clipped_frac 0.0 at init in all ten screen arms), so the
    # precondition the inversion violated holds here from step 0. FROZEN
    # READ: var_estimator_integrand/var_dt_log_p_tilde tail (3-5k) < 0.5
    # AND train health at base level => CV WORKS COLD at d256 (the paper's
    # lever needs no schedule on the right head — cross-estimator FVU
    # compared floor-corrected only); sustained ratio > 1 beyond ~1k steps
    # => the inversion is not init-mediated after all and the health-gated
    # switch-in is the only CV route left.
    "H2_d256_scr5k_fmo2_cv": replace(
        _scr5k_fmo2_cell("H2_d256_scr5k_fmo2_cv"),
        estimator="control_variate",
    ),
    # c_t decoupling arm (2026-08-19, Tier 3(a) of the standing queue,
    # user GO): the fmo2 screen base with c_t_batch=512 the ONLY change —
    # gradient batch stays 128, only the c_t estimator draws 512. The b512
    # arm showed batch 512 explains the ne512_b512 bundle (honest FVU
    # 0.0115 vs base 0.0286) and its B_crit readout (~45-91) put b512 ~10x
    # past the gradient-SNR knee, predicting the gain is c_t-estimator
    # variance, not gradient noise. This arm is the causal test: it buys
    # the c_t variance WITHOUT the gradient batch. Honest-FVU floor moves
    # with the c_t draw count: subtract 1/512 here (the b512 convention),
    # not 1/128. FROZEN BANDS (before launch, seed 42, stage-tail FVU
    # 3000-4999): honest FVU <= 0.015 = DECOUPLING CONFIRMED (matches the
    # b512/bundle 0.0115-0.0118 — the batch lever's whole gain is
    # c_t-side, and the cheap lever at 50k is c_t_batch, not batch);
    # >= 0.025 = GRADIENT-SIDE (matches base 0.0286 — the b512 gain needs
    # the gradient batch after all, B_crit read notwithstanding);
    # in between = SPLIT, both channels real, report the fractions.
    # VENUE (2026-08-19 s37): pulled off the DoC a100 queue and run on
    # Modal A100-80GB instead (tag 20260819-215249, seed 42). Not a scientific
    # change -- `train_remote` pins the same 80GB A100 class the anchors
    # ran on -- but a scheduling one: a comparable 5k d256 screen measured 18 minutes end-to-end
    # against a three-day queue wait, so the wait was ~150x the compute. Bands above unchanged, frozen
    # before either launch.
    "H2_d256_scr5k_fmo2_ctb512": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_ctb512", c_t_batch=512,
    ),
    "H2_d256_scr20k_fmo2": replace(
        # The horizon control, and the first flat-subcritical 16x16 run of
        # any length: no 16x16 model has ever trained at fixed sigma=0.10
        # beyond the ladder's first 5k steps (8x8 has exactly one such run,
        # the 50k floor at 0.00018 Var/site; 4x4 many). If the stage-1
        # deficit is slow convergence rather than a floor, THIS arm falls
        # toward 0.03 after 5k and every scr5k null gets re-read; if it
        # stays at ~0.119 for 4x the horizon, the 5k screen read stands.
        _scr5k_fmo2_cell("H2_d256_scr20k_fmo2"),
        train=replace(_scr5k_fmo2_cell("H2_d256_scr20k_fmo2").train,
                      n_steps=20_000),
    ),
    "H2_d256_scr5k_mo": _scr5k_mo_cell("H2_d256_scr5k_mo"),
    # --- 12x12 volume bracket of the archived failure (2026-08-15,
    # post-review; supersedes the fmo2 12x12 cell as the bracketing rung —
    # rationale in the builder docstring). Two seeds: this is a headline
    # scaling-curve point, not a screen arm.
    "H2_d144_c50_s223_letf_ma_50k_curr_naive": _d144_ma_bracket_cell(
        "H2_d144_c50_s223_letf_ma_50k_curr_naive",
    ),
    # --- clip-threshold continuation of the completed naive rescue -------
    "H2_d256_c50_s223_letf_ma_10k_sc_clip2000": _d256_clip2000_cont_cell(
        "H2_d256_c50_s223_letf_ma_10k_sc_clip2000",
    ),
    # Band-capacity push batch 1 (2026-07-08): three single-variable twins
    # of ma_50k_curr.
    # The discriminator: interval head, head_kind is the ONLY change.
    # Separates "shared band content is the bottleneck" (lands in the MA
    # band ~0.75-0.78) from "the MA aggregator is" (lands well above it).
    "H2_d64_c50_s223_letf_iv_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_iv_50k_curr", head_kind="interval",
    ),
    # H-width: double the band feature and attention widths, nothing else.
    "H2_d64_c50_s223_letf_ma_wide_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_wide_50k_curr", head_kind="masked_attention",
        band_feature_dim=32, attention_dim=64,
    ),
    # H-offsets: band also sees offset-2 / offset-2D bond families (second-
    # neighbour row/column pairs), beyond the energy's (1, D).
    "H2_d64_c50_s223_letf_ma_offs_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_offs_50k_curr", head_kind="masked_attention",
        pair_offsets=(1, 2, 8, 16),
    ),
    # Round-2 stencil family: the reported MA head + the 5-point
    # lattice-stencil band family (neighbours ±1, ±D on the flattened D x D
    # grid), narrow unary+offset families kept alongside to cover the collar.
    # use_stencil is the ONLY change vs ma_50k_curr -- the probe of whether a
    # richer 2D-local band content lifts the ~0.78 H-shared ceiling.
    "H2_d64_c50_s223_letf_ma_stencil_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_stencil_50k_curr", head_kind="masked_attention",
        use_stencil=True,
    ),
    # Horizon extension (2026-07-22). Judging the stencil
    # exposed that the 50k budget cuts BOTH heads off mid-descent: over the
    # final 10k steps the loss still falls 12.0% (ma) / 7.4% (stencil) and
    # train ESS is still climbing, so the 0.78/0.80 "ceiling" is read off
    # unconverged runs. n_steps is the ONLY change: the ladder holds absolute
    # start_steps (last rung 30k) and lr is warmup x per-stage value with no
    # total-budget normalisation, so the anneal does NOT stretch -- the extra
    # 50k steps all land on the final sigma=0.223 plateau (20k -> 70k), which
    # is the phase still descending, and each cell's first 50k steps stay
    # schedule-identical to its 50k twin. Both cells are needed: MA is the
    # control for whether the stencil's +0.024 survives a converged horizon
    # (MA's loss is falling FASTER at the cutoff, so it may close some gap).
    "H2_d64_c50_s223_letf_ma_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_100k_curr", head_kind="masked_attention",
        n_steps=100_000,
    ),
    "H2_d64_c50_s223_letf_ma_stencil_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_stencil_100k_curr", head_kind="masked_attention",
        n_steps=100_000, use_stencil=True,
    ),
    # The missing twin (judged 2026-07-22). It found the 50k cutoff lands
    # mid-descent for the one-pass heads -- but the SAME check on mo_50k_curr's
    # log shows mask_one was still descending fastest of the three at its own
    # cutoff (loss -25.7% over the final 15k, train ESS 0.864 -> 0.900). So the
    # reference rung is unconverged too, and "does a longer horizon close the
    # gap to MO's 0.9103" is unanswerable while only the one-pass heads get the
    # longer horizon: both sides must move before any gap is read.
    # Seed 42, NOT the record's 43 -- the rest of the d64 ladder is seed 42, and
    # the cross-seed comparison is a standing confound in every mo-vs-ma number
    # quoted so far (same-seed MA s43 is 0.7546, not the 0.7806 usually cited
    # against MO). This cell retires that confound at the same time as the
    # horizon one; it is NOT a twin of the 0.9103 run (seed differs by design),
    # so the 50k seed-42 mask_one point it implies is a second thing this run
    # buys back.
    "H2_d64_c50_s223_letf_mo_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_mo_100k_curr", head_kind="mask_one",
        n_steps=100_000,
    ),
    # Mixing-probe floor cell (sigma=0.10, 8x8) — the probe's control operating
    # point, where Kawasaki mixes happily; it exists to show the diagnostic
    # does not flag failure everywhere. Direct subcritical training with NO
    # curriculum, mirroring how the d16 sigma=0.10 cells trained: the sigma
    # ladder exists to reach sigma_c, and a floor cell that took the ladder
    # would measure the curriculum, not the operating point. Every other knob
    # is the d64 sigma_c shape verbatim (one-event n_euler=128 clip-safe per
    # the scout, 5000-draw final eval, seed 42 = the d64 ladder seed) so the
    # probe's floor-vs-headline contrast isolates sigma.
    "H2_d64_c50_s010_letf_ma_50k": _hard_cell(
        "H2_d64_c50_s010_letf_ma_50k", sigma=0.10, head_kind="masked_attention",
        D=8, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=256, n_eval_samples_training=512,
        use_sdpa_readout=True, eval_autocast_bf16=True,
    ),
    # The 16x16 rung — hard.tex §5.6's plan of record, the last training rung
    # (launched 2026-08-09). The reported masked-attention head on the
    # vertex-disjoint matching step (validated as a drop-in at the converged
    # 8x8 checkpoint: ESS 0.906 vs 0.910, composition bitwise-preserved), at
    # the 8x8 rung's 50k sigma-ladder recipe UNCHANGED — the ladder holds
    # absolute start_steps, so the schedule is identical, not stretched.
    # n_euler_steps stays 128 only BECAUSE the matching step decouples
    # trajectory length from the total rate: the clip-safe one-event budget
    # extrapolates to ~390 steps at this size (hard.tex subsec:rate-field).
    # Eval deltas are diagnostics-only: cadence 200 -> 500 and in-training
    # draw 512 -> 256 (each network pass scores ~16x the d64 cell's pairs, so
    # the d64 cadence would spend most of the job evaluating), chunk
    # 256 -> 64 to bound the eval batch at 4x the sites. The final eval keeps
    # the 5000-draw protocol the probe and the d64 ladder report on.
    "H2_d256_c50_s223_letf_ma_50k_curr": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
    ),
    # 50k rescue candidate (2026-08-12, prepped DURING the smoke wave): the
    # naive-estimator twin of the diverged cell above. Arm B's mechanism —
    # kill the inverted control variate (adds 2.3-70x variance at d=256 vs
    # an 8-30x reduction at d64) — is the only arm showing a converging loss
    # mid-smoke. DO NOT LAUNCH until (i) all five smoke verdicts are judged
    # against the pre-stated criteria and (ii) the user makes the launch
    # call (frozen amendment: launch by 14 Aug EOD or 16x16 degrades to
    # smoke-level evidence). Everything except the estimator is identical
    # to the diverged twin, so the comparison isolates the CV.
    "H2_d256_c50_s223_letf_ma_50k_curr_naive": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr_naive", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        estimator="naive_mc",
    ),
    # The definitive cold fmo2 ladder at 16x16 (2026-08-18, user GO after
    # the screen fleet was judged): the run every archived d256 number is
    # missing. The archived MA naive-rescue recipe verbatim with the head
    # family the only mechanism change (+ its riding EMA shadow and dual
    # site orderings, and the eval chunk sized to the factorised head's
    # measured memory — eval-only; all pinned in test_screen_pins), so any
    # difference from the rescue's Var[log w]/site 0.0707 / ESS/N 0.0031
    # is chargeable to the head. Bands FROZEN BEFORE LAUNCH, seeds 42+43:
    # stage-1 self-serves at ~5k (tail FVU, steps 3000-4999; screen base
    # band 0.036-0.044) — transfer CONFIRMED <= 0.05, >= 0.08 means the
    # ladder frame breaks the screen result (buffer/curriculum
    # interaction) and later rungs are not trusted until explained. Final
    # 5000-draw eval read on Var[log w]/site + ESS/N, EMA and raw, with
    # n_unique and top-weight mass alongside (the d144 lesson: one draw
    # carried 19.7% of the mass and swung ESS/N 6x): NULL <= 0.005 ESS/N
    # EMA (archived-MA scale — the head family alone does not move the
    # sigma_c endpoint), PARTIAL < 0.02, PASS >= 0.02 (cold + ladder +
    # naive matches the warm+CV 0.0219, the best d256 sigma_c number
    # owned), STRONG >= 0.10 (the usability bar the rescue failed).
    # Expectation set honestly by the sigma_c evidence: FVU parity does
    # not transfer to ESS across sigma (~70x at matched FVU ~0.12), so
    # ~0.02 is the realistic target scale, not the subcritical 0.5-0.9.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive": _d256_fmo2_ladder_cell(
        "H2_d256_c50_s223_letf_fmo2_50k_curr_naive", estimator="naive_mc",
    ),
    # The RECIPE runs (2026-08-18, registered ready-to-fire): the composed
    # best-effort sampler at 16x16, declared openly as a demonstration
    # arm, not a screen arm — attribution lives in the chain (archived MA
    # naive ladder vs the fmo2 naive ladder anchor isolates the head;
    # anchor vs recipe isolates the variance bundle; recipe_naive vs
    # recipe_cv isolates the estimator). Ingredients and their licences:
    # b512+ne512 moved as a pair holding per-slot gradient density at 1.0
    # (the screen's best arm: FVU 0.0138, ESS/N 0.92, clip-free at
    # sigma 0.1; sigma_c grid evidence: lambda_dt p99 0.98 with 1.5-2%
    # residual clipping on the warm run at ne128 = 0.5d, vs the builder's
    # ~2d rule); loss_microbatch_size=128 rides as the noise-scale
    # instrument (gradient-exact, parity-pinned) so B_crit is measured
    # along the whole ladder; horizon 50k matches the anchor so the
    # recipe-vs-anchor read is horizon-controlled (a sigma_c plateau
    # continuation off final.pt is the licensed extension if the tail is
    # still descending). LAUNCH GATES, frozen 2026-08-18: the naive
    # variant fires once the b512 screen arm reads healthy; the cv
    # variant additionally requires the cold-CV screen arm's frozen PASS
    # (tail var-ratio < 0.5 at base-level health) — cold CV at this size
    # inverted on the mis-scaled-init head and must never enter a long
    # run unlicensed. BANDS, frozen before any launch: final EMA eval
    # ESS/N >= 0.30 = STRONG (Var[log w] <= ~1.2 — the methodology bar:
    # a usable sampler at 256 sites); >= 0.10 = PASS (order of magnitude
    # over every archived d256 number); >= 0.02 = PARTIAL (no gain over
    # the single-lever warm-CV read — the composition added nothing);
    # < 0.02 = NULL. Read Var/site + n_unique + top-weight mass alongside,
    # per the d144 lesson.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive":
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive",
            estimator="naive_mc", n_euler_steps=512, batch_size=512,
            loss_microbatch_size=128,
        ),
    # No microbatch on the cv variant: loss_microbatch_size's
    # gradient-exactness is parity-pinned for the archived loss only —
    # a batch-coupled control variate would break the per-row
    # decomposition silently, so the CV recipe keeps the single backward
    # and the noise-scale measurement rides the naive variant alone.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_cv":
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_cv",
            estimator="control_variate", n_euler_steps=512, batch_size=512,
        ),
    # CV continuation of the landed recipe (2026-08-19, user GO after the
    # recipe pair judged NULL/NULL-grazing-PARTIAL per its own bands
    # above): the cv2 pattern — flat sigma_c, 20k steps, lr pinned to the
    # ladder's final 3e-4 — applied to the b512+ne512 recipe shape, run
    # with --init-from the seed-43 recipe run's final.pt (raw weights: the
    # EMA shadow restarts and is warmup-capped, so early eval_ema reads
    # are transient — say so at judging). estimator back to the control
    # variate is the arm variable; the cold-CV route stays unlicensed
    # (screen arm FAILED), continuation is the validated warm pattern
    # (var-ratio crossed 1 at ~step 914 on healing rates, 0.121 once
    # healthy). No loss_microbatch: gradient-exactness is parity-pinned
    # for the archived loss only, a batch-coupled CV would break the
    # per-row decomposition silently. TRIPWIRE ARMED (user-set 19-Aug):
    # halt_on_cv_inversion_after=2000 with the default window 10 — a
    # sustained controlled/naive integrand-variance inversion after step
    # 2000 halts the run; that halt IS the designed cost-capped negative
    # verdict, not an accident. FROZEN BANDS (before launch, vs the
    # parent's EMA eval ESS/N 0.0198 and the archived single-lever
    # warm-CV 0.0219): mechanism — trailing-median cv_var_ratio < 1 by
    # step 2000 and falling toward the ~0.12-0.14 healthy precedent;
    # endpoint (EMA eval, bootstrap CI alongside per the addenda) —
    # NULL < 0.03 (no separation from the parent read: the estimator
    # lever adds nothing at this scale and the weight-construction
    # residue stands as the whole story); PARTIAL 0.03-0.10; PASS
    # >= 0.10; STRONG >= 0.30. Var[log w]/site, n_unique and top-weight
    # mass read alongside (d144 lesson).
    "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne512": replace(
        _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne512",
            n_euler_steps=512,
        ),
        train=replace(
            _d256_fmo2_warm_cell(
                "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne512",
                n_euler_steps=512,
            ).train,
            batch_size=512,
            halt_on_cv_inversion_after=2000,
        ),
    ),
    # NAIVE twin of the CV continuation above (2026-08-20, s41, user GO at
    # judging): the same cell with `estimator` control_variate -> naive_mc
    # the ONLY declared change, continued with --init-from the SAME seed-43
    # recipe final.pt. Why it exists: the CV continuation judged PASS at
    # EMA eval ESS/N 0.2655 (bootstrap CI 0.2302-0.3043) against its
    # parent's 0.0198 — 13.4x, the largest quality number measured at this
    # size, and at sigma_c rather than a subcritical screen. But it trains
    # 20k steps BEYOND its parent, and no naive-estimator continuation from
    # that checkpoint exists, so the 13.4x cannot be split between (a) the
    # control variate and (b) 20k additional fixed-sigma_c steps. The
    # archived warm-CV reference (0.0219) does not serve as that control:
    # it is also control_variate AND warm-starts from a d64 EMA rather than
    # continuing a d256 checkpoint, so it differs on two axes at once.
    # This arm differs on exactly one.
    # DECLARED, and deliberately NOT a second variable: the parent cell's
    # halt_on_cv_inversion_after=2000 tripwire is dropped here because
    # cv_var_ratio has no meaning when the control variate is not in use.
    # It is a safety halt, never a training-affecting knob, and on the CV
    # run it never fired — so its absence cannot move this read.
    # FROZEN BANDS (before launch, seed 43, EMA eval ESS/N with bootstrap
    # CI, Var[log w]/site and top-weight read alongside; references: parent
    # 0.0198 / Var-site 0.00735, CV continuation 0.2655 / Var-site 0.00495):
    #   CONTINUATION-OWNS iff this arm's CI OVERLAPS the CV continuation's
    #                     (0.2302, 0.3043) -- the estimator is not the
    #                     lever, the finding is "20k more steps at fixed
    #                     sigma_c", and the CV chapter claim is withdrawn.
    #   ESTIMATOR-OWNS    iff EMA eval ESS/N < 0.10 (the PASS threshold the
    #                     CV arm cleared) with CI separation from it -- the
    #                     control variate owns the gain and the claim
    #                     stands as printed.
    #   SPLIT             otherwise -- report the recovered fraction
    #                     (naive - parent) / (cv - parent) explicitly
    #                     rather than rounding it to either story.
    "H2_d256_c50_s223_letf_fmo2_20k_sc_naive_b512_ne512": replace(
        _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_20k_sc_naive_b512_ne512",
            n_euler_steps=512,
        ),
        estimator="naive_mc",
        train=replace(
            _d256_fmo2_warm_cell(
                "H2_d256_c50_s223_letf_fmo2_20k_sc_naive_b512_ne512",
                n_euler_steps=512,
            ).train,
            batch_size=512,
        ),
    ),
    # ---- ne128 x CV composition family (2026-08-21, s42, user GO) -------
    # WHY THIS FAMILY EXISTS. s42 judging certified two d256 levers as
    # independent and, until now, uncomposed:
    #   * the GRID (training): `H2_..._b512_ne128_naive` reads GRID-HELPS
    #     at EMA Var[log w]/site 0.00485 (0.00463, 0.00508) against both
    #     ne512 parent seeds (0.00612 / 0.00735), and the 276127 grid probe
    #     removed the shared-field caveat -- the keystone keeps its gain
    #     when re-rolled at ne512 (0.00493), and the parent gains nothing
    #     when re-rolled at ne128 (0.00595, CI overlapping its own 0.00612).
    #     The advantage is in the TRAINED MODEL, not the eval sampler.
    #   * the ESTIMATOR (continuation): `H2_..._20k_sc_cv2_b512_ne512`
    #     reads PASS at EMA eval ESS/N 0.2655 (0.2302, 0.3043) vs its
    #     parent's 0.0198, and 276126's naive twin split the confound --
    #     ESTIMATOR-OWNS, 20k extra fixed-sigma_c steps alone recovering
    #     only 10.7% of the ESS gain (2.32x of the 13.41x).
    # Nobody has run them together: cvcont ran at ne512, and the keystone
    # says ne128 is both better AND 4x cheaper in rollout FLOPs.
    #
    # COST, measured not guessed (a100, b512): ne512 costs ~110 min / 5k
    # steps, confirmed three ways (5k screen 109 min, naivecont 20k
    # 7h20m, rw 50k 20h03m). Of that, 5000 x 0.464 s = 38 min is the inner
    # loss backward, which `wall_clock_step_s` measures and which reads
    # IDENTICALLY at ne128 and ne512 (0.4673 vs 0.4638) -- it is
    # grid-independent. Only the remaining ~72 min scales with the grid.
    # So ne128 buys ~2.0x WALL CLOCK, not 4x; the 4x is rollout FLOPs.
    # Budget at ne128: ~56 min / 5k steps => 20k ~3.7 h, 50k ~9.3 h,
    # 70k ~13 h, and ~15% more at h128/L3.
    #
    # LOSS MICROBATCHING at 128 rides on ALL SIX arms of this family, and
    # is NOT a declared variable on any of them. The keystone -- arm A's
    # continuation parent and the h32 comparator for arms B and P -- runs
    # loss_microbatch_size=128 itself, as does the ne512 recipe, so 128 is
    # this lineage's setting and omitting it is what would make an arm
    # differ from its own parent in an extra place. It is gradient-exact:
    # the swap loss is a per-row mean, c_t reaches it as a detached
    # per-row gather from a grid frozen for the cycle, and the identity is
    # pinned for ARBITRARY per-row c_t by
    # tests/test_loss_microbatch_parity.py -- which is why the estimator
    # cannot affect it (the control variate changes how c_t is COMPUTED,
    # in the outer no_grad rollout, never how the loss DECOMPOSES). The
    # `cv2` cells' historical "no loss_microbatch" note guards against a
    # BATCH-COUPLED control variate computed inside the loss; this code
    # computes c_t outside it, so the note does not bind here. It also
    # keeps the per-slice gradient-noise-scale instrument live on every
    # arm, and removes an OOM risk at h128/L3 (~6x the activation memory
    # of h32/L2).
    #
    # A: the CV continuation at the keystone's grid. One variable versus
    # the landed cvcont (n_euler_steps 512 -> 128); continued with
    # --init-from the keystone's own final.pt, so its parent is its own
    # seed rather than cvcont's.
    # PRIMARY DECLARED BEFORE THE READ, and deliberately NOT the project's
    # default Var/site: the two levers ALREADY COINCIDE on the bulk
    # statistic -- keystone 0.00485 (0.00463, 0.00508) and cvcont 0.00495
    # (0.00475, 0.00515) are indistinguishable -- so Var/site has no power
    # to resolve composition here. They differ 5x on the TAIL: EMA eval
    # ESS/N 0.0532 vs 0.2655, top weight 0.039 vs 0.0076. The tail
    # statistic is therefore the primary for this arm, with Var/site and
    # top weight read alongside.
    # FROZEN BANDS (before launch, seed 42, EMA eval ESS/N, bootstrap CI):
    #   COMPOSES   iff CI separated ABOVE cvcont's (0.2302, 0.3043) -- the
    #              estimator gain survives the cheaper grid and adds to it.
    #   REDUNDANT  iff CI OVERLAPS cvcont's -- both levers reach one
    #              ceiling; the recipe takes ne128 for the cost alone.
    #   INTERFERES iff CI separated BELOW cvcont's -- the coarse grid
    #              costs the control variate something, and ne512 stays
    #              the grid for CV arms specifically.
    #   Var/site CI wholly below 0.00463 is corroborating evidence of
    #   composition; a Var/site read stuck at ~0.0049 with ESS above
    #   0.3043 is the informative "tail-only" outcome and must be reported
    #   as such rather than rounded into either story.
    # SEED CAVEAT, recorded before the read: cvcont is seed 43 and this
    # arm is seed 42, and the measured d256 seed-to-seed spread is 18% of
    # mean on Var/site. The within-seed comparison (this arm vs its own
    # keystone parent) carries no such caveat and is the safer read.
    # TRIPWIRE ARMED at 2000, exactly as the ne512 twin: the continuation
    # is the validated warm pattern (var-ratio crossed 1 at ~step 914 on
    # healing rates, 0.121 once healthy), so 2000 is generous for a warm
    # start and the halt is the designed cost-capped negative verdict.
    "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne128": replace(
        _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne128",
            n_euler_steps=128,
        ),
        train=replace(
            _d256_fmo2_warm_cell(
                "H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne128",
                n_euler_steps=128,
            ).train,
            batch_size=512,
            loss_microbatch_size=128,
            halt_on_cv_inversion_after=2000,
        ),
    ),
    # B: the same total budget with the control variate on FROM STEP 0.
    # Matched to A by construction and that is the point: the keystone's
    # 50k is 30k of sigma ladder + 20k at sigma_c, and A adds 20k more at
    # sigma_c, so A is 30k ladder + 40k sigma_c = 70k. B runs 70k with the
    # SAME ladder, whose final stage starts at 30k, giving 30k ladder +
    # 40k sigma_c. The only difference left between A and B is WHEN the
    # control variate joins -- warm at 50k, or cold at 0.
    # ADVERSE PRIOR, recorded before launch: cold CV at d256 is the one
    # configuration already measured catastrophic -- eval ESS/N 0.0069
    # against the naive base's 0.5475 with raw FVU 2.06 (above 1 means the
    # control variate is INJECTING variance, not removing it). That was a
    # 5k screen at sigma=0.10 on the OLD ne512 recipe, so it does not
    # settle this arm, but sigma=0.10 is exactly this arm's first ladder
    # stage. The mechanism by which ne128 might change it -- a coarser
    # grid means larger dt and a different integrand variance -- is
    # PLAUSIBLE BUT NOT DERIVED, and is not claimed here.
    # TRIPWIRE ARMED at 5000, deliberately LATER than the warm arms' 2000:
    # a cold start has no healed model to recover toward, and the ~914
    # step healing precedent is a WARM number that must not be read across.
    # 5000 is one full ladder stage, so the arm is judged on a completed
    # stage rather than on a transient. If it fires, that halt IS the
    # verdict at ~55 min rather than ~13 h.
    # FROZEN BANDS (before launch, seed 42): frame check -- stage-1 tail
    # FVU <= 0.05 (the anchor transfer band; a frame break voids the read).
    #   WARM-JOIN-REQUIRED iff the tripwire fires, OR final EMA eval ESS/N
    #                      CI separated BELOW arm A's.
    #   COLD-CV-VIABLE     iff final EMA eval ESS/N CI overlaps or is
    #                      separated ABOVE arm A's -- the two-phase recipe
    #                      is then a convention, not a requirement, and
    #                      the chapter's "CV must join warm" claim is
    #                      withdrawn.
    "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2": (
        lambda _cell: replace(
            _cell,
            train=replace(
                _cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2",
            estimator="control_variate", n_euler_steps=128, batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # ---- capacity twins of the composition family (2026-08-21, user GO) --
    # WHY, AND AGAINST WHAT PRIOR. Capacity is a CLOSED door at d256 and
    # the evidence points the wrong way: the full-horizon arm
    # `H2_..._fmo2_h128_lr03_50k_curr_naive` read a REGRESSION, EMA
    # Var[log w]/site 0.0229 (0.0219, 0.0239) against the anchor's 0.0168
    # (0.0161, 0.0176), CI-disjoint WORSE, matching d64 (h128 0.695 vs
    # 0.810). That is recorded here so these arms cannot be read as if the
    # question were open.
    # What is genuinely new, and the only reason to re-ask: that arm ran on
    # the ANCHOR (batch 128, Var/site 0.0168) under naive_mc, a base 2.8x
    # worse than the current recipe's 0.0061. The coherent hypothesis is
    # that capacity could not bind while estimator variance was the binding
    # constraint, and the control variate moves exactly that constraint.
    # The hypothesis is NOT established and these arms are expected, on the
    # evidence, to read NULL or REGRESSION.
    # DECLARED FIELDS: hidden_dim 32 -> 128 and n_layers 2 -> 3, bundled.
    # The project's own convention keeps width and depth in separate cells
    # (see `_d64_fmo2_h128_cell`: "Depth is a separate cell and is
    # deliberately not bundled"), and that convention is knowingly set
    # aside here because the user's question is "does more capacity help",
    # not "which capacity knob helps" -- a bundled arm that reads NULL
    # closes both at once, and only a POSITIVE read would need unbundling.
    # `n_heads` stays 4, so head_dim rides 8 -> 32 as a consequence of
    # widening rather than as a further knob -- the same choice
    # `_d64_fmo2_h128_cell` made and for the same reason.
    # Loss microbatching rides at 128 as it does on all six arms of this
    # family; the rationale is stated once in the family header above.
    # LR: these arms keep the recipe ladder (1e-3 -> 3e-4 at stage 4), so
    # capacity is the only knob versus their h32 twins. THE RISK IS
    # MEASURED AND ON THE RECORD: the 5k screen found un-retuned lr 1e-3
    # penalises h128 specifically across sigma transitions (h128 tail FVU
    # 0.0498 vs h128+lr03's 0.0380-0.0406, base-like), which is why the
    # landed capacity arm flattened lr to 3e-4 everywhere. The `_lr03`
    # sibling below exists to separate that artefact from a capacity
    # verdict rather than leaving it as a caveat.
    #
    # P: the naive 50k parent. Serves twice -- as arm C's continuation
    # source (a continuation cannot start from the keystone, because every
    # weight matrix changes shape when hidden_dim moves), and as the
    # full-horizon capacity re-read at the CURRENT recipe under naive_mc,
    # which is the direct comparison to the landed anchor regression.
    # FROZEN BANDS (before launch, seed 42, EMA eval, bootstrap CI, vs its
    # one-variable h32/L2 twin the keystone: Var[log w]/site 0.00485
    # (0.00463, 0.00508), ESS/N 0.0532): frame check -- stage-1 tail FVU
    # <= 0.05, a break VOIDS the read.
    #   CAPACITY BINDS iff Var/site CI separated BELOW (0.00463, 0.00508).
    #   NULL           iff Var/site CI overlaps it -- capacity closes at
    #                  full horizon on the current recipe too, not just on
    #                  the anchor.
    #   REGRESSION     iff Var/site CI separated ABOVE it -- the anchor
    #                  finding reproduces on a 2.8x better base, which is
    #                  a stronger closure than the anchor read alone.
    "H2_d256_c50_s223_letf_fmo2_h128L3_50k_curr_b512_ne128_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_50k_curr_b512_ne128_naive",
            estimator="naive_mc", n_euler_steps=128, batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # P_lr03: P with the curriculum lr flattened to 3e-4 at EVERY stage,
    # matching the landed capacity arm's treatment exactly. This is the
    # lr/capacity de-confound at full horizon, the same pairing every
    # screen capacity arm shipped with, and it is judged twice.
    # FROZEN BANDS (before launch, seed 42, EMA eval Var/site, bootstrap
    # CI): the P bands above apply unchanged against the keystone; PLUS
    #   LR-ARTEFACT iff this arm's CI is separated BELOW P's -- the ladder
    #               lr was damaging h128 and P's read is an lr verdict
    #               rather than a capacity one, so P is void as a capacity
    #               read and arm D must be re-run at flat lr before its
    #               own read means anything.
    #   LR-FREE     iff the two CIs overlap -- the h128 lr sensitivity the
    #               5k screen measured does not survive to full horizon on
    #               this recipe, and P's capacity read stands as written.
    "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            curriculum=replace(
                _cell.curriculum,
                stages=tuple(
                    replace(stage, lr=3e-4)
                    for stage in _cell.curriculum.stages
                ),
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive",
            estimator="naive_mc", n_euler_steps=128, batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # C: the capacity twin of arm A -- 20k CV continuation at fixed
    # sigma_c. PARENT CHOSEN BY A CRITERION FROZEN HERE, BEFORE EITHER
    # PARENT LANDS, so the choice cannot be made with the answer in hand:
    # continue from whichever of P / P_lr03 has the lower EMA eval
    # Var[log w]/site point estimate, and if their CIs overlap take P (the
    # recipe-ladder arm), because a tie must not silently buy the extra
    # declared field. If BOTH fail the stage-1 tail FVU <= 0.05 frame
    # check, this arm does not launch.
    # FROZEN BANDS (before launch, EMA eval ESS/N with bootstrap CI,
    # Var/site and top weight alongside; primary is the tail statistic for
    # the same reason given at arm A):
    #   CAPACITY-BINDS-UNDER-CV iff CI separated ABOVE arm A's.
    #   NULL                    iff CI overlaps arm A's -- capacity does
    #                           not bind even once the estimator
    #                           constraint is lifted, which is the
    #                           strongest form of the closure.
    #   REGRESSION              iff CI separated BELOW arm A's.
    "H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            train=replace(
                _cell.train, batch_size=512, loss_microbatch_size=128,
                halt_on_cv_inversion_after=2000,
            ),
        )
    )(
        _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128",
            n_euler_steps=128,
        )
    ),
    # D: the capacity twin of arm B -- 70k with the control variate on
    # from step 0. Carries arm B's adverse cold-CV prior AND the capacity
    # prior above, so it is the least likely of the six to read positive;
    # it exists so the 2x2 {when CV joins} x {capacity} is complete and
    # "capacity binds only when CV is cold" is separable from "capacity
    # binds". Tripwire at 5000 for arm B's reason.
    # NOT LAUNCHED WITH AN lr03 SIBLING, deliberately: P vs P_lr03 is the
    # lr probe, and it is cheaper to read that first than to buy a second
    # 70k arm speculatively. If P_lr03 reads LR-ARTEFACT, D is void as a
    # capacity read and its flat-lr twin becomes owed.
    # FROZEN BANDS (before launch, seed 42): frame check -- stage-1 tail
    # FVU <= 0.05. Then, EMA eval ESS/N with bootstrap CI:
    #   CAPACITY-BINDS-UNDER-COLD-CV iff CI separated ABOVE arm B's.
    #   NULL                         iff CI overlaps arm B's.
    #   REGRESSION                   iff CI separated BELOW arm B's.
    # If arm B halts on its tripwire and this arm does not (or vice
    # versa), that ASYMMETRY is the finding and is reported as the primary
    # result for the pair, ahead of any endpoint number.
    "H2_d256_c50_s223_letf_fmo2_h128L3_70k_curr_b512_ne128_cv2": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            train=replace(
                _cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_70k_curr_b512_ne128_cv2",
            estimator="control_variate", n_euler_steps=128, batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # Buffer-depth-to-the-reference-invariant arm (2026-08-19, A6 of the
    # panel queue, user GO under the recipe-NULL clause): the recipe cell
    # verbatim with replay_buffer_cycles 8 -> 2 the ONLY change. The DNFS
    # reference bounds retention to ~1024 trajectories
    # (max_size = 1024 // outer_batch, FIFO — see the consult log at
    # _retain_chunks); our cycle-count invariant held cycles at 8 while
    # b512 quadrupled the batch, so the live recipe retains 4096
    # trajectories, 4x the reference invariant, at HALF the per-state
    # draw density (0.195 vs 0.78). cycles=2 at b512 restores 1024
    # exactly — the "fresh half" configuration the d64 replay2 smoke
    # already validated. FROZEN BANDS (before launch, seed 42, vs the
    # landed recipe s42: EMA eval ESS/N 0.0105, Var[log w]/site 0.0061):
    # CONFIRMED (buffer staleness binds at this scale) iff EMA ESS/N
    # >= 0.021 (2x) with bootstrap-CI separation from the parent read;
    # NULL iff within the parent's CI — the invariant is then refuted as
    # a d256 lever and the deviation needs only its existing one-line
    # defence. Stage-tail FVU per rung read alongside: a fresher buffer
    # should show faster post-boundary recovery if staleness is the
    # mechanism; unchanged recovery with a moved endpoint means the
    # mechanism claim is wrong even if the number moves.
    # CANCELLED 2026-08-19 (s37, user GO) before it ever started: the d64
    # loop battery measured the retention curve directly -- 256/1024/2048/
    # 4096 retained trajectories gave EMA 0.766/0.810/0.827/0.860,
    # monotone -- so the reference's ~1024 bound is a mid-point on a
    # continuing gain, not a target, and this arm's premise is refuted at
    # the size where it was measurable. The informative direction is the
    # cyc16 retention arm above. Transfer caveat, stated rather than
    # hidden: that curve is a d64 read in a healthy regime (EMA ~0.81),
    # and this arm would have been a d256 read in a broken one.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2",
            estimator="naive_mc", n_euler_steps=512, batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2",
                estimator="naive_mc", n_euler_steps=512, batch_size=512,
                loss_microbatch_size=128,
            ).train,
            replay_buffer_cycles=2,
        ),
    ),
    # KEYSTONE grid arm (2026-08-19 s37, user GO): the recipe cell
    # verbatim with n_euler_steps 512 -> 128 the ONLY change, so the read
    # is chargeable to the trajectory grid alone against the landed
    # recipe s42 (EMA eval ESS/N 0.0105, Var[log w]/site 0.0061, bootstrap
    # CI (0.0058, 0.0064)). Why now: the recipe adopted ne512 on a
    # clip-safety premise -- "lambda_dt p99 0.98 with 1.5-2% residual
    # clipping at ne128" -- and the d64 loop battery has since run a cell
    # at exactly that saturation (the ne32 grid arm: p99 0.982, 1.2% of
    # pairs clipping) which landed INSIDE the rung band, while its
    # ne32/ne128/ne256 arms span 8x of training grid within 0.020 EMA.
    # The eval-side twin is measured too: the sigma_c grid sweep moves
    # Var[log w]/site by 0.9% across 8x with all four CIs overlapping.
    # Neither is a d256 TRAINING-grid read, which has never been taken --
    # b512+ne512 vs b512+ne128 is the missing arm, and it is what
    # licenses printing the cheaper recipe.
    # FROZEN BANDS (before launch, seed 42; Var-PRIMARY because d256 ESS
    # reads are top-weight-dominated and do not resolve):
    #   GRID-FREE  iff Var[log w]/site CI overlaps the parent's
    #              (0.0058, 0.0064) -- ne128 becomes the standing d256
    #              grid, a 4x trajectory-compute cut, and downstream d256
    #              arms re-base onto this cell.
    #   GRID-BINDS iff Var/site CI lies wholly ABOVE the parent's -- the
    #              d64 insensitivity does not transfer and the
    #              d-dependence of quadrature is itself the finding.
    #   GRID-HELPS iff Var/site CI lies wholly BELOW -- coarser is better
    #              at d256 as it is on the eval side.
    # Read alongside: lambda_dt_p99 and lambda_dt_clipped_frac in the
    # sigma_c tail (the premise under test; the ne128 anchor reads p99
    # 0.84 with clipped_frac 0), stage-tail FVU per rung, n_unique and
    # top-weight mass.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive":
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive",
            estimator="naive_mc", n_euler_steps=128, batch_size=512,
            loss_microbatch_size=128,
        ),
    # Retention-depth arm (2026-08-19 s37, user GO): the recipe cell
    # verbatim with replay_buffer_cycles 8 -> 16 the ONLY change --
    # retention 4096 -> 8192 trajectories at UNCHANGED freshness, since
    # the rollout still draws outer_batch fresh trajectories per cycle,
    # so fresh-draws-per-gradient-step stays 5.12 and only the age of the
    # oldest retained chunk moves. This is the opposite direction to the
    # cancelled buf2 arm, and it is the direction the d64 battery
    # supports: retention 256/1024/2048/4096 gave EMA 0.766/0.810/0.827/
    # 0.860, monotone, which refutes the reference code's ~1024 bound as
    # a target rather than a mid-point on a continuing gain.
    # FROZEN BANDS (before launch, seed 42, vs the landed recipe s42:
    # EMA eval ESS/N 0.0105, Var[log w]/site 0.0061 CI (0.0058, 0.0064)),
    # Var-PRIMARY:
    #   DEPTH BINDS iff Var/site CI lies wholly BELOW the parent's;
    #   NULL        iff the CIs overlap -- depth is a d64-only lever and
    #               the transport wall owns the d256 residue;
    #   REGRESSION  iff the CI lies wholly ABOVE -- staleness costs more
    #               than coverage buys at this size.
    # One-sidedness declared: at fixed n_steps a deeper buffer changes
    # only retention, never the fresh-draw count, so a positive read
    # cannot be re-attributed to the freshness axis.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16",
            estimator="naive_mc", n_euler_steps=512, batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16",
                estimator="naive_mc", n_euler_steps=512, batch_size=512,
                loss_microbatch_size=128,
            ).train,
            replay_buffer_cycles=16,
        ),
    ),
    # Rank arm at the anchor recipe (2026-08-19, Tier 3(d) re-entered
    # under the recipe-NULL clause, user GO): the fmo2 naive ladder
    # anchor verbatim with bilinear_rank 8 -> 32 the ONLY change. Why 32:
    # the factorised-head forensics measured the REQUIRED effective rank
    # of the swap-rate field growing ~d/8 with lattice side — d/8 = 32 at
    # 256 sites, where the shipped default 8 was sized at the 4x4 gate.
    # Run at the anchor's b128/ne128 so the read is chargeable to rank
    # alone against the judged anchor pair (EMA eval ESS/N 0.0048/0.0058,
    # Var[log w]/site 0.0168/0.0159). FROZEN BANDS (before launch, seed
    # 42): frame check first — stage-1 tail FVU <= 0.05 (the anchor's own
    # transfer band; a frame break voids the rank read). RANK BINDS iff
    # EMA eval ESS/N >= 0.010 (2x the seed-42 anchor) with bootstrap-CI
    # separation, or Var[log w]/site <= 0.012; NULL iff within the
    # anchor's spread — the d/8 growth then stays a 4x4-to-d64 result
    # and expressivity is struck from the d256 residue list alongside
    # the other closed doors.
    # VENUE (2026-08-19 s37): pulled off the DoC a100 queue and run on
    # Modal A100-80GB instead (tag 20260819-215307, seed 42). Not a scientific
    # change -- `train_remote` pins the same 80GB A100 class the anchors
    # ran on -- but a scheduling one: the anchor shape (this cell minus the rank change) measured
    # 3h17m end-to-end against a three-day queue wait. Bands above unchanged, frozen
    # before either launch.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_rank32": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_rank32",
            estimator="naive_mc",
        ),
        bilinear_rank=32,
    ),
    # Third causal ordering at the anchor recipe (2026-08-19, user GO):
    # site_orderings ("row","col") -> ("row","col","diag"), the ONLY
    # change. Why this axis is live: at 8x8, adding the SECOND ordering
    # took the factorised rung from raw ESS/N 0.514 to 0.745 — the
    # largest single expressivity gain on the head's record — by
    # shrinking the deep-interior blind region to the intersection of the
    # per-ordering (prefix, suffix) intervals; a third, diagonal sweep
    # shrinks that intersection further. Deliberately NOT run at 8x8
    # first: the dual-ordering rung already sits 0.036 under its
    # masked-attention twin there, inside the verdict noise band, so a
    # third ordering has nothing measurable to buy at that size — the
    # same near-ceiling logic that made small-lattice rank arms
    # uninformative. 16x16 is where interior coverage could still bind;
    # this is the orderings-axis sibling of the rank-32 arm above, one
    # variable each against the same anchors. FROZEN BANDS (seed 42, vs
    # the anchor pair EMA eval ESS/N 0.0048/0.0058, Var[log w]/site
    # 0.0168/0.0159): frame check stage-1 tail FVU <= 0.05; ORDERINGS
    # BIND iff EMA eval ESS/N >= 0.010 with bootstrap-CI separation, or
    # Var[log w]/site <= 0.012; NULL iff within the anchor spread —
    # interior coverage then joins rank on the closed expressivity list.
    # CANCELLED 2026-08-19 (s37, user GO) before it started, on SLOT
    # TRIAGE and not on evidence -- the distinction matters and is
    # recorded deliberately. The d64 diag arm did land FLAT (EMA 0.8069
    # vs the rung's 0.8104, Var/site CIs overlapping), but this cell's own
    # reasoning above pre-declared exactly that read uninformative ("a
    # third ordering has nothing measurable to buy at that size"), and the
    # in-flight ledger licensed a cancel only on a d64 REGRESSION, which
    # did not occur. What actually decided it: the a100 queue was the
    # binding resource against the 8 Sept deadline, the neighbouring
    # expressivity axis is already covered by the rank-32 arm on the same
    # anchors, and the full-horizon capacity arm had just read a
    # REGRESSION. The question is NOT answered and may be relaunched.
    # RELAUNCHED the same evening on Modal A100-80GB (tag
    # 20260819-214730, seed 42), once the cost was measured rather than
    # assumed: the anchor cell -- this cell's shape minus the third
    # ordering -- runs 3h17m end-to-end on that hardware, so the a100
    # QUEUE was the real cost, not the compute, and Modal has no queue.
    # The extra ordering extends only the bilinear term, bounding the
    # multiplier under 1.5x. BANDS ABOVE ARE UNCHANGED and were frozen
    # before either launch; nothing about the read moves with the
    # hardware, which is the same A100-80GB class the anchor pair ran on
    # (modal_app.train_remote pins gpu="A100-80GB").
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_diag": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_diag",
            estimator="naive_mc",
        ),
        site_orderings=("row", "col", "diag"),
    ),
    # Capacity arm at the anchor recipe (2026-08-19, user GO — the held
    # full-horizon capacity read): the fmo2 naive ladder anchor with
    # hidden_dim 32 -> 128 AND flat lr 3e-4 across all curriculum stages.
    # Two declared fields, bundled deliberately: the 5k screen showed
    # un-retuned lr 1e-3 penalises h128 specifically across sigma
    # transitions (h128 tail FVU 0.0498 vs h128+lr03's 0.0380-0.0406 ~=
    # base), so a pure-capacity arm at the ladder's early lr would return
    # an lr artefact, not a capacity verdict — the same lesson that made
    # every screen capacity arm ship with an lr03 sibling. Capacity
    # context: at hidden 128 the factorised model is ~1.41M params, ~4.4x
    # the ViT the MDNS paper trains on 16x16 Ising (~318k), so a NULL
    # here also closes "the comparison family simply used a bigger
    # model". Cost measured, not guessed: fmo2 h128 inner steps run ~7%
    # slower than h32 (87.7 vs 82.9 ms on the screen's A100s), so the 50k
    # anchor-shaped run is ~3.5 h — cheap because the factorised head's
    # cost is dominated by d, not hidden width. FROZEN BANDS (before
    # launch, seed 42, vs the judged anchor pair EMA eval ESS/N
    # 0.0048/0.0058, Var[log w]/site 0.0168/0.0159): frame check —
    # stage-1 tail FVU <= 0.05 (the anchor transfer band; a frame break
    # voids the read); CAPACITY BINDS iff EMA eval ESS/N >= 0.010 (2x the
    # seed-42 anchor) with bootstrap-CI separation, or Var[log w]/site
    # <= 0.012; NULL iff within the anchor spread — capacity then joins
    # the closed doors at FULL horizon, not just the 5k screen.
    "H2_d256_c50_s223_letf_fmo2_h128_lr03_50k_curr_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128),
            curriculum=replace(
                _cell.curriculum,
                stages=tuple(
                    replace(stage, lr=3e-4)
                    for stage in _cell.curriculum.stages
                ),
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128_lr03_50k_curr_naive",
            estimator="naive_mc",
        )
    ),
    # ---- 8x8 outer/inner loop battery (2026-08-19) ----------------------
    # Eight single-variable twins of the archived fmo2 8x8 curriculum rung,
    # built from _d64_fmo2_loop_cell (its docstring carries the shared
    # frame: comparator raw ESS/N 0.7452 / EMA 0.8104 seed 42, and the
    # band convention INSENSITIVE = within +-0.03 EMA ESS/N of 0.8104).
    # The loop knobs opened here have been constants across essentially
    # the whole archive — varied about as often as hidden_dim once was —
    # so their scaling rules rest on provenance, not measurement. Key
    # shared quantity: per-state draw density = expected gradient draws
    # per buffer state over its lifetime = inner_steps x batch /
    # (euler_grid x rollout_width); the archived 8x8 shape sits at 0.78,
    # the reference-code default, while the 16x16 recipe that landed NULL
    # silently ran at 0.195 (batch and grid quadrupled, inner steps not).
    # Read on every arm: EMA + raw eval ESS/N with top-weight mass,
    # Var[log w]/site, per-rung train-ESS medians, stage-tail FVU.
    #
    # Gradient steps per outer cycle 100 -> 25: draw density 0.78 -> 0.195,
    # the exact broken density of the NULL 16x16 recipe, reproduced at the
    # healthy size with everything else untouched. Does breaking the
    # density invariant itself cost anything where training is otherwise
    # known-good? ONE-SIDED by construction, declared before launch: at
    # fixed n_steps, 4x more outer cycles also means 4x more fresh
    # trajectories, biasing this arm toward reading insensitive — a
    # SENSITIVE read is therefore strong evidence, an INSENSITIVE read
    # must carry the caveat. The inner500 cell below is the opposite
    # direction; together they bracket ~two decades of density around
    # 0.78. FROZEN BANDS: INSENSITIVE within +-0.03 EMA ESS/N of 0.8104;
    # SENSITIVE-UP >= +0.03 means the big-lattice recipe under-rolled per
    # state (fresh rollouts, not density per se, were the binding margin).
    "H2_d64_c50_s223_letf_fmo2_50k_curr_inner25": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_inner25",
        inner_steps_per_outer=25,
    ),
    # Replay depth 8 -> 2 cycles: 1024 -> 256 retained trajectories,
    # shrinking below the ~1024-trajectory retention bound the reference
    # code enforces. The one prior variation (a 12k masked-attention smoke
    # at this size) read WORSE at every rung — evidence that fresher-but-
    # fewer states hurt at 8x8 — but it confounded staleness with buffer
    # size and never ran the full horizon or the factorised family the
    # larger rungs use. Draw density is invariant to depth (cycle count
    # cancels), so depth is a pure staleness-vs-diversity dial. FROZEN
    # BANDS: SENSITIVE-DOWN <= -0.03 EMA ESS/N replicates the smoke's
    # direction on this family at full horizon; INSENSITIVE +-0.03. Read
    # alongside: the buffer is flushed at every sigma boundary and a
    # 2-cycle buffer refills 4x sooner, so rung-boundary train-ESS
    # recovery speed is the mechanism read, not just the endpoint.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_buf2": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_buf2",
        replay_buffer_cycles=2,
    ),
    # Stepping-protocol control: use_matching_step False -> True at the
    # unchanged 128-step grid. Every archived 8x8 run used one-event
    # stepping and every 12x12/16x16 run the vertex-disjoint matching
    # step, so no cross-size comparison has ever held the protocol fixed;
    # this cell retires that caveat with a measured number. Expected
    # INSENSITIVE (a frozen 8x8 checkpoint re-evaluated under the matching
    # step read 0.906 vs 0.910). Also the required comparator for the two
    # grid cells below: one-event stepping clips whenever total-rate x dt
    # exceeds 1, so a coarser grid under one-event stepping would confound
    # grid resolution with clipping — only under the matching step is the
    # grid a clean variable. FROZEN BAND: INSENSITIVE +-0.03 EMA ESS/N
    # of 0.8104.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match",
        use_matching_step=True,
    ),
    # Euler grid 128 -> 32 under the matching step: 0.5 x sites, the
    # grid-to-size ratio every archived 16x16 sigma_c run trained at. Read
    # against the matching-step control cell above, NOT the base — the
    # protocol change rides in both, leaving the grid the only difference.
    # This is the TRAINING-side grid question: the eval-only resolution
    # sweep on a frozen checkpoint cannot answer it, because a model
    # trained on a coarse grid learns to compensate that grid's bias.
    # Declared side effects, recorded before launch: (i) draw density
    # rises to 3.1 (fewer buffer rows per cycle at fixed inner steps),
    # biasing this arm AGAINST finding harm — a drop is strong evidence
    # for a grid-tracks-size rule, flat must carry the caveat; (ii) the
    # eval sampling grid moves with the training grid (the measured
    # eval-grid share is ~0.02 ESS/N, a rounding term at this comparator,
    # recorded not ignored). FROZEN BAND vs the matching control:
    # INSENSITIVE +-0.03 EMA ESS/N.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne32": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne32",
        use_matching_step=True,
        n_euler_steps=32,
    ),
    # Euler grid 128 -> 256 under the matching step: 2 x sites -> 4 x
    # sites, the finer-grid direction. With the matching control and the
    # 32-step cell this completes a three-point training-grid curve at
    # sigma_c (0.5d / 2d / 4d) — the first anywhere in the archive, which
    # contains no size at which the training grid was an isolated, tested
    # variable at sigma_c. A monotone rise crossing +0.03 EMA ESS/N =
    # training was grid-limited and the grid should track size; flat
    # closes the training-side grid door with a measurement. Same
    # eval-grid-moves-with-training-grid note as the 32-step cell. FROZEN
    # BAND vs the matching control: INSENSITIVE +-0.03 EMA ESS/N.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne256": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne256",
        use_matching_step=True,
        n_euler_steps=256,
    ),
    # Rollout/buffer width decoupled from the gradient batch: outer batch
    # None -> 32 with c_t_batch=128 — the FIRST run ever to set the
    # rollout width below the gradient batch. Three sample sizes that have
    # always been one number are separated: the buffer width falls 4x (32
    # trajectories per cycle, so rollout compute falls ~4x), the per-slot
    # normaliser c_t keeps its effective sample size at the base's 128
    # because c_t_batch pins the rollout row count there (the buffer takes
    # the first 32 rows; the trainer validates c_t_batch >= outer batch,
    # 128 >= 32), and the gradient batch stays 128. What remains isolated
    # is buffer width/diversity alone. If INSENSITIVE, rollout cost at any
    # size can be cut ~4x for free — the single biggest cost lever this
    # battery could license (rollouts were ~77% of wall on the widest
    # 16x16 recipe). FROZEN BANDS: INSENSITIVE +-0.03 EMA ESS/N licenses
    # narrow buffers at scale; SENSITIVE-DOWN <= -0.03 says buffer
    # diversity binds and retention should be restated in trajectories,
    # not cycles. Mechanism pinned tests-first in
    # tests/test_outer_batch_decoupling.py before this cell runs.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_m32ct128": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_m32ct128",
        outer_batch_size=32,
        c_t_batch=128,
    ),
    # Gradient steps per outer cycle 100 -> 500: draw density 3.9, fresh
    # trajectories 5x down — the over-reuse direction, paired with the
    # inner25 cell to bracket the archived 0.78. Insensitivity here
    # licenses cheap high-reuse recipes (rollout share of wall falls to
    # ~15% at this shape); a drop is the first direct evidence that
    # re-fitting the same finite buffer sample degrades the sampler. 500
    # rather than 400 because the curriculum validator requires every
    # sigma-stage start (multiples of 5000 on this ladder) to be a
    # multiple of the per-cycle inner-step count, and 400 does not divide
    # 5000. FROZEN BAND: INSENSITIVE +-0.03 EMA ESS/N of 0.8104.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_inner500": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_inner500",
        inner_steps_per_outer=500,
    ),
    # Replay depth 8 -> 16 cycles: 2048 retained trajectories, the deep
    # half that has never been tried at any size (the archive holds one
    # shallow variation and nothing above 8). With the buf2 cell this
    # completes the depth curve 256/1024/2048 around the reference-code
    # retention bound of ~1024. Depth is compute-free and memory-trivial
    # at this size, so the arm is pure information: any |delta| >= 0.03
    # EMA ESS/N in either direction revises the retention default;
    # INSENSITIVE keeps ~1024 trajectories as the defensible default it
    # currently is (provenance plus one confounded shallow smoke).
    "H2_d64_c50_s223_letf_fmo2_50k_curr_cyc16": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_cyc16",
        replay_buffer_cycles=16,
    ),
    # SMC-in-training arms (2026-08-20): first LIVE runs of
    # rollout_resample_ess_fraction on the swap route — ESS-triggered
    # resampling inside the buffer-rebuild rollout (LEAPS Alg. 1 lines
    # 11-14; the full c_t argument lives on the baseline TrainCfg field).
    # The d256 estimator wall (eval ESS/N 0.003) is the motivating target;
    # per the validation order d256 spends nothing until d64 shows a
    # signal. The builder makes each arm a true twin with tau the one
    # declared variable, and flag-off bit-identity is test-pinned, so the
    # archived rung comparator remains the valid control — no fresh
    # control run. The ESS endpoint is near-ceiling here (rung raw 0.745),
    # so per the diag-arm convention the read is NOT ESS-first. FROZEN
    # BANDS (seed 42, per arm): PRIMARY = EMA eval Var[log w] against the
    # archived rung's 0.2068, bootstrap 95% CI (0.1983, 0.2152); UPLIFT
    # iff the arm's CI sits wholly BELOW (0.1983, ...) — the variance-axis
    # signal that alone licenses a d256 twin; REGRESSION iff EMA eval
    # ESS/N <= 0.78 or the Var CI sits wholly above; INSENSITIVE
    # otherwise. Tripwire: rollout_resample_events ~ 0 beyond the first
    # sigma stage at BOTH taus = VACUOUS-AT-TAU, a trigger-calibration
    # finding (the healthy-d64 rollout may simply never dip under tau*M —
    # itself worth knowing before pricing a d256 arm, where it WILL fire).
    "H2_d64_c50_s223_letf_fmo2_50k_curr_smc03": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_smc03",
        rollout_resample_ess_fraction=0.3,
    ),
    "H2_d64_c50_s223_letf_fmo2_50k_curr_smc06": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_smc06",
        rollout_resample_ess_fraction=0.6,
    ),
    # Rollout width 128 -> 512 at fixed gradient batch 128, c_t riding at
    # 512 (the c_t_batch >= outer_batch constraint holding at equality):
    # the upward direction of the width decoupling. The reference
    # implementation itself rolled 2x its gradient batch; every run in
    # this repo has rolled 1x. Buys 4x fresh trajectories and 4x c_t
    # sample size per cycle at 4x rollout cost. DECLARED CONFOUND: at
    # fixed inner steps the per-state draw density falls 0.78 -> 0.195 as
    # a side effect (the same drop the 16x16 recipe made); the inner25
    # cell isolates density alone, so the difference between this arm and
    # inner25 is width-at-matched-density. FROZEN BAND: INSENSITIVE
    # +-0.03 EMA ESS/N of 0.8104; SENSITIVE-UP >= +0.03 says rollout
    # width was a starved axis and a wide-M d256 twin is licensed;
    # SENSITIVE-DOWN with inner25 also down says density, not width.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_m512ct512": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_m512ct512",
        outer_batch_size=512,
        c_t_batch=512,
    ),
    # Third causal ordering at 8x8 (2026-08-19, user call — the cheap
    # validation BEFORE the 16x16 sibling spends a100 hours on an untested
    # mechanism): site_orderings ("row","col") -> ("row","col","diag"),
    # the ONLY change vs the archived rung. The ESS endpoint is
    # near-ceiling here (rung raw 0.745 vs its masked-attention twin's
    # 0.781), so the read is deliberately NOT ESS-first: the primary
    # statistic is the EMA eval Var[log w], whose archived rung value is
    # 0.2068 with bootstrap 95% CI (0.1983, 0.2152) — variance CIs
    # resolve differences the ESS band cannot. FROZEN BANDS (seed 42):
    # UPLIFT DETECTED iff the arm's EMA Var[log w] CI sits wholly BELOW
    # (0.1983, ...), i.e. CI-disjoint downward; REGRESSION iff EMA eval
    # ESS/N <= 0.78 or the Var CI sits wholly above — a regression is the
    # early-warning verdict and the queued 16x16 third-ordering arm
    # should be cancelled before it runs; INSENSITIVE otherwise = the
    # orderings axis is saturated at two sweeps at this size and the
    # 16x16 arm carries the question alone.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_diag": replace(
        _d64_fmo2_loop_cell("H2_d64_c50_s223_letf_fmo2_50k_curr_diag"),
        site_orderings=("row", "col", "diag"),
    ),
    # Boundary shock at 8x8 (2026-08-19): the size-twin of the 16x16
    # `_rw` arm below, registered so the boundary question is answered in
    # hours at the size where training is known-healthy rather than only
    # on a day-long a100 run. Two changes off the rung and nothing else:
    # rewarmup_on_stage=True (a fresh 500-step LR ramp anchored at each of
    # the six sigma boundaries, so the transient — buffer flush plus a
    # target jump — is entered at a damped LR instead of full stride) and
    # the per-stage best-checkpoint instrument.
    #
    # ONE VARIABLE, verified in the trainer rather than inherited from the
    # sibling: the stage-best block sits inside the existing
    # `step % eval_every == 0` branch AFTER the eval that already runs
    # unconditionally, reads the `ess_value` that branch already computed
    # for the CSV, and does nothing but `statistics.median`,
    # `torch.save(head.state_dict())` and a JSON write. It draws no
    # samples, so it consumes no RNG and the next training step sees an
    # unchanged stream; it touches neither head, optimiser nor EMA; and
    # `wall_clock_step_s` is stamped before the eval branch, so even the
    # timing column is unperturbed. rewarmup_on_stage is therefore the
    # sole training-dynamics change.
    #
    # MECHANISM BAND (primary read; deliberately NOT the dead
    # peak-to-final-decay trigger, which did not survive a binned re-read
    # at 16x16). Statistic: FVU = loss / var_dt_log_p_tilde, per-step in
    # training_log.csv. Per boundary, "spike" means the MEDIAN of FVU over
    # the first 500 post-boundary steps divided by the pre-boundary stage
    # tail median, and "recovery" means the first step whose trailing-100
    # median falls within 10% of the new stage's tail median. The median,
    # not the max: on the two landed 16x16 parent seeds the per-boundary
    # PEAK varies 0.68x-3.49x seed-to-seed on an identical recipe while
    # the median-of-500 stays inside 0.82-1.22, so a peak-based band is
    # mostly reading seed noise. Any dip statistic is derived from FVU
    # too, never from `ess`: that column is written only every
    # eval_every=200 steps from 512 draws, so a 500-step window holds two
    # or three low-N points and cannot carry a verdict. (Raising the eval
    # cadence would fix that but is NOT free here — the in-training eval
    # draws through `target.sample_base` / `sample_swap_ctmc` with no RNG
    # save-restore around them, unlike the step-0 init diagnostic, so
    # eval_every is itself a training-dynamics field and moving it would
    # cost the twin.)
    #
    # SHOCK-DAMPED iff the spike ratio is reduced vs the rung at >= 4 of 6
    # boundaries AND recovery is faster wherever the rung took > 1k steps.
    # Recorded now, before launch, so the verdict is read at its true
    # strength: as a sign test that clause fires at 0.34 under the null;
    # the last boundaries carry little signal at 16x16 (the parent's
    # first-500 FVU is already at or below the stage tail there), so check
    # rather than assume how many of the six are informative at 8x8; and
    # the six boundaries are ONE trajectory, not six replicates — the arm
    # and the rung are configuration-identical only up to step 5,000, so
    # BOUNDARY 1 IS THE ONLY CONTROLLED COMPARISON and boundaries 2-6 are
    # descriptive. Pre-registered secondary, carrying the weight the sign
    # test cannot: the paired median-FVU-over-500 at the informative
    # boundaries, which must clear the 0.82-1.22 seed-to-seed spread
    # measured on the 16x16 parent pair (BORROWED noise scale — the 8x8
    # fmo2 comparator is single-seed, so it is unverified at this size).
    # Tie-break for the case the band is otherwise silent on — a smaller
    # spike but slower recovery, which a damped LR can plausibly produce
    # both halves of — is the stage-tail FVU (median of each stage's last
    # 1,000 steps), the quantity that actually survives to the endpoint.
    # NO-SHOCK iff the rung's own spikes are already < 1.5x and recover
    # within 500 steps at >= 4 of 6 boundaries, which would be the
    # informative negative: it localises the d256 boundary shock as a
    # SCALE effect (initialisation-scale gradients at 16x16, absent at
    # 8x8) rather than a property of the curriculum shape.
    #
    # PRIMARY ENDPOINT STATISTIC: EMA eval Var[log w], because ESS/N is
    # near its ceiling at this size and cannot resolve an improvement.
    # Rung reference 0.2068, bootstrap 95% CI (0.1983, 0.2152) —
    # population variance of the frozen 5000-draw EMA eval log-weights,
    # 2000-resample percentile bootstrap. UPLIFT iff the arm's CI sits
    # wholly BELOW that CI (disjoint downward); REGRESSION iff wholly
    # above. ENDPOINT READ ALONGSIDE (standard 8x8 convention):
    # INSENSITIVE iff EMA eval ESS/N is within +-0.03 of 0.8104. Bootstrap
    # 95% CIs on any near-band read.
    #
    # WHAT IT BUYS AT NULL: the boundary transient gets a measured price at
    # a healthy size for the first time, which is what makes the 16x16
    # sibling's read attributable — a live shock there against a null here
    # is a scale claim, whereas a null in both retires the boundary axis
    # for the whole ladder. The stage-best instrument separately gives the
    # first checkpoint-SELECTION read at a size where final.pt is known
    # good: if the sigma_c stage-best does not beat final.pt here but does
    # at 16x16, selection value is scale-tied too. Judged by an eval-only
    # pass on best_stage6.pt vs final.pt at the full 5,000-draw frozen
    # eval WITH a bootstrap CI, and a positive gap called only if it
    # clears that CI width — the rule picks the maximum over ~25 trailing
    # medians per 5,000-step stage at eval_every=200, and a best-of-many
    # maximum over a flat series carries upward selection bias that the
    # median-of-3 damps but does not remove.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_rw": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_rw",
        rewarmup_on_stage=True,
        stage_best_checkpoints=True,
    ),
    # No-flush at 8x8 (Tier 3(c), 2026-08-19): the rung with
    # flush_replay_on_stage False the ONLY change — the 8-cycle replay
    # window is allowed to carry states ACROSS a sigma boundary instead of
    # being emptied there. Depth stays 8, so this is not the buf2/cyc16
    # axis: the single variable is whether the retention window is
    # truncated at boundaries.
    #
    # REGISTERED, NOT QUEUED. The A5/no-flush decision memo (2026-08-19)
    # recommends closing this item with a derived reason instead of
    # spending the run; the argument is summarised below and the cell is
    # registered so the reopen condition has something to launch. Reopen
    # condition, recorded before any launch: the 16x16 `_rw` arm reads
    # NULL on its MECHANISM band — i.e. damping the LR does not touch the
    # boundary transient, leaving data starvation as the surviving
    # suspect. And if it is reopened, prefer the clean form in the
    # confound paragraph below over this one.
    #
    # MECHANISM, worked out against the loss as implemented rather than
    # assumed. loss_swap squares delta_t(x) = dt_log_p_tilde_t(x) - c_t +
    # sum_{i<j}([G]_+ - [-G]_+ exp(log ratio)), and BOTH target terms are
    # recomputed from `target` at the live sigma every gradient step
    # (`target.set_sigma` fires before the buffer is used). A retained
    # state is an evaluation point, not a stale label, so "the buffer
    # teaches the previous target" is FALSE in the label sense: nothing
    # carried across the boundary encodes the old sigma. The objective's
    # zero is delta_t(x) = 0 pointwise on the support of whatever law the
    # states are drawn from, and old-sigma states live on the same
    # fixed-composition manifold where the Gibbs target is strictly
    # positive at every sigma — support is shared exactly, so the
    # minimiser does not move and no-flush introduces no asymptotic bias.
    #
    # Two real effects remain. (i) COVERAGE, not bias: at finite capacity
    # and finite samples the loss is a weighted least-squares, and stale
    # states weight it toward where the PREVIOUS stage's model law put
    # mass; the plateaus here are small (0.100 -> 0.140 -> ... -> 0.223)
    # so overlap is high, and the stale fraction is evicted after exactly
    # 8 cycles = 800 inner steps. (ii) c_t OFFSET: c_t is estimated on the
    # current cycle's FRESH rollout while the loss averages over a buffer
    # mixing pre- and post-boundary states, so Delta_t = E_buffer[xi] - c_t
    # picks up a cross-sigma mismatch for those 800 steps. c_t reaches the
    # gradient ONLY through Delta_t, and as a uniform level shift of xi
    # rather than a distortion of its x-dependence — and the fixed point
    # re-pins that constant to dt log Z_t. So the damage is a bounded,
    # transient, self-healing offset, fully visible in `c_t_offset_rms`.
    # c_t itself is never biased: it never sees a retained state.
    #
    # DECLARED CONFOUND, and the reason a reopened arm should not take
    # this form. The consistent version of no-flush also moves c_t onto
    # the buffer, because the theta-step and the c_t-step are supposed to
    # average over the SAME reference law; leaving c_t on the fresh
    # rollout widens exactly that gap. So a NULL from this cell is
    # attributable to either half — retention itself, or the c_t measure
    # mismatch it induces — and only a c_t-over-buffer variant separates
    # them, at ~8x the c_t forward cost or on a subsample.
    #
    # STANDING EVIDENCE AGAINST, measured on the 16x16 parent logs before
    # this cell was written. If the flush drove the post-boundary
    # transient, recovery would be set by the constant 800-step refill
    # window. It is not: across the two landed seeds, per-boundary
    # recovery spans 100-3,057 steps and falls MONOTONICALLY down the
    # ladder in both, tracking the shrinking sigma jump (+40% at the first
    # boundary, +3.7% at the last); five of twelve boundary events recover
    # FASTER than the buffer refills; and the first-500-step excess FVU
    # area is zero or negative at the last two boundaries, including the
    # one into the sigma_c plateau the endpoint model is downstream of.
    # Upper bound on any benefit at 16x16: 6 x 800 = 4,800 of 50,000 steps
    # (9.6%), only 800 of them inside the 20,000-step sigma_c plateau.
    # NOT evidence in either direction: the DNFS reference never flushes
    # because it has no curriculum, so it never faced this choice.
    #
    # PRIMARY STATISTIC: EMA eval Var[log w] (ESS/N is near-ceiling at
    # this size). Rung reference 0.2068, bootstrap 95% CI (0.1983,
    # 0.2152) — population variance of the frozen 5000-draw EMA eval
    # log-weights, 2000-resample percentile bootstrap. UPLIFT iff the
    # arm's CI is disjoint BELOW; REGRESSION iff disjoint above.
    # ENDPOINT ALONGSIDE: INSENSITIVE iff EMA eval ESS/N within +-0.03 of
    # 0.8104. Bootstrap 95% CI on any near-band read. Given the evidence
    # above, INSENSITIVE is the PRE-REGISTERED EXPECTATION, not a
    # disappointment.
    # MECHANISM BAND, both directions declared before launch, on the
    # per-step columns: no-flush should SHORTEN post-boundary recovery
    # (steps for FVU = loss / var_dt_log_p_tilde to return within 10% of
    # the new stage's tail median) at >= 4 of 6 boundaries, and should
    # RAISE `c_t_offset_rms` over the first 800 steps after each boundary.
    # That rise is the predicted price and is the arm's real deliverable:
    # it is the only direct measurement of how far the loss measure and
    # the c_t measure may drift apart before the endpoint notices. If
    # neither column moves, the boundary is not buffer-limited and the
    # flush is a free convention. Same one-trajectory caveat as the `_rw`
    # arm: boundary 1 is the only controlled comparison against the rung.
    #
    # WHAT IT BUYS AT NULL: a never-varied convention stops being a
    # convention, with a number rather than a derivation behind it, and
    # the c_t_offset_rms profile prices the measure mismatch for any
    # future arm that wants to reuse states across a moving target
    # (SMC tempering, the c-conditioned route). A REGRESSION would be the
    # first direct evidence that the loss's sampling MEASURE — not just
    # its sample size — matters, which is the same axis the buf2/cyc16
    # depth arms probe from the other side.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_noflush": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_noflush",
        flush_replay_on_stage=False,
    ),
    # Boundary-shock arm (2026-08-19, user GO on corrected evidence): the
    # recipe cell with rewarmup_on_stage=True the ONLY training-dynamics
    # change — a fresh 500-step LR ramp from every sigma boundary, so the
    # boundary transient (buffer flush + target jump at initialisation-scale
    # gradients) is entered at a damped LR instead of full stride. The
    # per-stage best-checkpoint instrument rides as pure IO (trailing
    # median-of-3 train-eval ESS; see the TrainCfg field comment), so the
    # arm stays one-variable. EVIDENCE HONESTY, recorded before launch: the
    # original trigger (a x2.2-2.4 peak-to-final train-ESS decay in the
    # landed recipe's sigma_c stage) did NOT survive a binned re-read —
    # within-stage medians are flat-to-rising and the peak is a noise
    # excursion — so the live mechanism is the TRANSIENT COST at the six
    # boundaries (deep dips, 0-6.5k-step recoveries), not late decay.
    # FROZEN BANDS (seed 43, vs the landed parent's EMA eval ESS/N 0.0198
    # and its per-stage boundary profile): mechanism — first-500-step
    # post-boundary FVU spike and dip depth reduced vs the parent at >= 4
    # of 6 boundaries, and recovery-to-stage-median faster where the parent
    # took > 1k steps; endpoint — CONFIRMED iff EMA eval ESS/N >= 0.04 (2x
    # parent) with bootstrap-CI separation; NULL iff within the parent's
    # CI (expected under the corrected trigger; the arm then still buys the
    # measured boundary-transient cost + the stage-best instrument's
    # checkpoint-selection read, judged by an eval-only pass on the
    # sigma_c stage-best vs final).
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw",
            estimator="naive_mc", n_euler_steps=512, batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw",
                estimator="naive_mc", n_euler_steps=512, batch_size=512,
                loss_microbatch_size=128,
            ).train,
            rewarmup_on_stage=True,
            stage_best_checkpoints=True,
        ),
    ),
    # Phase-2 estimator switch (2026-08-14, user GO): CONTINUE the completed
    # naive 50k (launch with --init-from <naive run>/checkpoints/final.pt)
    # with the Stein control variate re-enabled. Mechanism, measured on the
    # trained naive checkpoint (CPU rollout harness that reproduces the d64
    # runs' logged CV ratios): the CV that ADDED 2.3-70x variance on the
    # DIVERGED model now REDUCES the c_t integrand variance 7x whole-run
    # (ratio 0.140; per-t-quarter 0.031/0.079/0.150/0.553) -- xi at the
    # optimum is a zero-variance constant, so the ratio improves further as
    # the model sharpens. Flat sigma_c (the ladder was already climbed),
    # lr pinned to the ladder's final 3e-4 with the standard short warmup,
    # dual-eval EMA rides. Bands FROZEN BEFORE LAUNCH: PASS = train-ESS
    # > 15/256 sustained AND rising over the final 5k steps, plus final
    # eval ess_fraction >= 0.05 (Var[log w] <= ~3.0); >= 0.20 = strong
    # pass; train-ESS flat in single digits = the estimator lever alone is
    # insufficient at this size and the next lever is rollout count.
    "H2_d256_c50_s223_letf_ma_20k_sc_cv2": _d256_cv2_cell(
        "H2_d256_c50_s223_letf_ma_20k_sc_cv2"
    ),
    # Clip-50 twin of the cell above (job 271892 diverged 2026-08-10). The
    # inherited clip of 500 was exceeded twelve-fold from initialisation at
    # d=256, so every step ran at the rescale-not-skip ceiling and the norm
    # escalated to ~2e5 without ever falling back — the same optimiser runaway
    # the soft chapter's amortised cells died of, where clip=50 took survival
    # from 1/4 to 4/4. Health criterion, stated before launch: success is the
    # pre-clip norm FALLING BACK between curriculum rungs and train ESS
    # climbing off ~1/128; a norm pinned at the threshold with flat ESS is the
    # runaway persisting, and the next move is a normalised/trust-region
    # update, not another threshold. Everything else identical to the twin.
    # RETIRED BEFORE LAUNCH (2026-08-11, adversarial-review verdict): with
    # AdamW, two thresholds that BOTH saturate every step produce gradient
    # sequences differing by a constant factor, which Adam's per-parameter
    # normalisation erases -- and the diverged twin's minimum pre-clip norm
    # over all 50k steps was 186, so clip=50 saturates always and would
    # near-exactly retrace the clip=500 trajectory. Kept as the record of a
    # rejected arm; superseded by the smoke12k ladder below. Do not launch.
    "H2_d256_c50_s223_letf_ma_50k_curr_clip50": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr_clip50", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        grad_clip_max_norm=50.0,
    ),
    # ---- 16x16 rescue smoke ladder (2026-08-11) -------------------------
    # Five 12k-step arms, ONE mechanism each, launched CONCURRENTLY after
    # the four-lens adversarial review of the diverged rung. 12k crosses the
    # sigma=0.14 (5k) and sigma=0.17 (10k) boundaries -- 0.17 is where the
    # diverged twin's per-rung re-ignition began, so every arm is scored on
    # (i) escaping the clip ceiling on rung 0 (grad_norm below ~500-scale by
    # step ~3k; the twin managed this once, loss 392->11) and (ii) surviving
    # the 10k transition (within-rung loss slope <= 0 on 10k-12k; the twin's
    # rose on every rung past the first). Pre-stated per-arm criteria live in
    # the launch plan; identical eval cadence keeps columns comparable.
    # 2026-08-12: the first wave (jobs 272749-53) all died at startup --
    # the cells carried the full 50k ladder, whose 15k stage trips the
    # strict start_step < n_steps validator at n_steps=12k. The smokes only
    # ever reach the first three stages, so they now share this truncated
    # view of the SAME ladder (identical sigmas/lrs/boundaries at 0/5k/10k).
    "H2_d256_smoke12k_unclip": _hard_cell(
        # Arm A: restore clip=500's d64 SEMANTICS (fires on spikes only) by
        # raising the threshold above the working-regime norm; under AdamW
        # this is also exactly the per-pair-normalised-loss arm (the two
        # differ by a constant the optimiser erases).
        "H2_d256_smoke12k_unclip", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        grad_clip_max_norm=20_000.0,
    ),
    "H2_d256_smoke12k_naive": _hard_cell(
        # Arm B: kill the inverted control variate (it ADDS variance at
        # d=256: integrand/naive variance ratio 2.3x at rung 0 -> ~70x late,
        # vs an 8-30x REDUCTION at d64) by estimating c_t naively. Also the
        # cheapest arm (~0.7x: skips the c_t head pass).
        "H2_d256_smoke12k_naive", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        estimator="naive_mc",
    ),
    "H2_d256_smoke12k_warm": _hard_cell(
        # Arm C: warm-start from the converged d64 ma_100k checkpoint
        # (110/116 keys shape-identical; positional tables bilinearly
        # interpolated by scripts/warm_start_swap_head.py, supplied via
        # run.py --init-from). Attacks the init-scale term directly: 96% of
        # the d256 init loss is the head's own coherent pair-sum noise.
        # Recipe otherwise UNCHANGED (clip 500) -- if this arm alone
        # escapes, initialisation was the story.
        "H2_d256_smoke12k_warm", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
    ),
    "H2_d256_smoke12k_stadamw": _hard_cell(
        # Arm D: StableAdamW -- per-tensor UPDATE clipping (unit-free,
        # size-invariant trust region), the "normalised or trust-region
        # update" the soft chapter's clip forensics already recommend in
        # print. Raw-gradient clip effectively disabled so the update
        # clipping is the only bounding mechanism (one variable per arm).
        "H2_d256_smoke12k_stadamw", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        grad_clip_max_norm=1e9,
        optimiser="stable_adamw",
    ),
    "H2_d256_smoke12k_rewarmup": _hard_cell(
        # Arm E: re-run the lr warmup ramp at every sigma transition. The
        # twin's one healthy window ended exactly at a transition (buffer
        # cleared + target jumped, no ramp); warmup was the only mechanism
        # that ever carried it through a transient. Clip left at the
        # inherited 500 to isolate the transition variable.
        "H2_d256_smoke12k_rewarmup", sigma=0.223,
        head_kind="masked_attention",
        D=16, n_steps=12_000, n_euler_steps=128, n_eval_samples=1000,
        eval_sample_chunk=64, n_eval_samples_training=256, eval_every=500,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        rewarmup_on_stage=True,
    ),
    # --- M-campaign Task 6 (2026-08-14): replay-buffer staleness ablation ---
    # Inner steps draw from an 8-cycle replay buffer while c_t is FRESH from
    # the latest outer cycle (swap_training.py); staleness is worst in the
    # fast early curriculum. Config-only falsification: the d64 MA
    # curriculum recipe at the 12k smoke horizon with the buffer window
    # 8 -> 2 the ONLY declared change (twin-ness pinned by
    # test_m6_replay2_smoke_mirrors_ma_recipe_except_declared_fields), read
    # against the archived MA twin's first 12k steps. The ladder truncates
    # to the shared 12k view because the validator rejects stages at or past
    # n_steps (the 2026-08-12 smoke-wave lesson).
    # Frozen bands (plan Task 6): INSENSITIVE (staleness confound ruled
    # out) if final train-ESS is within +-10% relative of the archived MA
    # twin at matched steps; SENSITIVE (>10% either way) -> full 50k arm +
    # re-think the buffer window in the d256 recipe.
    "H2_d64_smoke12k_replay2": _d64_smoke12k_replay2_cell(
        "H2_d64_smoke12k_replay2",
    ),
    # --- M-campaign Task 2 (2026-08-14): c_t EMA no-regression gate -------
    # USER CALL (session close 2026-08-14): launch WITHOUT waiting for cv2's
    # band — the gate reads against the ARCHIVED MA twin, so cv2's outcome
    # changes nothing about its attribution. The archived MA 50k curriculum
    # recipe with c_t_ema_halflife_cycles=4 the ONLY change (sigma-transition
    # resets keep the EMA honest across the ladder). Frozen band (plan Task
    # 2): NO-REGRESSION = final eval ESS frac within the archived MA twin's
    # seed spread, i.e. >= 0.755 (spread 0.755/0.769/0.781); below that the
    # across-cycle smoothing is not free and the lever is d256-only-judged.
    # VERDICT (2026-08-14 slate, back-annotated 2026-08-19): NO-REGRESSION
    # met but the lever is NEUTRAL — 1.05x, inside the c_t-noise family's
    # measured ~1.19x transfer ceiling (input variance moves 28x, Var[log w]
    # moves 1.19x, log-log slope 0.052). The whole family is deprioritised
    # by measurement, not argument.
    "H2_d64_c50_s223_letf_ma_50k_curr_ctema4": _d64_m2_ctema4_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr_ctema4",
    ),
    # --- c_t transfer function (2026-08-14): the missing naive arm -------
    "H2_d64_c50_s223_letf_ma_50k_curr_naive": _d64_naive_twin_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr_naive",
    ),
    # --- M-campaign Task 3 (2026-08-14): decoupled c_t rollout batch ------
    # Mechanism cell, same user call: d256-naive 12k smoke with c_t_batch
    # 128 -> 512 the ONLY change — c_t's per-slot standard error halves
    # (variance /4) while the inner batch and replay buffer stay at 128.
    # Frozen bands (plan Task 3), read vs the archived naive 50k run's first
    # 12k at matched steps: MEANINGFUL = train-ESS median on the sigma=0.17
    # rung (steps 10k-12k) >= 2x the archived run's, OR the logged per-slot
    # var_estimator_integrand down >= 2x. NEVER bundle with M2: one lever
    # per cell keeps attribution clean. Cost ~5 h a100 (trajectory phase
    # +2.25x on ~75% of wall); the no-grad MA pass at B=512 peaks ~20 GB,
    # in-cap on the 80 GB a100.
    # VERDICT (2026-08-14 slate, back-annotated 2026-08-19): NOT MEANINGFUL
    # — 1.09x against the >= 2x band; same ~1.19x c_t-noise transfer
    # ceiling as ctema4 above.
    "H2_d256_smoke12k_naive_ctb512": _d256_smoke12k_naive_ctb512_cell(
        "H2_d256_smoke12k_naive_ctb512",
    ),
    # d64 plumbing fallback for the cell above (~25 min a30): validates the
    # enlarged-rollout outer cycle end-to-end at small size FIRST if a100
    # scheduling blocks the d256 cell. No quality band — the unit tests
    # (test_c_t_batch.py) already pin the semantics; this is an integration
    # smoke.
    "H2_d64_smoke12k_ctb512": _d64_smoke12k_ctb512_cell(
        "H2_d64_smoke12k_ctb512",
    ),
    # RETIRED 2026-07-22 (user call, after batch 1 landed). Kept, not deleted:
    # these three cells are the only way to reproduce a NEGATIVE result the
    # writeup will cite, and the head + falsification suite they exercise stay
    # green at no run cost. Do NOT launch further ga cells at d64.
    #   ga8 0.58072 -- BELOW iv 0.6463, i.e. under the ladder floor, while
    #     costing MORE per inner step than its ma cost peer (28.8 vs 26.4 ms)
    #   ga16 0.71414 -- still below ma 0.7806, at 1.43x ma's step cost
    #   ga8_contig 0.00877 -- stable failure (trains at sigma=0.1, then cannot
    #     track the ladder to 0.223; flat loss ~5.14 for the final 20k)
    # Reading: k IS a real dial (ga8 -> ga16 = +0.133, in the expected
    # direction) but the whole family sits under the existing ladder at d64, so
    # no (k, cost) point here is worth having. Group SHAPE dominates k -- the
    # diagonal-vs-contiguous gap (0.572) is 4x the k gap, which vindicates the
    # independent-set argument far past what the original estimate predicted.
    # Grouped-anchor batch 1 (2026-07-22): the head family
    # sampled BETWEEN its endpoints for the first time -- k masked passes
    # instead of mask_one's d (= 64) or the one-pass heads' zero. Same 50k
    # curriculum as every other ladder rung, so the result reads directly
    # against iv 0.6463 < ma 0.7806 < stencil 0.8046 < mo 0.9103.
    # ga8: the headline. k=8 masks 12.5% of sites per pass at 1/8 of mask_one's
    # body cost, keeping FULL depth and global mixing over the rest.
    "H2_d64_c50_s223_letf_ga8_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga8_50k_curr", head_kind="grouped_anchor",
        n_groups=8,
    ),
    # ga16: the cost/quality dial. Halves the masked fraction to 6.25% for 2x
    # the passes -- with ga8 it gives the slope of quality against k, which is
    # what extrapolates to the D=16 choice of k.
    "H2_d64_c50_s223_letf_ga16_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga16_50k_curr", head_kind="grouped_anchor",
        n_groups=16,
    ),
    # ga8_contig: the shape control. Identical k, identical cost, but the
    # masked sites form a raster ROW instead of a dispersed Latin-square
    # diagonal -- the direct test of "dispersed >> line-shaped", which is a
    # prediction of the construction rather than an assumption in it.
    "H2_d64_c50_s223_letf_ga8_contig_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga8_contig_50k_curr", head_kind="grouped_anchor",
        n_groups=8, grouping="contiguous",
    ),
    "H2_d64_c50_s223_letf_mo": _hard_cell(
        "H2_d64_c50_s223_letf_mo", sigma=0.223, head_kind="mask_one",
        D=8, n_euler_steps=128, n_eval_samples=5000,
        # 256 samples x 64 anchor copies = 16k-row stacked passes under
        # no_grad -- a few GB transient on the L4, vs ~22 GB unchunked.
        eval_sample_chunk=256,
        # In-training ESS cadence is a diagnostic: 512 samples suffices and
        # the eval otherwise dominates the run (profile: 92% of GPU time).
        # run.py's final eval still draws the full 5000.
        n_eval_samples_training=512,
        # Tier-2 flags, user sign-off 2026-07-06 (perf-branch evidence
        # recorded 2026-07-05): SDPA readout everywhere, bf16 on the
        # in-training eval block only — the final 5000-sample eval runs fp32.
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    ),
}
