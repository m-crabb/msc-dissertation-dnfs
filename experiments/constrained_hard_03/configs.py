"""Hard-constraint configs: swap-move CTMC on the fixed-composition manifold.

SIGMA_C = ln(1+sqrt(2))/4 = 0.220343 (targets/ising.py, exact). Cells with
sigma=0.223/0.22305 ("s223", and the 0.223 ladder endpoint) are archived
runs at the legacy value; those literals are records and must not be edited
(stored run configs and the eval drift guard are pinned to them). New
sigma_c cells import SIGMA_C.

`HardStageCfg` adds `head_kind` (the swap-readout head) to the baseline
schema. Composition is enforced by the swap move set, so there is no penalty
or lambda curriculum.

Cell names are the results-directory names and are never renamed. Tokens:
  H2 / cuau16 / cuau64   S=2 Potts (Ising) lattice / 16- or 64-site Cu-Au cell
  d16 d64 d256 d400 d576 sites d = D^2 (4x4, 8x8, 16x16, 20x20, 24x24)
  c50                    target composition 0.50
  s010 s220 s223         coupling sigma 0.10 (floor), SIGMA_C, legacy 0.223;
                         T500 = Cu-Au 500 K
  letf                   leTF trunk; h128 / L3 / lr03 = hidden 128, 3 layers, lr 3e-4
  head tokens            dh doubly_hollow; mo mask_one; na non_antisym control;
                         iv interval; ma masked_attention (mal whole-lattice
                         window, mar relative pair code, masep separable scores,
                         mabef bonds + exact field, mamo2 two orderings);
                         ga8/ga16 grouped_anchor k=8/16; fab8/fab16 factorised
                         bilinear+global rank 8/16, fbil bilinear-only, fglo
                         global-only, fmp40 matched-parameter factor_dim 40;
                         fmo2 factorised, two site orderings (row, col);
                         fimo2 = fmo2 + prefix interior band; ef = exact-field
                         channel (fimo2ef); thp/thp2/thp3/thp4 two_hole_patch R=1..4
  10k ... 100k           training steps; scr5k / smoke12k = 5k screen / 12k smoke run
  curr / sc              sigma curriculum (the ladder) / trained flat at sigma_c
  b512 / ob512 / ctb512  batch_size 512 (loss batch; outer defaults to it) /
                         outer batch 512 only / c_t estimation batch 512
  ne128 ne512            Euler steps per rollout
  naive / cv / cv2       estimator: naive Monte Carlo / control variate / control
                         variate with the cv-inversion halt (tripwire) armed
  cyc16 buf2 rw noflush  replay 16 cycles / doubled buffer / re-warmup on stage /
                         replay not flushed on stage
  w2 w3 w4 (w2e w2sig)   campaign index at that lattice size, kept for key
                         uniqueness only (w2e eager head, w2sig legacy sigma)
  camort / house / lowlr composition-amortised / the house recipe / lr cut

Working vocabulary used in the comments below:
  house recipe   the shared training recipe a table row is built on
  twin           a cell that differs from its parent by the declared levers only
  arm            one cell of a family run for comparison; lever/knob = a setting
  rung / ladder  one lattice size / the sigma curriculum or the size sequence
  floor          sigma = 0.10, the easy end of the table; sigma_c = criticality
  gate / screen  a short run that must pass before the full cell / a 5k pilot
  smoke          a short run checking the pipeline, not a result
  rescue         the configuration that recovered a cell which had diverged
  tripwire       halt_on_cv_inversion_after: stop when the control variate inverts
  cold-CV        control variate from step 0; cvcont = CV continuation of a run
  chassis        the base cell a family is built from by `replace`
  judged / read  the recorded result of the run (EMA = EMA-weights eval)
  pinned by      enforced by the named test in tests/
  archived       a completed run whose literals must not change
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

from discrete_flow_sampler.constraints.exact_field_channel import ExactFieldSwapHead
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
from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
    bravais_patch_geometry,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.ising import SIGMA_C


@dataclass(frozen=True)
class HardStageCfg(StageCfg):
    """`StageCfg` plus `head_kind`, the swap-readout head `build_swap_head`
    instantiates: "doubly_hollow" (O(d^2) correctness gate), "mask_one" (O(d),
    numerically equal to doubly_hollow), "non_antisym" (negative control, never
    a real run), "interval" (one-pass three-interval band), "masked_attention"
    (same intervals, exclusion-mask attention band; the reported head),
    "grouped_anchor" (k masked passes), "factorised" (low-rank bilinear causal
    factors + hole-subtracted global context), "two_hole_patch" (ordering-free
    hollow torus patch + pooled levels).
    """

    head_kind: Literal[
        "doubly_hollow",
        "mask_one",
        "non_antisym",
        "interval",
        "masked_attention",
        "grouped_anchor",
        "factorised",
        "two_hole_patch",
    ] = "doubly_hollow"
    # Anchor-batch chunk for the mask_one head's vectorised forward; None =
    # unchunked (d=256 needs it: the (d*B)-row buffer OOMs the L4 otherwise).
    anchor_chunk_size: int | None = None
    # Opt-in torch.compile of the built head (default OFF). Head
    # only — the Euler loop's data-dependent sampling would graph-break.
    compile_head: bool = False
    # Band-capacity knobs for the interval / masked_attention heads. None =
    # the archived defaults: 16, 32 and (1, D) (row/column adjacency).
    band_feature_dim: int | None = None
    attention_dim: int | None = None
    pair_offsets: tuple[int, ...] | None = None
    # Adds the 5-point lattice-stencil band-feature family to the
    # masked_attention head; False = the depth-1 unary+offset band.
    use_stencil: bool = False
    # muP readout multiplier on G for the interval/masked_attention pair score:
    # 32/h for width h against the h32 reference; 1.0 = every archived cell.
    readout_score_scale: float = 1.0
    # Grouped-anchor head knobs: n_groups = k masked body passes (k = d is
    # mask_one bit-exactly). Read only when head_kind is "grouped_anchor".
    n_groups: int | None = None
    grouping: str = "diagonal"
    group_chunk_size: int | None = None
    # Factorised-head knobs; None = the head's defaults (rank 8, factor_dim 32,
    # global_feature_dim 16). The use_* switches are the head's ablation arms.
    bilinear_rank: int | None = None
    factor_dim: int | None = None
    global_feature_dim: int | None = None
    use_bilinear: bool = True
    use_global: bool = True
    # Causal-sweep directions for the factorised bilinear term; extras from
    # {"col", "diag"} give each pair a second blind split. ("row",) = archived.
    site_orderings: tuple[str, ...] = ("row",)
    # Interior band on the factorised head's per-pair path: "prefix" (interval
    # head's band) or "attention" (masked-attention band); None = archived head.
    interior_band: str | None = None
    # Terms the masked-attention band may pool over: "interval" = the open
    # (i, j) interval (archived cells); "lattice" = all terms clear of both holes.
    attention_window: str = "interval"
    # Position code of the band's query: "absolute" = per-site nn.Embedding
    # (archived cells); "relative" = one row per signed torus displacement.
    pair_position_mode: str = "absolute"
    # Masked-attention band without its (B, d^2, n) score tensor: an exact
    # identity (fp round-off), requires pair_position_mode="absolute".
    separable_band_scores: bool = False
    # Per-pair work of the interval / masked_attention / factorised heads on
    # the d(d-1)/2 unordered pairs (H_ji := H_ij); a memory lever for d=256.
    gather_triu_pairs: bool = False
    # Exterior combiner for the interval / masked-attention heads: "bilinear"
    # moves [P_i, S_j] from the per-pair MLP into a rank-`bilinear_rank` product.
    exterior_combiner: str = "mlp"
    # Whole-lattice bond sums (hole-touching bonds removed) added to the
    # factorised global term; requires interior_band, read only by that head.
    global_bond_features: bool = False
    # Adds gain(t) * sigma * Delta_ij (the closed-form Kawasaki energy change)
    # to any head's scores, gain = a + b t from zero; see exact_field_channel.py.
    exact_field_channel: bool = False
    # Two-hole patch head: hollow window radius R (2R+1 <= D) and pair-context
    # width; None = the head's defaults (R=1, feature_dim 32).
    patch_radius: int | None = None
    patch_feature_dim: int | None = None
    # Cluster-expansion cells only: window = count of Bravais neighbour shells
    # (1 = the twelve fcc nearest neighbours); patch_radius is ignored there.
    patch_shells: int | None = None
    # Dual-eval EMA shadow (discrete_flow_sampler.ema); 0.0 = off. Instrument
    # only: eval/ stays the raw primary, eval_ema/ is recorded alongside.
    ema_decay: float = 0.0
    # "ising" / "potts" = FixedComposition{Ising,Potts}Target. Defaults stay Ising
    # so run dirs written before Potts existed backfill under the drift guard.
    target_kind: Literal["ising", "potts", "cluster_expansion"] = "ising"
    # Per-species fractions (length S, sum 1, each times d integral). The single
    # source of S: vocab_size follows its length in `_hard_cell`.
    potts_composition: tuple[float, ...] | None = None
    # When set, the target is MixtureCompositionIsingTarget over these slices
    # (first = anchor), the head amortising through x alone; None = single slice.
    composition_mixture: tuple[float, ...] | None = None


class NonAntisymSwapHead(nn.Module):
    """Negative control: DoublyHollowSwapHead's readout on an unmasked body.

    G[:, i, j] = <H[:, j, :], omega_{x_i} - omega_{x_j}> with H from one
    unmasked leTF pass instead of a doubly-masked pass per (i, j). Same-spin
    pairs still read zero, so the swap CTMC conserves composition, but H sees
    the live x_i, so swap-antisymmetry G(i, j | x) = -G(i, j | Swap2(x, i, j))
    fails -- the property the ablation must falsify. Never use for a real run.
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


def build_swap_head(
    cfg: HardStageCfg, backbone: LeTFRateMatrix, target=None
) -> nn.Module:
    """Map `cfg.head_kind` to an instantiated swap-readout head. `target` is
    required only when `cfg.exact_field_channel` is set: the channel reads
    the target's adjacency and live sigma."""
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
            exterior_combiner=cfg.exterior_combiner,
            bilinear_rank=cfg.bilinear_rank or 8,
            gather_triu_pairs=cfg.gather_triu_pairs,
            site_orderings=cfg.site_orderings,
            lattice_side=cfg.ising.D,
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
            exterior_combiner=cfg.exterior_combiner,
            bilinear_rank=cfg.bilinear_rank or 8,
            gather_triu_pairs=cfg.gather_triu_pairs,
            attention_window=cfg.attention_window,
            pair_position_mode=cfg.pair_position_mode,
            separable_band_scores=cfg.separable_band_scores,
            site_orderings=cfg.site_orderings,
        )
    elif cfg.head_kind == "factorised":
        head = FactorisedSwapHead(
            backbone,
            bilinear_rank=cfg.bilinear_rank or 8,
            factor_dim=cfg.factor_dim or 32,
            global_feature_dim=cfg.global_feature_dim or 16,
            use_bilinear=cfg.use_bilinear,
            use_global=cfg.use_global,
            interior_band=cfg.interior_band,
            band_feature_dim=cfg.band_feature_dim or 16,
            pair_offsets=cfg.pair_offsets or (1, cfg.ising.D),
            attention_dim=cfg.attention_dim or 32,
            site_orderings=cfg.site_orderings,
            lattice_side=cfg.ising.D,
            gather_triu_pairs=cfg.gather_triu_pairs,
            global_bond_features=cfg.global_bond_features,
        )
    elif cfg.head_kind == "two_hole_patch":
        if cfg.target_kind == "cluster_expansion":
            # The expansion's supercell is not a square torus: build the
            # window and pooled balls from its translation group instead.
            geometry = bravais_patch_geometry(
                target.spec.positions,
                target.spec.cell,
                patch_shells=cfg.patch_shells or 1,
            )
            head = TwoHolePatchSwapHead(
                backbone,
                geometry=geometry,
                feature_dim=cfg.patch_feature_dim or 32,
            )
        else:
            head = TwoHolePatchSwapHead(
                backbone,
                lattice_side=cfg.ising.D,
                patch_radius=cfg.patch_radius or 1,
                feature_dim=cfg.patch_feature_dim or 32,
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
    if cfg.exact_field_channel:
        if target is None:
            raise ValueError("exact_field_channel needs the target at build time")
        head = ExactFieldSwapHead(head, target)
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
    """Shared shape for the D=4 gate cells (only sigma and head_kind vary) and,
    via the keyword knobs, the scaling rungs (n_euler_steps must be clip-safe
    for the one-event step, ~2d at d=64). `potts_composition` selects the
    S-species route: vocab_size = S and `sigma` is then the Potts coupling
    (Ising at sigma = S=2 Potts at 2 sigma, targets/potts.py)."""
    is_potts = potts_composition is not None
    return HardStageCfg(
        name=name,
        ising=IsingCfg(
            D=D,
            sigma=sigma,
            bias=0.0,
            target_composition=None if is_potts else 0.5,
            composition_penalty_strength=0.0,
        ),
        train=TrainCfg(
            n_steps=n_steps,
            batch_size=128,
            replay_buffer_cycles=8,
            lr=1e-3,
            seed=42,
            warmup_steps=500,
            grad_clip_max_norm=grad_clip_max_norm,
            optimiser=optimiser,
            rewarmup_on_stage=rewarmup_on_stage,
        ),
        ctmc=CTMCCfg(
            n_euler_steps=n_euler_steps,
            use_matching_step=use_matching_step,
        ),
        eval=EvalCfg(
            eval_every=eval_every,
            n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
            n_eval_samples_training=n_eval_samples_training,
            eval_autocast_bf16=eval_autocast_bf16,
        ),
        model=ModelCfg(
            kind="letf",
            hidden_dim=32,
            n_layers=2,
            n_heads=4,
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


# Sigma-plateau ladder for the d=64 sigma_c rung (baseline stage_3 recipe,
# final stage at the archived 0.223; lr drops at the 0.205 variance spike).
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

# The 12k smoke arms only reach the first three stages and the curriculum
# validator rejects stages starting beyond n_steps, so they share this view.
_SMOKE12K_SIGMA_LADDER = CurriculumCfg(stages=_D64_SIGMA_LADDER.stages[:3])


def _d64_curriculum_cell(
    name: str, head_kind: str, n_steps: int = 50_000, **head_knobs
) -> HardStageCfg:
    """The d=64 sigma_c curriculum recipe (D=8 rung); band-push cells share it
    verbatim and differ only in head_kind and the declared knobs (pinned by
    test_band_push_cells_mirror_ma_twin_except_declared_fields). The ladder is
    absolute start_steps, so n_steps=100_000 adds steps on the 0.223 plateau."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind=head_kind,
        D=8,
        n_steps=n_steps,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=256,
        n_eval_samples_training=512,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        curriculum=_D64_SIGMA_LADDER,
    )
    return replace(cell, **head_knobs)


def _d144_curriculum_cell(
    name: str, head_kind: str, n_steps: int = 50_000, **head_knobs
) -> HardStageCfg:
    """The 12x12 (d=144) rung between the healthy 8x8 and the 16x16 that does
    not train. Three fields differ from 8x8, each forced by volume:
    use_matching_step (Lambda*dt clips one-event steps), n_euler_steps=288 = 2d
    (the clip-safe rule), eval_sample_chunk=512 (eval-only). Ladder as 8x8."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind=head_kind,
        D=12,
        n_steps=n_steps,
        n_euler_steps=288,
        n_eval_samples=5000,
        eval_sample_chunk=512,
        n_eval_samples_training=512,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
    )
    return replace(cell, **head_knobs)


def _d64_fmo2_h128_cell(name: str) -> HardStageCfg:
    """The 8x8 factorised rung with hidden_dim 32 -> 128 the only change (n_heads
    stays 4, so head_dim rides 8 -> 32; n_layers stays 2). Capacity null at the
    healthy rung (rank headroom 32/(64/8) = 4x already present; holding it at
    256 sites needs hidden 128) and transfer source for a matched larger rung."""
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
    """The archived 8x8 factorised rung opened on its loop knobs: with no
    overrides it reproduces `H2_d64_c50_s223_letf_fmo2_50k_curr` bar the name
    (tests/test_screen_pins.py). `train_overrides` go straight into TrainCfg
    via `replace`. Comparator: EMA ESS/N 0.8104 (raw 0.7452); insensitive =
    within +-0.03; at the ESS ceiling read EMA Var[log w] against 0.2068
    (bootstrap 95% CI 0.1983-0.2152) instead."""
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


def _d256_fmo2_warm_cell(name: str, n_euler_steps: int = 128) -> HardStageCfg:
    """16x16 factorised rung warm-started from a trained 8x8 factorised model
    (`--init-from` a cross-volume transfer; positional tables resampled bicubically).
    No curriculum, flat sigma_c, lr pinned at 3e-4 as in the archived continuation.
    `n_euler_steps` is the arm variable: 128 = archived control (0.5d), 512 = 2d."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind="factorised",
        D=16,
        n_steps=20_000,
        n_euler_steps=n_euler_steps,
        n_eval_samples=5000,
        eval_sample_chunk=512,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
    )
    return replace(
        cell,
        ema_decay=0.9999,
        site_orderings=("row", "col"),
        train=replace(cell.train, lr=3e-4),
    )


def _d64_smoke12k_replay2_cell(name: str) -> HardStageCfg:
    """MA curriculum recipe at the 12k smoke horizon, replay_buffer_cycles 8 -> 2
    the only declared change versus the archived d64 MA twin."""
    cell = _d64_curriculum_cell(name, "masked_attention", n_steps=12_000)
    return replace(
        cell,
        train=replace(cell.train, replay_buffer_cycles=2),
        curriculum=_SMOKE12K_SIGMA_LADDER,
    )


def _d64_m2_ctema4_cell(name: str) -> HardStageCfg:
    """Archived d64 MA 50k curriculum recipe with c_t_ema_halflife_cycles 0.0 -> 4.0
    the only declared change (pinned by
    test_m2_ctema4_gate_mirrors_ma_twin_except_declared_fields). Dual-eval EMA not
    ridden: the comparison is raw vs the archived twin."""
    cell = _d64_curriculum_cell(name, "masked_attention")
    return replace(cell, train=replace(cell.train, c_t_ema_halflife_cycles=4.0))


def _d64_naive_twin_cell(name: str) -> HardStageCfg:
    """Archived MA 50k curriculum twin with estimator control_variate -> naive_mc
    the only declared change: measures the c_t-noise -> Var[log w] slope where the
    CV is known to work (no naive arm existed among the 21 archived d64 cells).
    Predictions pinned in test_ctv_naive_twin_mirrors_ma_twin_except_the_estimator."""
    cell = _d64_curriculum_cell(name, "masked_attention")
    return replace(cell, estimator="naive_mc")


def optimised_recipe(cell: HardStageCfg) -> HardStageCfg:
    """Optimisation bundle: compile_head=True and train.c_t_from_rollout=True, nothing
    else. Apply to new cells only: archived cells and their eager twins keep both
    flags off, since compiled runs are 1e-5-class vs eager, never bit-parity."""
    return replace(
        cell,
        compile_head=True,
        train=replace(cell.train, c_t_from_rollout=True),
    )


def _d64_smoke12k_ctb512_cell(name: str) -> HardStageCfg:
    """MA curriculum recipe at the 12k smoke horizon with c_t_batch=512 the only
    mechanism change: validates the enlarged-rollout plumbing at d64 before d256."""
    cell = _d64_curriculum_cell(name, "masked_attention", n_steps=12_000)
    return replace(
        cell,
        train=replace(cell.train, c_t_batch=512),
        curriculum=_SMOKE12K_SIGMA_LADDER,
    )


def _d256_smoke12k_naive_ctb512_cell(name: str) -> HardStageCfg:
    """Smoke12k naive-estimator recipe with c_t_batch=512 the only declared change
    (pinned by test_m3_ctb512_smoke_mirrors_naive_arm_except_declared_fields)."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        estimator="naive_mc",
    )
    return replace(cell, train=replace(cell.train, c_t_batch=512))


def _d256_cv2_cell(name: str) -> HardStageCfg:
    """Phase-2 twin of the d=256 naive rescue: estimator back to the control variate,
    no curriculum (continues from the naive run's final checkpoint via --init-from),
    lr pinned to the ladder's final 3e-4, dual-eval EMA riding."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=20_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
    )
    return replace(cell, ema_decay=0.9999, train=replace(cell.train, lr=3e-4))


def _d256_scr5k_cell(
    name: str,
    head_kind: str,
    *,
    eval_sample_chunk: int,
    n_euler_steps: int = 128,
) -> HardStageCfg:
    """One arm of the 16x16 stage-1 screen: flat sigma=0.10, 5,000 steps, naive c_t,
    the rescue recipe's shape otherwise verbatim. The archived 16x16 failure is fully
    expressed in the rescue's first 5k steps (stage-tail FVU 0.119 vs 8x8's 0.033).
    Read: FVU over steps 3,000-4,999; LIVE <= 0.08, PARITY <= 0.04, NULL >= 0.10."""
    return _hard_cell(
        name,
        sigma=0.10,
        head_kind=head_kind,
        D=16,
        n_steps=5_000,
        n_euler_steps=n_euler_steps,
        n_eval_samples=1000,
        eval_sample_chunk=eval_sample_chunk,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        estimator="naive_mc",
    )


def _scr5k_fmo2_cell(name: str) -> HardStageCfg:
    """Factorised-multi-order screen arm: the scr5k shape with the fmo2 family's
    conventions (EMA shadow, two causal orderings) and an eval chunk sized to the
    head (eval-only; cannot move the trained model)."""
    cell = _d256_scr5k_cell(name, "factorised", eval_sample_chunk=512)
    return replace(cell, ema_decay=0.9999, site_orderings=("row", "col"))


def _scr5k_fmo2_with(
    name: str, *, n_euler_steps: int = 128, **overrides
) -> HardStageCfg:
    """One fmo2 screen arm = the fmo2 base with the named overrides the only changes.
    Overrides route to the model or training dataclass by field name; anything else
    is a typo and must fail loudly rather than produce an undeclared twin."""
    model_field_names = {"hidden_dim", "n_layers"}
    train_field_names = {
        "lr",
        "grad_clip_max_norm",
        "batch_size",
        "loss_microbatch_size",
        "c_t_batch",
    }
    unknown = set(overrides) - model_field_names - train_field_names
    if unknown:
        raise ValueError(f"unknown screen-arm override(s): {sorted(unknown)}")
    cell = _scr5k_fmo2_cell(name)
    model_overrides = {k: v for k, v in overrides.items() if k in model_field_names}
    train_overrides = {k: v for k, v in overrides.items() if k in train_field_names}
    if n_euler_steps != 128:
        cell = replace(cell, ctmc=replace(cell.ctmc, n_euler_steps=n_euler_steps))
    if model_overrides:
        cell = replace(cell, model=replace(cell.model, **model_overrides))
    if train_overrides:
        cell = replace(cell, train=replace(cell.train, **train_overrides))
    return cell


def _scr5k_ma_h128_lr03_cell(name: str) -> HardStageCfg:
    """MA bridge capacity arm: hidden 32 -> 128 with lr 1e-3 -> 3e-4 co-varied,
    deliberately bundled (the 8x8 fmo2 h128 arm lost ground under lr 1e-3, so a
    frozen-lr arm would reproduce that false negative; fmo2 carries the full 2x2).
    loss_microbatch_size=64 is a gradient-exact memory fit, not a recipe variable."""
    cell = _d256_scr5k_cell(name, "masked_attention", eval_sample_chunk=128)
    return replace(
        cell,
        model=replace(cell.model, hidden_dim=128),
        train=replace(cell.train, lr=3e-4, loss_microbatch_size=64),
    )


def _scr5k_ma_clip60k_cell(name: str) -> HardStageCfg:
    """MA screen base with the grad-clip threshold rescaled to d=256 gradient units,
    the only change; derivation and expected ranges at the cell's entry below."""
    cell = _d256_scr5k_cell(name, "masked_attention", eval_sample_chunk=128)
    return replace(cell, train=replace(cell.train, grad_clip_max_norm=60_000.0))


def _scr5k_mo_cell(name: str) -> HardStageCfg:
    """mask_one screen arm: the best 8x8 head (0.00129 Var/site at 100k, 2.2x under
    the MA twin), never before run at 16x16. loss_microbatch_size=16 is a
    gradient-exact memory fit (the stacked anchor passes retain ~d trunk graphs per
    row), not a recipe variable; slicing moves no batch variable."""
    cell = _d256_scr5k_cell(name, "mask_one", eval_sample_chunk=64)
    return replace(cell, train=replace(cell.train, loss_microbatch_size=16))


def _d256_fmo2_ladder_cell(
    name: str,
    *,
    estimator: str,
    n_euler_steps: int = 128,
    batch_size: int = 128,
    loss_microbatch_size: int | None = None,
) -> HardStageCfg:
    """16x16 fmo2 sigma-ladder shape: the archived MA naive-rescue recipe with the
    factorised family's conventions (EMA shadow, dual site orderings, own eval chunk).
    Bare call = anchor cell; batch/grid/microbatch knobs build the composed cells."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind="factorised",
        D=16,
        n_steps=50_000,
        n_euler_steps=n_euler_steps,
        n_eval_samples=5000,
        eval_sample_chunk=512,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
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
    """12x12 volume bracket of the archived failure: the 16x16 naive rescue recipe
    with the lattice side the only mechanism change (the earlier fmo2 12x12 rung sat
    three variables away). Read on Var[log w]/site, never ESS: 8x8 naive 0.00472 ->
    12x12 -> 16x16 naive 0.0707. Eval chunk 128 and 512-draw train evals, eval-only."""
    return _hard_cell(
        name,
        sigma=0.223,
        head_kind="masked_attention",
        D=12,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=128,
        n_eval_samples_training=512,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        estimator="naive_mc",
    )


def _d256_clip2000_cont_cell(name: str) -> HardStageCfg:
    """Clip continuation: the completed naive rescue continued via --init-from for 10k
    steps at flat sigma_c, lr 3e-4, with grad_clip_max_norm 500 -> 2000 the only
    mechanism change (the clip fires on ~25% of the rescue's final-plateau steps, p99
    ~1050). Null read: the plateau's FVU 0.137 +- 0.005 continuing flat."""
    cell = _hard_cell(
        name,
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=10_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=128,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        estimator="naive_mc",
        grad_clip_max_norm=2_000.0,
    )
    return replace(cell, ema_decay=0.9999, train=replace(cell.train, lr=3e-4))


CONFIGS: dict[str, HardStageCfg] = {
    # First Potts cell (S=3), CPU-smokeable. D=3 is the smallest lattice that is both
    # non-degenerate and divisible by 3 (exact equal three-way slice). sigma = 0.5025
    # = ln(1+sqrt 3)/2, the 3-state Potts critical coupling halved for the
    # double-counted A (cf. Ising's ln(1+sqrt 2)/2 = 0.223); doubly_hollow is free here.
    "H3_d9_c33_s503_letf_dh": _hard_cell(
        "H3_d9_c33_s503_letf_dh",
        sigma=0.5025,
        head_kind="doubly_hollow",
        D=3,
        potts_composition=(1 / 3, 1 / 3, 1 / 3),
    ),
    "H2_d16_c50_s010_letf_dh": _hard_cell(
        "H2_d16_c50_s010_letf_dh",
        sigma=0.10,
        head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s223_letf_dh": _hard_cell(
        "H2_d16_c50_s223_letf_dh",
        sigma=0.223,
        head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s040_letf_dh": _hard_cell(
        "H2_d16_c50_s040_letf_dh",
        sigma=0.40,
        head_kind="doubly_hollow",
    ),
    # Antisymmetry negative control: same sigma as the floor rung,
    # NonAntisymSwapHead instead of DoublyHollowSwapHead, same loss.
    "H2_d16_c50_s010_letf_na": _hard_cell(
        "H2_d16_c50_s010_letf_na",
        sigma=0.10,
        head_kind="non_antisym",
    ),
    # 4x4 demo cells: 10k-step MA/MO twins of the 2k dh ladder at the floor and
    # critical rungs. Only head_kind and n_steps differ from the _dh cells (pinned by
    # test_demo_4x4_cells_mirror_dh_ladder_except_declared_fields); no curriculum,
    # the 2k sigma_c dh cell already trains cold.
    "H2_d16_c50_s010_letf_ma_10k": _hard_cell(
        "H2_d16_c50_s010_letf_ma_10k",
        sigma=0.10,
        head_kind="masked_attention",
        n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_ma_10k": _hard_cell(
        "H2_d16_c50_s223_letf_ma_10k",
        sigma=0.223,
        head_kind="masked_attention",
        n_steps=10_000,
    ),
    "H2_d16_c50_s010_letf_mo_10k": _hard_cell(
        "H2_d16_c50_s010_letf_mo_10k",
        sigma=0.10,
        head_kind="mask_one",
        n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_mo_10k": _hard_cell(
        "H2_d16_c50_s223_letf_mo_10k",
        sigma=0.223,
        head_kind="mask_one",
        n_steps=10_000,
    ),
    # Factorised-head 4x4 gate cells: 10k twins of the MA demo cells, only head_kind
    # and the declared factorised knobs differ (pinned by
    # test_factorised_gate_cells_mirror_ma_twin_except_declared_fields). fab8/fab16 =
    # bilinear+global at rank 8/16, fbil = bilinear-only, fglo = global-only.
    "H2_d16_c50_s010_letf_fab8_10k": _hard_cell(
        "H2_d16_c50_s010_letf_fab8_10k",
        sigma=0.10,
        head_kind="factorised",
        n_steps=10_000,
    ),
    "H2_d16_c50_s223_letf_fab8_10k": _hard_cell(
        "H2_d16_c50_s223_letf_fab8_10k",
        sigma=0.223,
        head_kind="factorised",
        n_steps=10_000,
    ),
    "H2_d16_c50_s010_letf_fab16_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fab16_10k",
            sigma=0.10,
            head_kind="factorised",
            n_steps=10_000,
        ),
        bilinear_rank=16,
    ),
    "H2_d16_c50_s223_letf_fab16_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fab16_10k",
            sigma=0.223,
            head_kind="factorised",
            n_steps=10_000,
        ),
        bilinear_rank=16,
    ),
    "H2_d16_c50_s010_letf_fbil_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fbil_10k",
            sigma=0.10,
            head_kind="factorised",
            n_steps=10_000,
        ),
        use_global=False,
    ),
    "H2_d16_c50_s223_letf_fbil_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fbil_10k",
            sigma=0.223,
            head_kind="factorised",
            n_steps=10_000,
        ),
        use_global=False,
    ),
    "H2_d16_c50_s010_letf_fglo_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fglo_10k",
            sigma=0.10,
            head_kind="factorised",
            n_steps=10_000,
        ),
        use_bilinear=False,
    ),
    "H2_d16_c50_s223_letf_fglo_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fglo_10k",
            sigma=0.223,
            head_kind="factorised",
            n_steps=10_000,
        ),
        use_bilinear=False,
    ),
    # Matched-param arm: factor_dim 40 raises the head-owned count to 36,384 ~ the MA
    # head's 34,272 readout-work params (21,440 head-owned + 12,832 attention_readout).
    # Rules out head-parameter deficit as the cause of fab8's partial read.
    "H2_d16_c50_s010_letf_fmp40_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fmp40_10k",
            sigma=0.10,
            head_kind="factorised",
            n_steps=10_000,
        ),
        factor_dim=40,
    ),
    "H2_d16_c50_s223_letf_fmp40_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fmp40_10k",
            sigma=0.223,
            head_kind="factorised",
            n_steps=10_000,
        ),
        factor_dim=40,
    ),
    # Multi-order arm: fab8 + the column-major causal stream
    # (site_orderings=("row","col")), adding coverage where the rank/width arms
    # only added capacity. Read 0.883/0.938/0.879 at s223 vs fab8's 0.911 bar:
    # inconclusive at 4x4 (interiors <= 14 sites); see the d64 rung.
    "H2_d16_c50_s010_letf_fmo2_10k": replace(
        _hard_cell(
            "H2_d16_c50_s010_letf_fmo2_10k",
            sigma=0.10,
            head_kind="factorised",
            n_steps=10_000,
        ),
        site_orderings=("row", "col"),
    ),
    "H2_d16_c50_s223_letf_fmo2_10k": replace(
        _hard_cell(
            "H2_d16_c50_s223_letf_fmo2_10k",
            sigma=0.223,
            head_kind="factorised",
            n_steps=10_000,
        ),
        site_orderings=("row", "col"),
    ),
    # Exterior-vs-interior separation cells: hold an existing interior fixed
    # and change only the exterior combiner, so the factorisation's own price
    # is measured. Arms: fib = bilinear exterior + interval prefix-sum band
    # (twin: interval); fatt = bilinear + masked-attention band (twin: MA);
    # fimo2 / fmoatt = the same with the column ordering added; iv = the
    # interval head itself at 4x4. Parity bar >= 0.955 on 2/3 seeds.
    **{
        f"H2_d16_c50_{sigma_label}_letf_{arm}_10k": replace(
            _hard_cell(
                f"H2_d16_c50_{sigma_label}_letf_{arm}_10k",
                sigma=sigma,
                head_kind="factorised",
                n_steps=10_000,
            ),
            **knobs,
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
        for arm, knobs in {
            "fib": {"interior_band": "prefix"},
            "fatt": {"interior_band": "attention"},
            "fimo2": {"interior_band": "prefix", "site_orderings": ("row", "col")},
            "fmoatt": {"interior_band": "attention", "site_orderings": ("row", "col")},
        }.items()
    },
    **{
        f"H2_d16_c50_{sigma_label}_letf_iv_10k": _hard_cell(
            f"H2_d16_c50_{sigma_label}_letf_iv_10k",
            sigma=sigma,
            head_kind="interval",
            n_steps=10_000,
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
    },
    # Literal factorisation test: the archived MA / interval heads with only
    # [P_i, S_j] moved from the per-pair MLP into a rank-8 bilinear product
    # (exterior_combiner="bilinear"). mab ~ MA means the factorisation is free.
    **{
        f"H2_d16_c50_{sigma_label}_letf_{arm}_10k": replace(
            _hard_cell(
                f"H2_d16_c50_{sigma_label}_letf_{arm}_10k",
                sigma=sigma,
                head_kind=head_kind,
                n_steps=10_000,
            ),
            exterior_combiner="bilinear",
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
        for arm, head_kind in (("mab", "masked_attention"), ("ivb", "interval"))
    },
    # Exact-field channel: fimo2 and mab chassis with the closed-form
    # sigma*Delta_ij added as a fixed score channel behind a learned gain
    # (exact_field_channel=True, nothing else changes). Twins at sigma_c:
    # fimo2 0.924/0.925/0.903, mab 0.970/0.973/0.973.
    **{
        f"H2_d16_c50_{sigma_label}_letf_{arm}ef_10k": replace(
            _hard_cell(
                f"H2_d16_c50_{sigma_label}_letf_{arm}ef_10k",
                sigma=sigma,
                head_kind=head_kind,
                n_steps=10_000,
            ),
            exact_field_channel=True,
            **knobs,
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
        for arm, head_kind, knobs in (
            (
                "fimo2",
                "factorised",
                {"interior_band": "prefix", "site_orderings": ("row", "col")},
            ),
            ("mab", "masked_attention", {"exterior_combiner": "bilinear"}),
        )
    },
    # Periodic-RoPE backbone at 4x4: the fimo2 twins with only the backbone
    # changed (model.kind="rope_vit", patch_size 1 or 2). A correctness check
    # (energy-marginal TV <= 0.02, antisymmetry zero) with the fimo2 twins'
    # sigma_c range 0.903-0.925 as parity reference; no equivariance claim.
    **{
        f"H2_d16_c50_{sigma_label}_rope{patch_size}_fimo2_10k": (
            lambda _cell, _p: replace(
                _cell,
                model=replace(_cell.model, kind="rope_vit", patch_size=_p),
            )
        )(
            replace(
                _hard_cell(
                    f"H2_d16_c50_{sigma_label}_rope{patch_size}_fimo2_10k",
                    sigma=sigma,
                    head_kind="factorised",
                    n_steps=10_000,
                ),
                interior_band="prefix",
                site_orderings=("row", "col"),
            ),
            patch_size,
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
        for patch_size in (1, 2)
    },
    # Two-hole patch head at 4x4: the fimo2 twin protocol with only the head
    # changed to "two_hole_patch" (R=1, feature_dim 32). Same 4x4 correctness
    # bars and parity reference (fimo2 sigma_c 0.903-0.925) as the rope twins;
    # the one head whose pair rate is torus translation-equivariant in one pass.
    **{
        f"H2_d16_c50_{sigma_label}_letf_thp_10k": _hard_cell(
            f"H2_d16_c50_{sigma_label}_letf_thp_10k",
            sigma=sigma,
            head_kind="two_hole_patch",
            n_steps=10_000,
        )
        for sigma_label, sigma in (("s010", 0.10), ("s223", 0.223))
    },
    # First non-enumerable rung: 8x8 at sigma_c, mask_one head (bit-exact ==
    # doubly_hollow), one-event step n_euler=128 (clip-safe per the scout),
    # 5000-draw eval. Budget probe at 12.5x the 2000-step cell that was cut off
    # mid-descent (ESS 0.0005-0.001 vs the 4x4 sigma_c reference 0.68-0.80).
    "H2_d64_c50_s223_letf_mo_25k": _hard_cell(
        "H2_d64_c50_s223_letf_mo_25k",
        sigma=0.223,
        head_kind="mask_one",
        D=8,
        n_steps=25_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=256,
        n_eval_samples_training=512,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    ),
    # Curriculum + budget rung: the 25k cold probe lifted final ESS frac
    # 0.001 -> 0.12 with the loss still descending, so this adds the sigma-
    # plateau ladder the unconstrained baseline needed at sigma_c (stage_3
    # recipe: 5k plateaus, LR drop at sigma=0.205, 40% of budget on the final).
    "H2_d64_c50_s223_letf_mo_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_mo_50k_curr",
        head_kind="mask_one",
    ),
    # Masked-attention twin of the 50k curriculum rung: only head_kind differs.
    # End-to-end A/B vs the mask_one record, re-validates the head before 16x16,
    # and retrains the 8x8 mixing-probe trend point on one architecture.
    "H2_d64_c50_s223_letf_ma_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr",
        head_kind="masked_attention",
    ),
    # Factorised-head d64 rung: single-variable twin of the MA curriculum rung
    # above; only head_kind and the ema_decay instrument differ (EMA never
    # touches training, so the raw-eval comparison stays single-variable). Seed 42.
    "H2_d64_c50_s223_letf_fab8_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_fab8_50k_curr",
            head_kind="factorised",
        ),
        ema_decay=0.9999,
    ),
    # Rank-at-scale arm: rank-16 twin of the fab8 d64 rung (bilinear_rank=16).
    # Judged: raw 0.5615, CI (0.5333, 0.5871), EMA 0.5920 -- rank binding
    # excluded (CI upper < the 0.60 bar); +0.05 raw over fab8, ~19% of the
    # 0.27 gap to the MA twin's 0.781.
    "H2_d64_c50_s223_letf_fab16_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_fab16_50k_curr",
            head_kind="factorised",
        ),
        ema_decay=0.9999,
        bilinear_rank=16,
    ),
    # Multi-order streams at scale: the fab8 d64 rung + the column-major causal
    # stream (site_orderings=("row","col")), seed 42; single-variable twin pinned
    # by test_fmo2_d64_rung_mirrors_fab8_rung_except_orderings.
    # Judged: raw 0.745 / EMA 0.810, clearing the >= 0.70 strong bar (MA twin 0.781).
    "H2_d64_c50_s223_letf_fmo2_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_fmo2_50k_curr",
            head_kind="factorised",
        ),
        ema_decay=0.9999,
        site_orderings=("row", "col"),
    ),
    # Exterior-vs-interior separation rungs: twins of the fmo2 rung above (EMA
    # instrument kept) with an interior band on the per-pair path; arm table in
    # the 4x4 block. Bars vs fmo2 0.745 raw / MA 0.781: strong >= 0.78,
    # null <= 0.745; fatt's twin is MA itself, fib's the interval rung (0.646).
    **{
        f"H2_d64_c50_s223_letf_{arm}_50k_curr": replace(
            _d64_curriculum_cell(
                f"H2_d64_c50_s223_letf_{arm}_50k_curr",
                head_kind="factorised",
            ),
            ema_decay=0.9999,
            **knobs,
        )
        for arm, knobs in {
            "fib": {"interior_band": "prefix"},
            "fatt": {"interior_band": "attention"},
            "fimo2": {"interior_band": "prefix", "site_orderings": ("row", "col")},
        }.items()
    },
    # Exact-field channel rungs: the fimo2 rung and the mab literal cell with
    # exact_field_channel=True, nothing else changed. Bars: fimo2ef vs fimo2
    # 0.750 raw / 0.830 EMA, mabef vs mab 0.769 / 0.834; null = no lift. The
    # learned gain (a, b) is read off checkpoints/final.pt after the run.
    "H2_d64_c50_s223_letf_fimo2ef_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_fimo2ef_50k_curr",
            head_kind="factorised",
        ),
        ema_decay=0.9999,
        exact_field_channel=True,
        interior_band="prefix",
        site_orderings=("row", "col"),
    ),
    "H2_d64_c50_s223_letf_mabef_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_mabef_50k_curr",
            head_kind="masked_attention",
        ),
        ema_decay=0.9999,
        exact_field_channel=True,
        exterior_combiner="bilinear",
    ),
    # Periodic-RoPE / patch-key backbone twins of the fimo2 rung
    # (models/rope_vit.py): absolute position tables replaced by rotary phases
    # 2*pi*m/L; at p=2 the far keys are pooled 2x2 patches. Head unchanged, so
    # only the backbone's position code is tested. Read on EMA Var[log w]/site
    # vs the fimo2 rung's 0.00281 (0.00271, 0.00292).
    **{
        f"H2_d64_c50_s223_rope{patch_size}_fimo2_50k_curr": (
            lambda _cell, _p: replace(
                _cell,
                model=replace(_cell.model, kind="rope_vit", patch_size=_p),
            )
        )(
            replace(
                _d64_curriculum_cell(
                    f"H2_d64_c50_s223_rope{patch_size}_fimo2_50k_curr",
                    head_kind="factorised",
                ),
                ema_decay=0.9999,
                interior_band="prefix",
                site_orderings=("row", "col"),
            ),
            patch_size,
        )
        for patch_size in (1, 2)
    },
    # Two-hole patch head twin of the fimo2 rung: only head_kind differs (R=1,
    # feature_dim 32). At 4x4 it reached sigma_c ESS 0.997/0.993/0.988 vs the
    # fimo2 twins' 0.924/0.925/0.903; this rung asks whether blindness-by-locality
    # survives d=64. Bars vs fimo2's EMA ESS/N 0.830: strong >= 0.84, null < 0.70.
    "H2_d64_c50_s223_letf_thp_50k_curr": replace(
        _d64_curriculum_cell(
            "H2_d64_c50_s223_letf_thp_50k_curr",
            head_kind="two_hole_patch",
        ),
        ema_decay=0.9999,
    ),
    # Scaling slate: the fmo2 rung cleared its 8x8 range (raw 0.745 / EMA 0.810)
    # and is cheaper than the MA twin at 16x16, so the head no longer limits
    # volume; what does is unlocated (every c_t lever <= 1.19x, 2.9x less
    # integrand variance moved 16x16 eval variance 3%, +20k steps 3% more, vs
    # the 7.9x a usable 16x16 needs). Four cells, one variable each.
    # 1. Volume: 8x8 works, 16x16 does not, nothing between has run.
    "H2_d144_c50_s223_letf_fmo2_50k_curr": replace(
        _d144_curriculum_cell(
            "H2_d144_c50_s223_letf_fmo2_50k_curr",
            head_kind="factorised",
        ),
        ema_decay=0.9999,
        site_orderings=("row", "col"),
    ),
    # 2. Capacity: hidden_dim has been 32 at every volume ever run.
    "H2_d64_c50_s223_letf_fmo2_h128_50k_curr": _d64_fmo2_h128_cell(
        "H2_d64_c50_s223_letf_fmo2_h128_50k_curr",
    ),
    # 3. Transfer x resolution, as a two-arm pair one variable apart.
    "H2_d256_c50_s223_letf_fmo2_20k_sc_warm": _d256_fmo2_warm_cell(
        "H2_d256_c50_s223_letf_fmo2_20k_sc_warm",
    ),
    # Cancelled, never ran: its question ("does training at the finer grid beat
    # re-rolling finer?") was answered by the b512+ne512 pair (endpoint null)
    # and the eval-only grid sweep.
    "H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512": _d256_fmo2_warm_cell(
        "H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512",
        n_euler_steps=512,
    ),
    # --- 16x16 stage-1 screen: flat sigma=0.10, 5k steps, naive c_t, one
    # variable per arm, read on stage-tail FVU (ranges in the _d256_scr5k_cell
    # docstring). The MA pair anchors the archived failure (base 0.119); the
    # fmo2 arms screen the head any scaled run would use. lr has never been
    # varied across the archive, so every capacity arm (h128, L3) ships with an
    # lr=3e-4 sibling: the one capacity variation ever run failed on un-retuned lr.
    "H2_d256_scr5k_ma": _d256_scr5k_cell(
        "H2_d256_scr5k_ma",
        "masked_attention",
        eval_sample_chunk=128,
    ),
    "H2_d256_scr5k_ma_h128_lr03": _scr5k_ma_h128_lr03_cell(
        "H2_d256_scr5k_ma_h128_lr03",
    ),
    # muP-init arm: the bridge arm with readout_score_scale = 32/128 the only
    # change, the causal test of the muP diagnosis (readout dot has no fan-in
    # compensation; G ~ sqrt(h) at init). Plumbing check via init_diagnostics
    # (0.0023753 = 0.25x the bridge's 0.0095), not the first CSV row (0.00164).
    "H2_d256_scr5k_ma_h128_lr03_mup": replace(
        _scr5k_ma_h128_lr03_cell("H2_d256_scr5k_ma_h128_lr03_mup"),
        readout_score_scale=32 / 128,
    ),
    # d-scaled clip arm: the MA screen base with grad_clip_max_norm 500 -> 60,000
    # the only change, the causal test of "clip changes meaning with lattice
    # size". Scale set by measurement: early median grad_norm 925-1108 at d64 vs
    # 1.0-1.2e5 at d256 (~120x), so 60,000 restores p50(grad)/clip ~ 2.
    "H2_d256_scr5k_ma_clip60k": _scr5k_ma_clip60k_cell(
        "H2_d256_scr5k_ma_clip60k",
    ),
    "H2_d256_scr5k_fmo2": _scr5k_fmo2_cell("H2_d256_scr5k_fmo2"),
    "H2_d256_scr5k_fmo2_h128": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_h128",
        hidden_dim=128,
    ),
    "H2_d256_scr5k_fmo2_h128_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_h128_lr03",
        hidden_dim=128,
        lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_lr03",
        lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_L3": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_L3",
        n_layers=3,
    ),
    "H2_d256_scr5k_fmo2_L3_lr03": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_L3_lr03",
        n_layers=3,
        lr=3e-4,
    ),
    "H2_d256_scr5k_fmo2_clip2000": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_clip2000",
        grad_clip_max_norm=2_000.0,
    ),
    "H2_d256_scr5k_fmo2_ne512_b512": _scr5k_fmo2_with(
        # n_grid also sets c_t slots, buffer size and the loss's t-support, and
        # inner_batch/n_grid is the per-slot gradient density, so grid and batch
        # move together to hold density at 1.0 (else the arm tests starvation).
        "H2_d256_scr5k_fmo2_ne512_b512",
        n_euler_steps=512,
        batch_size=512,
    ),
    # --- screen phase 2: two arms, one variable each against the same base pair
    # (0.0364/0.0442).
    # b512: batch 512 at the base grid, the batch-only decomposition of the
    # ne512_b512 bundle (honest FVU 0.0118 vs base 0.0286; its grid-stress
    # instruments read unstressed). loss_microbatch_size=128 rides for the B_crit
    # readout (McCandlish two-batch identity), not memory; gradient-exact.
    "H2_d256_scr5k_fmo2_b512": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_b512",
        batch_size=512,
        loss_microbatch_size=128,
    ),
    # cv: the published control-variate estimator, cold, on the factorised head;
    # the archived "CV inverts cold at d256" was measured on MA and tracked rate
    # mis-scaling, not size. fmo2's cold init is clean (lambda_dt_clipped_frac 0),
    # so the precondition holds from step 0. Read: integrand var-ratio tail < 0.5.
    "H2_d256_scr5k_fmo2_cv": replace(
        _scr5k_fmo2_cell("H2_d256_scr5k_fmo2_cv"),
        estimator="control_variate",
    ),
    # c_t decoupling arm: the fmo2 screen base with c_t_batch=512 the only change;
    # gradient batch stays 128. b512 explained the bundle (honest FVU 0.0115 vs
    # 0.0286) with B_crit ~45-91, predicting the gain is c_t-estimator variance.
    # Honest-FVU floor is 1/512 here (b512 convention). Modal run, tag 20260819-215249.
    "H2_d256_scr5k_fmo2_ctb512": _scr5k_fmo2_with(
        "H2_d256_scr5k_fmo2_ctb512",
        c_t_batch=512,
    ),
    "H2_d256_scr20k_fmo2": replace(
        # The horizon control: the first flat-subcritical 16x16 run past 5k steps.
        # If the stage-1 deficit is slow convergence rather than a floor, this arm
        # falls toward 0.03 after 5k and every scr5k null gets re-read; if it
        # stays at ~0.119 for 4x the horizon, the 5k screen read holds.
        _scr5k_fmo2_cell("H2_d256_scr20k_fmo2"),
        train=replace(_scr5k_fmo2_cell("H2_d256_scr20k_fmo2").train, n_steps=20_000),
    ),
    "H2_d256_scr5k_mo": _scr5k_mo_cell("H2_d256_scr5k_mo"),
    # --- 12x12 volume bracket of the archived failure (replaces the fmo2 12x12
    # cell as the bracketing rung; see the builder docstring). Two seeds: a
    # headline scaling-curve point, not a screen arm.
    "H2_d144_c50_s223_letf_ma_50k_curr_naive": _d144_ma_bracket_cell(
        "H2_d144_c50_s223_letf_ma_50k_curr_naive",
    ),
    # --- clip-threshold continuation of the completed naive rescue
    "H2_d256_c50_s223_letf_ma_10k_sc_clip2000": _d256_clip2000_cont_cell(
        "H2_d256_c50_s223_letf_ma_10k_sc_clip2000",
    ),
    # Band-capacity push, batch 1: three single-variable twins of ma_50k_curr.
    # Discriminator: the interval head, head_kind the only change. Separates
    # "shared band content is the bottleneck" (lands in the MA band ~0.75-0.78)
    # from "the MA aggregator is" (lands well above it).
    "H2_d64_c50_s223_letf_iv_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_iv_50k_curr",
        head_kind="interval",
    ),
    # Literal factorisation test at d64: the MA / interval rungs with
    # exterior_combiner="bilinear" and the EMA instrument (never touches
    # training). Bars: mab vs MA 0.781, parity >= 0.76 raw; ivb vs interval 0.646.
    **{
        f"H2_d64_c50_s223_letf_{arm}_50k_curr": _d64_curriculum_cell(
            f"H2_d64_c50_s223_letf_{arm}_50k_curr",
            head_kind=head_kind,
            exterior_combiner="bilinear",
            ema_decay=0.9999,
        )
        for arm, head_kind in (("mab", "masked_attention"), ("ivb", "interval"))
    },
    # H-width: double the band feature and attention widths, nothing else.
    "H2_d64_c50_s223_letf_ma_wide_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_wide_50k_curr",
        head_kind="masked_attention",
        band_feature_dim=32,
        attention_dim=64,
    ),
    # H-offsets: band also sees offset-2 / offset-2D bond families (second-
    # neighbour row/column pairs), beyond the energy's (1, D).
    "H2_d64_c50_s223_letf_ma_offs_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_offs_50k_curr",
        head_kind="masked_attention",
        pair_offsets=(1, 2, 8, 16),
    ),
    # Round-2 stencil family: the reported MA head + the 5-point lattice-stencil
    # band family (neighbours ±1, ±D on the flattened D x D grid), narrow unary+
    # offset families kept for the collar. use_stencil is the only change vs
    # ma_50k_curr: does richer 2D-local band content lift the ~0.78 ceiling?
    "H2_d64_c50_s223_letf_ma_stencil_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_stencil_50k_curr",
        head_kind="masked_attention",
        use_stencil=True,
    ),
    # Horizon extension: the 50k budget cuts both heads off mid-descent (loss
    # still falling 12.0% ma / 7.4% stencil over the final 10k), so the 0.78/0.80
    # "ceiling" is read off unconverged runs. n_steps is the only change; the
    # ladder pins absolute start_steps, so the extra 50k land on the final plateau.
    "H2_d64_c50_s223_letf_ma_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_100k_curr",
        head_kind="masked_attention",
        n_steps=100_000,
    ),
    "H2_d64_c50_s223_letf_ma_stencil_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_stencil_100k_curr",
        head_kind="masked_attention",
        n_steps=100_000,
        use_stencil=True,
    ),
    # The missing twin: mo_50k_curr was still descending fastest at its cutoff
    # (loss -25.7% over the final 15k, train ESS 0.864 -> 0.900), so the
    # reference rung is unconverged too. Seed 42, not the record's 43: same-seed
    # MA s43 is 0.7546, not the 0.7806 usually cited against MO's 0.9103.
    "H2_d64_c50_s223_letf_mo_100k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_mo_100k_curr",
        head_kind="mask_one",
        n_steps=100_000,
    ),
    # Mixing-probe floor cell (sigma=0.10, 8x8): the probe's control operating
    # point, where Kawasaki mixes happily. Direct subcritical training with no
    # curriculum (a floor cell on the ladder would measure the curriculum, not
    # the operating point); every other knob is the d64 sigma_c shape verbatim.
    "H2_d64_c50_s010_letf_ma_50k": _hard_cell(
        "H2_d64_c50_s010_letf_ma_50k",
        sigma=0.10,
        head_kind="masked_attention",
        D=8,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=256,
        n_eval_samples_training=512,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    ),
    # The 16x16 rung, the last training rung: the masked-attention head on
    # the vertex-disjoint matching step (drop-in at the 8x8 checkpoint: ESS
    # 0.906 vs 0.910) at the 8x8 50k sigma-ladder recipe unchanged.
    # n_euler_steps stays 128 because the matching step decouples trajectory
    # length from total rate. Eval deltas are diagnostics-only: cadence 200 ->
    # 500, in-training draw 512 -> 256, chunk 256 -> 64; final eval 5000 draws.
    "H2_d256_c50_s223_letf_ma_50k_curr": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
    ),
    # 50k rescue: the naive-estimator twin of the diverged cell above;
    # estimator is the only change, isolating the inverted control variate
    # (adds 2.3-70x variance at d=256 vs an 8-30x reduction at d64).
    "H2_d256_c50_s223_letf_ma_50k_curr_naive": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr_naive",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        estimator="naive_mc",
    ),
    # Cold fmo2 ladder at 16x16: the archived MA naive-rescue recipe with the
    # head family (+ riding EMA shadow, dual site orderings, eval chunk) the
    # only change, so any difference from the rescue's Var[log w]/site 0.0707 /
    # ESS/N 0.0031 is chargeable to the head. Pinned in test_screen_pins.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive": _d256_fmo2_ladder_cell(
        "H2_d256_c50_s223_letf_fmo2_50k_curr_naive",
        estimator="naive_mc",
    ),
    # The recipe run: the composed best-effort sampler at 16x16, a demonstration
    # arm. Levers vs the anchor: batch_size 512 + n_euler_steps 512 (moved as a
    # pair, per-slot gradient density 1.0) and loss_microbatch_size 128 (noise-scale
    # instrument); 50k horizon matches the anchor.
    # Read: EMA ESS/N 0.0105, Var/site 0.0061.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive": _d256_fmo2_ladder_cell(
        "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive",
        estimator="naive_mc",
        n_euler_steps=512,
        batch_size=512,
        loss_microbatch_size=128,
    ),
    # CV twin of the recipe; estimator the only change. No loss_microbatch:
    # defined when a batch-coupled CV was feared to break the per-row
    # decomposition (since resolved, c_t is a detached per-row gather; the
    # archived run stays as it ran, new CV cells microbatch freely).
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_cv": _d256_fmo2_ladder_cell(
        "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_cv",
        estimator="control_variate",
        n_euler_steps=512,
        batch_size=512,
    ),
    # CV continuation of the recipe: flat sigma_c, 20k steps, lr 3e-4, --init-from
    # the seed-43 recipe final.pt; estimator control_variate is the arm variable,
    # halt_on_cv_inversion_after=2000 the cost-capped tripwire. No loss_microbatch
    # (archived; see the cv variant). Judged PASS: EMA ESS/N 0.2655 vs parent 0.0198.
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
    # Naive twin of the CV continuation: estimator control_variate -> naive_mc the
    # only declared change, same seed-43 final.pt, splitting the CV arm's 13.4x
    # (0.2655 vs 0.0198) between estimator and 20k extra sigma_c steps; tripwire
    # dropped (meaningless without a CV). Judged ESTIMATOR-OWNS: naive recovers 10.7%.
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
    # ---- ne128 x CV composition family -------
    # Composes two levers measured independently: the ne128 training grid
    # (keystone `H2_..._b512_ne128_naive`, GRID-HELPS at EMA Var/site 0.00485
    # vs ne512 0.00612) and the CV continuation (`..._20k_sc_cv2_b512_ne512`,
    # ESTIMATOR-OWNS at EMA ESS/N 0.2655 vs 0.0198). loss_microbatch_size=128
    # rides on all six arms, not a declared variable (the keystone runs it;
    # gradient-exact).
    # cvcont-ne128: CV continuation of the keystone, --init-from its own
    # final.pt; n_euler_steps 512 -> 128 the one variable vs cvcont. Primary
    # statistic is the tail (EMA ESS/N, top weight), since Var/site already
    # coincides (0.00485 vs 0.00495). Tripwire 2000 as the ne512 twin; seed 42 vs
    # cvcont's 43.
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
    # cold-cv70k: control variate on from step 0 at the matched budget --
    # 70k with the same ladder = 30k ladder + 40k sigma_c, as cvcont-ne128's
    # 50k + 20k. Tripwire 5000 (one full stage; a cold start has no healed model,
    # the ~914-step healing precedent is warm). Judged: EMA ESS/N 0.381 (0.346, 0.419).
    "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2": (
        lambda _cell: replace(
            _cell,
            train=replace(_cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2",
            estimator="control_variate",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # ---- exact-field twin of cold-cv70k ----------
    # One variable vs cold-cv70k: exact_field_channel=True. Prior from d64:
    # +0.09 raw ESS on this prefix-band chassis (fimo2ef 0.839 vs fimo2 0.750),
    # null on the attention-band one. Primary Var/site vs the twin's
    # (0.00344, 0.00372); single seed, 18% d256 seed-spread caveat. Tripwire 5000.
    "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2_ef": (
        lambda _cell: replace(
            _cell,
            train=replace(_cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000),
            exact_field_channel=True,
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2_ef",
            estimator="control_variate",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # ---- capacity twins of the composition family --
    # Levers: hidden_dim 32 -> 128 and n_layers 2 -> 3, bundled (n_heads stays
    # 4, so head_dim rides 8 -> 32); loss_microbatch 128 rides as on the family.
    # Prior is negative: `..._fmo2_h128_lr03_50k_curr_naive` read REGRESSION
    # (Var/site 0.0229 vs anchor 0.0168, CI-disjoint), but on a base 2.8x worse
    # than the recipe; re-asked because the CV moves the estimator-variance constraint.
    # h128L3 parent: the naive 50k parent, continuation source for the h128L3
    # cvcont (weight shapes change with hidden_dim) and the full-horizon capacity
    # re-read vs the keystone (Var/site 0.00485 (0.00463, 0.00508)). Keeps the
    # ladder lr; the lr03 sibling separates the un-retuned-lr artefact. Judged: 0.00574.
    "H2_d256_c50_s223_letf_fmo2_h128L3_50k_curr_b512_ne128_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_50k_curr_b512_ne128_naive",
            estimator="naive_mc",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # h128L3 lr03 parent: the h128L3 parent with curriculum lr flattened to
    # 3e-4 at every stage (the landed capacity arm's treatment), the lr/capacity
    # de-confound at full horizon. Judged LR-ARTEFACT: EMA Var/site 0.00295 vs the
    # ladder-lr parent's 0.00574, CI-disjoint, so the parent's capacity read is void.
    "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            curriculum=replace(
                _cell.curriculum,
                stages=tuple(
                    replace(stage, lr=3e-4) for stage in _cell.curriculum.stages
                ),
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive",
            estimator="naive_mc",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # h128L3 cvcont: capacity twin of cvcont-ne128, 20k CV continuation at
    # fixed sigma_c run from both h128L3 parents (tags 20260822-C-cvcont-h128L3
    # -fromP and -fromP03); both endpoints reported, Bonferroni factor 2 on
    # the comparison. Primary statistic the tail, as at cvcont-ne128.
    "H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            train=replace(
                _cell.train,
                batch_size=512,
                loss_microbatch_size=128,
                halt_on_cv_inversion_after=2000,
            ),
        )
    )(
        _d256_fmo2_warm_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128",
            n_euler_steps=128,
        )
    ),
    # h128L3 cold-cv70k: capacity twin of cold-cv70k (CV from step 0, 70k),
    # completing the 2x2 {when CV joins} x {capacity}. Tripwire 5000 as
    # cold-cv70k; a tripwire asymmetry between the pair is reported ahead of
    # any endpoint. Judged NULL vs cold-cv70k: EMA ESS/N 0.430.
    "H2_d256_c50_s223_letf_fmo2_h128L3_70k_curr_b512_ne128_cv2": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            train=replace(_cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_70k_curr_b512_ne128_cv2",
            estimator="control_variate",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # h128L3 lr03 cold-cv70k: the h128L3 cold-cv70k arm with curriculum lr
    # flattened to 3e-4; exists because the lr03 parent read LR-ARTEFACT, so the
    # ladder-lr arm's NULL is void as a capacity read. Primary vs cold-cv70k's
    # (0.346, 0.419); secondary vs the ladder-lr arm's 0.430. Tripwire 5000 unchanged.
    "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_70k_curr_b512_ne128_cv2": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128, n_layers=3),
            train=replace(_cell.train, n_steps=70_000, halt_on_cv_inversion_after=5000),
            curriculum=replace(
                _cell.curriculum,
                stages=tuple(
                    replace(stage, lr=3e-4) for stage in _cell.curriculum.stages
                ),
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_70k_curr_b512_ne128_cv2",
            estimator="control_variate",
            n_euler_steps=128,
            batch_size=512,
            loss_microbatch_size=128,
        )
    ),
    # buf2: the recipe cell with replay_buffer_cycles 8 -> 2 the only change,
    # restoring the ~1024 retained trajectories of the reference invariant.
    # Cancelled before it started: the d64 battery's retention curve
    # (256/1024/2048/4096 -> EMA 0.766/0.810/0.827/0.860, monotone) refutes the
    # premise; see cyc16.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2",
            estimator="naive_mc",
            n_euler_steps=512,
            batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_buf2",
                estimator="naive_mc",
                n_euler_steps=512,
                batch_size=512,
                loss_microbatch_size=128,
            ).train,
            replay_buffer_cycles=2,
        ),
    ),
    # Keystone grid arm: the recipe cell with n_euler_steps 512 -> 128 the only
    # change, the missing d256 training-grid read vs the recipe run (Var/site
    # 0.0061 (0.0058, 0.0064)); Var-primary since d256 ESS is top-weight-dominated.
    # Judged GRID-HELPS: EMA Var/site 0.00485 (0.00463, 0.00508); ne128 is the
    # standing d256 grid.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive": _d256_fmo2_ladder_cell(
        "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive",
        estimator="naive_mc",
        n_euler_steps=128,
        batch_size=512,
        loss_microbatch_size=128,
    ),
    # cyc16: the recipe cell with replay_buffer_cycles 8 -> 16 the only change,
    # retention 4096 -> 8192 trajectories at unchanged freshness (fresh draws
    # per step stay 5.12), the direction the d64 retention curve supports.
    # Var-primary vs the recipe run's 0.0061 (0.0058, 0.0064).
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16",
            estimator="naive_mc",
            n_euler_steps=512,
            batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_cyc16",
                estimator="naive_mc",
                n_euler_steps=512,
                batch_size=512,
                loss_microbatch_size=128,
            ).train,
            replay_buffer_cycles=16,
        ),
    ),
    # Rank arm at the anchor recipe: the fmo2 naive ladder anchor with
    # bilinear_rank 8 -> 32 the only change (d/8 = 32 at 256 sites; the
    # factorised-head forensics measured required rank growing ~d/8).
    # Binds iff EMA eval ESS/N >= 0.010 vs the anchor pair 0.0048/0.0058.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_rank32": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_rank32",
            estimator="naive_mc",
        ),
        bilinear_rank=32,
    ),
    # Third causal ordering at the anchor recipe: site_orderings
    # ("row","col") -> ("row","col","diag"), the only change. Binds iff EMA
    # eval ESS/N >= 0.010 vs the anchor pair 0.0048/0.0058; the d64 diag arm
    # landed flat (EMA 0.8069 vs the rung's 0.8104).
    "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_diag": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_naive_diag",
            estimator="naive_mc",
        ),
        site_orderings=("row", "col", "diag"),
    ),
    # Capacity arm at the anchor recipe: hidden_dim 32 -> 128 and flat lr
    # 3e-4 across all curriculum stages, bundled because un-retuned lr 1e-3
    # penalised h128 on the 5k screen (tail FVU 0.0498 vs 0.0380-0.0406).
    # Binds iff EMA eval ESS/N >= 0.010 vs the anchor pair 0.0048/0.0058.
    "H2_d256_c50_s223_letf_fmo2_h128_lr03_50k_curr_naive": (
        lambda _cell: replace(
            _cell,
            model=replace(_cell.model, hidden_dim=128),
            curriculum=replace(
                _cell.curriculum,
                stages=tuple(
                    replace(stage, lr=3e-4) for stage in _cell.curriculum.stages
                ),
            ),
        )
    )(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_h128_lr03_50k_curr_naive",
            estimator="naive_mc",
        )
    ),
    # ---- 8x8 outer/inner loop battery ----------------------
    # Single-variable twins of the archived fmo2 8x8 curriculum rung via
    # _d64_fmo2_loop_cell (comparator raw ESS/N 0.7452 / EMA 0.8104, seed 42;
    # insensitive = within +-0.03 EMA ESS/N of 0.8104). Draw density =
    # inner_steps x batch / (euler_grid x rollout_width): 0.78 at the archived
    # 8x8 shape, 0.195 on the 16x16 recipe that landed null.
    #
    # Gradient steps per outer cycle 100 -> 25: draw density 0.78 -> 0.195,
    # the 16x16 recipe's density at the healthy size. One-sided: 4x more
    # fresh trajectories bias the arm toward reading insensitive.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_inner25": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_inner25",
        inner_steps_per_outer=25,
    ),
    # Replay depth 8 -> 2 cycles: 1024 -> 256 retained trajectories, below
    # the reference code's ~1024 retention bound. Density is invariant to
    # depth, so this is a pure staleness-vs-diversity dial.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_buf2": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_buf2",
        replay_buffer_cycles=2,
    ),
    # Stepping-protocol control: use_matching_step False -> True at the 128
    # grid; the comparator for the two grid cells below (one-event stepping
    # clips when total-rate x dt > 1, so the grid is clean only under the
    # matching step). A frozen 8x8 checkpoint re-read 0.906 vs 0.910.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match",
        use_matching_step=True,
    ),
    # Euler grid 128 -> 32 under the matching step (0.5 x sites, the archived
    # 16x16 ratio); read against the match control, not the base. Draw
    # density rises to 3.1 as a side effect, biasing against finding harm.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne32": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne32",
        use_matching_step=True,
        n_euler_steps=32,
    ),
    # Euler grid 128 -> 256 under the matching step (4 x sites): with match
    # and match_ne32 a three-point training-grid curve 0.5d / 2d / 4d.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne256": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_match_ne256",
        use_matching_step=True,
        n_euler_steps=256,
    ),
    # Rollout width decoupled from the gradient batch: outer_batch_size 32
    # with c_t_batch 128, so the buffer width falls 4x while c_t and the
    # gradient batch stay at 128. Mechanism pinned in
    # tests/test_outer_batch_decoupling.py.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_m32ct128": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_m32ct128",
        outer_batch_size=32,
        c_t_batch=128,
    ),
    # Gradient steps per outer cycle 100 -> 500: draw density 3.9, the
    # over-reuse direction bracketing 0.78 with inner25. 500 not 400 because
    # every sigma-stage start (multiples of 5000) must divide by the cycle.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_inner500": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_inner500",
        inner_steps_per_outer=500,
    ),
    # Replay depth 8 -> 16 cycles: 2048 retained trajectories, completing
    # the depth curve 256/1024/2048 with buf2.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_cyc16": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_cyc16",
        replay_buffer_cycles=16,
    ),
    # SMC-in-training arms: rollout_resample_ess_fraction (tau) live on the
    # swap route (LEAPS Alg. 1 lines 11-14); flag-off bit-identity is
    # test-pinned, so the archived rung is the control. Primary read is EMA
    # eval Var[log w] vs the rung's 0.2068 (95% CI 0.1983-0.2152), not ESS.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_smc03": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_smc03",
        rollout_resample_ess_fraction=0.3,
    ),
    "H2_d64_c50_s223_letf_fmo2_50k_curr_smc06": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_smc06",
        rollout_resample_ess_fraction=0.6,
    ),
    # Rollout width 128 -> 512 at gradient batch 128, c_t_batch 512: the
    # upward direction of the width decoupling (the reference rolled 2x its
    # batch). Confound: draw density falls 0.78 -> 0.195 at fixed inner
    # steps; inner25 isolates density alone.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_m512ct512": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_m512ct512",
        outer_batch_size=512,
        c_t_batch=512,
    ),
    # Third causal ordering at 8x8: site_orderings ("row","col") ->
    # ("row","col","diag"), the only change vs the rung. Read on EMA eval
    # Var[log w] vs the rung's 0.2068 (95% CI 0.1983-0.2152); regression iff
    # EMA ESS/N <= 0.78 is the early warning against the 16x16 diag arm.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_diag": replace(
        _d64_fmo2_loop_cell("H2_d64_c50_s223_letf_fmo2_50k_curr_diag"),
        site_orderings=("row", "col", "diag"),
    ),
    # Boundary shock at 8x8, size-twin of the 16x16 `_rw` arm below:
    # rewarmup_on_stage=True (fresh 500-step LR ramp at each sigma boundary)
    # plus stage_best_checkpoints=True, which is pure IO. Read: median FVU
    # over the first 500 post-boundary steps vs the stage tail (median, not
    # peak: peak varies 0.68x-3.49x seed-to-seed, median-of-500 0.82-1.22).
    "H2_d64_c50_s223_letf_fmo2_50k_curr_rw": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_rw",
        rewarmup_on_stage=True,
        stage_best_checkpoints=True,
    ),
    # No-flush at 8x8: flush_replay_on_stage False the only change (depth
    # stays 8; the replay window carries states across sigma boundaries).
    # Defined but not run: 16x16 parent recovery spans 100-3,057 steps and
    # falls monotonically down the ladder, not the constant 800-step refill
    # a flush-driven transient would give. Reopen iff the 16x16 `_rw` is null.
    "H2_d64_c50_s223_letf_fmo2_50k_curr_noflush": _d64_fmo2_loop_cell(
        "H2_d64_c50_s223_letf_fmo2_50k_curr_noflush",
        flush_replay_on_stage=False,
    ),
    # Boundary-shock arm on the b512/ne512 naive recipe: rewarmup_on_stage
    # the only training-dynamics change, stage_best_checkpoints riding as IO.
    # Confirmed iff EMA eval ESS/N >= 0.04 (2x the parent's 0.0198); null is
    # expected, the peak-to-final decay trigger not surviving a binned re-read.
    "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw": replace(
        _d256_fmo2_ladder_cell(
            "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw",
            estimator="naive_mc",
            n_euler_steps=512,
            batch_size=512,
            loss_microbatch_size=128,
        ),
        train=replace(
            _d256_fmo2_ladder_cell(
                "H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne512_naive_rw",
                estimator="naive_mc",
                n_euler_steps=512,
                batch_size=512,
                loss_microbatch_size=128,
            ).train,
            rewarmup_on_stage=True,
            stage_best_checkpoints=True,
        ),
    ),
    # Phase-2 estimator switch: continue the completed naive 50k (--init-from
    # its checkpoints/final.pt) with the Stein control variate re-enabled at
    # flat sigma_c, lr 3e-4. On the trained naive checkpoint the CV cuts the
    # c_t integrand variance 7x (ratio 0.140) where it added 2.3-70x on the
    # diverged model. Pass = train-ESS > 15/256 rising, eval ess_fraction >= 0.05.
    "H2_d256_c50_s223_letf_ma_20k_sc_cv2": _d256_cv2_cell(
        "H2_d256_c50_s223_letf_ma_20k_sc_cv2"
    ),
    # Clip-50 twin of the diverged cell above (its inherited clip 500 was
    # exceeded twelve-fold from init at d=256). Never run: under AdamW two
    # thresholds that both saturate every step differ by a constant the
    # optimiser erases, and the twin's minimum pre-clip norm was 186. Do not launch.
    "H2_d256_c50_s223_letf_ma_50k_curr_clip50": _hard_cell(
        "H2_d256_c50_s223_letf_ma_50k_curr_clip50",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_D64_SIGMA_LADDER,
        grad_clip_max_norm=50.0,
    ),
    # ---- 16x16 rescue smoke ladder -------------------------
    # 12k-step arms, one mechanism each, after the review of the diverged
    # rung; 12k crosses the sigma=0.14 (5k) and 0.17 (10k) boundaries, where
    # the twin's per-rung re-ignition began. Scored on escaping the clip
    # ceiling on rung 0 and a within-rung loss slope <= 0 over 10k-12k.
    # _SMOKE12K_SIGMA_LADDER is the 50k ladder truncated to the stages reached.
    "H2_d256_smoke12k_unclip": _hard_cell(
        # unclip: restore clip=500's d64 semantics (fires on spikes only) by
        # raising the threshold above the working-regime norm.
        "H2_d256_smoke12k_unclip",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        grad_clip_max_norm=20_000.0,
    ),
    "H2_d256_smoke12k_naive": _hard_cell(
        # naive: kill the inverted control variate (it adds variance at
        # d=256, ratio 2.3x at rung 0 -> ~70x late) by estimating c_t naively.
        "H2_d256_smoke12k_naive",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        estimator="naive_mc",
    ),
    "H2_d256_smoke12k_warm": _hard_cell(
        # warm: warm-start from the converged d64 ma_100k checkpoint via
        # scripts/warm_start_swap_head.py and run.py --init-from; recipe
        # otherwise unchanged (clip 500).
        "H2_d256_smoke12k_warm",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
    ),
    "H2_d256_smoke12k_stadamw": _hard_cell(
        # stadamw: StableAdamW per-tensor update clipping; raw-gradient clip
        # effectively disabled so update clipping is the only bound.
        "H2_d256_smoke12k_stadamw",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        grad_clip_max_norm=1e9,
        optimiser="stable_adamw",
    ),
    "H2_d256_smoke12k_rewarmup": _hard_cell(
        # rewarmup: re-run the lr warmup ramp at every sigma transition (the
        # smoke twin's one healthy window ended at a transition). Clip left at
        # the inherited 500 to isolate the transition variable.
        "H2_d256_smoke12k_rewarmup",
        sigma=0.223,
        head_kind="masked_attention",
        D=16,
        n_steps=12_000,
        n_euler_steps=128,
        n_eval_samples=1000,
        eval_sample_chunk=64,
        n_eval_samples_training=256,
        eval_every=500,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
        use_matching_step=True,
        curriculum=_SMOKE12K_SIGMA_LADDER,
        rewarmup_on_stage=True,
    ),
    # --- replay-buffer staleness ablation ---
    # d64 MA curriculum recipe at the 12k smoke horizon with the replay buffer
    # window 8 -> 2 the only lever, read against the archived MA twin's first
    # 12k steps. Insensitive iff final train-ESS is within +-10% of the twin.
    "H2_d64_smoke12k_replay2": _d64_smoke12k_replay2_cell(
        "H2_d64_smoke12k_replay2",
    ),
    # --- c_t EMA no-regression check ---
    # Archived MA 50k curriculum recipe with c_t_ema_halflife_cycles=4 the only
    # lever. Judged: no regression (>= 0.755 bar met) but neutral at 1.05x,
    # inside the c_t-noise family's ~1.19x transfer ceiling.
    "H2_d64_c50_s223_letf_ma_50k_curr_ctema4": _d64_m2_ctema4_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr_ctema4",
    ),
    # --- c_t transfer function: the missing naive arm -------
    "H2_d64_c50_s223_letf_ma_50k_curr_naive": _d64_naive_twin_cell(
        "H2_d64_c50_s223_letf_ma_50k_curr_naive",
    ),
    # --- decoupled c_t rollout batch ---
    # d256-naive 12k smoke with c_t_batch 128 -> 512 the only lever (c_t
    # per-slot standard error halves; inner batch and replay stay 128). Bar:
    # 2x on the sigma=0.17 rung's train-ESS. Judged: not meaningful, 1.09x.
    "H2_d256_smoke12k_naive_ctb512": _d256_smoke12k_naive_ctb512_cell(
        "H2_d256_smoke12k_naive_ctb512",
    ),
    # d64 plumbing twin of the cell above: integration smoke for the enlarged
    # rollout outer cycle; no quality bar (test_c_t_batch.py pins semantics).
    "H2_d64_smoke12k_ctb512": _d64_smoke12k_ctb512_cell(
        "H2_d64_smoke12k_ctb512",
    ),
    # --- grouped-anchor family: retired after batch 1, kept as the negative
    # result the writeup cites; do not launch further ga cells at d64 ---
    # k masked passes per step on the 50k curriculum, read against
    # iv 0.6463 < ma 0.7806 < stencil 0.8046 < mo 0.9103. Judged: ga8 0.581
    # (under the ladder floor), ga16 0.714, ga8_contig 0.009; group shape
    # dominates k (diagonal-vs-contiguous gap 0.572 vs k gap 0.133).
    # ga8: k=8 masks 12.5% of sites per pass at 1/8 of mask_one's body cost.
    "H2_d64_c50_s223_letf_ga8_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga8_50k_curr",
        head_kind="grouped_anchor",
        n_groups=8,
    ),
    # ga16: k=16 halves the masked fraction for 2x the passes (the slope in k).
    "H2_d64_c50_s223_letf_ga16_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga16_50k_curr",
        head_kind="grouped_anchor",
        n_groups=16,
    ),
    # ga8_contig: same k and cost, but the masked sites form a raster row
    # instead of a dispersed Latin-square diagonal (the shape control).
    "H2_d64_c50_s223_letf_ga8_contig_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ga8_contig_50k_curr",
        head_kind="grouped_anchor",
        n_groups=8,
        grouping="contiguous",
    ),
    "H2_d64_c50_s223_letf_mo": _hard_cell(
        "H2_d64_c50_s223_letf_mo",
        sigma=0.223,
        head_kind="mask_one",
        D=8,
        n_euler_steps=128,
        n_eval_samples=5000,
        # 256 samples x 64 anchor copies stacked under no_grad (~22 GB unchunked).
        eval_sample_chunk=256,
        # In-training ESS is a diagnostic; 512 keeps the eval from dominating
        # the run. run.py's final eval still draws the full 5000.
        n_eval_samples_training=512,
        # SDPA readout everywhere; bf16 on the in-training eval only, final fp32.
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    ),
}


# ---- two-hole patch twins of the cold-cv70k arm ----
# Parent = the h32 cold-cv70k arm (the h128/L3 lever lives in the causal
# stacks, which this head never runs); each twin moves one lever by `replace`:
# head_kind (R=1), + exact_field_channel, patch_radius=2. Pinned by
# test_thp_d256_twins_of_arm_b. Read (EMA ESS/N, seed 42): strong > 0.430
# (the h128/L3 cold-CV record), pass in (0.381, 0.430] (beats parent), else null.
_ARM_B = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2"]
CONFIGS.update(
    {
        "H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2": replace(
            _ARM_B,
            name="H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2",
            head_kind="two_hole_patch",
        ),
        "H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2_ef": replace(
            _ARM_B,
            name="H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2_ef",
            head_kind="two_hole_patch",
            exact_field_channel=True,
        ),
        "H2_d256_c50_s223_letf_thp2_70k_curr_b512_ne128_cv2": replace(
            _ARM_B,
            name="H2_d256_c50_s223_letf_thp2_70k_curr_b512_ne128_cv2",
            head_kind="two_hole_patch",
            patch_radius=2,
        ),
    }
)

# ---- wave-2 house-table fill: 4x4 and 8x8 rungs at single provenance ----
# Arms mo/ma/fimo2ef/thp x sigma {0.10 floor, SIGMA_C exact} x seeds 42/43/44
# on the optimised recipe (s220 = exact 0.220343; archived s223 cells stay).
# d64 s220 reuses the ladder with only the endpoint moved (0.215 < 0.220343
# keeps stage order); d64 s010 trains flat, no curriculum. Pinned by
# test_wave2_house_cells_mirror_archived_twins_except_declared_fields.
_WAVE2_ARM_KNOBS: dict[str, dict] = {
    "mo": {"head_kind": "mask_one"},
    "ma": {"head_kind": "masked_attention"},
    "fimo2ef": {
        "head_kind": "factorised",
        "exact_field_channel": True,
        "interior_band": "prefix",
        "site_orderings": ("row", "col"),
    },
    # fmo2ef: ef on the global-interior chassis (archived fmo2, no interior_band,
    # + exact_field_channel); no archived namesake. Read: parity iff within the
    # w2 fimo2ef sibling's seed spread; plain parent fmo2 d64 s223 seed 42 =
    # 0.745 raw / 0.810 EMA alongside.
    "fmo2ef": {
        "head_kind": "factorised",
        "exact_field_channel": True,
        "site_orderings": ("row", "col"),
    },
    "thp": {"head_kind": "two_hole_patch"},
}

_D64_SIGMA_LADDER_SC = CurriculumCfg(
    stages=_D64_SIGMA_LADDER.stages[:-1]
    + (replace(_D64_SIGMA_LADDER.stages[-1], sigma=SIGMA_C),)
)


# 4x4-only arms, kept out of _WAVE2_ARM_KNOBS so they never acquire an 8x8
# cell: the doubly-hollow oracle is ~1,170x the masked-attention forward at
# d=64, which is why the 8x8 eval table has no oracle row.
_D16_ONLY_ARM_KNOBS: dict[str, dict] = {
    "dh": {"head_kind": "doubly_hollow"},
}


def _wave2_d16_cell(arm: str, sigma_label: str, sigma: float) -> HardStageCfg:
    knobs = dict({**_WAVE2_ARM_KNOBS, **_D16_ONLY_ARM_KNOBS}[arm])
    name = f"H2_d16_c50_{sigma_label}_letf_{arm}_10k_w2"
    cell = _hard_cell(
        name,
        sigma=sigma,
        head_kind=knobs.pop("head_kind"),
        n_steps=10_000,
    )
    return optimised_recipe(replace(cell, **knobs))


def _wave2_d64_critical_cell(arm: str) -> HardStageCfg:
    knobs = dict(_WAVE2_ARM_KNOBS[arm])
    name = f"H2_d64_c50_s220_letf_{arm}_50k_curr_w2"
    cell = _d64_curriculum_cell(name, head_kind=knobs.pop("head_kind"))
    cell = replace(
        cell,
        ising=replace(cell.ising, sigma=SIGMA_C),
        curriculum=_D64_SIGMA_LADDER_SC,
        ema_decay=0.9999,
        **knobs,
    )
    return optimised_recipe(cell)


def _wave2_d64_floor_cell(arm: str) -> HardStageCfg:
    knobs = dict(_WAVE2_ARM_KNOBS[arm])
    name = f"H2_d64_c50_s010_letf_{arm}_50k_w2"
    cell = _hard_cell(
        name,
        sigma=0.10,
        head_kind=knobs.pop("head_kind"),
        D=8,
        n_steps=50_000,
        n_euler_steps=128,
        n_eval_samples=5000,
        eval_sample_chunk=256,
        n_eval_samples_training=512,
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    )
    return optimised_recipe(replace(cell, ema_decay=0.9999, **knobs))


CONFIGS.update(
    {
        **{
            cell.name: cell
            for arm in _WAVE2_ARM_KNOBS
            for cell in (
                _wave2_d16_cell(arm, "s010", 0.10),
                _wave2_d16_cell(arm, "s220", SIGMA_C),
                _wave2_d64_critical_cell(arm),
                _wave2_d64_floor_cell(arm),
            )
        },
    }
)

# Composition-amortisation cells: the thp sigma_c cell with composition_mixture
# the only moved field; grid = the d256 zero-shot fractions realised at d64
# (n+ = 32/30/28/24/20), read per-slice via probe_zero_shot_transfer. Pinned by
# test_camort_cell_is_the_thp_critical_twin_plus_the_mixture_knob.
CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                _wave2_d64_critical_cell("thp"),
                name="H2_d64_camort_s220_letf_thp_50k_curr",
                composition_mixture=(0.5, 0.46875, 0.4375, 0.375, 0.3125),
            ),
            # D=4 gate twin: the fractions integral at d=16 (n+ = 8/7/6/5), read
            # per-slice against exact enumeration by gate_camort_4x4.py.
            replace(
                _wave2_d16_cell("thp", "s220", SIGMA_C),
                name="H2_d16_camort_s220_letf_thp_10k",
                composition_mixture=(0.5, 0.4375, 0.375, 0.3125),
            ),
        )
    }
)

# Oracle row of the 4x4 eval table on the w2 recipe at exact SIGMA_C (the
# legacy s223 dh cells predate both). The oracle matches mask_one numerically,
# so the row exists for its FLOP/es column. compile_head=False is the one
# further deviation: the d^2 = 256 unrolled pair forwards never finish
# compiling (26 min without reaching step 1), so the row differs from mask-one
# in head_kind and compile_head; compiled and eager forwards agree to ~1e-5.
CONFIGS.update(
    {
        cell.name: replace(cell, compile_head=False)
        for arm in _D16_ONLY_ARM_KNOBS
        for cell in (
            _wave2_d16_cell(arm, "s010", 0.10),
            _wave2_d16_cell(arm, "s220", SIGMA_C),
        )
    }
)

# Sigma-vs-recipe twins (Modal, seeds 42-44, tag 20260826-hold-fimo2ef). The
# w2 fimo2ef sigma_c cells landed raw ESS 0.637/0.794/0.831 against the
# archived namesake's 0.926-0.938 while every other arm passed; each twin
# walks one delta back (test_hold_twins_isolate_sigma_from_recipe_for_fimo2ef):
#   `_w2sig` = the w2 recipe at the legacy sigma 0.223;
#   `_eager` = exact SIGMA_C with compile_head and c_t_from_rollout off.
_W2_FIMO2EF_SC = CONFIGS["H2_d16_c50_s220_letf_fimo2ef_10k_w2"]
CONFIGS.update(
    {
        "H2_d16_c50_s223_letf_fimo2ef_10k_w2sig": replace(
            _W2_FIMO2EF_SC,
            name="H2_d16_c50_s223_letf_fimo2ef_10k_w2sig",
            ising=replace(_W2_FIMO2EF_SC.ising, sigma=0.223),
        ),
        "H2_d16_c50_s220_letf_fimo2ef_10k_eager": replace(
            _W2_FIMO2EF_SC,
            name="H2_d16_c50_s220_letf_fimo2ef_10k_eager",
            compile_head=False,
            train=replace(_W2_FIMO2EF_SC.train, c_t_from_rollout=False),
        ),
    }
)

# Round 2: round 1 localised the drop to recipe x exact sigma_c x ef
# (interaction -0.132; both-on 0.637-0.831 vs both-off 0.874-0.944). One twin
# per open attribution, same seeds and tag:
#   `_cmpl` = c_t_from_rollout off, compile_head kept;
#   `_w2rec` = the plain fimo2 chassis (ef off) on the full w2 recipe.
CONFIGS.update(
    {
        "H2_d16_c50_s220_letf_fimo2ef_10k_cmpl": replace(
            _W2_FIMO2EF_SC,
            name="H2_d16_c50_s220_letf_fimo2ef_10k_cmpl",
            train=replace(_W2_FIMO2EF_SC.train, c_t_from_rollout=False),
        ),
        "H2_d16_c50_s220_letf_fimo2_10k_w2rec": replace(
            _W2_FIMO2EF_SC,
            name="H2_d16_c50_s220_letf_fimo2_10k_w2rec",
            exact_field_channel=False,
        ),
    }
)


# Eager factorised cells (`_w2e`): the w2 house cells with compile_head=False
# the one declared deviation. The investigation above localised a ~40%
# catastrophic-seed rate to factorised x compile_head x SIGMA_C (5/12 bad
# seeds vs 0/21 elsewhere, Fisher p=0.0033) and exonerated c_t_from_rollout,
# so c_t stays. Judged: fmo2ef seed 44 = 0.531 tripped the 0.70 reopen bar;
# both 4x4 cells print with a dagger and fmo2ef is retired from forward waves.
_W2_FMO2EF_SC = CONFIGS["H2_d16_c50_s220_letf_fmo2ef_10k_w2"]
_W2_D64_FIMO2EF_SC = CONFIGS["H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2"]
_W2_D64_FMO2EF_SC = CONFIGS["H2_d64_c50_s220_letf_fmo2ef_50k_curr_w2"]
CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                _W2_FIMO2EF_SC,
                name="H2_d16_c50_s220_letf_fimo2ef_10k_w2e",
                compile_head=False,
            ),
            replace(
                _W2_FMO2EF_SC,
                name="H2_d16_c50_s220_letf_fmo2ef_10k_w2e",
                compile_head=False,
            ),
            replace(
                _W2_D64_FIMO2EF_SC,
                name="H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2e",
                compile_head=False,
            ),
            replace(
                _W2_D64_FMO2EF_SC,
                name="H2_d64_c50_s220_letf_fmo2ef_50k_curr_w2e",
                compile_head=False,
            ),
        )
    }
)


# ---- 16x16 house-table fill at exact sigma_c ----
# thp/thp2/fimo2ef/ma x {0.10 floor 50k flat, SIGMA_C 100k ladder}, seeds
# 42-44, each a `replace` on _ARM_B (pinned by
# test_d256_house_cells_are_declared_transforms_of_arm_b); per-arm levers in
# the dicts below. Judged: fimo2ef EMA ESS 0.515/0.577/0.503, eager holds at
# d256; ma 0.0035/0.0422/0.0002, the head fails at sigma_c (all three printed).
_D256_HOUSE_ARM_KNOBS: dict[str, dict] = {
    "thp": {"head_kind": "two_hole_patch"},
    "thp2": {"head_kind": "two_hole_patch", "patch_radius": 2},
    "fimo2ef": {
        "head_kind": "factorised",
        "exact_field_channel": True,
        "interior_band": "prefix",
        "site_orderings": ("row", "col"),
        # gather_triu_pairs: label-symmetric pair contexts, so d(d-1)/2 unordered
        # pairs replace the d^2 grid; opt-in so archived cells stay byte-identical.
        "gather_triu_pairs": True,
    },
    # site_orderings pinned: ("row","col") inherited from the fmo2 parent rode
    # inertly; without the pin these archived ma cells rebuild as two-ordering.
    "ma": {
        "head_kind": "masked_attention",
        "gather_triu_pairs": True,
        "site_orderings": ("row",),
    },
}

# Loss microbatching per arm: off on the thp arms (single-shot fits and is
# 40% faster), 128 where the gradient-noise-scale instrument rides.
_D256_HOUSE_MICROBATCH: dict[str, int | None] = {
    "thp": None,
    "thp2": None,
    "fimo2ef": 128,
    "ma": 128,
}

# Eval-only, ma alone: the archived d256 masked-attention chunk.
_D256_HOUSE_EVAL_CHUNK: dict[str, int] = {"ma": 128}


def _d256_house_cell(
    arm: str,
    name: str,
    *,
    sigma: float,
    n_steps: int,
    curriculum: CurriculumCfg | None,
) -> HardStageCfg:
    """One house-table cell: _ARM_B with the arm's head knobs, the coupling
    and its schedule, and the per-arm microbatch/eval-chunk fields. Everything
    else rides from the parent, so the row is attributable to the head and
    the coupling."""
    cell = replace(
        _ARM_B,
        name=name,
        ising=replace(_ARM_B.ising, sigma=sigma),
        curriculum=curriculum,
        train=replace(
            _ARM_B.train,
            n_steps=n_steps,
            loss_microbatch_size=_D256_HOUSE_MICROBATCH[arm],
            # Not inherited from _ARM_B: there the halt is a screening cell's
            # answer; on a production run it is a silent truncation (it halted
            # two w3 sigma=0.1 arms at step 5000 of 50000).
            halt_on_cv_inversion_after=None,
        ),
        eval=replace(
            _ARM_B.eval,
            eval_sample_chunk=_D256_HOUSE_EVAL_CHUNK.get(
                arm, _ARM_B.eval.eval_sample_chunk
            ),
        ),
        **_D256_HOUSE_ARM_KNOBS[arm],
    )
    return optimised_recipe(cell)


def _d256_house_critical_cell(arm: str) -> HardStageCfg:
    """sigma_c arm: exact SIGMA_C at the cell and the ladder endpoint, 100k
    steps landing entirely on the final plateau. The eager exception applies
    here only: the factorised arm trains eager."""
    cell = _d256_house_cell(
        arm,
        f"H2_d256_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w3",
        sigma=SIGMA_C,
        n_steps=100_000,
        curriculum=_D64_SIGMA_LADDER_SC,
    )
    is_factorised = _D256_HOUSE_ARM_KNOBS[arm]["head_kind"] == "factorised"
    return replace(cell, compile_head=not is_factorised)


def _d256_house_floor_cell(arm: str) -> HardStageCfg:
    """sigma=0.10 floor arm: flat coupling, no curriculum (a ladder at the
    floor would measure the curriculum), 50k steps as at d64. Compiled
    factorised cells at the floor were healthy, so every floor arm keeps the
    full optimised recipe."""
    return _d256_house_cell(
        arm,
        f"H2_d256_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w3",
        sigma=0.10,
        n_steps=50_000,
        curriculum=None,
    )


CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            *(_d256_house_critical_cell(arm) for arm in _D256_HOUSE_ARM_KNOBS),
            *(_d256_house_floor_cell(arm) for arm in _D256_HOUSE_ARM_KNOBS),
        )
    }
)

# d256 camort confirmation: the thp2 sigma_c cell plus one lever, the mixture,
# at the d64 camort grid's fractions (n+/256 = 128/120/112/96/80), so it
# confirms the d64 design at size. Not the soft chapter's randomised draw set:
# a confirmation must not move the grid. Before: d256 null 0.019 at c=0.25.
_D256_CAMORT_PARENT = _d256_house_critical_cell("thp2")
CONFIGS["H2_d256_camort_s220_letf_thp2_100k_curr"] = replace(
    _D256_CAMORT_PARENT,
    name="H2_d256_camort_s220_letf_thp2_100k_curr",
    composition_mixture=(0.5, 0.46875, 0.4375, 0.375, 0.3125),
)


# --- 20x20 radius probe (w4) ---------------------------------
# Does the two-hole patch head hold one rung above 16x16, and does its radius
# keep paying (d256: R=1 -> R=2 took sigma_c EMA ESS 0.638 -> 0.826)? Floor
# coupling first: a rate-load failure leaves the floor clean and sigma_c
# broken, a statistical failure breaks both. Chassis, backbone, batch 512,
# n_euler 128 and the 50k horizon ride _ARM_B; loss microbatching off.
_D400_RADIUS_ARM_KNOBS: dict[str, dict] = {
    "thp2": {"head_kind": "two_hole_patch", "patch_radius": 2},
    "thp3": {"head_kind": "two_hole_patch", "patch_radius": 3},
}


def _d400_radius_cell(arm: str) -> HardStageCfg:
    """One 20x20 floor cell: _ARM_B with the lattice, the flat 0.10 coupling
    and the arm's radius, and nothing else. The tripwire is cleared as on every
    d256 house cell (a screening halt, a silent truncation on a production run)."""
    cell = replace(
        _ARM_B,
        name=f"H2_d400_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w4",
        ising=replace(_ARM_B.ising, D=20, sigma=0.10),
        curriculum=None,
        train=replace(
            _ARM_B.train,
            n_steps=50_000,
            loss_microbatch_size=None,
            halt_on_cv_inversion_after=None,
        ),
        **_D400_RADIUS_ARM_KNOBS[arm],
    )
    return optimised_recipe(cell)


def _d400_bf16_cell(arm: str) -> HardStageCfg:
    """The bf16 twin of a 20x20 floor cell: its fp32 sibling with
    `train_autocast_bf16` and nothing else, so a bf16-vs-fp32 gap is chargeable
    to the precision. The estimator is not at risk: G is read out in fp32, the
    swap log-ratio is bit-identical under bf16, and the frozen eval stays fp32."""
    cell = _d400_radius_cell(arm)
    return replace(
        cell,
        name=cell.name + "bf16",
        train=replace(cell.train, train_autocast_bf16=True),
    )


# --- 20x20 at the critical coupling --------------------------
# The floor wave sat on the sampling floor (raw ESS 0.988-0.9995, all eight
# cells) and could not discriminate; sigma_c is where the instrument has range
# and where the Kawasaki chain is actually slow. Three deviations from the
# floor cell, all the d256 critical convention: exact SIGMA_C, the sigma ladder
# (reused unrescaled, absolute start_steps), 100k steps. Three seeds, not two.
def _d400_critical_cell(arm: str) -> HardStageCfg:
    """One 20x20 sigma_c cell: its sigma = 0.10 sibling at the exact critical
    coupling, on the reused sigma ladder, for 100k steps. Both arms are patch
    heads, so the factorised eager exception does not reach here; they keep the
    compiled step (test_every_new_probe_cell_rides_the_optimised_recipe)."""
    cell = _d400_radius_cell(arm)
    return replace(
        cell,
        name=f"H2_d400_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w4",
        ising=replace(cell.ising, sigma=SIGMA_C),
        curriculum=_D64_SIGMA_LADDER_SC,
        train=replace(cell.train, n_steps=100_000),
    )


def _d400_critical_bf16_cell(arm: str) -> HardStageCfg:
    """The bf16 twin of a 20x20 sigma_c cell: its fp32 sibling plus
    `train_autocast_bf16` and nothing else. The floor settled the cost half; the
    quality half needs sigma_c, where weight variance can separate. bf16 moves
    the proposal only (swap log-ratio pinned bit-identical under autocast)."""
    cell = _d400_critical_cell(arm)
    return replace(
        cell,
        name=cell.name + "bf16",
        train=replace(cell.train, train_autocast_bf16=True),
    )


# --- 24x24 at the critical coupling --------------------------
# The d400 sigma_c recipe moved to D=24 with R=3 as the anchor (judged at d400:
# R=3 EMA ESS 0.789-0.810 vs R=2 0.633-0.731, disjoint over six seeds) and R=4
# as the one continuation. Capacity held (hidden 32 / 2 layers; wider was a
# regression at d256). bf16 only, no fp32 twin (null at both d400 couplings).
# One new lever: loss microbatching at 128, matching the d256 house microbatch.
_D576_RADIUS_ARM_KNOBS: dict[str, dict] = {
    "thp3": {"patch_radius": 3},
    "thp4": {"patch_radius": 4},
}
_D576_MICROBATCH = 128


def _d576_critical_bf16_cell(arm: str) -> HardStageCfg:
    """One 24x24 sigma_c cell: the d400 R=3 bf16 critical cell moved to D=24,
    with loss microbatching at 128 and the arm's radius, and nothing else. So
    d576-vs-d400 at R=3 differs in lattice and (exact) microbatch only, and
    R=4-vs-R=3 at d576 in the radius only."""
    parent = _d400_critical_bf16_cell("thp3")
    return replace(
        parent,
        name=f"H2_d576_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w5bf16",
        ising=replace(parent.ising, D=24),
        train=replace(parent.train, loss_microbatch_size=_D576_MICROBATCH),
        **_D576_RADIUS_ARM_KNOBS[arm],
    )


# --- The whole-lattice attention window (`mal`) --------------
# The fourth cell of the head's window x weights grid: global x learned, on the
# same unary and bond families the band uses. Each `mal` cell is its `ma`
# sibling at the same size, coupling and budget with `attention_window` its
# only declared change. Judged: won at the 4x4 gate on disjoint seeds, null at
# 8x8 sigma_c (EMA 0.831 +- 0.030 vs `ma` 0.846 +- 0.022; the softmax dilutes).
_MAL_TWINS: dict[str, str] = {
    "H2_d16_c50_s010_letf_ma_10k_w2": "H2_d16_c50_s010_letf_mal_10k_win",
    "H2_d16_c50_s220_letf_ma_10k_w2": "H2_d16_c50_s220_letf_mal_10k_win",
    "H2_d64_c50_s010_letf_ma_50k_w2": "H2_d64_c50_s010_letf_mal_50k_win",
    "H2_d64_c50_s220_letf_ma_50k_curr_w2": "H2_d64_c50_s220_letf_mal_50k_curr_win",
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(CONFIGS[parent], name=name, attention_window="lattice")
            for parent, name in _MAL_TWINS.items()
        )
    }
)


# --- Relative pair position code (`mar`) ---------------------
# The window is not the lever (`mal` null at 8x8), so this arm tests the one
# structural difference left between the raster and patch families: `ma`
# indexes `pair_position_embedding` by absolute site, `thp` by signed torus
# displacement. Each cell is its `ma` sibling with `pair_position_mode` its only
# change. Judged: 8x8 sigma_c EMA 0.788 +- 0.025, disjoint below `ma`'s 0.846.
_MAR_TWINS: dict[str, str] = {
    "H2_d16_c50_s010_letf_ma_10k_w2": "H2_d16_c50_s010_letf_mar_10k_rel",
    "H2_d16_c50_s220_letf_ma_10k_w2": "H2_d16_c50_s220_letf_mar_10k_rel",
    "H2_d64_c50_s010_letf_ma_50k_w2": "H2_d64_c50_s010_letf_mar_50k_rel",
    "H2_d64_c50_s220_letf_ma_50k_curr_w2": "H2_d64_c50_s220_letf_mar_50k_curr_rel",
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(CONFIGS[parent], name=name, pair_position_mode="relative")
            for parent, name in _MAR_TWINS.items()
        )
    }
)


# --- Periodic-RoPE backbone under the masked-attention head ---
# The last untested layer of the raster head's position code: `mar` moved the
# band query's pair code; RoPE moves the backbone's site code (torus rotary
# phases in place of d free position vectors), reaching the band's keys and
# values through P_i and S_j. Each cell is its `ma` sigma_c sibling with
# `model.kind` and `patch_size` its only change; p=1 is the single-variable
# read, p=2 pools far keys 2x2. Smoke first: this backbone was never compiled.
_MAROPE_PARENT = "H2_d64_c50_s220_letf_ma_50k_curr_w2"
_MAROPE_TWINS: dict[str, int] = {
    f"H2_d64_c50_s220_rope{patch_size}_ma_50k_curr_w2": patch_size
    for patch_size in (1, 2)
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS[_MAROPE_PARENT],
                name=name,
                model=replace(
                    CONFIGS[_MAROPE_PARENT].model,
                    kind="rope_vit",
                    patch_size=patch_size,
                ),
            )
            for name, patch_size in _MAROPE_TWINS.items()
        )
    }
)


# --- Bonds in the global term (4x4 gate) --------------
# `fimo2ef` carries bonds only inside the band; the global term is unary. Does
# the exterior bond sum help? Three cells: bond (bonds added), bond1o (bonds,
# row ordering only: can bonds retire the second ordering?), wide (no bonds,
# global term widened to the bond arm's parameter count, the capacity control).
# The sigma_c parent is the eager twin: the eager exception is d16-specific
# (5/12 catastrophic compiled seeds there against 0/21 elsewhere).
_ARM_B_PARENTS = {
    "s010": "H2_d16_c50_s010_letf_fimo2ef_10k_w2",
    "s220": "H2_d16_c50_s220_letf_fimo2ef_10k_w2e",
}
# global_feature_dim matching the bond arm's +576 to within 33 parameters
# (138,482 -> 139,058 with bonds, 139,091 widened) at the 4x4 gate's width.
_ARM_B_MATCHED_GLOBAL_DIM = 19
_ARM_B_ARMS: dict[str, dict] = {
    "fimo2efb_10k_bond": {"global_bond_features": True},
    "fiefb_10k_bond1o": {"global_bond_features": True, "site_orderings": ("row",)},
    "fimo2efw_10k_wide": {"global_feature_dim": _ARM_B_MATCHED_GLOBAL_DIM},
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS[parent],
                name=f"H2_d16_c50_{sigma_label}_letf_{arm}",
                **knobs,
            )
            for sigma_label, parent in _ARM_B_PARENTS.items()
            for arm, knobs in _ARM_B_ARMS.items()
        )
    }
)


# --- Bonds in the global term at the 8x8 rung: the discriminating scale ---
# 4x4 filters, 8x8 discriminates (seed spread five times tighter). Gate read:
# bond1o 0.9107 +- 0.0481 vs baseline 0.8511 +- 0.0423, wide worst at 0.7781.
# bond1o changes two things (adds bonds, drops the second ordering), so the 1o
# arm walks back one: baseline -> 1o costs the ordering, 1o -> bond1o asks if
# bonds recover it. Compiled parent (the `_w2` cell); sigma_c only, the floor
# saturates. Run all three arms on one card so comparisons are within-venue.
_ARM_B_D64_PARENT = "H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2"
_ARM_B_D64_ARMS: dict[str, dict] = {
    # bond: the hypothesis on the two-ordering chassis. Null at the 4x4 gate
    # (0.8546 +- 0.0640 vs 0.8511 +- 0.0423), carried anyway: the gate cannot
    # discriminate in either direction (`mal` reversed between these rungs).
    "fimo2efb_50k_curr_bond": {"global_bond_features": True},
    # bond1o: the arm the gate flagged; bonds, second ordering retired.
    "fiefb_50k_curr_bond1o": {
        "global_bond_features": True,
        "site_orderings": ("row",),
    },
    # 1o: the control -- second ordering retired, no bonds.
    "fief_50k_curr_1o": {"site_orderings": ("row",)},
}
# Matched-parameter controls not run (the bond family adds 576 parameters,
# 0.4%). The 4x4 wide cell does not settle capacity: seeds 0.6418/0.8196/0.8728,
# one low seed carried the mean. Run the control before believing any lift here.

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS[_ARM_B_D64_PARENT], name=f"H2_d64_c50_s220_letf_{arm}", **knobs
            )
            for arm, knobs in _ARM_B_D64_ARMS.items()
        )
    }
)


# Band-only interior: `fbil` (bilinear exterior, no global) was seed-unstable at
# the 4x4 gate with no interior mechanism at all. If the prefix band alone
# stabilises it, the global term is one of two interchangeable interior
# suppliers, not special. Same chassis, seeds and venue as the four cells above;
# baseline 0.8373 +- 0.0236 raw / 0.8848 +- 0.0055 EMA on this rung.
_BAND_ONLY_D64_ARMS = {
    "fimo2e_50k_curr_noglobal": {"use_global": False},
    # One-ordering twin, crossing the ordering axis as the 1o arm does.
    "fie_50k_curr_noglobal1o": {
        "use_global": False,
        "site_orderings": ("row",),
    },
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS[_ARM_B_D64_PARENT], name=f"H2_d64_c50_s220_letf_{arm}", **knobs
            )
            for arm, knobs in _BAND_ONLY_D64_ARMS.items()
        )
    }
)


# Separable-scores trajectory check. `separable_band_scores` is the band's exact
# function (forward agreement 1.5e-7, gradients matched), so this is not a
# quality arm: it asks whether the trajectory lands in the same outcome
# distribution. `compile_head`, a ~1e-5-class numerics change, gave catastrophic
# seeds at ~40% (5/12 vs 0/21, Fisher p = 0.0033). One variable against the
# archived dense twin `H2_d64_c50_s220_letf_ma_50k_curr_w2` (tag
# 20260825-hard-w2-d64), run compiled for recipe parity. Seeds 42-47:
# P(0 of 3 clean | p = 0.4) = 0.6^3 = 22%, 0.6^6 = 5%.
_ARM_C_SEED_CHECK_ARMS = {
    "masep_50k_curr_w2": {"separable_band_scores": True},
}

CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS["H2_d64_c50_s220_letf_ma_50k_curr_w2"],
                name=f"H2_d64_c50_s220_letf_{arm}",
                **knobs,
            )
            for arm, knobs in _ARM_C_SEED_CHECK_ARMS.items()
        )
    }
)


# Unfactorised orderings, 8x8 at exact sigma_c: the strongest head with the MLP
# combiner, factorisation then an optimisation applied to it. Coverage is not
# the mechanism (`fimo2ef` tiles the lattice and still loses 0.144 raw, disjoint,
# without its second ordering); a second ordering buys a deep read of a region
# the band reads shallowly. The chain, one field per step, both bands:
#   ma      -> mamo2    orderings on the attention band
#   mamo2   -> mamo2ef  the exact field channel, given orderings
#   iv      -> ivmo2    orderings on the prefix band
#   ivmo2   -> ivmo2ef  the exact field channel, given orderings
# `iv` is in the chain because no interval cell exists on this chassis (the
# 0.646 d64 figure is another wave's). Roster keyed by bare arm: each cell is
# its `ma` sibling at the rung with only these knobs replaced (8x8 critical
# cells ran under 20260828-rasterord-d64;
# test_raster_ladder_roster_covers_both_rungs pins them).
_RASTER_LADDER_ARMS = {
    "mamo2": {"site_orderings": ("row", "col")},
    "mamo2ef": {
        "site_orderings": ("row", "col"),
        "exact_field_channel": True,
    },
    "iv": {"head_kind": "interval"},
    "ivmo2": {
        "head_kind": "interval",
        "site_orderings": ("row", "col"),
    },
    "ivmo2ef": {
        "head_kind": "interval",
        "site_orderings": ("row", "col"),
        "exact_field_channel": True,
    },
}

# The 4x4 rung is a gate, not a ranking: bare `ma` already reads 0.997 at
# sigma=0.10 and 0.975 at sigma_c, so the 0.150 the ladder spans at 8x8 cannot
# fit; it inherits the wave-2 gate chassis (no EMA instrument). The 8x8 floor
# rung (flat parent, no `_curr` infix) completes the sigma=0.10 column as the
# control: a head merely better trained, rather than matched to critical
# structure, would separate at both couplings.
_RASTER_LADDER_PARENTS = {
    "H2_d64_c50_s220_letf_{arm}_50k_curr_w2": "H2_d64_c50_s220_letf_ma_50k_curr_w2",
    "H2_d64_c50_s010_letf_{arm}_50k_w2": "H2_d64_c50_s010_letf_ma_50k_w2",
    "H2_d16_c50_s010_letf_{arm}_10k_w2": "H2_d16_c50_s010_letf_ma_10k_w2",
    "H2_d16_c50_s220_letf_{arm}_10k_w2": "H2_d16_c50_s220_letf_ma_10k_w2",
    # The 16x16 rung asks whether the ladder rescues the rung where bare `ma`
    # fails (archived sigma_c row EMA ESS 0.0035/0.0422/0.0002). It anchors on
    # the archived dense `ma` sigma_c cell (separable scores are an exact
    # rewrite); the floor `ma` anchor is retrained as the `masep` cell below.
    "H2_d256_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w3": "H2_d256_c50_s220_letf_ma_100k_curr_b512_ne128_cv2_w3",
    "H2_d256_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w3": "H2_d256_c50_s010_letf_ma_50k_b512_ne128_cv2_w3",
}

# The floor rung runs separable, as every new masked-attention cell does from
# here (exact function; 0/6 catastrophic seeds in the trajectory check). It is a
# rung knob, not an arm knob: the sigma_c and 4x4 patterns already ran dense
# under 20260828-rasterord-d64, and putting the flag in _RASTER_LADDER_ARMS
# would redefine archived configs. The floor `ma` anchor moves with them (the
# `masep` floor cell below).
_RASTER_LADDER_RUNG_KNOBS = {
    "H2_d64_c50_s010_letf_{arm}_50k_w2": {"separable_band_scores": True},
    # The d256 rungs ride the triu gather true from the parent: gather-off OOM'd
    # a 183 GB B200 at step 0, since the rollout runs the head at the full batch
    # 512, where the ungathered pair slab is (512, 256, 256, 144) fp32 = 18 GiB.
    "H2_d256_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w3": {
        "separable_band_scores": True,
    },
    "H2_d256_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w3": {
        "separable_band_scores": True,
    },
}

# Rung knobs only an attention-band head can read: `IntervalSwapHead` has no
# score tensor to factorise, so an `iv*` cell must not record the flag.
_ATTENTION_ONLY_RUNG_KNOBS = frozenset({"separable_band_scores"})


def _raster_ladder_cell(arm: str, arm_knobs: dict, pattern: str):
    """One ladder cell: its `ma` sibling at this rung with the arm's knobs
    moved, plus any rung knobs the arm's band can actually read."""
    parent = CONFIGS[_RASTER_LADDER_PARENTS[pattern]]
    head_kind = arm_knobs.get("head_kind", parent.head_kind)
    rung_knobs = {
        knob: value
        for knob, value in _RASTER_LADDER_RUNG_KNOBS.get(pattern, {}).items()
        if head_kind == "masked_attention" or knob not in _ATTENTION_ONLY_RUNG_KNOBS
    }
    return replace(parent, name=pattern.format(arm=arm), **arm_knobs, **rung_knobs)


CONFIGS.update(
    {
        cell.name: cell
        for arm, arm_knobs in _RASTER_LADDER_ARMS.items()
        for pattern in _RASTER_LADDER_PARENTS
        for cell in (_raster_ladder_cell(arm, arm_knobs, pattern),)
    }
)

# The floor rung's separable `ma` anchor, so the chain there moves one field per
# step within one contraction order. The d256 floor `masep` replaces a
# compromised anchor: the archived d256 floor `ma` row is a two-seed mean
# (0.862 +- 0.057) with a degenerate seed excluded. Its sigma_c sibling is not
# retrained (same function under an exact rewrite).
CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                CONFIGS["H2_d64_c50_s010_letf_ma_50k_w2"],
                name="H2_d64_c50_s010_letf_masep_50k_w2",
                separable_band_scores=True,
            ),
            replace(
                CONFIGS["H2_d256_c50_s010_letf_ma_50k_b512_ne128_cv2_w3"],
                name="H2_d256_c50_s010_letf_masep_50k_b512_ne128_cv2_w3",
                separable_band_scores=True,
            ),
        )
    }
)


CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            *(_d400_radius_cell(arm) for arm in _D400_RADIUS_ARM_KNOBS),
            *(_d400_bf16_cell(arm) for arm in _D400_RADIUS_ARM_KNOBS),
            *(_d400_critical_cell(arm) for arm in _D400_RADIUS_ARM_KNOBS),
            *(_d400_critical_bf16_cell(arm) for arm in _D400_RADIUS_ARM_KNOBS),
            *(_d576_critical_bf16_cell(arm) for arm in _D576_RADIUS_ARM_KNOBS),
        )
    }
)


# ---------------------------------------------------------------------------
# Cu-Au alloy rungs: the canonical sampler on the MetaDNS/Damewood Cu-Au fcc
# cluster expansion in data/ce/ (experiments/alloy_ce/export_binary_expansion.py).
# `sigma` = beta/2 = 1/(2 k_B T) in 1/eV; the curriculum cools 1200 K -> 500 K
# into the L1_2 (x_Au = 0.25) / L1_0 (x_Au = 0.5) ordered regime. Head =
# mask_one, the one swap head with no 2D-torus assumption (fcc adjacency, not a
# raster). 16-site cell = the enumerable gate (C(16,4) = 1820, C(16,8) = 12870).
K_B_EV = 8.617333262e-5


def cuau_sigma(temperature_K: float) -> float:
    """beta/2 in 1/eV at the given temperature (11.602 at 500 K)."""
    return 1.0 / (2.0 * K_B_EV * temperature_K)


_CUAU_TEMPERATURE_LADDER_K = (1200.0, 800.0, 600.0, 500.0)


def _cuau_curriculum(n_steps: int) -> CurriculumCfg:
    """Four equal stages cooling to 500 K; lr eases at the last two, as the
    Ising sigma ladder does approaching sigma_c."""
    stage = n_steps // 4
    return CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(
                start_step=k * stage,
                sigma=cuau_sigma(T),
                lr=1e-3 if k < 2 else 3e-4,
            )
            for k, T in enumerate(_CUAU_TEMPERATURE_LADDER_K)
        )
    )


def _cuau_hard_cell(
    name,
    *,
    sites: int,
    composition: float,
    n_steps: int,
    n_euler_steps: int,
    n_eval_samples: int,
    eval_sample_chunk: int | None,
    hidden_dim: int,
    n_layers: int,
) -> HardStageCfg:
    side = {16: 4, 64: 8}[sites]  # D is a label here: d comes from the file
    cell = _hard_cell(
        name,
        cuau_sigma(_CUAU_TEMPERATURE_LADDER_K[0]),
        "mask_one",
        D=side,
        n_steps=n_steps,
        n_euler_steps=n_euler_steps,
        n_eval_samples=n_eval_samples,
        eval_sample_chunk=eval_sample_chunk,
        eval_every=500,
        curriculum=_cuau_curriculum(n_steps),
    )
    return replace(
        cell,
        ising=replace(
            cell.ising,
            target_composition=composition,
            expansion_json=f"data/ce/cuau_fcc_{'2x2x4' if sites == 16 else '4x4x4'}.json",
        ),
        model=replace(cell.model, hidden_dim=hidden_dim, n_layers=n_layers),
        target_kind="cluster_expansion",
    )


for _sites, _steps, _ne, _n_eval, _chunk, _hidden, _layers in (
    (16, 10_000, 50, 5_000, None, 64, 3),
    (64, 50_000, 128, 5_000, 256, 128, 3),
):
    for _c, _c_tag in ((0.25, "c25"), (0.5, "c50")):
        _name = f"H2_cuau{_sites}_{_c_tag}_T500_mask_one_{_steps // 1000}k_curr"
        CONFIGS[_name] = _cuau_hard_cell(
            _name,
            sites=_sites,
            composition=_c,
            n_steps=_steps,
            n_euler_steps=_ne,
            n_eval_samples=_n_eval,
            eval_sample_chunk=_chunk,
            hidden_dim=_hidden,
            n_layers=_layers,
        )


# Desk-check wave on the c=0.5 identity-flow collapse: the archived H2_cuau16_c50
# cell collapses at the 1200 -> 800 K step (loss settles at
# Var_uniform-slice[beta E], eval samples are uniform slice draws). One knob per
# cell against that control:
#   rewarm     -- re-run the lr warmup ramp at every stage boundary
#   lowlr      -- lr 1e-4 from the 800 K stage on (MetaDNS trains Cu-Au at 1e-4)
#   keepreplay -- retain the replay window across the boundary
#   ladder6    -- six stages 1200/1000/900/800/600/500 K (the 1200 -> 800 K
#                 step carries 0.79 nats of KL)
#   direct500  -- MetaDNS-style: 500 K from init at lr 1e-4, no ladder
def _cuau_ladder(temps_lrs, n_steps):
    n_stage = len(temps_lrs)
    boundaries = [round(k * n_steps / n_stage / 100) * 100 for k in range(n_stage)]
    return CurriculumCfg(
        stages=tuple(
            CurriculumStageCfg(start_step=start, sigma=cuau_sigma(T), lr=lr)
            for start, (T, lr) in zip(boundaries, temps_lrs)
        )
    )


_CUAU16_C50_CONTROL = CONFIGS["H2_cuau16_c50_T500_mask_one_10k_curr"]
_HOUSE_LADDER = [(1200.0, 1e-3), (800.0, 1e-3), (600.0, 3e-4), (500.0, 3e-4)]
for _variant, _curriculum, _train_overrides, _ising_overrides in (
    ("rewarm", _cuau_ladder(_HOUSE_LADDER, 10_000), dict(rewarmup_on_stage=True), {}),
    (
        "lowlr",
        _cuau_ladder(
            [(1200.0, 1e-3), (800.0, 1e-4), (600.0, 1e-4), (500.0, 1e-4)], 10_000
        ),
        {},
        {},
    ),
    (
        "keepreplay",
        _cuau_ladder(_HOUSE_LADDER, 10_000),
        dict(flush_replay_on_stage=False),
        {},
    ),
    (
        "ladder6",
        _cuau_ladder(
            [
                (1200.0, 1e-3),
                (1000.0, 1e-3),
                (900.0, 1e-3),
                (800.0, 1e-3),
                (600.0, 3e-4),
                (500.0, 3e-4),
            ],
            10_000,
        ),
        {},
        {},
    ),
    (
        "direct500",
        _cuau_ladder([(500.0, 1e-4)], 10_000),
        dict(lr=1e-4),
        dict(sigma=cuau_sigma(500.0)),
    ),
):
    _name = f"H2_cuau16_c50_T500_mask_one_10k_{_variant}"
    CONFIGS[_name] = replace(
        _CUAU16_C50_CONTROL,
        name=_name,
        curriculum=_curriculum,
        train=replace(_CUAU16_C50_CONTROL.train, **_train_overrides),
        ising=replace(_CUAU16_C50_CONTROL.ising, **_ising_overrides),
    )

# Follow-up: lr 1e-4 from the first step down rescued c=0.5 in 2/2 seeds (eval
# ESS 0.29 / 0.34); ladder6 alone 1/2; rewarm, keepreplay and direct500 stayed
# at the identity flow. Two combinations: low lr on a doubled budget, and on
# ladder6.
_LOWLR_LADDER = [(1200.0, 1e-3), (800.0, 1e-4), (600.0, 1e-4), (500.0, 1e-4)]
_LOWLR_LADDER6 = [
    (1200.0, 1e-3),
    (1000.0, 1e-4),
    (900.0, 1e-4),
    (800.0, 1e-4),
    (600.0, 1e-4),
    (500.0, 1e-4),
]
for _variant, _ladder, _n_steps in (
    ("lowlr", _LOWLR_LADDER, 20_000),
    ("ladder6lowlr", _LOWLR_LADDER6, 10_000),
):
    _name = f"H2_cuau16_c50_T500_mask_one_{_n_steps // 1000}k_{_variant}"
    CONFIGS[_name] = replace(
        _CUAU16_C50_CONTROL,
        name=_name,
        curriculum=_cuau_ladder(_ladder, _n_steps),
        train=replace(_CUAU16_C50_CONTROL.train, n_steps=_n_steps),
    )


# House-strength Cu-Au cells: what H2_d64_c50_s220_letf_mo_50k_curr_w2 gets
# (50k steps, ne128, EMA 0.9999), a seven-stage ladder linear in beta 1200 ->
# 500 K, and lr 1e-4 from the first step down (lr 1e-3 there shrinks every swap
# rate to the identity flow). The 16-site gate at a quarter recipe collapsed at c=0.5.
def _cuau_house_ladder(
    n_stages=7, T_hot=1200.0, T_cold=500.0, lr_hot=1e-3, lr_cold=1e-4
):
    beta_hot, beta_cold = 1.0 / T_hot, 1.0 / T_cold
    temps = [
        1.0 / (beta_hot + k * (beta_cold - beta_hot) / (n_stages - 1))
        for k in range(n_stages)
    ]
    return [(T, lr_hot if k == 0 else lr_cold) for k, T in enumerate(temps)]


for _c, _c_tag in ((0.25, "c25"), (0.5, "c50")):
    _control = CONFIGS[f"H2_cuau16_{_c_tag}_T500_mask_one_10k_curr"]
    _name = f"H2_cuau16_{_c_tag}_T500_mask_one_50k_house"
    CONFIGS[_name] = replace(
        _control,
        name=_name,
        ema_decay=0.9999,
        curriculum=_cuau_ladder(_cuau_house_ladder(), 50_000),
        train=replace(_control.train, n_steps=50_000),
        ctmc=replace(_control.ctmc, n_euler_steps=128),
    )

# Composition sweep for the 16-site canonical F(c): the house c=0.5 recipe at
# n_Au = 5, 6, 7 of 16, so F(c) is read off each cell's weights against exact
# enumeration of the 2^16 states (the 64-site c=0.5 slice is out of reach at
# 500 K); the c25 / c50 house cells complete the set.
for _c, _c_tag in ((0.3125, "c31"), (0.375, "c38"), (0.4375, "c44")):
    _parent = CONFIGS["H2_cuau16_c50_T500_mask_one_50k_house"]
    _name = f"H2_cuau16_{_c_tag}_T500_mask_one_50k_house"
    CONFIGS[_name] = replace(
        _parent,
        name=_name,
        ising=replace(_parent.ising, target_composition=_c),
    )


# Composition-amortised 16-site cell: the house c=0.5 recipe with the slice
# mixture as the only moved field (n_Au = 8..4 of 16, anchor 0.5 first). One
# checkpoint reads F(c) at all five compositions; the sweep above is the control.
_CUAU16_HOUSE = CONFIGS["H2_cuau16_c50_T500_mask_one_50k_house"]
CONFIGS["H2_cuau16_camort_T500_mask_one_50k_house"] = replace(
    _CUAU16_HOUSE,
    name="H2_cuau16_camort_T500_mask_one_50k_house",
    composition_mixture=(0.5, 0.4375, 0.375, 0.3125, 0.25),
)


# The 64-site cells (4x4x4 repeats = the MetaDNS benchmark cell) take the house
# ladder, lr cut and EMA in place; their first definition above is four-stage.
for _c_tag in ("c25", "c50"):
    _name = f"H2_cuau64_{_c_tag}_T500_mask_one_50k_curr"
    CONFIGS[_name] = replace(
        CONFIGS[_name],
        ema_decay=0.9999,
        curriculum=_cuau_ladder(_cuau_house_ladder(), 50_000),
    )

# Two-hole patch twins of the 64-site cells (mask-one at 64 sites was killed at
# step 9k). Declared deviations besides the head: in-training eval draws 256
# not 5000 (final eval still 5000), and a two-shell window (18 sites: pair
# terms reach 9.3 A; reach probe R^2 0.80 one shell vs 0.95 two). No exact-field
# channel (lost to its plain twin 3/3 on the 16-site alloy cell).
for _c_tag in ("c25", "c50"):
    _parent = CONFIGS[f"H2_cuau64_{_c_tag}_T500_mask_one_50k_curr"]
    _name = f"H2_cuau64_{_c_tag}_T500_thp_50k_curr"
    CONFIGS[_name] = replace(
        _parent,
        name=_name,
        head_kind="two_hole_patch",
        patch_shells=2,
        eval=replace(_parent.eval, n_eval_samples_training=256),
    )

# Revival arms for the 64-site cells: the first thp wave ordered c=0.25 (ESS
# 0.17-0.20) but gave up on c=0.5 down the ladder (loss / static-flow loss ->
# ~1, samples 9.5 swaps from the nearest L1_0); reach is exonerated (two-shell
# R^2 0.95-0.99). Levers are the ladder and the trajectory, one change each:
#   l14        fourteen rungs linear in beta 1200 -> 500 K, at 50k or 100k
#   ne256      4d Euler steps instead of the house 2d
#   l14_ne256  both, the ceiling arm
_THP64_PARENTS = {
    c: CONFIGS[f"H2_cuau64_{c}_T500_thp_50k_curr"] for c in ("c25", "c50")
}
for _c_tag, _variant, _n_stages, _n_steps, _n_euler in (
    ("c50", "50k_l14", 14, 50_000, 128),
    ("c50", "100k_l14", 14, 100_000, 128),
    ("c50", "50k_ne256", 7, 50_000, 256),
    ("c50", "100k_l14_ne256", 14, 100_000, 256),
    ("c25", "50k_ne256", 7, 50_000, 256),
):
    _parent = _THP64_PARENTS[_c_tag]
    _name = f"H2_cuau64_{_c_tag}_T500_thp_{_variant}"
    CONFIGS[_name] = replace(
        _parent,
        name=_name,
        curriculum=_cuau_ladder(_cuau_house_ladder(n_stages=_n_stages), _n_steps),
        train=replace(_parent.train, n_steps=_n_steps),
        ctmc=replace(_parent.ctmc, n_euler_steps=_n_euler),
    )


# MetaDNS temperature-grid cells: MetaDNS reports its 4x4x4 Cu-Au cell at 1200,
# 680 and 500 K, and 500 K at c=0.5 is out of reach, so the 64-site rows sit on
# the comparator's grid. The ladder stops at the row's temperature: one 1200 K
# stage at lr 1e-3, or four stages linear in beta 1200 -> 680 K at the house lr cut.
for _c_tag in ("c25", "c50"):
    _parent = _THP64_PARENTS[_c_tag]
    for _variant, _ladder, _n_steps in (
        ("T1200_thp_10k", [(1200.0, 1e-3)], 10_000),
        ("T680_thp_30k_l4", _cuau_house_ladder(n_stages=4, T_cold=680.0), 30_000),
    ):
        _name = f"H2_cuau64_{_c_tag}_{_variant}"
        CONFIGS[_name] = replace(
            _parent,
            name=_name,
            curriculum=_cuau_ladder(_ladder, _n_steps),
            train=replace(_parent.train, n_steps=_n_steps),
        )


# Fresh-trajectory probe: the 20k lr-cut cell reached ESS 0.79 where 10k gave
# 0.34, and MetaDNS trains on ~200x more distinct rollouts. Same 10k gradient
# steps, five times the rollouts (inner 100 -> 20) or four times the width (outer 512).
_LOWLR_10K = CONFIGS["H2_cuau16_c50_T500_mask_one_10k_lowlr"]
CONFIGS["H2_cuau16_c50_T500_mask_one_10k_lowlr_inner20"] = replace(
    _LOWLR_10K,
    name="H2_cuau16_c50_T500_mask_one_10k_lowlr_inner20",
    train=replace(_LOWLR_10K.train, inner_steps_per_outer=20),
)
CONFIGS["H2_cuau16_c50_T500_mask_one_10k_lowlr_ob512"] = replace(
    _LOWLR_10K,
    name="H2_cuau16_c50_T500_mask_one_10k_lowlr_ob512",
    train=replace(_LOWLR_10K.train, outer_batch_size=512),
)


# Exact-field channel twins on the hard alloy cells: the swap channel
# now reads the target's own swap log-ratio (-beta Delta E_swap on an
# expansion), one declared change from the c=0.5 recipe cells.
for _parent_name in (
    "H2_cuau16_c50_T500_mask_one_20k_lowlr",
    "H2_cuau16_c50_T500_mask_one_50k_house",
):
    _parent = CONFIGS[_parent_name]
    CONFIGS[f"{_parent_name}_ef"] = replace(
        _parent, name=f"{_parent_name}_ef", exact_field_channel=True
    )
