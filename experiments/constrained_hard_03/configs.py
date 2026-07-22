"""Hard-constraint configs: swap-move CTMC on the fixed-composition manifold.

Imports the shared schema dataclasses from the baseline experiment (same
pattern as `constrained_soft_02/configs.py`) and adds `HardStageCfg`, which
carries a `head_kind` selecting the swap-readout head. Unlike the soft-02
cells, the target here is `FixedCompositionIsingTarget`: composition is
enforced exactly by the swap move set (n_plus == N_A always), so there is no
`composition_penalty_strength` or `lambda_curriculum` to anneal.

Cell-name format: `H2_d<dim>_c<c_target_x100>_s<sigma label>_letf_<head tag>`.
The sigma-ladder cells (`_dh` suffix) probe the swap-CTMC across the Ising
phase transition (σ_c ≈ 0.22305) with the correctness-gate
`DoublyHollowSwapHead`; the `_na` cell pairs the subcritical floor rung with
`NonAntisymSwapHead`, a deliberately antisymmetry-breaking negative control
for Task 9's ablation.
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
    attention -- bit-exact blindness, decision 2026-07-07).
    """

    head_kind: Literal[
        "doubly_hollow", "mask_one", "non_antisym", "interval", "masked_attention",
        "grouped_anchor",
    ] = "doubly_hollow"
    # Anchor-batch chunk for the mask_one head's vectorised forward; None =
    # unchunked. d=256 needs this: the stacked d-anchor-copies pass would
    # otherwise build a (d*B)-row buffer that OOMs the L4.
    anchor_chunk_size: int | None = None
    # Opt-in torch.compile of the built head (Tier 2, default OFF). Head
    # only — the Euler loop's data-dependent sampling would graph-break.
    compile_head: bool = False
    # Band-capacity knobs for the interval / masked_attention heads
    # (band-capacity push, design docs/design/2026-07-08-band-capacity-push-
    # design.md). None = the constructions every prior run used:
    # band_feature_dim 16, attention_dim 32, pair_offsets (1, D) — row and
    # column adjacency of the flattened D x D lattice. attention_dim is
    # masked_attention-only (the interval head has no attention).
    band_feature_dim: int | None = None
    attention_dim: int | None = None
    pair_offsets: tuple[int, ...] | None = None
    # Round-2 stencil family (design §5.i): add the 5-point lattice-stencil
    # band-feature family to the masked_attention head. False = the depth-1
    # unary+offset band every prior run used. lattice_side is cfg.ising.D
    # (the flattened D x D grid), so no separate field is needed.
    use_stencil: bool = False
    # Grouped-anchor head knobs (avenues doc 2026-07-22 §4c). n_groups is k,
    # the number of masked body passes: k = d reproduces mask_one bit-exactly,
    # smaller k trades masked-site count for passes. Only read when head_kind
    # is "grouped_anchor", so every other cell stays byte-identical.
    n_groups: int | None = None
    grouping: str = "diagonal"
    group_chunk_size: int | None = None


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
    This is exactly the property Task 9's antisymmetry-ablation control must
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
    use_sdpa_readout: bool = False,
    eval_autocast_bf16: bool = False,
    curriculum: CurriculumCfg | None = None,
) -> HardStageCfg:
    """Shared shape for the sigma-ladder + control cells: the D=4 gate cells fix
    only sigma and head_kind (all otherwise identical). The keyword knobs open
    the same shape to the non-enumerable scaling rungs (§7 mixing probe): D sets
    the lattice, n_euler_steps must be clip-safe for the one-event step at that D
    (scout: ~2d at d=64), n_eval_samples sizes the IS-ESS eval drawn on the GPU
    job itself (evals-ride-the-gpu-job), and eval_sample_chunk streams that eval
    in slices so the vectorised head's d-anchor-copies batch fits GPU memory."""
    return HardStageCfg(
        name=name,
        ising=IsingCfg(
            D=D, sigma=sigma, bias=0.0,
            target_composition=0.5, composition_penalty_strength=0.0,
        ),
        train=TrainCfg(
            n_steps=n_steps, batch_size=128, replay_buffer_cycles=8,
            lr=1e-3, seed=42, warmup_steps=500,
        ),
        ctmc=CTMCCfg(n_euler_steps=n_euler_steps),
        eval=EvalCfg(
            eval_every=200, n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
            n_eval_samples_training=n_eval_samples_training,
            eval_autocast_bf16=eval_autocast_bf16,
        ),
        model=ModelCfg(
            kind="letf", hidden_dim=32, n_layers=2, n_heads=4, vocab_size=2,
            use_sdpa_readout=use_sdpa_readout,
        ),
        estimator="control_variate",
        head_kind=head_kind,
        wandb_project="dnfs-constraints",
        curriculum=curriculum,
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


def _d64_curriculum_cell(
    name: str, head_kind: str, n_steps: int = 50_000, **head_knobs
) -> HardStageCfg:
    """The d=64 sigma_c curriculum recipe — the shape of the PASSED D=8 rung.
    Every band-capacity-push cell shares it verbatim and differs only in
    head_kind and the declared head knobs, so outcome differences are
    attributable to the declared change (design 2026-07-08; twin-ness is
    pinned by test_band_push_cells_mirror_ma_twin_except_declared_fields).

    `n_steps` defaults to the 50k budget every batch-1 / round-2 cell used.
    The horizon-extension cells (avenues doc §4a) pass 100_000: the sigma
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


CONFIGS: dict[str, HardStageCfg] = {
    "H2_d16_c50_s010_letf_dh": _hard_cell(
        "H2_d16_c50_s010_letf_dh", sigma=0.10, head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s223_letf_dh": _hard_cell(
        "H2_d16_c50_s223_letf_dh", sigma=0.223, head_kind="doubly_hollow",
    ),
    "H2_d16_c50_s040_letf_dh": _hard_cell(
        "H2_d16_c50_s040_letf_dh", sigma=0.40, head_kind="doubly_hollow",
    ),
    # Task 9's antisymmetry negative control: same sigma as the floor rung,
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
    # Band-capacity push batch 1 (design docs/design/2026-07-08-band-
    # capacity-push-design.md): three single-variable twins of ma_50k_curr.
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
    # Round-2 stencil family (design §5.i): the reported MA head + the 5-point
    # lattice-stencil band family (neighbours ±1, ±D on the flattened D x D
    # grid), narrow unary+offset families kept alongside to cover the collar.
    # use_stencil is the ONLY change vs ma_50k_curr -- the probe of whether a
    # richer 2D-local band content lifts the ~0.78 H-shared ceiling.
    "H2_d64_c50_s223_letf_ma_stencil_50k_curr": _d64_curriculum_cell(
        "H2_d64_c50_s223_letf_ma_stencil_50k_curr", head_kind="masked_attention",
        use_stencil=True,
    ),
    # Horizon extension (avenues doc 2026-07-22 §4a). Judging the stencil
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
    # Grouped-anchor batch 1 (avenues doc 2026-07-22 §4c): the head family
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
        # Tier-2 flags, user sign-off 2026-07-06 (perf-branch evidence in
        # the 2026-07-05 findings doc): SDPA readout everywhere, bf16 on the
        # in-training eval block only — the final 5000-sample eval runs fp32.
        use_sdpa_readout=True,
        eval_autocast_bf16=True,
    ),
}
