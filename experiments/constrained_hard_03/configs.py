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

from dataclasses import dataclass
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

from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
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
    real run -- see that class's docstring), or "interval"
    (IntervalSwapHead, the one-pass three-interval spike head -- property-
    equivalent to doubly_hollow, not numerically equal: different H).
    """

    head_kind: Literal[
        "doubly_hollow", "mask_one", "non_antisym", "interval"
    ] = "doubly_hollow"
    # Anchor-batch chunk for the mask_one head's vectorised forward; None =
    # unchunked. d=256 needs this: the stacked d-anchor-copies pass would
    # otherwise build a (d*B)-row buffer that OOMs the L4.
    anchor_chunk_size: int | None = None
    # Opt-in torch.compile of the built head (Tier 2, default OFF). Head
    # only — the Euler loop's data-dependent sampling would graph-break.
    compile_head: bool = False


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
        # Band pair-feature offsets (1, D): row and column adjacency of the
        # flattened D x D lattice -- the interactions the Ising energy uses.
        head = IntervalSwapHead(backbone, pair_offsets=(1, cfg.ising.D))
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
    "H2_d64_c50_s223_letf_mo_50k_curr": _hard_cell(
        "H2_d64_c50_s223_letf_mo_50k_curr", sigma=0.223, head_kind="mask_one",
        D=8, n_steps=50_000, n_euler_steps=128, n_eval_samples=5000,
        eval_sample_chunk=256, n_eval_samples_training=512,
        use_sdpa_readout=True, eval_autocast_bf16=True,
        curriculum=CurriculumCfg(
            stages=(
                CurriculumStageCfg(start_step=0, sigma=0.100, lr=1e-3),
                CurriculumStageCfg(start_step=5_000, sigma=0.140, lr=1e-3),
                CurriculumStageCfg(start_step=10_000, sigma=0.170, lr=1e-3),
                CurriculumStageCfg(start_step=15_000, sigma=0.190, lr=1e-3),
                CurriculumStageCfg(start_step=20_000, sigma=0.205, lr=3e-4),
                CurriculumStageCfg(start_step=25_000, sigma=0.215, lr=3e-4),
                CurriculumStageCfg(start_step=30_000, sigma=0.223, lr=3e-4),
            )
        ),
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
