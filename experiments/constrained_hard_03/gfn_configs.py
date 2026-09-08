"""Config cells for the GFlowNet comparator (hard chapter).

A flat dataclass, separate from HardStageCfg: the GFN has no time grid, Euler
steps, swap head or CTMC knobs. Run dirs, the eval/metrics.json schema and
composition observables are shared, so the table scripts ingest GFN rows
unchanged. Plain cells are the correctness cells (hidden 128 / 3 layers,
597.6k params); `_par` cells are the comparison cells at measured parameter
parity with the wave-2 heads (hidden 64 / 2 layers / 4 heads, 101.4k params
~= ma's 101.0k). sigma_stages is the annealing ladder; 4x4 cells train flat.
"""

from dataclasses import dataclass, replace

from discrete_flow_sampler.targets.ising import SIGMA_C

GFN_OBJECTIVES = ("tb", "fldb")


@dataclass(frozen=True)
class GFNCellCfg:
    name: str
    objective: str  # "tb" (trajectory balance) | "fldb" (forward-looking DB)
    sigma: float
    D: int = 4
    target_composition: float = 0.5
    # Policy network (see models/raster_gfn_policy.py for the O(d^2)-vs-O(d^3) note).
    hidden_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    with_flow_head: bool = False  # forced True for fldb in the builder below
    # Separate flow MLP over the one-hot prefix state (the torchgfn convention)
    # instead of the shared-trunk readout. False = every archived cell.
    standalone_flow_head: bool = False
    # Training.
    n_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    # Separate Adam lr for TB's scalar log_z (None = share learning_rate); at
    # flat 1e-3 it sat 2.4 nats under the exact 10.81 at 10k. ~100x = torchgfn.
    log_z_learning_rate: float | None = None
    # FL-DB analogue for the flow head (None = share learning_rate). At 1e-3 the
    # d64 sigma_c centres still climbed at 50k; global 3e-3 destabilised the policy.
    flow_head_learning_rate: float | None = None
    epsilon: float = 0.05  # uniform behaviour mix; off-policy, TB-tolerated
    # torch.compile the scoring path (the compile_head analogue). Default False:
    # archived cells never retro-flip. The sampler stays eager (KV-cache shapes vary).
    compile_policy: bool = False
    sigma_stages: tuple[float, ...] = ()  # annealing ladder; () = train flat
    # House-recipe levers added for the d64 rung; the 0 / None / False defaults
    # reproduce the d16 waves byte-identically.
    # Linear lr ramp over the first `warmup_steps` updates (house d64 value 500),
    # network group only: ramping TB's log_z group would re-starve the scalar.
    warmup_steps: int = 0
    # clip_grad_norm_ max-norm (house d64 value 500.0); None = no clipping. The
    # pre-clip norm is logged as `grad_norm` either way.
    grad_clip_max_norm: float | None = None
    # Eval-side EMA shadow (house 0.9999, warmup-corrected); 0.0 = off. When on
    # the final eval runs twice: raw weights into eval/, shadow into eval_ema/.
    ema_decay: float = 0.0
    # bf16 autocast around eval sampling+scoring (house d64 evals carry
    # eval_autocast_bf16=true); False = fp32 end to end (the d16 waves).
    eval_autocast_bf16: bool = False
    # In-training frozen-ESS diagnostic every `eval_every` steps (house 200); 0 = off.
    # On-policy ESS cannot serve: reverse-KL blindness (Whitammer et al. 2023 Prop. 1).
    eval_every: int = 0
    n_eval_samples_training: int = 512
    # Eval (house protocol: 5000 draws, chunked).
    n_eval_samples: int = 5000
    eval_sample_chunk: int = 512
    # Bookkeeping.
    log_every: int = 100
    checkpoint_every: int = 2000
    wandb_project: str = "dnfs-constraints"

    def __post_init__(self):
        if self.objective not in GFN_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {GFN_OBJECTIVES}, got {self.objective!r}"
            )
        if self.sigma_stages and self.sigma_stages[-1] != self.sigma:
            raise ValueError(
                "sigma_stages must end at the cell's own sigma "
                f"(got stages {self.sigma_stages} for sigma={self.sigma})"
            )


def _gfn_d16_cell(objective: str, sigma_label: str, sigma: float) -> GFNCellCfg:
    """4x4 rung, both arms, both house couplings; flat (no annealing)."""
    cell = GFNCellCfg(
        name=f"GFN_d16_c50_{sigma_label}_{objective}_10k",
        objective=objective,
        sigma=sigma,
    )
    if objective == "fldb":
        cell = replace(cell, with_flow_head=True)
    return cell


def _gfn_d16_parity_cell(objective: str, sigma_label: str, sigma: float) -> GFNCellCfg:
    """Parity cells: measured-parameter parity with the wave-2 heads plus the
    split log_z lr on the TB arm. hidden 64 / 2 layers = 101,378 params, within
    0.4% of the masked-attention head (100,960); the house d16 sizing (hidden
    32 / 2 layers) would give only 26.1k. The plain cells stay archived."""
    cell = replace(
        _gfn_d16_cell(objective, sigma_label, sigma),
        hidden_dim=64,
        n_layers=2,
        batch_size=128,
    )
    cell = replace(cell, name=f"{cell.name}_par")
    if objective == "tb":
        cell = replace(cell, log_z_learning_rate=0.1)
    return cell


# Fair-tuning grid around the `_par` recipe: lr x epsilon at sigma_c, centre
# excluded (it is the `_par` cell). The 4x4 gate filters rather than
# discriminates, so this only shows the recipe sits in no hole. Warmup and
# grad clip are not folded in: changing the centre would orphan the `_par` cells.
_SWEEP_LR_GRID = {"l3e4": 3e-4, "l1e3": 1e-3, "l3e3": 3e-3}
_SWEEP_EPSILON_GRID = {"e000": 0.0, "e005": 0.05, "e010": 0.1}
_SWEEP_CENTRE = ("l1e3", "e005")


def _gfn_d16_sweep_cell(objective: str, lr_key: str, epsilon_key: str) -> GFNCellCfg:
    base = _gfn_d16_parity_cell(objective, "s220", SIGMA_C)
    return replace(
        base,
        name=base.name.replace("_par", f"_{lr_key}_{epsilon_key}_swp"),
        learning_rate=_SWEEP_LR_GRID[lr_key],
        epsilon=_SWEEP_EPSILON_GRID[epsilon_key],
    )


# The 8x8 rung. The sigma_c ladder mirrors the house _D64_SIGMA_LADDER incl.
# its 20k final plateau: _stage_sigma gives equal n_steps/len shares, so ten
# 5k stages with sigma_c repeated four times reproduce the house start-steps
# 0/5k/.../30k with 20k on the cell's coupling. The GFN trains flat-lr (house
# drops 1e-3 -> 3e-4 at 20k), a declared deviation the star's 3e-4 arm brackets.
_D64_GFN_SIGMA_STAGES = (
    0.100,
    0.140,
    0.170,
    0.190,
    0.205,
    0.215,
    SIGMA_C,
    SIGMA_C,
    SIGMA_C,
    SIGMA_C,
)

# Trimmed star around the d64 centre, sigma_c only (house ESS spans
# 0.735-0.953 there). Arms from the 4x4 grid: the lr axis first (epsilon was
# flat at 4x4), 3e-4 because 50k removes the "under-trained" excuse, 3e-3
# because the FLDB triplet sat disjoint above its centre.
_D64_STAR_ARMS = (
    ("l3e4", "e005"),
    ("l3e3", "e005"),
    ("l1e3", "e000"),
    ("l1e3", "e010"),
)


def _gfn_d64_parity_cell(objective: str, sigma_label: str, sigma: float) -> GFNCellCfg:
    """8x8 centre: the 4x4 parity recipe at the house d64 budget. Parity is
    measured params again: 104,450 at D=8, 3.5% under the d64 ma head's 108,256.
    Budget levers match the wave-2 recipe field by field; declared deviations:
    flat lr (no 20k drop to 3e-4) and no lr ramp on the TB log_z group."""
    cell = replace(
        _gfn_d16_parity_cell(objective, sigma_label, sigma),
        D=8,
        n_steps=50_000,
        warmup_steps=500,
        grad_clip_max_norm=500.0,
        ema_decay=0.9999,
        eval_autocast_bf16=True,
        eval_every=200,
        compile_policy=True,
        checkpoint_every=5000,
        sigma_stages=(_D64_GFN_SIGMA_STAGES if sigma_label == "s220" else ()),
    )
    return replace(cell, name=f"GFN_d64_c50_{sigma_label}_{objective}_50k_par")


def _gfn_d64_star_cell(objective: str, lr_key: str, epsilon_key: str) -> GFNCellCfg:
    base = _gfn_d64_parity_cell(objective, "s220", SIGMA_C)
    return replace(
        base,
        name=base.name.replace("_par", f"_{lr_key}_{epsilon_key}_swp"),
        learning_rate=_SWEEP_LR_GRID[lr_key],
        epsilon=_SWEEP_EPSILON_GRID[epsilon_key],
    )


# Flow-head split-lr arms, FL-DB at sigma_c only (the s010 fldb centre already
# sits at 0.97); see flow_head_learning_rate. 1e-1 is the log_z precedent
# (~100x), 1e-2 a conservative 10x. One lever off the fldb centre (pinned by
# test_flow_lr_cells_are_fldb_centre_twins_plus_one_lever).
_FLOW_LR_GRID = {"flr1e1": 1e-1, "flr1e2": 1e-2}


def _gfn_d64_flow_lr_cell(flr_key: str) -> GFNCellCfg:
    base = _gfn_d64_parity_cell("fldb", "s220", SIGMA_C)
    return replace(
        base,
        name=base.name.replace("_par", f"_{flr_key}"),
        flow_head_learning_rate=_FLOW_LR_GRID[flr_key],
    )


GFN_CONFIGS = {
    cell.name: cell
    for objective in GFN_OBJECTIVES
    for cell in (
        _gfn_d16_cell(objective, "s010", 0.10),
        _gfn_d16_cell(objective, "s220", SIGMA_C),
        _gfn_d16_parity_cell(objective, "s010", 0.10),
        _gfn_d16_parity_cell(objective, "s220", SIGMA_C),
        *(
            _gfn_d16_sweep_cell(objective, lr_key, epsilon_key)
            for lr_key in _SWEEP_LR_GRID
            for epsilon_key in _SWEEP_EPSILON_GRID
            if (lr_key, epsilon_key) != _SWEEP_CENTRE
        ),
        _gfn_d64_parity_cell(objective, "s010", 0.10),
        _gfn_d64_parity_cell(objective, "s220", SIGMA_C),
        *(
            _gfn_d64_star_cell(objective, lr_key, epsilon_key)
            for lr_key, epsilon_key in _D64_STAR_ARMS
        ),
    )
}
# fldb-only arms live outside the per-objective comprehension.
GFN_CONFIGS.update(
    {cell.name: cell for cell in map(_gfn_d64_flow_lr_cell, _FLOW_LR_GRID)}
)

# Budget-doubled FLDB diagnostic, one lever (n_steps 100k): settles "slow vs
# broken" -- seed-42 logs showed the centre still climbing at 50k (ESS +0.06/5k,
# DB loss ~0.007 by 15k) and a hot flow head worse. The ladder dilates with the
# budget (10k/stage, 40k plateau). Never a table row: breaks budget parity.
GFN_CONFIGS.update(
    {
        cell.name: cell
        for cell in (
            replace(
                _gfn_d64_parity_cell("fldb", "s220", SIGMA_C),
                name="GFN_d64_c50_s220_fldb_100k_par",
                n_steps=100_000,
            ),
            # Standalone-flow arm: the torchgfn parameterisation at the matched
            # 50k budget, one lever off the centre; the added flow-MLP params
            # (~17k at d64) are a declared delta inside the fldb family, not a
            # parity break against the house heads.
            replace(
                _gfn_d64_parity_cell("fldb", "s220", SIGMA_C),
                name="GFN_d64_c50_s220_fldb_50k_sfh",
                standalone_flow_head=True,
            ),
        )
    }
)


# The 16x16 rung: 256-step trajectories are the regime the GFN literature calls
# hard for TB (Madan/Pan expect FL>TB there), so the d64 "FL>TB has not appeared
# by 64 steps" is a scoped claim only if this rung tests the scope. The sigma_c
# ladder mirrors the house d256 100k curriculum under the equal-share rule: 20
# stages of 5k = house start-steps 0/5k/.../30k with the final 70k at sigma_c.
# Budget parity is per rung (50k floor / 100k sigma_c); flat lr stays a deviation.
_D256_GFN_SIGMA_STAGES = (
    0.100,
    0.140,
    0.170,
    0.190,
    0.205,
    0.215,
) + (SIGMA_C,) * 14


def _gfn_d256_parity_cell(objective: str, sigma_label: str) -> GFNCellCfg:
    """16x16 centre: the d64 recipe with only the rung levers moved. Parity is
    measured params a third time: hidden 64 carried up gives 116,738 (12.6% under
    thp2's 133,632); hidden 68 gives 130,562, within 2.3% of thp2 and 5.0% of ma.
    Eval cadence moves to the house d256 regime (500 / 256 draws, bf16)."""
    sigma_c = sigma_label == "s220"
    cell = replace(
        _gfn_d64_parity_cell(objective, sigma_label, SIGMA_C if sigma_c else 0.10),
        D=16,
        hidden_dim=68,
        n_steps=100_000 if sigma_c else 50_000,
        sigma_stages=_D256_GFN_SIGMA_STAGES if sigma_c else (),
        # House d256 in-training eval cadence: within-rung parity outranks
        # cross-rung GFN consistency. The 5000-draw final eval is untouched.
        eval_every=500,
        n_eval_samples_training=256,
    )
    steps_label = "100k" if sigma_c else "50k"
    return replace(
        cell, name=f"GFN_d256_c50_{sigma_label}_{objective}_{steps_label}_par"
    )


GFN_CONFIGS.update(
    {
        cell.name: cell
        for objective in GFN_OBJECTIVES
        for cell in (
            _gfn_d256_parity_cell(objective, "s010"),
            _gfn_d256_parity_cell(objective, "s220"),
        )
    }
)


# The 20x20 rung: the TB comparator carried to the chapter's largest printed
# swap-head table. Only TB is run: FL-DB was dead at 256-step trajectories with
# the same construction. The fldb cell is registered so the rung's gate needs
# no new code.
def _gfn_d400_parity_cell(objective: str, sigma_label: str) -> GFNCellCfg:
    """20x20 centre: the d256 recipe with the lattice and the parity re-size
    moved. hidden 68 carried up gives 140,354, 11.6% under both house anchors
    (thp2 158,848 / thp3 159,616); hidden 72 gives 155,522, -2.1%/-2.6%, at
    parity. Budgets, ladder, eval cadence and compile ride the d256 cell."""
    steps_label = "100k" if sigma_label == "s220" else "50k"
    cell = _gfn_d256_parity_cell(objective, sigma_label)
    return replace(
        cell,
        name=f"GFN_d400_c50_{sigma_label}_{objective}_{steps_label}_par",
        D=20,
        hidden_dim=72,
    )


GFN_CONFIGS.update(
    {
        cell.name: cell
        for objective in GFN_OBJECTIVES
        for cell in (
            _gfn_d400_parity_cell(objective, "s010"),
            _gfn_d400_parity_cell(objective, "s220"),
        )
    }
)


# Critical-only 24x24 rung. Hidden 76 gives 184,834 policy params, within 2.9%
# of both house anchors (R=3: 189,184; R=4: 190,208); hidden 72 is -11%, 80 is
# +6--7%. Budget, ladder, optimiser and eval ride d400. Only TB is scheduled;
# FLDB is registered for the size/venue gate.
GFN_CONFIGS.update(
    {
        cell.name: cell
        for objective in GFN_OBJECTIVES
        for cell in (
            replace(
                _gfn_d400_parity_cell(objective, "s220"),
                name=f"GFN_d576_c50_s220_{objective}_100k_par",
                D=24,
                hidden_dim=76,
            ),
        )
    }
)
