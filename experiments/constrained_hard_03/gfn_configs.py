"""Config cells for the GFlowNet comparator (hard chapter).

A deliberately flat dataclass, SEPARATE from HardStageCfg: the GFN has no
time grid, no Euler steps, no swap head and no CTMC knobs, so inheriting the
swap-stack config would carry dead fields that a recipe-parity audit then has
to explain away. The two families meet at the artefact level instead — run
dirs, eval/metrics.json schema and composition observables are shared, so
the table scripts ingest GFN rows unchanged.

Fairness protocol (design doc 2026-08-30-gfn-comparator): the s92
correctness wave ran hidden 128 / 3 layers, which the s93 parity audit
measured at 597.6k params against 79.5k-101k for the wave-2 d16 heads —
the `_par` cells are the judging wave at measured parameter parity
(hidden 64 / 2 layers / 4 heads = 101.4k params ~= the ma head's 101.0k;
batch 128). sigma_stages is the
beta-annealing knob (the VAN-line criticality mitigation); the 4x4 cells
train flat, mirroring the wave-2 house convention that the floor/4x4 rung
measures the operating point, not the curriculum.
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
    # Standalone flow module instead of the shared-trunk linear readout
    # (s100): the torchgfn-conventional parameterisation — a separate MLP
    # over the one-hot prefix state — which also DECOUPLES flow gradients
    # from the policy trunk (the shielding mechanism the flow-lr arms
    # surfaced). False = every archived cell, byte-identical.
    standalone_flow_head: bool = False
    # Training.
    n_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    # Separate Adam lr for the TB arm's scalar log_z (None = share
    # learning_rate). Adam moves a scalar at ~lr/step under a consistent
    # gradient, so at the flat 1e-3 the s92 wave's log Z (init 0) climbed at
    # its measured speed limit (+7e-4/step) and sat 2.4 nats below the exact
    # slice value 10.81 at 10k steps — it arithmetically could not arrive.
    # ~100x on log_z alone is the Malkin et al. / torchgfn convention.
    log_z_learning_rate: float | None = None
    # The FL-DB analogue of the split above (s100). The flow head is the
    # FL-DB arm's normaliser: its output must reach the tens-of-nats
    # completion-entropy scale (log C(64,32) ~ 43 nats enters the prefix
    # flows), and Adam moves a final-layer bias at ~lr/step, so at the
    # shared 1e-3 the d64 sigma_c centres were STILL CLIMBING at 50k
    # (frozen ESS 0.45 -> 0.75 over the last 25k while TB converged by
    # 35k, DB residual loss already ~1e-3 — near-satisfied objective,
    # starved normaliser). A GLOBAL lr raise is ruled out by the star
    # (3e-3 destabilised the policy: one seed spent 10k steps near ESS
    # 0.04). None = share learning_rate; every archived cell reproduced
    # exactly.
    flow_head_learning_rate: float | None = None
    epsilon: float = 0.05  # uniform behaviour mix; off-policy, TB-tolerated
    # torch.compile the SCORING path (site_log_probs and the FL-DB variant)
    # — the GFN analogue of optimised_recipe's compile_head. Default False:
    # archived cells never retro-flip, and the flag flips for new 8x8+ cells
    # only after a GPU numerical-parity gate at the 8x8 launch bench (the
    # house compile certification pattern). The SAMPLER stays eager either
    # way: its per-step shapes vary with the KV-cache length, which is
    # recompile territory, and the cache already took the rollout from
    # O(d^3) to O(d^2) attention.
    compile_policy: bool = False
    sigma_stages: tuple[float, ...] = ()  # annealing ladder; () = train flat
    # House-recipe training levers, added for the d64 rung (s94). Every
    # default is ARCHIVED-INERT: 0 / None / False reproduces the d16 waves
    # byte-identically, so archived cells never retro-flip and the sweep
    # twin test's field-by-field comparison needs no exemptions.
    # Linear lr ramp over the first `warmup_steps` updates (house d64 value
    # 500), applied to the NETWORK group only: the TB log_z group exists
    # because Adam starves a scalar at the shared lr (see
    # log_z_learning_rate above), and re-throttling it for the ramp would
    # re-create a mild version of that failure at the start of every run.
    warmup_steps: int = 0
    # clip_grad_norm_ max-norm (house d64 value 500.0); None = no clipping.
    # The pre-clip total norm is logged as `grad_norm` either way — whether
    # the rail engaged is readable from the training log.
    grad_clip_max_norm: float | None = None
    # Eval-side EMA shadow (house ema_decay 0.9999, warmup-corrected);
    # 0.0 = off. When on, the final eval runs TWICE — raw weights into
    # eval/, shadow weights into eval_ema/ — mirroring run.py's dual-eval
    # instrument so the table scripts ingest GFN rows unchanged.
    ema_decay: float = 0.0
    # bf16 autocast around eval sampling+scoring (house d64 evals carry
    # eval_autocast_bf16=true); False = fp32 end to end (the d16 waves).
    eval_autocast_bf16: bool = False
    # In-training frozen-ESS diagnostic every `eval_every` steps (house
    # eval_every=200): n_eval_samples_training draws at epsilon=0, raw
    # weights, CURRENT stage sigma. 0 = off. This is the telltale the
    # behaviour-batch ESS cannot be — on-policy draws score their own
    # policy healthily even when it has mode-collapsed (reverse-KL
    # blindness; Malkin et al. 2023 Prop. 1).
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
    """s93 judging wave: parameter parity with the wave-2 heads and the
    split log_z lr on the TB arm. The plain cells above are the s92
    correctness wave, kept as archived configs (never retro-flipped).

    Parity is measured in PARAMETERS, not copied hyperparameters: the house
    d16 sizing (hidden 32 / 2 layers) gives this policy only 26.1k params
    because the swap cells' 79.5k-101k live in a backbone+head stack the
    policy doesn't have. hidden 64 / 2 layers = 101,378 params, within 0.4%
    of the masked-attention head (100,960) — the flagship comparator row."""
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


# Fair-tuning grid around the `_par` recipe (s93): lr x epsilon, sigma_c
# only (the discriminating coupling), centre EXCLUDED because the centre IS
# the `_par` cell. The 4x4 gate is a FILTER, not a signal (the window-arm
# lesson: a disjoint 4x4 separation reversed at 8x8), so this grid exists
# to show the house recipe sits in no hole and to exclude configs that
# break cheaply — the discriminating sweep belongs at 8x8. Warmup and grad
# clip are deliberately NOT folded in: all 24 GFN cells to date converged
# without them, so there is nothing for them to rescue, and changing the
# centre would orphan the judged `_par` wave.
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


# The 8x8 rung (s94). The sigma_c stage ladder mirrors the house
# _D64_SIGMA_LADDER exactly, INCLUDING its 20k final plateau: _stage_sigma
# gives every entry an equal n_steps/len share, so ten 5k-step stages with
# the final sigma repeated four times reproduce the house start-steps
# 0/5k/10k/15k/20k/25k/30k with 20k on the cell's own coupling. (The house
# ladder also drops lr 1e-3 -> 3e-4 at step 20k; the GFN trains flat-lr —
# a declared deviation the star's flat 3e-4 arm brackets.)
_D64_GFN_SIGMA_STAGES = (
    0.100, 0.140, 0.170, 0.190, 0.205, 0.215,
    SIGMA_C, SIGMA_C, SIGMA_C, SIGMA_C,
)

# Trimmed star around the d64 centre, sigma_c only (the discriminating
# coupling — d64 house ESS spans 0.735-0.953 there, against the saturated
# 4x4 gate). Arms from the s94 4x4 grid verdict: the lr axis first (epsilon
# was flat at 4x4 across both objectives), 3e-4 because "under-trained at
# 10k" no longer excuses it at 50k, 3e-3 because the FLDB triplet sat
# disjoint above its centre.
_D64_STAR_ARMS = (
    ("l3e4", "e005"), ("l3e3", "e005"), ("l1e3", "e000"), ("l1e3", "e010"),
)


def _gfn_d64_parity_cell(
    objective: str, sigma_label: str, sigma: float
) -> GFNCellCfg:
    """8x8 centre: the validated 4x4 parity recipe at the house d64 budget.

    Parity is measured params again: the SAME hidden 64 / 2 layers / 4
    heads policy lands at 104,450 params at D=8, within 3.5% (under) of
    the wave-2 d64 masked-attention head's 108,256. Budget levers match
    the wave-2 recipe field by field (50k steps, batch 128, warmup 500,
    grad clip 500, EMA 0.9999 dual eval, bf16 eval autocast, in-training
    frozen eval every 200 steps on 512 draws); compile_policy ships ON,
    gated by the GPU numerical-parity check at the launch bench.
    Declared deviations from the house recipe: flat lr (no 20k-step drop
    to 3e-4 — the star's flat 3e-4 arm brackets it) and no lr ramp on the
    TB log_z group (see warmup_steps above).
    """
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
        sigma_stages=(
            _D64_GFN_SIGMA_STAGES if sigma_label == "s220" else ()
        ),
    )
    return replace(
        cell, name=f"GFN_d64_c50_{sigma_label}_{objective}_50k_par"
    )


def _gfn_d64_star_cell(objective: str, lr_key: str, epsilon_key: str) -> GFNCellCfg:
    base = _gfn_d64_parity_cell(objective, "s220", SIGMA_C)
    return replace(
        base,
        name=base.name.replace("_par", f"_{lr_key}_{epsilon_key}_swp"),
        learning_rate=_SWEEP_LR_GRID[lr_key],
        epsilon=_SWEEP_EPSILON_GRID[epsilon_key],
    )


# Flow-head split-lr arms, FL-DB at sigma_c only (s100): the fairness fix
# the training logs point at — see flow_head_learning_rate on the cfg. Two
# values bracket the unknown: 1e-1 is the log_z precedent (~100x), 1e-2 a
# conservative 10x. ONE LEVER off the judged fldb centre (pinned by
# test_flow_lr_cells_are_fldb_centre_twins_plus_one_lever); everything else
# would confound the diagnosis. sigma_c only because the s010 fldb centre
# already sits at 0.97.
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

# Budget-doubled FLDB diagnostic (s100): settles "slow vs broken". The
# seed-42 flow-lr logs showed the centre's slow ESS climb is the POLICY
# converging (DB loss ~0.007 by 15k, ESS still +0.06/5k at 50k) and a hot
# flow head makes things WORSE (dose-monotone), so the remaining question
# is whether FLDB simply needs more steps. One lever: n_steps 100k. NOTE
# the sigma ladder DILATES with it (equal step shares: 10k/stage, 40k
# final plateau) — declared; the endpoint question is unaffected but
# mid-training step-matched comparisons against the 50k centre are
# confounded during the ladder. NEVER a table row (breaks budget parity);
# it buys the sentence "comparable quality at twice the budget, TB
# dominates at matched budget" — or refutes it.
GFN_CONFIGS.update({
    cell.name: cell
    for cell in (
        replace(
            _gfn_d64_parity_cell("fldb", "s220", SIGMA_C),
            name="GFN_d64_c50_s220_fldb_100k_par",
            n_steps=100_000,
        ),
        # Standalone-flow arm (s100): the torchgfn-conventional
        # parameterisation at the matched 50k budget, ONE lever off the
        # judged centre. Judged against the centre (policy identical), so
        # the added flow-MLP params (~17k at d64) are a declared delta,
        # not a parity break — parity with the house heads binds the
        # PRINTED centre rows, and this arm's comparison never leaves the
        # fldb family.
        replace(
            _gfn_d64_parity_cell("fldb", "s220", SIGMA_C),
            name="GFN_d64_c50_s220_fldb_50k_sfh",
            standalone_flow_head=True,
        ),
    )
})


# The 16x16 rung (s102): the fairness claim the comparator subsection
# still owes — 256-step trajectories are the regime the GFN literature
# documents as hard for TB (Madan/Pan expect FL>TB there), so the d64
# verdict "FL>TB has not appeared by 64-step trajectories" is only
# defensible as a scoped claim if this rung tests the scope.
#
# The sigma_c ladder mirrors the house d256 100k curriculum exactly under
# the equal-share rule: 20 stages of 5k = the house start-steps
# 0/5k/10k/15k/20k/25k/30k with the final 70k on the exact critical
# coupling. Budget parity is per rung: the house d256 cells train 50k at
# the floor and 100k at sigma_c, so these do too (the d64 wave's 50k/50k
# matched ITS rung's house cells). Flat lr stays a declared deviation, as
# at d64.
_D256_GFN_SIGMA_STAGES = (
    0.100, 0.140, 0.170, 0.190, 0.205, 0.215,
) + (SIGMA_C,) * 14


def _gfn_d256_parity_cell(objective: str, sigma_label: str) -> GFNCellCfg:
    """16x16 centre: the judged d64 recipe with only the rung levers moved.

    Parity is measured params a third time, and this rung is the first
    where the policy must be RE-SIZED to keep it: the d64 sizing's only
    d-dependent parameters are the position embedding (256 x hidden), so
    carrying hidden 64 up unchanged lands at 116,738 params — 12.6% under
    the chapter's thp2 stack (133,632) and 15.1% under the ma cell
    (137,440), against the +0.4%/-3.5% precedent at d16/d64. hidden 68
    (17 dims per head) gives 130,562: within 2.3% of thp2 and 5.0% of ma,
    i.e. at parity with BOTH candidate anchors, so the anchor question
    (the flagship ma comparator vs the chapter's head at a rung where ma
    fails) dissolves rather than needing a defence. Alternatives rejected:
    hidden 72 overshoots both anchors (+8.6%/+5.6%); a third layer blows
    past by 25%.

    Every other lever rides the judged d64 `_par` recipe unchanged
    (batch 128, warmup 500, grad clip 500, EMA 0.9999 dual eval, bf16
    eval autocast, in-training frozen eval every 200 steps, compile ON
    gated by the launch bench at THIS size on the venue stack).
    NOTE (cost-ladder tripwire, s100): registering this cell at hidden 68
    obsoletes tab:head-cost-ladder's d256 GFN entries, which price the
    d64 recipe re-realised at D=16 — profile_swap's gfn modes re-point
    here automatically; re-run modal_app::bench when the rows are next
    touched."""
    sigma_c = sigma_label == "s220"
    cell = replace(
        _gfn_d64_parity_cell(
            objective, sigma_label, SIGMA_C if sigma_c else 0.10
        ),
        D=16,
        hidden_dim=68,
        n_steps=100_000 if sigma_c else 50_000,
        sigma_stages=_D256_GFN_SIGMA_STAGES if sigma_c else (),
        # In-training eval cadence moved to the HOUSE d256 regime
        # (eval_every=500, 256 draws, bf16 autocast — the swap-head cells'
        # exact schedule) rather than riding the d64 wave's 200/512:
        # these rows sit beside the house cells in tab:eval-hard-16x16,
        # so within-rung parity outranks cross-rung GFN consistency. The
        # headline 5000-draw final eval is untouched. bf16 eval autocast
        # is already inherited from the d64 parity cell.
        eval_every=500,
        n_eval_samples_training=256,
    )
    steps_label = "100k" if sigma_c else "50k"
    return replace(
        cell, name=f"GFN_d256_c50_{sigma_label}_{objective}_{steps_label}_par"
    )


GFN_CONFIGS.update({
    cell.name: cell
    for objective in GFN_OBJECTIVES
    for cell in (
        _gfn_d256_parity_cell(objective, "s010"),
        _gfn_d256_parity_cell(objective, "s220"),
    )
})
