from experiments.constrained_soft_02.configs import CONFIGS as CONSTRAINED_CONFIGS
from experiments.dnfs_baseline_01.configs import CONFIGS as BASELINE_CONFIGS
from experiments.dnfs_baseline_01.configs import IsingCfg


def test_ising_cfg_base_composition_defaults_to_half():
    assert IsingCfg().base_composition == 0.5


def test_constrained_configs_use_constraints_wandb_project():
    cfg = CONSTRAINED_CONFIGS["S2_d4_c03_l50_letf"]

    assert cfg.wandb_project == "dnfs-constraints"
    assert cfg.ising.target_composition == 0.3
    assert cfg.ising.composition_penalty_strength == 50.0


def test_baseline_configs_keep_baseline_wandb_project():
    assert BASELINE_CONFIGS["stage_2_d4"].wandb_project == "dnfs-baseline"


def test_letf_constrained_cells_use_let_arch_and_penalty():
    """leTF constrained cells: correct arch + penalty fields + wandb project."""
    for cell_name, expected_d, expected_hidden in [
        ("S2_d4_c03_l50_letf", 4, 64),
        ("S2_d10_c03_l50_letf_ne128", 10, 128),
    ]:
        cfg = CONSTRAINED_CONFIGS[cell_name]
        assert cfg.model.kind == "let"
        assert cfg.model.hidden_dim == expected_hidden
        assert cfg.model.n_heads == 4
        assert cfg.ising.D == expected_d
        assert cfg.ising.target_composition == 0.3
        assert cfg.ising.composition_penalty_strength == 50.0
        assert cfg.wandb_project == "dnfs-constraints"
        assert cfg.estimator == "control_variate"


def test_c05_d4_letf_cell_mirrors_c03_with_only_target_changed():
    """Supervisor-requested c_target=0.5 cell at D=4: shape-identical to the
    c=0.3 D=4 leTF cell except for `target_composition`. Pinned so any
    future refactor of the c=0.5 cell stays paired with its c=0.3 sibling."""
    base = CONSTRAINED_CONFIGS["S2_d4_c03_l50_letf"]
    twin = CONSTRAINED_CONFIGS["S2_d4_c05_l50_letf"]
    assert twin.ising.target_composition == 0.5
    assert twin.ising.D == base.ising.D
    assert twin.ising.sigma == base.ising.sigma
    assert twin.ising.bias == base.ising.bias
    assert (
        twin.ising.composition_penalty_strength
        == base.ising.composition_penalty_strength
    )
    assert twin.model == base.model
    assert twin.train == base.train
    assert twin.ctmc == base.ctmc
    assert twin.eval == base.eval
    assert twin.estimator == base.estimator
    assert twin.wandb_project == base.wandb_project


def test_d4_critical_cell_mirrors_d4_with_only_sigma_changed():
    """4x4 row at the critical coupling for the Stage-4 table, giving the
    critical operating point an exact-enumeration reference. Shape-identical
    to stage_4_d4 except sigma, and trains direct with no curriculum: the
    sigma-transition collapse that motivated the d10 curriculum was a D=10
    finding, and the 4x4 lattice has no phase transition to fight."""
    base = BASELINE_CONFIGS["stage_4_d4"]
    cfg = BASELINE_CONFIGS["stage_4_d4_critical"]
    assert cfg.ising.sigma == 0.22305
    assert cfg.ising.D == base.ising.D
    assert cfg.ising.bias == base.ising.bias
    assert cfg.model == base.model
    assert cfg.train == base.train
    assert cfg.ctmc == base.ctmc
    assert cfg.eval == base.eval
    assert cfg.estimator == base.estimator
    assert cfg.curriculum is None
    assert cfg.wandb_project == base.wandb_project


def test_letf_d10_ne128_cell_carries_stability_stack():
    """leTF d=10 ne128 cell carries the stability stack (warmup=2000, ne=128).

    Pre-stability-stack siblings (`S2_d10_c03_l50_letf` with ne=64, warmup=500)
    were removed 2026-05-22 in the constrained-side cleanup; this test now
    just asserts the absolute values on the surviving cell rather than the
    relative deviation from a removed baseline.
    """
    cfg = CONSTRAINED_CONFIGS["S2_d10_c03_l50_letf_ne128"]
    assert cfg.ctmc.n_euler_steps == 128
    assert cfg.train.warmup_steps == 2000
    assert cfg.model.kind == "let"
    assert cfg.model.hidden_dim == 128
    assert cfg.ising.D == 10
    assert cfg.wandb_project == "dnfs-constraints"


def test_d10_c05_ne64_cell_is_report_witness():
    """D=10 c=0.5 headline cell: inherits the c=0.3 ne128 stability stack
    (warmup=2000, lambda=50, let/h128) but runs the paper-faithful n_euler=64
    grid rather than the unvalidated 128. c=0.3 is dropped from the report,
    so this is the D=10 soft witness; pinned so a refactor keeps it paired
    with its c=0.3 sibling on everything except target and n_euler."""
    base = CONSTRAINED_CONFIGS["S2_d10_c03_l50_letf_ne128"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64"]
    assert cfg.ising.target_composition == 0.5
    assert cfg.ctmc.n_euler_steps == 64
    assert cfg.ising.D == base.ising.D
    assert (
        cfg.ising.composition_penalty_strength
        == base.ising.composition_penalty_strength
    )
    assert cfg.train.warmup_steps == base.train.warmup_steps
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.wandb_project == base.wandb_project


def test_d10_c05_lambda_sweep_cells_mirror_l50_with_only_penalty_changed():
    """λ-sweep cells at c=0.5 d=10: clones of the l50 ne64 witness, λ only."""
    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64"]
    for lam in (5.0, 10.0, 100.0):
        cfg = CONSTRAINED_CONFIGS[f"S2_d10_c05_l{int(lam)}_letf_ne64"]
        assert cfg.ising.composition_penalty_strength == lam
        assert cfg.ising.D == base.ising.D
        assert cfg.ising.sigma == base.ising.sigma
        assert cfg.ising.bias == base.ising.bias
        assert cfg.ising.target_composition == base.ising.target_composition
        assert cfg.train == base.train
        assert cfg.ctmc == base.ctmc
        assert cfg.eval == base.eval
        assert cfg.model == base.model
        assert cfg.estimator == base.estimator
        assert cfg.wandb_project == base.wandb_project


def test_d4_c05_lambda_sweep_cells_mirror_l50_with_only_penalty_changed():
    """λ-sweep cells at c=0.5 d=4: clones of the l50 cell, λ only."""
    base = CONSTRAINED_CONFIGS["S2_d4_c05_l50_letf"]
    for lam in (5.0, 10.0, 100.0):
        cfg = CONSTRAINED_CONFIGS[f"S2_d4_c05_l{int(lam)}_letf"]
        assert cfg.ising.composition_penalty_strength == lam
        assert cfg.ising.D == base.ising.D
        assert cfg.ising.sigma == base.ising.sigma
        assert cfg.ising.bias == base.ising.bias
        assert cfg.ising.target_composition == base.ising.target_composition
        assert cfg.train == base.train
        assert cfg.ctmc == base.ctmc
        assert cfg.eval == base.eval
        assert cfg.model == base.model
        assert cfg.estimator == base.estimator
        assert cfg.wandb_project == base.wandb_project


def test_d10_c05_l50_ne128_mirrors_ne64_with_only_euler_steps_changed():
    """Recipe-ladder rung 1 (2026-06-11): l50 witness with ne128 only."""
    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne128"]
    assert cfg.ctmc.n_euler_steps == 128
    assert cfg.ising == base.ising
    assert cfg.train == base.train
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.wandb_project == base.wandb_project


def test_d10_c05_l50_anneal_mirrors_ne64_with_lambda_curriculum_only():
    """Recipe-ladder anneal rung (2026-06-11): l50 ne64 witness plus a
    10->25->50 lambda curriculum; the final stage must land on the cell's
    own penalty strength so eval reports the operating point."""
    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64_anneal"]
    stages = cfg.lambda_curriculum.stages
    assert [
        (s.start_step, s.composition_penalty_strength) for s in stages
    ] == [(0, 10.0), (10_000, 25.0), (20_000, 50.0)]
    assert stages[-1].composition_penalty_strength == (
        cfg.ising.composition_penalty_strength
    )
    assert cfg.ising == base.ising
    assert cfg.train == base.train
    assert cfg.ctmc == base.ctmc
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.wandb_project == base.wandb_project


def test_fc_gate_anneal_cells_mirror_c05_anneal_with_only_composition_changed():
    """F(c) campaign gate windows (2026-06-13): off-centre clones of the won
    c=0.5 anneal rung varying only target_composition, so the off-centre runs
    are a controlled test of whether the annealed recipe generalises."""
    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64_anneal"]
    for key, ct in (
        ("S2_d10_c030_l50_letf_ne64_anneal", 0.30),
        ("S2_d10_c055_l50_letf_ne64_anneal", 0.55),
        ("S2_d10_c060_l50_letf_ne64_anneal", 0.60),
        ("S2_d10_c065_l50_letf_ne64_anneal", 0.65),
        ("S2_d10_c080_l50_letf_ne64_anneal", 0.80),
    ):
        cfg = CONSTRAINED_CONFIGS[key]
        assert cfg.ising.target_composition == ct
        # only the composition moves; the rest of the ising block is the anchor's
        assert cfg.ising.D == base.ising.D
        assert cfg.ising.sigma == base.ising.sigma
        assert cfg.ising.bias == base.ising.bias
        assert (
            cfg.ising.composition_penalty_strength
            == base.ising.composition_penalty_strength
        )
        assert cfg.train == base.train
        assert cfg.ctmc == base.ctmc
        assert cfg.eval == base.eval
        assert cfg.model == base.model
        assert cfg.estimator == base.estimator
        assert cfg.lambda_curriculum == base.lambda_curriculum
        assert cfg.wandb_project == base.wandb_project


def test_fc_c080_ne128_anneal_mirrors_ne64_with_only_euler_steps_changed():
    """F(c) gate fallback rung (2026-06-16): the c=0.80 stress window with a
    finer Euler grid only (ne64 -> ne128), a controlled test of whether finer
    integration rescues seed survival at the most off-centre target."""
    base = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne64_anneal"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne128_anneal"]
    assert cfg.ctmc.n_euler_steps == 128
    assert base.ctmc.n_euler_steps == 64
    assert cfg.ising == base.ising
    assert cfg.train == base.train
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.lambda_curriculum == base.lambda_curriculum
    assert cfg.wandb_project == base.wandb_project


def test_matched_anneal_mirrors_c080_ne128_with_only_base_composition():
    base = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne128_anneal"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne128_matched_anneal"]
    assert cfg.ising.base_composition == 0.80
    assert base.ising.base_composition == 0.5
    assert cfg.ising.target_composition == base.ising.target_composition == 0.80
    assert cfg.ising.D == base.ising.D
    assert cfg.ising.sigma == base.ising.sigma
    assert cfg.ising.composition_penalty_strength == (
        base.ising.composition_penalty_strength
    )
    assert cfg.train == base.train
    assert cfg.ctmc == base.ctmc
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.lambda_curriculum == base.lambda_curriculum
    assert cfg.wandb_project == base.wandb_project


def test_matched_fixed50_drops_anneal_keeps_matched_base():
    base = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne128_matched_anneal"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c080_l50_letf_ne128_matched_fixed50"]
    assert cfg.lambda_curriculum is None
    assert cfg.ising == base.ising  # same matched base + target + lambda=50
    assert cfg.train == base.train
    assert cfg.ctmc == base.ctmc
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator


def test_hard_cell_is_fixed_composition_no_penalty():
    from experiments.constrained_hard_03.configs import CONFIGS

    cfg = CONFIGS["H2_d16_c50_s010_letf_dh"]
    assert cfg.ising.D == 4
    assert cfg.ising.target_composition == 0.5
    assert cfg.ising.composition_penalty_strength == 0.0
    assert cfg.ising.sigma < 0.2                       # subcritical floor rung
    assert cfg.model.kind == "letf"
    assert cfg.head_kind == "doubly_hollow"


def test_hard_ladder_covers_three_sigmas_plus_control():
    from experiments.constrained_hard_03.configs import CONFIGS

    # Scoped to the binary D=16 ladder this test is about: `_dh` alone now
    # also catches the Potts gate cell (H3_d9, a different lattice, species
    # count and sigma convention), which is not a rung of this ladder.
    ladder = [
        k for k in CONFIGS if k.startswith("H2_d16") and k.endswith("_dh")
    ]
    assert sorted(CONFIGS[k].ising.sigma for k in ladder) == [0.10, 0.223, 0.40]
    control = CONFIGS["H2_d16_c50_s010_letf_na"]
    assert control.head_kind == "non_antisym"


def test_hard_d64_cell_enables_tier2_flags():
    """User sign-off 2026-07-06 (perf-branch evidence): the D=8 scaling cell
    runs with the SDPA readout and bf16 IN-TRAINING evals only — run.py's
    final 5,000-sample eval stays fp32. The D=4 gate cells stay flag-off:
    Tier-2 enablement is a per-cell decision, never a global default."""
    from experiments.constrained_hard_03.configs import CONFIGS

    cfg = CONFIGS["H2_d64_c50_s223_letf_mo"]
    assert cfg.model.use_sdpa_readout is True
    assert cfg.eval.eval_autocast_bf16 is True
    gate = CONFIGS["H2_d16_c50_s223_letf_dh"]
    assert gate.model.use_sdpa_readout is False
    assert gate.eval.eval_autocast_bf16 is False


def test_d64_25k_budget_probe_mirrors_base_cell_except_n_steps():
    """The 25k budget probe (2026-07-06) must isolate ONE variable: same cell
    as H2_d64_c50_s223_letf_mo in every respect except the training budget,
    so a converged/stalled outcome is attributable to budget alone."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    base = CONFIGS["H2_d64_c50_s223_letf_mo"]
    probe = CONFIGS["H2_d64_c50_s223_letf_mo_25k"]
    assert probe.train.n_steps == 25_000
    normalised = replace(
        probe, name=base.name, train=replace(probe.train, n_steps=base.train.n_steps)
    )
    assert normalised == base


def test_d64_50k_curriculum_cell_mirrors_base_except_budget_and_ladder():
    """The curriculum rung (2026-07-06) changes exactly TWO things vs the base
    d=64 cell — budget (50k) and the sigma-plateau ladder — so its outcome is
    attributable to those levers. The ladder is the proven baseline recipe
    (stage_3 conv critical, itself 50k steps) with the final stage at the hard
    cells' 0.223; boundaries must align with outer cycles for train_swap."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    base = CONFIGS["H2_d64_c50_s223_letf_mo"]
    cell = CONFIGS["H2_d64_c50_s223_letf_mo_50k_curr"]
    assert cell.train.n_steps == 50_000
    stages = cell.curriculum.stages
    assert (stages[0].start_step, stages[0].sigma) == (0, 0.100)
    assert stages[-1].sigma == cell.ising.sigma == 0.223
    assert all(
        later.start_step > earlier.start_step and later.sigma > earlier.sigma
        for earlier, later in zip(stages, stages[1:])
    )
    assert all(
        stage.start_step < cell.train.n_steps
        and stage.start_step % cell.train.inner_steps_per_outer == 0
        for stage in stages
    )
    normalised = replace(
        cell, name=base.name, curriculum=None,
        train=replace(cell.train, n_steps=base.train.n_steps),
    )
    assert normalised == base


def test_hard_cfg_anchor_chunk_size_reaches_mask_one_head():
    """d=256 cannot run the mask_one head unchunked (the anchor-batched pass
    builds a (d*B)-row buffer); the head's anchor_chunk_size knob must be
    settable from HardStageCfg, defaulting to None (unchunked, behaviour
    unchanged)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    from discrete_flow_sampler.models.letf import LeTFRateMatrix

    cfg = CONFIGS["H2_d64_c50_s223_letf_mo"]
    backbone = LeTFRateMatrix(
        d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    assert cfg.anchor_chunk_size is None
    assert build_swap_head(cfg, backbone).anchor_chunk_size is None
    chunked = replace(cfg, anchor_chunk_size=32)
    assert build_swap_head(chunked, backbone).anchor_chunk_size == 32


def test_hard_cfg_band_capacity_knobs_reach_band_heads():
    """Band-capacity push (2026-07-08): band_feature_dim /
    attention_dim / pair_offsets must be settable from HardStageCfg,
    defaulting to None = the constructions every prior run used (offsets
    (1, D), feature width 16, attention width 32), so existing cells build
    byte-identical heads."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    from discrete_flow_sampler.models.letf import LeTFRateMatrix

    cfg = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    backbone = LeTFRateMatrix(
        d=64, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )

    assert (cfg.band_feature_dim, cfg.attention_dim, cfg.pair_offsets) == (
        None, None, None,
    )
    default_head = build_swap_head(cfg, backbone)
    assert default_head.pair_offsets == (1, cfg.ising.D)
    assert default_head.band_unary_features[-1].out_features == 16
    assert default_head.band_query_projections[0].out_features == 32

    tuned = replace(
        cfg, band_feature_dim=32, attention_dim=64, pair_offsets=(1, 2, 8, 16)
    )
    tuned_head = build_swap_head(tuned, backbone)
    assert tuned_head.pair_offsets == (1, 2, 8, 16)
    assert len(tuned_head.band_pair_features) == 4
    assert tuned_head.band_unary_features[-1].out_features == 32
    assert tuned_head.band_query_projections[0].out_features == 64

    interval = replace(tuned, head_kind="interval")
    interval_head = build_swap_head(interval, backbone)
    assert interval_head.pair_offsets == (1, 2, 8, 16)
    assert interval_head.band_unary_features[-1].out_features == 32

    # Stencil family: use_stencil defaults off (byte-identical
    # head) and, when set, builds a stencil MLP addressing the D x D grid.
    assert cfg.use_stencil is False
    assert not hasattr(default_head, "band_stencil_features")
    stencil_head = build_swap_head(replace(cfg, use_stencil=True), backbone)
    assert stencil_head.stencil_side == cfg.ising.D
    assert stencil_head.band_stencil_features[0].in_features == 5 * 16


def test_band_push_cells_mirror_ma_twin_except_declared_fields():
    """Band-capacity push batch 1 (2026-07-08): each cell must be a
    single-variable twin of H2_d64_c50_s223_letf_ma_50k_curr so its outcome
    is attributable to the declared change alone (the discriminator changes
    the head kind; the other two change exactly one capacity axis)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]

    discriminator = CONFIGS["H2_d64_c50_s223_letf_iv_50k_curr"]
    assert discriminator.head_kind == "interval"
    assert replace(discriminator, name=twin.name, head_kind=twin.head_kind) == twin

    wide = CONFIGS["H2_d64_c50_s223_letf_ma_wide_50k_curr"]
    assert (wide.band_feature_dim, wide.attention_dim) == (32, 64)
    assert (
        replace(wide, name=twin.name, band_feature_dim=None, attention_dim=None)
        == twin
    )

    offsets = CONFIGS["H2_d64_c50_s223_letf_ma_offs_50k_curr"]
    assert offsets.pair_offsets == (1, 2, 8, 16)
    assert replace(offsets, name=twin.name, pair_offsets=None) == twin

    # Round-2 stencil family: use_stencil is the only change.
    stencil = CONFIGS["H2_d64_c50_s223_letf_ma_stencil_50k_curr"]
    assert stencil.use_stencil is True and twin.use_stencil is False
    assert replace(stencil, name=twin.name, use_stencil=False) == twin


def test_horizon_100k_cells_mirror_50k_twins_except_n_steps():
    """Horizon extension (2026-07-22): the 50k curriculum
    stops while both heads are still improving (loss -12.0% / -7.4% over the
    final 10k steps, train ESS still climbing), so the 0.78/0.80 ceiling is
    read off unconverged runs. These cells double the budget and change
    NOTHING else — same sigma ladder, same lr drop, so the extra 50k steps all
    land on the final sigma=0.223 plateau (20k -> 70k) rather than stretching
    the anneal. n_steps must be the sole difference from the 50k twin, and the
    stencil pair must differ from each other by use_stencil alone, or the
    ablation ladder stops being attributable."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    pairs = {
        "H2_d64_c50_s223_letf_ma_100k_curr": "H2_d64_c50_s223_letf_ma_50k_curr",
        "H2_d64_c50_s223_letf_ma_stencil_100k_curr": (
            "H2_d64_c50_s223_letf_ma_stencil_50k_curr"
        ),
        # mask_one gets the same treatment: judging the first two exposed that
        # the reference rung is unconverged at 50k as well, so the horizon
        # change has to reach it or the ladder compares heads at two budgets.
        "H2_d64_c50_s223_letf_mo_100k_curr": "H2_d64_c50_s223_letf_mo_50k_curr",
    }
    for long_name, short_name in pairs.items():
        long_cell, short_cell = CONFIGS[long_name], CONFIGS[short_name]
        assert long_cell.train.n_steps == 100_000
        assert short_cell.train.n_steps == 50_000
        rebuilt = replace(
            long_cell,
            name=short_cell.name,
            train=replace(long_cell.train, n_steps=short_cell.train.n_steps),
        )
        assert rebuilt == short_cell

    # The ladder is deliberately NOT stretched: identical stage boundaries at
    # both horizons, so the lr drop still fires at 20k and the final plateau
    # absorbs the whole extra budget.
    long_ma = CONFIGS["H2_d64_c50_s223_letf_ma_100k_curr"]
    assert long_ma.curriculum == CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"].curriculum
    assert long_ma.curriculum.stages[-1].start_step == 30_000

    # The 100k pair is itself a single-variable comparison (the MA twin is the
    # control for "did the stencil's +0.024 survive a converged horizon").
    long_stencil = CONFIGS["H2_d64_c50_s223_letf_ma_stencil_100k_curr"]
    assert long_stencil.use_stencil is True and long_ma.use_stencil is False
    assert replace(long_stencil, name=long_ma.name, use_stencil=False) == long_ma


def test_d256_rung_mirrors_ma_twin_except_declared_scale_fields():
    """The 16x16 rung (hard.tex §5.6 plan of record): the d64
    masked-attention curriculum cell rescaled and nothing else — same head,
    same 50k sigma ladder on absolute start_steps, same n_euler=128. That
    Euler budget is only clip-safe at d=256 because the matching step is
    declared canonical (CTMCCfg.use_matching_step; clip-safe one-event
    extrapolates to ~390 steps), so the knob must be True and must be the
    ONLY trajectory-step difference. Eval deltas are diagnostics-only —
    cadence and in-training draw shrink because the per-pass cost is ~16x
    the d64 cell's; the final eval keeps the 5000-draw protocol."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    d256 = CONFIGS["H2_d256_c50_s223_letf_ma_50k_curr"]

    assert d256.ising.D == 16
    assert d256.ctmc.use_matching_step is True
    assert twin.ctmc.use_matching_step is False
    assert d256.ctmc.n_euler_steps == twin.ctmc.n_euler_steps == 128
    assert d256.curriculum == twin.curriculum

    # Declared eval deltas, stated exactly.
    assert d256.eval.n_eval_samples == 5000
    assert d256.eval.eval_every == 500
    assert d256.eval.n_eval_samples_training == 256
    assert d256.eval.eval_sample_chunk == 64

    # Nothing else moved.
    rebuilt = replace(
        d256,
        name=twin.name,
        ising=replace(d256.ising, D=8),
        ctmc=replace(d256.ctmc, use_matching_step=False),
        eval=twin.eval,
    )
    assert rebuilt == twin


def test_grouped_anchor_cells_mirror_ma_twin_except_declared_fields():
    """Grouped-anchor batch 1 (2026-07-22): each cell must be a
    single-variable twin of H2_d64_c50_s223_letf_ma_50k_curr apart from the
    head selection and its declared knobs, so the result reads against the
    existing ladder rungs rather than against a different recipe. ga16 differs
    from ga8 by k alone (the cost dial) and ga8_contig by grouping alone (the
    dispersal control) -- both must hold, or neither comparison is clean."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    ga8 = CONFIGS["H2_d64_c50_s223_letf_ga8_50k_curr"]
    ga16 = CONFIGS["H2_d64_c50_s223_letf_ga16_50k_curr"]
    contiguous = CONFIGS["H2_d64_c50_s223_letf_ga8_contig_50k_curr"]

    for cell, n_groups, grouping in (
        (ga8, 8, "diagonal"), (ga16, 16, "diagonal"), (contiguous, 8, "contiguous"),
    ):
        assert cell.head_kind == "grouped_anchor"
        assert (cell.n_groups, cell.grouping) == (n_groups, grouping)
        normalised = replace(
            cell, name=twin.name, head_kind=twin.head_kind,
            n_groups=None, grouping="diagonal",
        )
        assert normalised == twin

    # The two comparisons the batch is built to make, each single-variable.
    assert replace(ga16, name=ga8.name, n_groups=8) == ga8
    assert replace(contiguous, name=ga8.name, grouping="diagonal") == ga8


def test_build_swap_head_wires_the_grouped_anchor_knobs():
    """head_kind='grouped_anchor' must reach the head with cfg.ising.D as the
    lattice side (so 'diagonal' can disperse across the raster), and must fail
    loudly rather than silently defaulting when n_groups is unset.

    All three launch cells are built, not just ga8: `site_groups` REFUSES an
    n_groups it cannot balance on the raster, so constructing each cell's head
    is what turns "this k is legal at this D" from an assumption into a test.
    A cell whose k the head rejects would otherwise fail at run start, on the
    GPU, after the job had been paid for."""
    from dataclasses import replace

    import pytest
    import torch
    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    from discrete_flow_sampler.models.letf import LeTFRateMatrix

    def _backbone(cfg):
        return LeTFRateMatrix(
            d=cfg.ising.D**2, vocab_size=2, hidden_dim=8, n_layers=2, n_heads=2
        )

    expected = {"ga8": (8, "diagonal"), "ga16": (16, "diagonal"),
                "ga8_contig": (8, "contiguous")}
    for tag, (n_groups, grouping) in expected.items():
        cfg = CONFIGS[f"H2_d64_c50_s223_letf_{tag}_50k_curr"]
        head = build_swap_head(cfg, _backbone(cfg))
        assert (head.n_groups, head.grouping) == (n_groups, grouping)
        assert head.group_of_site.shape == (cfg.ising.D**2,)

    # The dispersal the ga8 cell is actually buying, asserted rather than
    # gestured at: on the D x D raster with n_groups = D, each group must hit
    # every row exactly once and every column exactly once (the Latin-square
    # diagonal). This is a property of the RASTER, so it holds only if
    # lattice_side arrived as cfg.ising.D -- a different side reshapes the
    # grid and the one-per-row-and-column structure dies.
    cfg = CONFIGS["H2_d64_c50_s223_letf_ga8_50k_curr"]
    head = build_swap_head(cfg, _backbone(cfg))
    side = cfg.ising.D
    site = torch.arange(side * side)
    rows, cols = site // side, site % side
    for group_id in range(head.n_groups):
        member = head.group_of_site == group_id
        assert sorted(rows[member].tolist()) == list(range(side))
        assert sorted(cols[member].tolist()) == list(range(side))

    with pytest.raises(ValueError, match="requires n_groups"):
        build_swap_head(replace(cfg, n_groups=None), _backbone(cfg))


def test_demo_4x4_cells_mirror_dh_ladder_except_declared_fields():
    """4x4 supervisor-demo cells (2026-07-08): single-variable twins of the
    2k dh ladder — only name, head_kind and n_steps may differ, so head and
    budget effects in the demo stay attributable."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    demo_cells = {
        "H2_d16_c50_s010_letf_ma_10k": ("H2_d16_c50_s010_letf_dh", "masked_attention"),
        "H2_d16_c50_s223_letf_ma_10k": ("H2_d16_c50_s223_letf_dh", "masked_attention"),
        "H2_d16_c50_s010_letf_mo_10k": ("H2_d16_c50_s010_letf_dh", "mask_one"),
        "H2_d16_c50_s223_letf_mo_10k": ("H2_d16_c50_s223_letf_dh", "mask_one"),
    }
    for demo_name, (ladder_name, head_kind) in demo_cells.items():
        demo, ladder = CONFIGS[demo_name], CONFIGS[ladder_name]
        assert demo.head_kind == head_kind
        assert demo.train.n_steps == 10_000
        rebuilt = replace(
            demo,
            name=ladder.name,
            head_kind=ladder.head_kind,
            train=replace(demo.train, n_steps=ladder.train.n_steps),
        )
        assert rebuilt == ladder


def test_walkback_d8_baseline_twin_mirrors_d10_except_lattice_side():
    """Walk-back-to-8x8 slate (2026-08-12). The three experiment chapters
    shared no non-enumerable lattice size — baseline and soft ran 10x10,
    hard ran 8x8 and 16x16 — so 8x8 (d = 64 sites, the lattice the hard
    cells name d64) becomes the shared cross-chapter comparison size, and
    10x10 keeps its paper-replication role. This cell must be a D=8 twin of
    the d10 critical paper-curriculum record: EVERY knob except the lattice
    side copied — same sigma ladder and LR drops, same 200k budget, same
    ne64 — so any difference against the d10 four-seed family is
    attributable to lattice size alone."""
    from dataclasses import replace

    base = BASELINE_CONFIGS["stage_4_d10_critical_paper_curriculum"]
    twin = BASELINE_CONFIGS["stage_4_d8_critical_paper_curriculum"]
    assert twin.ising.D == 8
    rebuilt = replace(twin, name=base.name, ising=replace(twin.ising, D=10))
    assert rebuilt == base


def test_walkback_d8_soft_twins_mirror_d10_except_lattice_side():
    """Soft-family walk-back twins (2026-08-12): same recipe, same sigma,
    same lambda — only the lattice side moves, so differences vs the d10
    records are attributable to size. The c03 twin mirrors the relaunch
    recipe (warmup 2000) that the current d10 entry records; the superseded
    first c03 batch ran warmup 500."""
    from dataclasses import replace

    pairs = {
        "S2_d8_c05_l10_letf_ne64": "S2_d10_c05_l10_letf_ne64",
        "S2_d8_c05_l50_letf_ne64": "S2_d10_c05_l50_letf_ne64",
        "S2_d8_c03_l50_letf_ne128": "S2_d10_c03_l50_letf_ne128",
    }
    for twin_name, base_name in pairs.items():
        base = CONSTRAINED_CONFIGS[base_name]
        twin = CONSTRAINED_CONFIGS[twin_name]
        assert twin.ising.D == 8, twin_name
        rebuilt = replace(twin, name=base.name, ising=replace(twin.ising, D=10))
        assert rebuilt == base, twin_name


def test_c05_ne128_anneal_control_mirrors_ne64_anneal_euler_only():
    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne64_anneal"]
    cfg = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne128_anneal"]
    assert cfg.ctmc.n_euler_steps == 128
    assert base.ctmc.n_euler_steps == 64
    assert cfg.ising == base.ising  # base_composition stays 0.5 both sides
    assert cfg.train == base.train
    assert cfg.eval == base.eval
    assert cfg.model == base.model
    assert cfg.estimator == base.estimator
    assert cfg.lambda_curriculum == base.lambda_curriculum
    assert cfg.wandb_project == base.wandb_project
