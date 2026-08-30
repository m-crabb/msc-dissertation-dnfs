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


def test_fc_ne128_retrain_windows_mirror_c05_ne128_with_only_composition_changed():
    """F(c) retrain at ne128 (2026-08-23): the four missing windows are the
    c=0.5 ne128 anneal cell with only target_composition moved, so the
    retrained curve is one recipe end to end."""
    from dataclasses import replace

    base = CONSTRAINED_CONFIGS["S2_d10_c05_l50_letf_ne128_anneal"]
    for key, ct in (
        ("S2_d10_c030_l50_letf_ne128_anneal", 0.30),
        ("S2_d10_c055_l50_letf_ne128_anneal", 0.55),
        ("S2_d10_c060_l50_letf_ne128_anneal", 0.60),
        ("S2_d10_c065_l50_letf_ne128_anneal", 0.65),
    ):
        cfg = CONSTRAINED_CONFIGS[key]
        assert cfg.ising.target_composition == ct
        assert replace(cfg, name=base.name, ising=base.ising) == base
        assert cfg.ctmc.n_euler_steps == 128


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
    """Signed off 2026-07-06 (perf-branch evidence): the D=8 scaling cell
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


def test_m6_replay2_smoke_mirrors_ma_recipe_except_declared_fields():
    """M6 (2026-08-14, plan Task 6): the replay2 12k smoke is the archived
    MA curriculum recipe with exactly the declared deviations — the 12k
    budget, the forced ladder truncation to the first three stages (the
    validator rejects stages at or past n_steps), and replay_buffer_cycles
    8 -> 2 — so its outcome attributes to the buffer window alone."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    cell = CONFIGS["H2_d64_smoke12k_replay2"]
    assert cell.train.replay_buffer_cycles == 2
    assert cell.train.n_steps == 12_000
    rebuilt_twin = replace(
        cell,
        name=twin.name,
        train=replace(
            cell.train,
            n_steps=twin.train.n_steps,
            replay_buffer_cycles=twin.train.replay_buffer_cycles,
        ),
        curriculum=twin.curriculum,
    )
    assert rebuilt_twin == twin


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


def test_factorised_gate_cells_mirror_ma_twin_except_declared_fields():
    """Factorised-head 4x4 gate (2026-08-13): single-variable twins of the
    MA demo cells — only name, head_kind and the declared factorised knobs
    may differ, so the head effect stays attributable to the declared
    change. The knob dict below is also the gate's arm table."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    arm_knobs = {
        "fab8": {},
        "fab16": {"bilinear_rank": 16},
        "fbil": {"use_global": False},
        "fglo": {"use_bilinear": False},
        "fmp40": {"factor_dim": 40},
        "fmo2": {"site_orderings": ("row", "col")},
        "fib": {"interior_band": "prefix"},
        "fatt": {"interior_band": "attention"},
        "fimo2": {"interior_band": "prefix", "site_orderings": ("row", "col")},
        "fmoatt": {"interior_band": "attention", "site_orderings": ("row", "col")},
    }
    factorised_fields = (
        "bilinear_rank", "factor_dim", "global_feature_dim",
        "use_bilinear", "use_global", "site_orderings", "interior_band",
    )
    for arm, knobs in arm_knobs.items():
        for sigma_label in ("s010", "s223"):
            name = f"H2_d16_c50_{sigma_label}_letf_{arm}_10k"
            twin_name = f"H2_d16_c50_{sigma_label}_letf_ma_10k"
            cell, twin = CONFIGS[name], CONFIGS[twin_name]
            assert cell.head_kind == "factorised"
            for field, value in knobs.items():
                assert getattr(cell, field) == value, f"{name}: {field}"
            rebuilt = replace(
                cell,
                name=twin.name,
                head_kind=twin.head_kind,
                **{f: getattr(twin, f) for f in factorised_fields},
            )
            assert rebuilt == twin, f"{name} drifts from its MA twin"


def test_fab8_d64_rung_mirrors_ma_curriculum_twin_except_declared_fields():
    """d64 scaling rung (2026-08-13): the factorised cell must be a
    single-variable twin of the archived MA 50k-curriculum rung — only
    name, head_kind and the declared dual-eval EMA instrument may differ,
    so the transfer read stays attributable to the head."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d64_c50_s223_letf_fab8_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    assert cell.head_kind == "factorised"
    assert cell.ema_decay == 0.9999 and twin.ema_decay == 0.0
    rebuilt = replace(
        cell, name=twin.name, head_kind=twin.head_kind, ema_decay=0.0
    )
    assert rebuilt == twin


def test_fab16_d64_rung_mirrors_fab8_rung_except_rank():
    """Rank-at-scale arm (2026-08-14): the fab16 d64 cell must differ from
    the fab8 rung in name and bilinear_rank ALONE, so the rank read at the
    0.27-deficit operating point stays single-variable."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d64_c50_s223_letf_fab16_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_fab8_50k_curr"]
    assert cell.bilinear_rank == 16
    rebuilt = replace(
        cell, name=twin.name, bilinear_rank=twin.bilinear_rank
    )
    assert rebuilt == twin


def test_fmo2_d64_rung_mirrors_fab8_rung_except_orderings():
    """A-prime at scale (2026-08-14, logged amendment): the fmo2 d64 cell
    must differ from the fab8 rung in name and site_orderings ALONE, so the
    interior-coverage read at the 0.27-deficit operating point stays
    single-variable."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d64_c50_s223_letf_fmo2_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_fab8_50k_curr"]
    assert cell.site_orderings == ("row", "col")
    rebuilt = replace(
        cell, name=twin.name, site_orderings=twin.site_orderings
    )
    assert rebuilt == twin


def test_d256_cv2_cell_mirrors_naive_rescue_except_declared_fields():
    """Phase-2 estimator switch (2026-08-14): the cv2 cell must be the naive
    rescue's shape with exactly the declared deltas — estimator back to the
    control variate, 20k flat-sigma_c steps in place of the 50k ladder
    (training continues from the naive checkpoint via --init-from), lr
    pinned to the ladder's final 3e-4, and the dual-eval EMA instrument —
    so the estimator read stays attributable."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d256_c50_s223_letf_ma_20k_sc_cv2"]
    twin = CONFIGS["H2_d256_c50_s223_letf_ma_50k_curr_naive"]
    assert cell.estimator == "control_variate"
    assert cell.curriculum is None and cell.ising.sigma == 0.223
    assert cell.train.n_steps == 20_000 and cell.train.lr == 3e-4
    assert cell.ema_decay == 0.9999
    rebuilt = replace(
        cell,
        name=twin.name,
        estimator=twin.estimator,
        curriculum=twin.curriculum,
        ema_decay=twin.ema_decay,
        train=replace(
            cell.train, n_steps=twin.train.n_steps, lr=twin.train.lr
        ),
    )
    assert rebuilt == twin


def test_d144_fmo2_rung_mirrors_d64_fmo2_rung_except_volume_scaled_fields():
    """The 12x12 rung (2026-08-15). 8x8 trains to Var/site 0.0040 and 16x16
    sits at 0.0707; no volume in between has ever been run, so the wall is
    unbracketed. This cell must be the 8x8 factorised rung with ONLY the
    volume-forced deltas: the lattice side, the multi-event step (the
    one-event step clips once Lambda ~ d^2/2 outgrows the budget), the
    clip-safe 2d Euler grid the shared builder's own rule asks for, and the
    eval chunk the factorised head's memory allows. Head, sigma ladder,
    batch, lr, replay depth and seed stay verbatim, so volume is the read."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d144_c50_s223_letf_fmo2_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_fmo2_50k_curr"]
    assert cell.ising.D == 12 and twin.ising.D == 8
    # 2d at 144 sites; the 8x8 rung honours the same rule at 128 = 2*64.
    assert cell.ctmc.n_euler_steps == 288
    assert cell.ctmc.use_matching_step and not twin.ctmc.use_matching_step
    assert cell.eval.eval_sample_chunk == 512
    assert cell.head_kind == "factorised"
    assert cell.site_orderings == ("row", "col")
    assert cell.curriculum == twin.curriculum
    assert cell.train == twin.train
    rebuilt = replace(
        cell,
        name=twin.name,
        ising=replace(cell.ising, D=twin.ising.D),
        ctmc=twin.ctmc,
        eval=twin.eval,
    )
    assert rebuilt == twin


def test_d64_fmo2_h128_mirrors_fmo2_rung_except_hidden_dim():
    """Capacity arm (2026-08-15): hidden_dim 32 -> 128 the ONLY change
    against the 8x8 factorised rung. hidden_dim is set once in the shared
    cell builder and every cell at every volume has used 32, so it has never
    appeared in a cell diff; this pin makes the first variation of it
    single-variable. n_heads and n_layers must NOT move with it — head_dim
    riding 8 -> 32 is the consequence of widening, not a second knob."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d64_c50_s223_letf_fmo2_h128_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_fmo2_50k_curr"]
    assert cell.model.hidden_dim == 128 and twin.model.hidden_dim == 32
    assert cell.model.n_heads == twin.model.n_heads == 4
    assert cell.model.n_layers == twin.model.n_layers == 2
    rebuilt = replace(
        cell,
        name=twin.name,
        model=replace(cell.model, hidden_dim=twin.model.hidden_dim),
    )
    assert rebuilt == twin


def test_d256_fmo2_warm_mirrors_cv2_continuation_shape_except_head():
    """Cross-volume transfer arm (2026-08-15). The archived phase-2 cell
    continued 16x16 from a 16x16 checkpoint, which cannot test transfer at
    all; this one starts from an 8x8 factorised model resampled onto the
    larger torus. It must share the continuation SHAPE with that archived
    cell — no curriculum, flat sigma_c, lr at the ladder's final 3e-4, the
    dual-eval EMA riding — so the schedule is not a second variable, and
    differ in the head and the eval chunk the factorised head's memory
    allows. n_euler stays at the 128 every archived 16x16 cell used even
    though the builder's clip-safe rule asks 2d = 512 here: training on a
    finer grid AND from a transfer would confound them, and the resolution
    axis is read afterwards off the frozen checkpoint instead."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d256_c50_s223_letf_fmo2_20k_sc_warm"]
    twin = CONFIGS["H2_d256_c50_s223_letf_ma_20k_sc_cv2"]
    assert cell.head_kind == "factorised" and cell.site_orderings == ("row", "col")
    assert cell.curriculum is None and cell.ising.sigma == 0.223
    assert cell.train.n_steps == 20_000 and cell.train.lr == 3e-4
    assert cell.ema_decay == 0.9999
    assert cell.ctmc.n_euler_steps == twin.ctmc.n_euler_steps == 128
    assert cell.eval.eval_sample_chunk == 512
    rebuilt = replace(
        cell,
        name=twin.name,
        head_kind=twin.head_kind,
        site_orderings=twin.site_orderings,
        eval=twin.eval,
    )
    assert rebuilt == twin


def test_d256_fmo2_warm_ne512_differs_from_its_twin_in_the_grid_alone():
    """Resolution-at-training arm (2026-08-15). A frozen-checkpoint sweep can
    only ask how an already-trained model behaves when re-rolled on a finer
    grid; it cannot separate "the grid is coarse" from "the model was fitted
    to a coarse grid", since a model trained under a biased discretisation
    learns to compensate that bias. This arm trains at 2d = 512, the
    clip-safe budget this module's builder states, against a 128 twin that
    is what every archived 16x16 cell ran. n_euler_steps must be the ONLY
    difference, so any gain is attributable to the grid rather than to the
    cross-volume transfer the pair shares."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    cell = CONFIGS["H2_d256_c50_s223_letf_fmo2_20k_sc_warm_ne512"]
    twin = CONFIGS["H2_d256_c50_s223_letf_fmo2_20k_sc_warm"]
    assert cell.ctmc.n_euler_steps == 512 == 2 * cell.ising.D**2
    assert twin.ctmc.n_euler_steps == 128
    rebuilt = replace(
        cell, name=twin.name, ctmc=replace(cell.ctmc, n_euler_steps=128)
    )
    assert rebuilt == twin


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


def test_d16_unconstrained_control_mirrors_d8_walkback_except_declared():
    """Unconstrained 16x16 control (2026-08-18): the d8 walkback cell with
    exactly three declared changes — lattice side, the 50k budget, and the
    compressed sigma ladder (the hard chapter's frame: stages every 5k,
    lr 1e-3 -> 3e-4 on reaching 0.205, final 40% at sigma_c). Everything
    else (engine, batch, replay, clip, ne64, CV estimator, the full
    5000-draw eval protocol) is the archived unconstrained recipe, so the
    d256 read is chargeable to size and the cross-family read to
    machinery. The 5000-draw evals size the CARD (dense readout ~9.8 GiB
    scores at d=256 — A100, not L4), deliberately not the recipe."""
    from dataclasses import replace

    base = BASELINE_CONFIGS["stage_4_d8_critical_paper_curriculum"]
    cell = BASELINE_CONFIGS["stage_4_d16_critical_50k_ladder"]
    assert cell.ising.D == 16
    assert cell.train.n_steps == 50_000
    ladder = cell.curriculum.stages
    assert [s.sigma for s in ladder] == [
        0.100, 0.140, 0.170, 0.190, 0.205, 0.215, 0.22305
    ]
    assert [s.start_step for s in ladder] == list(range(0, 35_000, 5_000))
    # lr drops to 3e-4 exactly on reaching sigma=0.205, per the hard recipe.
    assert [s.lr for s in ladder] == [1e-3] * 4 + [3e-4] * 3
    rebuilt = replace(
        cell,
        name=base.name,
        ising=replace(cell.ising, D=8),
        train=replace(cell.train, n_steps=200_000),
        curriculum=base.curriculum,
    )
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


def test_m2_ctema4_gate_mirrors_ma_twin_except_declared_fields():
    """M2 no-regression gate (2026-08-14, plan Task 2): the gate cell must be
    the archived MA 50k curriculum twin with c_t_ema_halflife_cycles
    0.0 -> 4.0 the ONLY declared change — no horizon, ladder, or instrument
    deltas — so the no-regression read against the archived twin's seed
    spread (band >= 0.755 of 0.755-0.781) attributes to the c_t EMA alone.
    The dual-eval EMA instrument is deliberately NOT ridden: the gate is
    judged raw-vs-archived-twin, and a pure twin keeps the read clean."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    cell = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr_ctema4"]
    assert cell.train.c_t_ema_halflife_cycles == 4.0
    assert twin.train.c_t_ema_halflife_cycles == 0.0
    assert cell.ema_decay == twin.ema_decay == 0.0
    rebuilt = replace(
        cell,
        name=twin.name,
        train=replace(
            cell.train,
            c_t_ema_halflife_cycles=twin.train.c_t_ema_halflife_cycles,
        ),
    )
    assert rebuilt == twin


def test_ctv_naive_twin_mirrors_ma_twin_except_the_estimator():
    """c_t transfer-function cell (2026-08-14): the whole M2/M3/cv2 thrust
    reduces c_t NOISE, but the quantity that has to fall ~8x for a usable
    d256 is per-site Var[log w]. Nobody has measured the transfer between
    them — every one of the 21 archived d64 cells runs control_variate, so
    the slope is unmeasured in BOTH directions at every healthy size.

    This cell is the archived MA 50k curriculum twin with the estimator
    control_variate -> naive_mc as the ONLY declared change, so the
    difference in final Var[log w]/site IS the transfer function, measured
    where the control variate is known to work (a healthy run) rather than
    where it inverted (the diverged d256).

    Pre-registered predictions (the naive integrand variance 26.0 is
    already logged as the Var[dt log p tilde] column of the CV runs, so
    these are arithmetic, not guesses; twin ESS is 0.781):
      c_t noise does not drive log-weight variance -> ESS ~ 0.78
      transfer linear in c_t standard error        -> ESS ~ 0.25
      transfer linear in c_t variance              -> ESS ~ 0.001
      training destabilises                        -> the CV is a STABILITY
        crutch, not only a variance reducer, which would mean the archived
        d256 naive rescue ran 50k steps without one.
    """
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    cell = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr_naive"]
    assert cell.estimator == "naive_mc"
    assert twin.estimator == "control_variate"
    # The knobs the campaign added must all be OFF: this cell has to be a
    # pure single-variable read against an archived twin that predates them.
    assert cell.train.c_t_ema_halflife_cycles == 0.0
    assert cell.train.c_t_batch is None
    assert cell.ema_decay == twin.ema_decay == 0.0
    rebuilt = replace(cell, name=twin.name, estimator=twin.estimator)
    assert rebuilt == twin


def test_m3_ctb512_smoke_mirrors_naive_arm_except_declared_fields():
    """M3 mechanism cell (2026-08-14, plan Task 3): the d256 12k smoke must
    be the smoke12k Arm-B naive recipe with c_t_batch=512 the ONLY declared
    change, so the read against the archived naive 50k run's first 12k
    (train-ESS median on the sigma=0.17 rung, var_estimator_integrand)
    attributes to the decoupled c_t rollout batch alone. NEVER bundle with
    M2: attribution stays clean one lever per cell."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d256_smoke12k_naive"]
    cell = CONFIGS["H2_d256_smoke12k_naive_ctb512"]
    assert cell.train.c_t_batch == 512
    assert twin.train.c_t_batch is None
    assert cell.train.c_t_ema_halflife_cycles == 0.0
    rebuilt = replace(
        cell,
        name=twin.name,
        train=replace(cell.train, c_t_batch=twin.train.c_t_batch),
    )
    assert rebuilt == twin


def test_m3_ctb512_d64_smoke_mirrors_ma_recipe_except_declared_fields():
    """M3 d64 plumbing smoke (2026-08-14, plan Task 3 fallback): the d64 MA
    curriculum recipe at the 12k smoke horizon (the forced ladder truncation
    of every smoke arm) with c_t_batch=512 the ONLY mechanism change —
    validates the enlarged-rollout plumbing (buffer prefix, c_t over the
    full set, resume) on a 25-min a30 job before the d256 cell."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    twin = CONFIGS["H2_d64_c50_s223_letf_ma_50k_curr"]
    cell = CONFIGS["H2_d64_smoke12k_ctb512"]
    assert cell.train.c_t_batch == 512
    assert cell.train.n_steps == 12_000
    assert cell.train.c_t_ema_halflife_cycles == 0.0
    rebuilt = replace(
        cell,
        name=twin.name,
        train=replace(
            cell.train, n_steps=twin.train.n_steps, c_t_batch=None
        ),
        curriculum=twin.curriculum,
    )
    assert rebuilt == twin


def test_swap_route_never_reads_the_ising_log_ratio_clamp_field():
    """Dead-field pin (adversarial panel, 2026-08-18). Every hard-route
    config.json carries `ising.log_ratio_clamp` (default 5.0), but the swap
    path clamps at the hardcoded SWAP_LOG_RATIO_CLAMP = 30.0 and the
    fixed-composition target never even receives the field — the config
    value is single-site-route-only and CANNOT govern any hard-chapter run.
    Behavioural proof: perturbing the target's attribute to an absurd value
    must leave the swap residual bit-identical. The value is left unplumbed
    deliberately: making the swap clamp configurable would trip eval_only's
    config-drift guard on every archived run dir (stored 5.0 vs registry),
    a blast radius the never-firing clamp (0 of 141,000 logged rows across
    six d64-d256 runs; physical bound ~7.14 nats at sigma_c) does not earn.
    This pin keeps the trap documented instead."""
    import torch

    from discrete_flow_sampler.constraints.swap_readout import (
        LeTFMaskOneSwapHead,
    )
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.samplers.swap_kolmogorov import residual_swap
    from discrete_flow_sampler.targets.ising import (
        FixedCompositionIsingTarget,
    )

    torch.manual_seed(0)
    head = LeTFMaskOneSwapHead(
        LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2,
                       n_heads=2)
    )
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.223, target_composition=0.5
    )
    x = target.sample_base(8, device="cpu")
    t = torch.full((8,), 0.7)
    reference = residual_swap(x, t, 0.0, head, target)
    target.log_ratio_clamp = 1e-6  # absurd; would zero every ratio if read
    assert torch.equal(residual_swap(x, t, 0.0, head, target), reference)


def test_ne128_cv_family_arms_are_one_variable_twins_of_their_comparators():
    """The ne128 x CV composition family (2026-08-21) exists to compose two
    levers s42 certified as independent — the training grid (keystone,
    GRID-HELPS) and the c_t estimator (cvcont, ESTIMATOR-OWNS) — plus
    capacity twins. Each arm is only interpretable if it differs from its
    comparator by the fields its registry comment declares and no others,
    so every relationship in the family is pinned here by rebuilding the
    comparator from the arm and asserting equality.

    `loss_microbatch_size` is deliberately NOT a declared deviation
    anywhere in the family: the keystone parent already carries it, so 128
    is the lineage setting and it is gradient-exact
    (test_loss_microbatch_parity pins the identity for arbitrary per-row
    c_t, which is why the control variate cannot disturb it — c_t is
    computed in the outer no_grad rollout, not inside the loss)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    keystone = CONFIGS["H2_d256_c50_s223_letf_fmo2_50k_curr_b512_ne128_naive"]
    cvcont = CONFIGS["H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne512"]
    arm_a = CONFIGS["H2_d256_c50_s223_letf_fmo2_20k_sc_cv2_b512_ne128"]
    arm_b = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2"]
    arm_p = CONFIGS["H2_d256_c50_s223_letf_fmo2_h128L3_50k_curr_b512_ne128_naive"]
    arm_p03 = CONFIGS[
        "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_50k_curr_b512_ne128_naive"
    ]
    arm_c = CONFIGS["H2_d256_c50_s223_letf_fmo2_h128L3_20k_sc_cv2_b512_ne128"]
    arm_d = CONFIGS["H2_d256_c50_s223_letf_fmo2_h128L3_70k_curr_b512_ne128_cv2"]
    arm_d_lr03 = CONFIGS[
        "H2_d256_c50_s223_letf_fmo2_h128L3_lr03_70k_curr_b512_ne128_cv2"
    ]

    # Every arm trains at the keystone's grid and batch, under the schedule
    # its parent already used.
    arm_b_ef = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2_ef"]
    # s55: the exact-field twin of B differs by the channel flag alone.
    assert arm_b_ef.exact_field_channel and not arm_b.exact_field_channel
    assert replace(arm_b_ef, name=arm_b.name, exact_field_channel=False) == arm_b
    for arm in (arm_a, arm_b, arm_p, arm_p03, arm_c, arm_d, arm_d_lr03,
                arm_b_ef):
        assert arm.ctmc.n_euler_steps == 128, arm.name
        assert arm.train.batch_size == 512, arm.name
        assert arm.train.loss_microbatch_size == 128, arm.name

    # A vs the landed ne512 CV continuation: the grid is the only lever,
    # with the backward schedule riding for the reason above.
    assert arm_a.estimator == "control_variate"
    assert arm_a.train.halt_on_cv_inversion_after == 2000
    assert replace(
        arm_a,
        name=cvcont.name,
        ctmc=replace(arm_a.ctmc, n_euler_steps=cvcont.ctmc.n_euler_steps),
        train=replace(
            arm_a.train,
            loss_microbatch_size=cvcont.train.loss_microbatch_size,
        ),
    ) == cvcont

    # B vs the keystone: the estimator, the horizon and the cold-start
    # tripwire are the three declared changes. A and B then carry the SAME
    # budget shape — 30k of ladder plus 40k at sigma_c — so the only thing
    # separating them is when the control variate joins.
    assert arm_b.estimator == "control_variate"
    assert arm_b.train.n_steps == 70_000
    assert arm_b.train.halt_on_cv_inversion_after == 5000
    assert arm_b.curriculum == keystone.curriculum
    assert arm_b.curriculum.stages[-1].start_step == 30_000
    assert replace(
        arm_b,
        name=keystone.name,
        estimator=keystone.estimator,
        train=replace(
            arm_b.train,
            n_steps=keystone.train.n_steps,
            halt_on_cv_inversion_after=keystone.train.halt_on_cv_inversion_after,
        ),
    ) == keystone

    # P vs the keystone: capacity only, width and depth bundled as declared.
    assert (arm_p.model.hidden_dim, arm_p.model.n_layers) == (128, 3)
    assert arm_p.model.n_heads == keystone.model.n_heads == 4
    assert arm_p.estimator == "naive_mc"
    assert replace(
        arm_p,
        name=keystone.name,
        model=replace(
            arm_p.model,
            hidden_dim=keystone.model.hidden_dim,
            n_layers=keystone.model.n_layers,
        ),
    ) == keystone

    # P03 vs P: the curriculum lr is the only change, flattened to the
    # ladder's final value at every stage so the h128 lr artefact the 5k
    # screen measured is separable from a capacity verdict.
    assert {stage.lr for stage in arm_p03.curriculum.stages} == {3e-4}
    assert replace(
        arm_p03, name=arm_p.name, curriculum=arm_p.curriculum
    ) == arm_p

    # C vs A and D vs B: capacity only, on each h32 arm respectively.
    for capacity_arm, base_arm in ((arm_c, arm_a), (arm_d, arm_b)):
        assert (capacity_arm.model.hidden_dim, capacity_arm.model.n_layers) == (
            128,
            3,
        ), capacity_arm.name
        assert replace(
            capacity_arm,
            name=base_arm.name,
            model=replace(
                capacity_arm.model,
                hidden_dim=base_arm.model.hidden_dim,
                n_layers=base_arm.model.n_layers,
            ),
        ) == base_arm, capacity_arm.name

    # D_lr03 vs D: the curriculum lr is the only change, the same flat
    # 3e-4 treatment as P03 — owed because P03 read LR-ARTEFACT (s55),
    # voiding D's capacity read at the ladder lr.
    assert {stage.lr for stage in arm_d_lr03.curriculum.stages} == {3e-4}
    assert replace(
        arm_d_lr03, name=arm_d.name, curriculum=arm_d.curriculum
    ) == arm_d


def test_interior_separation_cells_are_single_variable_twins():
    """Exterior-vs-interior separation (2026-08-23): the 4x4 interval cell
    differs from the MA demo cell by head_kind alone, and each d64 band rung
    differs from the fmo2 rung by the declared interior knobs alone."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    for sigma_label in ("s010", "s223"):
        iv = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_iv_10k"]
        ma = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_ma_10k"]
        assert iv.head_kind == "interval"
        assert replace(iv, name=ma.name, head_kind=ma.head_kind) == ma

    fmo2 = CONFIGS["H2_d64_c50_s223_letf_fmo2_50k_curr"]
    for arm, band, orderings in (
        ("fib", "prefix", ("row",)),
        ("fatt", "attention", ("row",)),
        ("fimo2", "prefix", ("row", "col")),
    ):
        cell = CONFIGS[f"H2_d64_c50_s223_letf_{arm}_50k_curr"]
        assert (cell.interior_band, cell.site_orderings) == (band, orderings)
        assert replace(
            cell, name=fmo2.name, interior_band=None, site_orderings=("row", "col")
        ) == fmo2


def test_bilinear_exterior_cells_change_only_the_combiner():
    """Literal factorisation test (2026-08-23): mab / ivb differ from the
    archived MA / interval cells by exterior_combiner alone (plus the EMA
    instrument at d64, which never touches training)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    for sigma_label in ("s010", "s223"):
        for arm, twin_arm in (("mab", "ma"), ("ivb", "iv")):
            cell = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_{arm}_10k"]
            twin = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_{twin_arm}_10k"]
            assert cell.exterior_combiner == "bilinear" and twin.exterior_combiner == "mlp"
            assert replace(cell, name=twin.name, exterior_combiner="mlp") == twin
    for arm, twin_arm in (("mab", "ma"), ("ivb", "iv")):
        cell = CONFIGS[f"H2_d64_c50_s223_letf_{arm}_50k_curr"]
        twin = CONFIGS[f"H2_d64_c50_s223_letf_{twin_arm}_50k_curr"]
        assert replace(cell, name=twin.name, exterior_combiner="mlp", ema_decay=twin.ema_decay) == twin


def test_thp_d256_twins_of_arm_b():
    """s57: the three 16x16 two-hole patch arms differ from arm B by the
    declared head fields alone (head_kind; + exact_field_channel; + R=2)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    arm_b = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2"]
    for name, fields in {
        "H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2": {},
        "H2_d256_c50_s223_letf_thp_70k_curr_b512_ne128_cv2_ef": {"exact_field_channel": False},
        "H2_d256_c50_s223_letf_thp2_70k_curr_b512_ne128_cv2": {"patch_radius": None},
    }.items():
        cell = CONFIGS[name]
        assert cell.head_kind == "two_hole_patch"
        assert replace(cell, name=arm_b.name, head_kind=arm_b.head_kind, **fields) == arm_b
        assert cell.train.loss_microbatch_size == 128 and cell.train.batch_size == 512


def test_wave1_sigma_c_twins_mirror_their_archived_parents():
    """s58 sigma_c migration, Wave 1 (launched s63): each `_sc` cell differs
    from its archived 0.22305 parent by the coupling alone — `ising.sigma`
    and the final curriculum stage move to the exact SIGMA_C, the sigma
    ladder below the endpoint stays verbatim — plus the two declared
    optimised_recipe flags (compile_model, c_t_from_rollout; s60 standing
    rule for every new cell). Pinned so the retrain twins can never drift
    from the cells whose printed rows they replace."""
    from dataclasses import replace

    from discrete_flow_sampler.targets.ising import SIGMA_C

    for parent_name in [
        "stage_4_d4_critical",
        "stage_4_d10_critical_paper_curriculum",
        "stage_4_d8_critical_paper_curriculum",
    ]:
        parent = BASELINE_CONFIGS[parent_name]
        twin = BASELINE_CONFIGS[parent_name + "_sc"]

        assert twin.ising.sigma == SIGMA_C
        assert twin.model.compile_model and twin.train.c_t_from_rollout

        if parent.curriculum is not None:
            *twin_ladder, twin_final = twin.curriculum.stages
            *parent_ladder, parent_final = parent.curriculum.stages
            assert twin_final.sigma == SIGMA_C
            assert tuple(twin_ladder) == tuple(parent_ladder)
            assert (twin_final.start_step, twin_final.lr) == (
                parent_final.start_step,
                parent_final.lr,
            )

        # Everything not declared above is identical to the archived parent.
        assert replace(
            twin,
            name=parent.name,
            ising=replace(twin.ising, sigma=parent.ising.sigma),
            train=replace(twin.train, c_t_from_rollout=False),
            model=replace(twin.model, compile_model=False),
            curriculum=parent.curriculum,
        ) == parent


def test_amort_specialist_twins_mirror_c05_except_composition():
    """4x4 specialist twins for tab:amort-4x4 (s64): the conditioned rows at
    c = 0.30/0.70/0.80 each get a specialist comparator. The recipe is
    byte-identical to the c=0.50 specialist -- target_composition is the
    ONLY change -- and optimised_recipe is deliberately NOT applied: every
    row these compare against is eager, and one recipe per table governs
    over the standing cost rule (decision on record, s64)."""
    from dataclasses import replace

    base = CONSTRAINED_CONFIGS["S2_d4_c05_50k_l50_letf_anneal_offset_clip50"]
    for c_target, tag in ((0.30, "c03"), (0.70, "c07"), (0.80, "c08")):
        name = f"S2_d4_{tag}_50k_l50_letf_anneal_offset_clip50"
        twin = CONSTRAINED_CONFIGS[name]
        assert twin.ising.target_composition == c_target, name
        rebuilt = replace(
            twin, name=base.name,
            ising=replace(twin.ising,
                          target_composition=base.ising.target_composition),
        )
        assert rebuilt == base, name


def test_flat_window_ablation_mirrors_conditioned_cell_except_curriculum():
    """Uniform-from-start ablation of the widening curriculum (s64): the
    printed conditioned cell justified its widening by analogy with the
    sigma/lambda curricula, never by ablation. This twin draws c from the
    FULL final window from step 0 -- half_width 0.30, no curriculum -- and
    everything else is byte-identical, so any difference vs the printed
    conditioned rows is attributable to the widening schedule alone."""
    from dataclasses import replace

    base = CONSTRAINED_CONFIGS["S2_d4_camort_50k_l50_letf_anneal_offset_clip50"]
    twin = CONSTRAINED_CONFIGS[
        "S2_d4_camort_50k_l50_letf_anneal_offset_clip50_flatw30"]
    assert twin.composition.curriculum is None
    assert twin.composition.half_width == 0.30
    rebuilt = replace(
        twin, name=base.name,
        composition=replace(twin.composition,
                            half_width=base.composition.half_width,
                            curriculum=base.composition.curriculum),
    )
    assert rebuilt == base


def test_wave2_house_cells_mirror_archived_twins_except_declared_fields():
    """Wave-2 house-table fill (s68): every cell in the 4x4/8x8 fresh matrix
    must be the declared transform of its archived namesake and NOTHING else
    -- the s220 cells move sigma from the legacy 0.223 to the exact SIGMA_C
    (cell sigma AND curriculum endpoint, ladder reused-not-rescaled), every
    cell takes the s60 optimised recipe (compile_head + c_t_from_rollout),
    and the d64 cells carry the dual-eval EMA instrument. Any other field
    drifting would make the retrained table unattributable to the sigma
    correction."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS
    from discrete_flow_sampler.targets.ising import SIGMA_C, SIGMA_C_LEGACY

    def deoptimised(cell):
        return replace(
            cell,
            compile_head=False,
            train=replace(cell.train, c_t_from_rollout=False),
        )

    # --- d16: archived twins exist at BOTH sigma labels. The fmo2ef arm has
    # NO archived namesake at any size (ef never ran on the global chassis),
    # so it pins against the archived plain-fmo2 parent with the ef flag the
    # one extra declared delta. ------------------------------------------------
    for arm, archived_arm, extra in (
        ("mo", "mo", {}),
        ("ma", "ma", {}),
        ("fimo2ef", "fimo2ef", {}),
        ("fmo2ef", "fmo2", {"exact_field_channel": False}),
        ("thp", "thp", {}),
    ):
        for lbl, archived_lbl in (("s010", "s010"), ("s220", "s223")):
            w2 = CONFIGS[f"H2_d16_c50_{lbl}_letf_{arm}_10k_w2"]
            twin = CONFIGS[f"H2_d16_c50_{archived_lbl}_letf_{archived_arm}_10k"]
            assert w2.compile_head and w2.train.c_t_from_rollout, w2.name
            rebuilt = replace(deoptimised(w2), **extra)
            if lbl == "s220":
                assert w2.ising.sigma == SIGMA_C, w2.name
                assert twin.ising.sigma == 0.223, twin.name
                rebuilt = replace(
                    rebuilt, ising=replace(rebuilt.ising, sigma=twin.ising.sigma)
                )
            rebuilt = replace(rebuilt, name=twin.name)
            assert rebuilt == twin, w2.name

    # --- d64 s220: curriculum cells against the legacy-0.223 namesakes (the
    # fmo2ef arm against its plain-fmo2 parent, as above) --------------------
    for arm, archived, extra in (
        ("mo", "H2_d64_c50_s223_letf_mo_50k_curr", {}),
        ("ma", "H2_d64_c50_s223_letf_ma_50k_curr", {}),
        ("fimo2ef", "H2_d64_c50_s223_letf_fimo2ef_50k_curr", {}),
        ("fmo2ef", "H2_d64_c50_s223_letf_fmo2_50k_curr",
         {"exact_field_channel": False}),
        ("thp", "H2_d64_c50_s223_letf_thp_50k_curr", {}),
    ):
        w2 = CONFIGS[f"H2_d64_c50_s220_letf_{arm}_50k_curr_w2"]
        twin = CONFIGS[archived]
        assert w2.ising.sigma == SIGMA_C, w2.name
        assert w2.ema_decay == 0.9999, w2.name
        # Ladder reused, not rescaled: only the final stage moves to SIGMA_C.
        assert w2.curriculum.stages[:-1] == twin.curriculum.stages[:-1], w2.name
        final_w2, final_twin = w2.curriculum.stages[-1], twin.curriculum.stages[-1]
        assert final_w2.sigma == SIGMA_C and final_twin.sigma == 0.223, w2.name
        assert (final_w2.start_step, final_w2.lr) == (
            final_twin.start_step, final_twin.lr), w2.name
        rebuilt = replace(
            deoptimised(w2),
            name=twin.name,
            ema_decay=twin.ema_decay,
            ising=replace(w2.ising, sigma=twin.ising.sigma),
            curriculum=twin.curriculum,
            **extra,
        )
        assert rebuilt == twin, w2.name

    # --- d64 s010: flat floor cells against the archived MA floor cell ------
    floor_twin = CONFIGS["H2_d64_c50_s010_letf_ma_50k"]
    declared = {
        "mo": {"head_kind": "mask_one"},
        "ma": {"head_kind": "masked_attention"},
        "fimo2ef": {
            "head_kind": "factorised",
            "exact_field_channel": True,
            "interior_band": "prefix",
            "site_orderings": ("row", "col"),
        },
        "fmo2ef": {
            "head_kind": "factorised",
            "exact_field_channel": True,
            "interior_band": None,
            "site_orderings": ("row", "col"),
        },
        "thp": {"head_kind": "two_hole_patch"},
    }
    for arm, fields in declared.items():
        w2 = CONFIGS[f"H2_d64_c50_s010_letf_{arm}_50k_w2"]
        assert w2.curriculum is None, w2.name
        assert w2.ising.sigma == 0.10, w2.name
        assert w2.ema_decay == 0.9999, w2.name
        for field_name, value in fields.items():
            assert getattr(w2, field_name) == value, (w2.name, field_name)
        rebuilt = replace(
            deoptimised(w2),
            name=floor_twin.name,
            ema_decay=floor_twin.ema_decay,
            head_kind=floor_twin.head_kind,
            exact_field_channel=False,
            interior_band=None,
            site_orderings=("row",),
        )
        assert rebuilt == floor_twin, w2.name

    # The sigma shift itself, stated once: -1.2% from the legacy label.
    assert abs(SIGMA_C / SIGMA_C_LEGACY - 1) < 0.013


def test_hold_twins_isolate_sigma_from_recipe_for_fimo2ef():
    """s70 HOLD investigation (2026-08-26): the w2 fimo2ef sigma_c cells
    landed raw ESS 0.637-0.831 against the archived namesake's 0.926-0.938
    with indistinguishable training curves, so the frozen clause held them
    from print. The archived-vs-w2 config diff has exactly two live deltas
    (sigma 0.223 -> SIGMA_C, and the s60 optimised recipe), and each twin
    must walk exactly ONE of them back: `_w2sig` = the full w2 cell at the
    archived legacy sigma 0.223; `_eager` = the w2 cell with only the two
    recipe flags off. Any other field drifting re-confounds the
    sigma-vs-recipe attribution the twins exist to separate."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS
    from discrete_flow_sampler.targets.ising import SIGMA_C

    w2 = CONFIGS["H2_d16_c50_s220_letf_fimo2ef_10k_w2"]

    sigma_twin = CONFIGS["H2_d16_c50_s223_letf_fimo2ef_10k_w2sig"]
    assert sigma_twin.ising.sigma == 0.223
    assert sigma_twin.compile_head and sigma_twin.train.c_t_from_rollout
    rebuilt = replace(
        sigma_twin, name=w2.name,
        ising=replace(sigma_twin.ising, sigma=SIGMA_C),
    )
    assert rebuilt == w2

    eager_twin = CONFIGS["H2_d16_c50_s220_letf_fimo2ef_10k_eager"]
    assert eager_twin.ising.sigma == SIGMA_C
    assert not eager_twin.compile_head
    assert not eager_twin.train.c_t_from_rollout
    rebuilt = replace(
        eager_twin, name=w2.name, compile_head=True,
        train=replace(eager_twin.train, c_t_from_rollout=True),
    )
    assert rebuilt == w2


def test_hold_round2_twins_isolate_compile_and_ef():
    """Round-2 HOLD twins (s70): round 1 localised the fimo2ef sigma_c
    depression to the recipe x exact-criticality x ef corner but not which
    recipe flag carries it, nor whether the ef channel is truly necessary.
    `_cmpl` walks back c_t_from_rollout ALONE (compile kept); `_w2rec` is
    the plain fimo2 chassis with the ef channel ALONE walked back. Any
    other field drifting re-confounds exactly the attribution each twin
    exists to make."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    w2 = CONFIGS["H2_d16_c50_s220_letf_fimo2ef_10k_w2"]

    compile_twin = CONFIGS["H2_d16_c50_s220_letf_fimo2ef_10k_cmpl"]
    assert compile_twin.compile_head
    assert not compile_twin.train.c_t_from_rollout
    rebuilt = replace(
        compile_twin, name=w2.name,
        train=replace(compile_twin.train, c_t_from_rollout=True),
    )
    assert rebuilt == w2

    no_ef_twin = CONFIGS["H2_d16_c50_s220_letf_fimo2_10k_w2rec"]
    assert not no_ef_twin.exact_field_channel
    assert no_ef_twin.compile_head and no_ef_twin.train.c_t_from_rollout
    rebuilt = replace(no_ef_twin, name=w2.name, exact_field_channel=True)
    assert rebuilt == w2


def test_decision_c_cells_are_compile_only_walks_of_the_w2_cells():
    """Decision (c) (s70): factorised arms at exact sigma_c train eager —
    the hold investigation localised a ~40% catastrophic-seed rate to
    factorised x compile_head x SIGMA_C and exonerated everything else.
    Each `_w2e` cell must be its w2 namesake with compile_head=False the
    ONE deviation (c_t_from_rollout stays on — it was exonerated); any
    other drift would make the eager refill unattributable to the compile
    decision."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    for w2e_name in (
        "H2_d16_c50_s220_letf_fimo2ef_10k_w2e",
        "H2_d16_c50_s220_letf_fmo2ef_10k_w2e",
        "H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2e",
        "H2_d64_c50_s220_letf_fmo2ef_50k_curr_w2e",
    ):
        w2e = CONFIGS[w2e_name]
        w2 = CONFIGS[w2e_name.removesuffix("_w2e") + "_w2"]
        assert not w2e.compile_head, w2e_name
        assert w2e.train.c_t_from_rollout, w2e_name
        rebuilt = replace(w2e, name=w2.name, compile_head=True)
        assert rebuilt == w2, w2e_name


def test_wave2_house_cells_build_their_heads():
    """Construction check for the wave-2 house cells: build_swap_head must
    instantiate every arm (the ef cells need the target for the field
    channel's adjacency), so a knob typo fails here and not on the GPU.

    The census is DERIVED from `_WAVE2_ARM_KNOBS` (each arm contributes four
    cells: the d16 gate at both couplings, the d64 sigma_c rung and the d64
    floor) rather than matched on the `_w2` suffix. A suffix match silently
    swept in the RoPE-backbone probes, which share the wave-2 parent and its
    name but are not house cells -- and would in any case be built here on
    the wrong backbone, since this harness hands every cell a plain leTF."""
    from experiments.constrained_hard_03.configs import (
        CONFIGS, _WAVE2_ARM_KNOBS, build_swap_head,
    )
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    wave2 = [
        name for arm in _WAVE2_ARM_KNOBS for name in (
            f"H2_d16_c50_s010_letf_{arm}_10k_w2",
            f"H2_d16_c50_s220_letf_{arm}_10k_w2",
            f"H2_d64_c50_s220_letf_{arm}_50k_curr_w2",
            f"H2_d64_c50_s010_letf_{arm}_50k_w2",
        )
    ]
    assert len(wave2) == 4 * len(_WAVE2_ARM_KNOBS) == 20
    assert all(cfg.model.kind == "letf" for cfg in map(CONFIGS.get, wave2))
    for name in wave2:
        cfg = CONFIGS[name]
        d = cfg.ising.D ** 2
        backbone = LeTFRateMatrix(
            d=d, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
        )
        target = FixedCompositionIsingTarget(
            D=cfg.ising.D, sigma=cfg.ising.sigma, target_composition=0.5
        )
        head = build_swap_head(cfg, backbone, target=target)
        assert head is not None, name


def test_d256_house_cells_are_declared_transforms_of_arm_b():
    """16x16 house-table fill (s71, 2026-08-26): the eight `_w3` cells must
    be _ARM_B -- the d256 lineage chassis -- transformed by exactly the
    fields their registry block declares, and nothing else. Declared, per
    cell: the head knobs; the coupling (SIGMA_C at the cell AND the ladder
    endpoint, ladder reused-not-rescaled, or flat 0.10 with NO curriculum);
    the horizon (100k on the sigma_c arms, all of the extra 30k landing on
    the final plateau because the ladder's start_steps are absolute; 50k at
    the floor); the s60 optimised recipe; loss_microbatch_size, off on the
    thp arms where single-shot is 40% faster and fits at 24.9 GB and kept
    at 128 on the two pair-slab arms; gather_triu_pairs on the two heads
    that read it; the archived MA eval chunk; and (s73)
    halt_on_cv_inversion_after cleared to None. That last one is a DECLARED
    deviation, not drift: the tripwire is _ARM_B's cost-capped negative
    verdict as a cold-CV SCREENING cell, and on a production house cell it is
    a silent truncation -- it stopped the ma and fimo2ef sigma=0.1 arms at
    step 5000 of 50000 on trailing cv_var_ratios of only 1.08-1.76, whose
    evals then read ESS 0.0009 and looked exactly like divergence. Any other
    field drifting would make the row unattributable to the head and the
    coupling."""
    from dataclasses import replace

    from discrete_flow_sampler.targets.ising import SIGMA_C
    from experiments.constrained_hard_03.configs import CONFIGS

    arm_b = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2"]
    declared = {
        "thp": {"head_kind": "two_hole_patch"},
        "thp2": {"head_kind": "two_hole_patch", "patch_radius": 2},
        "fimo2ef": {
            "head_kind": "factorised", "exact_field_channel": True,
            "interior_band": "prefix", "site_orderings": ("row", "col"),
            "gather_triu_pairs": True,
        },
        # site_orderings PINNED to ('row',) 2026-08-28. The cell inherits
        # ('row','col') from the fmo2 parent, and it rode INERTLY while the
        # raster heads ignored the field. They no longer do, so the pin is
        # what keeps these two ARCHIVED cells the single-ordering heads they
        # were trained as. thp/thp2 need no pin: the patch head still does
        # not read the field.
        "ma": {
            "head_kind": "masked_attention", "gather_triu_pairs": True,
            "site_orderings": ("row",),
        },
    }
    microbatch = {"thp": None, "thp2": None, "fimo2ef": 128, "ma": 128}
    eval_chunk = {"ma": 128}
    # Fields the arms set that _ARM_B does not; reset to rebuild the parent.
    reset = {
        "head_kind": arm_b.head_kind, "patch_radius": None,
        "exact_field_channel": False, "interior_band": None,
        "gather_triu_pairs": False, "site_orderings": arm_b.site_orderings,
    }

    cells = [
        (f"H2_d256_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w3", arm,
         "s220") for arm in declared
    ] + [
        (f"H2_d256_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w3", arm, "s010")
        for arm in declared
    ]
    assert len(cells) == 8

    for name, arm, label in cells:
        cell = CONFIGS[name]
        for field, value in declared[arm].items():
            assert getattr(cell, field) == value, (name, field)
        assert cell.train.c_t_from_rollout, name
        # Production cells run to full budget; the screening tripwire is off.
        assert cell.train.halt_on_cv_inversion_after is None, name
        assert cell.train.loss_microbatch_size == microbatch[arm], name
        assert cell.eval.eval_sample_chunk == eval_chunk.get(
            arm, arm_b.eval.eval_sample_chunk), name
        if label == "s220":
            assert cell.ising.sigma == SIGMA_C, name
            assert cell.train.n_steps == 100_000, name
            # Ladder reused, not rescaled: only the final stage's sigma moves.
            assert cell.curriculum.stages[:-1] == arm_b.curriculum.stages[:-1]
            final, b_final = cell.curriculum.stages[-1], arm_b.curriculum.stages[-1]
            assert final.sigma == SIGMA_C and b_final.sigma == 0.223, name
            assert (final.start_step, final.lr) == (
                b_final.start_step, b_final.lr), name
            # Decision (c) is scoped to factorised x exact sigma_c ONLY.
            assert cell.compile_head == (arm != "fimo2ef"), name
        else:
            assert cell.curriculum is None, name
            assert cell.ising.sigma == 0.10, name
            assert cell.train.n_steps == 50_000, name
            assert cell.compile_head, name
        rebuilt = replace(
            cell,
            name=arm_b.name,
            compile_head=False,
            ising=replace(cell.ising, sigma=arm_b.ising.sigma),
            curriculum=arm_b.curriculum,
            train=replace(
                cell.train, n_steps=arm_b.train.n_steps,
                loss_microbatch_size=arm_b.train.loss_microbatch_size,
                c_t_from_rollout=False,
                halt_on_cv_inversion_after=(
                    arm_b.train.halt_on_cv_inversion_after
                ),
            ),
            eval=replace(
                cell.eval, eval_sample_chunk=arm_b.eval.eval_sample_chunk
            ),
            **reset,
        )
        assert rebuilt == arm_b, name


def test_d256_house_twins_isolate_the_radius_and_the_coupling():
    """The two twin relationships the 16x16 fill is read through.

    (1) thp vs thp2 at either coupling differ ONLY in the patch radius, so
    the R=1/R=2 comparison is chargeable to the head's one architectural
    knob. (2) Each floor cell is its sigma_c sibling with the coupling and
    its schedule walked back -- sigma, curriculum, n_steps -- plus, on
    fimo2ef ALONE, the decision-(c) compile deviation, which is scoped to
    exact sigma_c because the compiled factorised cells at the 0.10 floor
    were healthy. Any third difference would confound the floor row with a
    recipe change."""
    from dataclasses import replace

    from discrete_flow_sampler.targets.ising import SIGMA_C
    from experiments.constrained_hard_03.configs import CONFIGS

    for template in (
        "H2_d256_c50_s220_letf_{}_100k_curr_b512_ne128_cv2_w3",
        "H2_d256_c50_s010_letf_{}_50k_b512_ne128_cv2_w3",
    ):
        thp = CONFIGS[template.format("thp")]
        thp2 = CONFIGS[template.format("thp2")]
        assert thp.patch_radius is None and thp2.patch_radius == 2
        assert replace(thp2, name=thp.name, patch_radius=None) == thp

    for arm in ("thp", "thp2", "fimo2ef", "ma"):
        critical = CONFIGS[
            f"H2_d256_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w3"]
        floor = CONFIGS[f"H2_d256_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w3"]
        # The compile deviation is the fimo2ef sigma_c cell's alone.
        assert (floor.compile_head != critical.compile_head) == (
            arm == "fimo2ef"), arm
        rebuilt = replace(
            floor,
            name=critical.name,
            compile_head=critical.compile_head,
            ising=replace(floor.ising, sigma=SIGMA_C),
            curriculum=critical.curriculum,
            train=replace(floor.train, n_steps=100_000),
        )
        assert rebuilt == critical, arm


def test_d256_fimo2ef_head_matches_the_smaller_fimo2ef_cells():
    """The fimo2ef chassis must be the SAME head at 4x4, 8x8 and 16x16, or
    the house table's fimo2ef column is three different architectures. Every
    head-shaping field is pinned across the three sizes; the two legitimate
    size-scoped differences are asserted explicitly rather than allowed to
    pass silently -- gather_triu_pairs is a d256 memory lever (bit-class
    equivalent, ~1e-7 fp32) that archived cells deliberately do not carry,
    and pair_offsets is (1, D), the lattice's own row/column adjacency."""
    from experiments.constrained_hard_03.configs import CONFIGS

    d256 = CONFIGS[
        "H2_d256_c50_s220_letf_fimo2ef_100k_curr_b512_ne128_cv2_w3"]
    shaping = (
        "head_kind", "exact_field_channel", "interior_band", "site_orderings",
        "bilinear_rank", "factor_dim", "global_feature_dim", "use_bilinear",
        "use_global", "band_feature_dim", "attention_dim", "pair_offsets",
        "exterior_combiner", "readout_score_scale",
    )
    for smaller_name in (
        "H2_d16_c50_s220_letf_fimo2ef_10k_w2e",
        "H2_d64_c50_s220_letf_fimo2ef_50k_curr_w2e",
    ):
        smaller = CONFIGS[smaller_name]
        for field in shaping:
            assert getattr(d256, field) == getattr(smaller, field), (
                smaller_name, field)
        assert not smaller.gather_triu_pairs, smaller_name
        # Decision (c) holds at every size for factorised x exact sigma_c.
        assert not smaller.compile_head and not d256.compile_head
    assert d256.gather_triu_pairs
    assert d256.model.hidden_dim == 32 and d256.model.n_layers == 2


def test_d256_house_cells_build_their_heads():
    """Construction check for the nineteen 16x16 house cells -- the eight
    original arms plus the ten raster-ladder cells and the floor `masep`
    anchor (2026-08-30): build_swap_head must instantiate every arm at d=256
    (the ef arms need the target for the field channel's adjacency, and the
    gather and separable flags must survive the constructors), so a knob typo
    fails here and not eighteen hours into a GPU run."""
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget
    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    house = [name for name in CONFIGS if name.endswith("_w3")]
    assert len(house) == 19
    for name in house:
        cfg = CONFIGS[name]
        assert cfg.ising.D == 16, name
        backbone = LeTFRateMatrix(
            d=cfg.ising.D ** 2, vocab_size=2, hidden_dim=16, n_layers=2,
            n_heads=2,
        )
        target = FixedCompositionIsingTarget(
            D=cfg.ising.D, sigma=cfg.ising.sigma, target_composition=0.5
        )
        assert build_swap_head(cfg, backbone, target=target) is not None, name

def test_d256_house_cells_never_carry_the_cold_cv_tripwire():
    """Production house cells must reach their full budget.

    `halt_on_cv_inversion_after` is a designed cost-capped NEGATIVE VERDICT
    for the cold-CV screening arms, where a sustained controlled/naive
    integrand-variance inversion is the answer being bought. The d256 house
    cells inherit their parent `_ARM_B` wholesale (that is what makes a row
    attributable to the head and the coupling), and in s73 the tripwire rode
    across with it and silently truncated the `ma` and `fimo2ef` sigma=0.1
    arms at step 5000 of 50000 on trailing cv_var_ratios of 1.08-1.76. The
    evals that followed read ESS 0.00094 / 0.00087 / 0.00021 and were very
    nearly recorded as divergence. An early-inverted CV is a reason to watch
    a production run, not to kill it -- the healthy thp2 twin at the same
    size and coupling opened at cv_var_ratio 1.86 and reached eval ESS 0.998.
    """
    from experiments.constrained_hard_03.configs import CONFIGS

    house = [c for n, c in CONFIGS.items() if "d256" in n and "_w3" in n]
    assert house, "no d256 w3 house cells found -- has the naming changed?"
    for cell in house:
        assert getattr(cell.train, "halt_on_cv_inversion_after", None) is None, (
            f"{cell.name} carries a cold-CV tripwire; production house cells "
            f"must run to their full n_steps={cell.train.n_steps}"
        )


def test_d400_radius_cells_are_declared_transforms_of_arm_b():
    """20x20 radius probe (2026-08-27): the two `_w4` cells must be _ARM_B --
    the same d256 lineage chassis the 16x16 rung is built on -- transformed
    by exactly the fields their registry block declares, and nothing else.

    WHY THIS TEST AND NOT A FRESH BUILDER. The rung changes one physical
    thing, the lattice, and one architectural thing, the patch radius. If
    any other field drifts, a d400-vs-d256 read stops being chargeable to
    the size and an R=3-vs-R=2 read stops being chargeable to the radius --
    which is the entire question the six jobs are being spent on. Declared,
    per cell: `ising.D = 20` and the flat 0.10 coupling with NO curriculum
    (the floor convention -- a ladder at the easy target would measure the
    curriculum); the head knobs; 50k steps; the s60 optimised recipe; the
    cold-CV tripwire cleared as on every production cell; and
    loss_microbatch_size OFF, which is a MEASUREMENT not a guess -- profiled
    2026-08-27 on a Modal A100-80GB (the same card class as the DoC a100
    partition) at 58.25 GB peak for R=2 and 63.41 GB for R=3 over 512 rows,
    against a d256 R=2 control that reproduced its recorded 24.93 GB
    exactly.

    NETWORK SIZE IS HELD FIXED ON PURPOSE. `model` must be byte-identical to
    the d256 parent: the probe asks what the lattice and the radius do, so
    width and depth are not allowed to move underneath them."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    arm_b = CONFIGS["H2_d256_c50_s223_letf_fmo2_70k_curr_b512_ne128_cv2"]
    declared = {
        "thp2": {"head_kind": "two_hole_patch", "patch_radius": 2},
        "thp3": {"head_kind": "two_hole_patch", "patch_radius": 3},
    }
    reset = {
        "head_kind": arm_b.head_kind, "patch_radius": None,
        "exact_field_channel": False, "interior_band": None,
        "gather_triu_pairs": False,
    }

    for arm, knobs in declared.items():
        name = f"H2_d400_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w4"
        cell = CONFIGS[name]
        for field, value in knobs.items():
            assert getattr(cell, field) == value, (name, field)
        assert cell.ising.D == 20, name
        assert cell.ising.sigma == 0.10, name
        assert cell.curriculum is None, name
        assert cell.train.n_steps == 50_000, name
        assert cell.train.c_t_from_rollout, name
        assert cell.compile_head, name
        assert cell.train.halt_on_cv_inversion_after is None, name
        # Both radii fit an 80 GB card single-shot; see the docstring.
        assert cell.train.loss_microbatch_size is None, name
        # The network is the control variable, not a knob.
        assert cell.model == arm_b.model, name
        rebuilt = replace(
            cell,
            name=arm_b.name,
            compile_head=False,
            ising=replace(cell.ising, D=arm_b.ising.D, sigma=arm_b.ising.sigma),
            curriculum=arm_b.curriculum,
            train=replace(
                cell.train, n_steps=arm_b.train.n_steps,
                loss_microbatch_size=arm_b.train.loss_microbatch_size,
                c_t_from_rollout=False,
                halt_on_cv_inversion_after=(
                    arm_b.train.halt_on_cv_inversion_after
                ),
            ),
            **reset,
        )
        assert rebuilt == arm_b, name


def test_d400_radius_cells_isolate_the_radius():
    """The twin relationship the probe is read through: the two `_w4` cells
    differ in `patch_radius` and in NOTHING else, so an R=3-vs-R=2 gap at
    20x20 is chargeable to the head's one architectural knob -- the same
    reading the 16x16 rung gets from its thp/thp2 pair."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    r2 = CONFIGS["H2_d400_c50_s010_letf_thp2_50k_b512_ne128_cv2_w4"]
    r3 = CONFIGS["H2_d400_c50_s010_letf_thp3_50k_b512_ne128_cv2_w4"]
    assert (r2.patch_radius, r3.patch_radius) == (2, 3)
    assert replace(r3, name=r2.name, patch_radius=2) == r2


def test_d400_critical_cells_are_their_floor_siblings_at_sigma_c():
    """20x20 sigma_c rung (2026-08-29): each critical cell must be its OWN
    sigma = 0.10 sibling transformed by exactly three declared fields --
    the coupling, the ladder and the horizon -- and nothing else.

    WHY THIS IS THE RIGHT PARENT. The floor cell already carries every d400
    decision that was argued and measured: the lattice, the head knobs, the
    held-fixed backbone, batch 512, n_euler 128, the single-shot backward
    (58.25 / 63.41 GB peak, profiled) and the cleared cold-CV tripwire.
    Rebuilding from _ARM_B instead would re-open all of them. Chaining off
    the floor cell means a critical-vs-floor read at d400 is chargeable to
    the COUPLING, which is the entire question the rung is being spent on.

    THE THREE DEVIATIONS ARE THE d256 CRITICAL CONVENTION, not new choices:
    exact SIGMA_C, `_D64_SIGMA_LADDER_SC`, and 100k steps -- the same triple
    `_d256_house_critical_cell` applies one rung down."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import (
        CONFIGS, SIGMA_C, _D64_SIGMA_LADDER_SC,
    )

    for arm in ("thp2", "thp3"):
        floor = CONFIGS[f"H2_d400_c50_s010_letf_{arm}_50k_b512_ne128_cv2_w4"]
        crit = CONFIGS[
            f"H2_d400_c50_s220_letf_{arm}_100k_curr_b512_ne128_cv2_w4"
        ]
        assert crit.ising.sigma == SIGMA_C, arm
        assert crit.curriculum is _D64_SIGMA_LADDER_SC, arm
        assert crit.train.n_steps == 100_000, arm
        # Everything the floor cell settled must ride unchanged.
        assert crit.ising.D == 20, arm
        assert crit.patch_radius == floor.patch_radius, arm
        assert crit.model == floor.model, arm
        assert crit.train.loss_microbatch_size is None, arm
        assert crit.train.halt_on_cv_inversion_after is None, arm
        assert crit.compile_head, arm
        assert crit.train.c_t_from_rollout, arm
        rebuilt = replace(
            crit,
            name=floor.name,
            ising=replace(crit.ising, sigma=floor.ising.sigma),
            curriculum=floor.curriculum,
            train=replace(crit.train, n_steps=floor.train.n_steps),
        )
        assert rebuilt == floor, arm


def test_d400_critical_ladder_is_reused_not_rescaled():
    """The ladder must be the SHARED d64 object with its endpoint at
    SIGMA_C, not a d400 copy.

    WHY IT MATTERS THAT IT IS NOT RESCALED. `start_step` is ABSOLUTE, so
    lengthening the cell from 50k to 100k does not stretch the schedule: the
    boundaries stay at 0/5k/10k/15k/20k/25k/30k and the extra 50k lands
    entirely on the final sigma_c plateau. lr is the stage value times a
    fixed-step warmup ramp and is never normalised by n_steps, so the first
    50k steps of a 100k cell are schedule-identical to a 50k cell's.

    WHY IT IS LATTICE-INDEPENDENT AND MAY CROSS RUNGS AT ALL. The ladder
    varies only `sigma` and `lr`; no stage field mentions D or d. That is
    what licenses a d64-authored curriculum on a 20x20 cell, and it is
    asserted rather than assumed because a lattice-dependent stage appearing
    later would silently make the d256 and d400 critical cells
    incomparable."""
    from experiments.constrained_hard_03.configs import (
        CONFIGS, SIGMA_C, _D64_SIGMA_LADDER, _D64_SIGMA_LADDER_SC,
    )

    stages = _D64_SIGMA_LADDER_SC.stages
    assert [s.start_step for s in stages] == [
        s.start_step for s in _D64_SIGMA_LADDER.stages
    ]
    assert [s.start_step for s in stages] == [
        0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000
    ]
    assert stages[-1].sigma == SIGMA_C
    # No stage carries a lattice-dependent field.
    for stage in stages:
        assert not any(
            f in vars(stage) for f in ("D", "d", "lattice_side")
        ), stage

    # The d256 and d400 critical cells share the object, so the schedule
    # cannot drift between the two rungs.
    d256 = CONFIGS["H2_d256_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w3"]
    d400 = CONFIGS["H2_d400_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w4"]
    assert d256.curriculum is d400.curriculum


def test_d400_critical_cells_isolate_the_radius():
    """The R=3-vs-R=2 read at sigma_c, mirroring the floor rung's own twin
    test. The radius was a NULL at the floor -- tied on ESS, at floor on
    every error column, +16% FLOP/es for nothing -- but that null was
    measured where every cell sat on the sampling ceiling, so it licenses
    nothing about sigma_c. Same lesson as the saturated 4x4 gate and the
    `mal` window arm, whose 4x4 separation reversed one rung up."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS

    r2 = CONFIGS["H2_d400_c50_s220_letf_thp2_100k_curr_b512_ne128_cv2_w4"]
    r3 = CONFIGS["H2_d400_c50_s220_letf_thp3_100k_curr_b512_ne128_cv2_w4"]
    assert (r2.patch_radius, r3.patch_radius) == (2, 3)
    assert replace(r3, name=r2.name, patch_radius=2) == r2


def test_d400_radius_cells_build_their_heads():
    """Construction check at the real lattice: build_swap_head must
    instantiate both radii at d=400, so an illegal window (the head requires
    2R+1 <= D, which is 7 <= 20 here) or a knob typo fails in two seconds
    rather than after an a100 queue wait. Also pins the pooled-level count:
    the patch head derives its radii as powers of two whose box fits the
    torus, so D=20 earns a fourth level (1, 2, 4, 8) that D=16 does not --
    free extra context that comes with the rung and is NOT a declared knob.
    """
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget
    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    cells = [name for name in CONFIGS if name.endswith("_w4")]
    # 2 floor arms + 2 sigma_c arms (2026-08-29).
    assert len(cells) == 4
    for name in cells:
        cfg = CONFIGS[name]
        backbone = LeTFRateMatrix(
            d=cfg.ising.D ** 2, vocab_size=2, hidden_dim=16, n_layers=2,
            n_heads=2,
        )
        target = FixedCompositionIsingTarget(
            D=cfg.ising.D, sigma=cfg.ising.sigma, target_composition=0.5
        )
        head = build_swap_head(cfg, backbone, target=target)
        assert head is not None, name
        assert head.pooling_radii == (1, 2, 4, 8), (name, head.pooling_radii)


def test_every_new_probe_cell_rides_the_optimised_recipe():
    """Standing rule: every NEW cell goes out on the s60 optimised recipe --
    `compile_head=True` and `train.c_t_from_rollout=True`. Archived cells and
    their eager twins are never retro-flipped, and the ONE exception is
    decision (c), factorised arms at the EXACT critical coupling, which train
    eager because factorised x compile x sigma_c produced catastrophic seeds
    at ~40% (5/12 against 0/21 elsewhere, Fisher p=0.0033).

    None of these cells is factorised -- `mal` is masked attention, `thp2`
    and `thp3` are patch heads -- so the exception does not reach them and
    every one must be compiled. This is a cheap pin on a rule that is easy to
    lose when a cell is built by `replace`-ing a parent rather than by
    calling `optimised_recipe` directly."""
    from experiments.constrained_hard_03.configs import CONFIGS

    probes = [n for n in CONFIGS
              if n.endswith("_win") or n.endswith("_rel") or "_w4" in n]
    # 4 `mal` window twins + 4 `mar` relative-position twins (4x4 and 8x8,
    # both couplings each) + 8 d400 radius x precision arms (4 at the 0.10
    # floor, 4 at sigma_c). Update deliberately when a probe is added, so a
    # cell cannot join the set without someone reading this rule.
    assert len(probes) == 16, sorted(probes)
    for name in probes:
        cell = CONFIGS[name]
        assert cell.head_kind != "factorised", name
        assert cell.compile_head, name
        assert cell.train.c_t_from_rollout, name


def test_arm_b_cells_are_single_variable_and_the_control_is_matched():
    """Arm B (2026-08-28), pinned before any GPU spend.

    Each cell must be its `fimo2ef` parent transformed by EXACTLY the fields
    its registry block declares -- bonds, bonds-minus-the-second-ordering, or
    a widened global term -- and B0 must actually match B1's parameter count.
    The matched control exists because this campaign has attributed a
    capacity effect to form before (`fab16` at double the rank came in worse
    at 0.5615), and a control that does not match is not a control.

    Also pins decision (c), which is easy to lose when a cell is built by
    `replace`-ing a parent: the exception is d16-SPECIFIC and this gate is
    d16, so the sigma_c arms must train EAGER (compile x factorised x sigma_c
    gave catastrophic seeds at ~40% there, 5/12 against 0/21) while the floor
    arms keep the full optimised recipe. Getting this wrong would not crash
    -- it would spend nine runs and read as a null."""
    from dataclasses import replace

    import torch

    from experiments.constrained_hard_03.configs import (
        CONFIGS, _ARM_B_ARMS, _ARM_B_PARENTS,
    )
    from experiments.constrained_hard_03.run import build_target_and_head

    assert len(_ARM_B_ARMS) == 3, "B1, B2 and the matched control"
    counts = {}
    for sigma_label, parent_name in _ARM_B_PARENTS.items():
        parent = CONFIGS[parent_name]
        assert parent.interior_band is not None, "bonds share the band's modules"
        assert not parent.global_bond_features, parent_name
        # decision (c): eager at exact criticality, full recipe at the floor
        assert parent.compile_head == (sigma_label != "s220"), parent_name
        assert parent.train.c_t_from_rollout, "c_t reuse was exonerated"
        for arm, knobs in _ARM_B_ARMS.items():
            cell = CONFIGS[f"H2_d16_c50_{sigma_label}_letf_{arm}"]
            # single variable: undo the declared knobs and the parent returns
            undone = {field: getattr(parent, field) for field in knobs}
            assert replace(cell, name=parent.name, **undone) == parent, arm
            _, head = build_target_and_head(cell, torch.device("cpu"))
            counts[(sigma_label, arm)] = sum(p.numel() for p in head.parameters())
        bonds = counts[(sigma_label, "fimo2efb_10k_bond")]
        widened = counts[(sigma_label, "fimo2efw_10k_wide")]
        assert abs(bonds - widened) / bonds < 0.001, (bonds, widened)
        # B2 drops an ordering, so it must be strictly CHEAPER than B1 --
        # that is the cost story the cell exists to tell.
        assert counts[(sigma_label, "fiefb_10k_bond1o")] < bonds


def test_arm_b_d64_triangle_isolates_bonds_from_the_ordering():
    """The 8x8 rung (2026-08-28): three arms that differ by ONE field each
    along the chain baseline -> B3 -> B2, so a lift can be attributed.

    baseline (2 orderings, no bonds) -> B3 (1 ordering, no bonds) prices the
    second causal ordering; B3 -> B2 (1 ordering, + bonds) prices the bond
    family against it. B2 alone against the baseline cannot separate them,
    which is exactly the gap the 4x4 gate left.

    Also pins the recipe: decision (c) is d16-SPECIFIC, so unlike the 4x4
    cells these train COMPILED (at d64 compiled is marginally higher with
    zero catastrophic seeds, and eager costs ~50% more per step)."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import (
        CONFIGS, _ARM_B_D64_ARMS, _ARM_B_D64_PARENT,
    )

    parent = CONFIGS[_ARM_B_D64_PARENT]
    assert parent.site_orderings == ("row", "col") and not parent.global_bond_features
    assert parent.compile_head, "decision (c) does not reach d64"

    b2 = CONFIGS["H2_d64_c50_s220_letf_fiefb_50k_curr_bond1o"]
    b3 = CONFIGS["H2_d64_c50_s220_letf_fief_50k_curr_1o"]
    b1 = CONFIGS["H2_d64_c50_s220_letf_fimo2efb_50k_curr_bond"]
    assert len(_ARM_B_D64_ARMS) == 3, "B1, B2, B3; baseline is an existing cfg"
    # B1 keeps the parent's two orderings and adds bonds -- the uncontaminated
    # "do bonds help" question the B2/B3 pair cannot ask.
    assert b1.site_orderings == parent.site_orderings and b1.global_bond_features
    for cell, knobs in ((b1, _ARM_B_D64_ARMS["fimo2efb_50k_curr_bond"]),
                        (b2, _ARM_B_D64_ARMS["fiefb_50k_curr_bond1o"]),
                        (b3, _ARM_B_D64_ARMS["fief_50k_curr_1o"])):
        undone = {field: getattr(parent, field) for field in knobs}
        assert replace(cell, name=parent.name, **undone) == parent, cell.name
        assert cell.compile_head and cell.train.c_t_from_rollout, cell.name
    # B2 and B3 differ in the BOND FLAG ALONE -- the comparison the rung exists
    # for, and the one that stays valid whatever card they run on.
    assert replace(b2, name=b3.name, global_bond_features=False) == b3
    assert b2.site_orderings == b3.site_orderings == ("row",)


def test_raster_ladder_roster_covers_both_rungs():
    """The sweep-ladder arms exist at every rung, and each is its `ma` sibling
    with ONLY the declared knobs moved.

    Two things this pins. First, the chain is only readable one field at a
    time if every ladder cell differs from its parent in exactly the roster's
    knobs and nothing else -- an extra field silently inherited from a
    different parent would make `ma -> mamo2` price two changes and the
    printed anchors invalid. Second, the 8x8 critical cells are ALREADY RUN
    (tag 20260828-rasterord-d64, seeds 42/43/44, printed in
    tab:eval-hard-8x8), so the roster restructure that added the 4x4 rung must
    leave their configs untouched; deriving both rungs from one roster is what
    makes that checkable rather than hoped for.

    The heads are built, not just constructed as configs, because the `ef`
    arms need the target for the exact-field channel's adjacency -- a knob
    typo there fails here and not after a GPU launch.
    """
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import (
        CONFIGS, _RASTER_LADDER_ARMS, _RASTER_LADDER_PARENTS,
        _RASTER_LADDER_RUNG_KNOBS, build_swap_head,
    )
    from discrete_flow_sampler.models.letf import LeTFRateMatrix
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    assert set(_RASTER_LADDER_ARMS) == {
        "mamo2", "mamo2ef", "iv", "ivmo2", "ivmo2ef"}
    assert len(_RASTER_LADDER_PARENTS) == 6

    for arm, knobs in _RASTER_LADDER_ARMS.items():
        for pattern, parent_name in _RASTER_LADDER_PARENTS.items():
            name = pattern.format(arm=arm)
            cfg = CONFIGS[name]
            parent = CONFIGS[parent_name]
            # Rung knobs ride on top of the arm's, and only where the band
            # can read them: `separable_band_scores` is attention-only (the
            # prefix-sum arms have no score tensor and must NOT pick it up),
            # while `gather_triu_pairs` reaches both bands -- the interval
            # head assembles the same symmetric pair slab.
            rung = _RASTER_LADDER_RUNG_KNOBS.get(pattern, {})
            if knobs.get("head_kind", parent.head_kind) != "masked_attention":
                rung = {k: v for k, v in rung.items()
                        if k != "separable_band_scores"}
            # Only the roster's knobs may differ from the `ma` parent.
            assert cfg == replace(parent, name=name, **knobs, **rung), name

            d = cfg.ising.D ** 2
            backbone = LeTFRateMatrix(
                d=d, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
            )
            target = FixedCompositionIsingTarget(
                D=cfg.ising.D, sigma=cfg.ising.sigma, target_composition=0.5
            )
            assert build_swap_head(cfg, backbone, target=target) is not None, name
