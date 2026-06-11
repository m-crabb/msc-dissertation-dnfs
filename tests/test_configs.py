from experiments.constrained_soft_02.configs import CONFIGS as CONSTRAINED_CONFIGS
from experiments.dnfs_baseline_01.configs import CONFIGS as BASELINE_CONFIGS


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
