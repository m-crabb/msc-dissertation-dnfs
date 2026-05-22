from experiments.dnfs_baseline_01.configs import CONFIGS as BASELINE_CONFIGS
from experiments.constrained_soft_02.configs import CONFIGS as CONSTRAINED_CONFIGS


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
    assert twin.ising.composition_penalty_strength == base.ising.composition_penalty_strength
    assert twin.model == base.model
    assert twin.train == base.train
    assert twin.ctmc == base.ctmc
    assert twin.eval == base.eval
    assert twin.estimator == base.estimator
    assert twin.wandb_project == base.wandb_project


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
