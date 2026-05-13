from experiments.dnfs_baseline_02.configs import CONFIGS


def test_constrained_configs_use_constraints_wandb_project():
    cfg = CONFIGS["constrained_stage_2_d4_c03_l50"]

    assert cfg.wandb_project == "dnfs-constraints"
    assert cfg.ising.target_composition == 0.3
    assert cfg.ising.composition_penalty_strength == 50.0


def test_baseline_configs_keep_baseline_wandb_project():
    assert CONFIGS["stage_2_d4"].wandb_project == "dnfs-baseline"
