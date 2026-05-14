from experiments.dnfs_baseline_01.configs import CONFIGS as BASELINE_CONFIGS
from experiments.constrained_soft_02.configs import CONFIGS as CONSTRAINED_CONFIGS


def test_constrained_configs_use_constraints_wandb_project():
    cfg = CONSTRAINED_CONFIGS["S2_d4_c03_l50"]

    assert cfg.wandb_project == "dnfs-constraints"
    assert cfg.ising.target_composition == 0.3
    assert cfg.ising.composition_penalty_strength == 50.0


def test_baseline_configs_keep_baseline_wandb_project():
    assert BASELINE_CONFIGS["stage_2_d4"].wandb_project == "dnfs-baseline"
