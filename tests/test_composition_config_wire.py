"""Config wiring for the amortised cells.

Pins the things that silently produce a wrong-but-running job: a cell that
asks for amortisation while building an unconditioned model (the composition
would be rejected at train time, but only after the job is queued), curriculum
stages that do not land on outer-cycle boundaries, and a control cell whose
value set has drifted away from the specialists it is meant to be compared to.
"""
import pytest
from experiments.constrained_soft_02.configs import CONFIGS
from experiments.dnfs_baseline_01.run import _build_model

from discrete_flow_sampler.targets.ising import IsingTarget

AMORTISED_CELLS = (
    "S2_d10_camort_l50_letf_ne128_anneal",
    "S2_d10_cgrid_l50_letf_ne128_anneal",
)

# The compositions with archived per-composition specialists; the grid cell
# exists to amortise over exactly these, so drift here breaks the comparison.
SPECIALIST_COMPOSITIONS = (0.30, 0.50, 0.55, 0.60, 0.65, 0.80)


@pytest.mark.parametrize("cell_name", AMORTISED_CELLS)
def test_amortised_cell_builds_a_conditioned_model(cell_name):
    cfg = CONFIGS[cell_name]
    assert cfg.composition is not None
    assert cfg.model.condition_on_composition is True

    target = IsingTarget(
        D=cfg.ising.D,
        sigma=cfg.ising.sigma,
        target_composition=cfg.ising.target_composition,
        composition_penalty_strength=cfg.ising.composition_penalty_strength,
    )
    model = _build_model(cfg, target)
    assert model.condition_on_composition is True
    assert hasattr(model, "comp_embedder")


@pytest.mark.parametrize("cell_name", AMORTISED_CELLS)
def test_amortised_cell_inherits_the_surviving_recipe(cell_name):
    """λ anneal and ne128 are what made this leg train on 4/4 seeds."""
    cfg = CONFIGS[cell_name]
    assert cfg.ctmc.n_euler_steps == 128
    assert cfg.lambda_curriculum is not None
    strengths = [
        stage.composition_penalty_strength
        for stage in cfg.lambda_curriculum.stages
    ]
    assert strengths == [10.0, 25.0, 50.0]


def test_composition_curriculum_widens_and_aligns_to_outer_cycles():
    cfg = CONFIGS["S2_d10_camort_l50_letf_ne128_anneal"]
    stages = cfg.composition.curriculum
    assert stages is not None
    widths = [stage.half_width for stage in stages]
    assert widths == sorted(widths) and widths[0] < widths[-1]
    # Final window must cover every specialist composition, or the amortised
    # model is never trained where its comparators live.
    assert cfg.composition.centre - widths[-1] <= min(SPECIALIST_COMPOSITIONS)
    assert cfg.composition.centre + widths[-1] >= max(SPECIALIST_COMPOSITIONS)

    for stage in stages:
        assert stage.start_step % cfg.train.inner_steps_per_outer == 0
        assert stage.start_step < cfg.train.n_steps


def test_grid_control_draws_exactly_the_specialist_compositions():
    cfg = CONFIGS["S2_d10_cgrid_l50_letf_ne128_anneal"]
    assert cfg.composition.values == SPECIALIST_COMPOSITIONS
    assert cfg.composition.curriculum is None


def test_specialist_cells_are_untouched():
    """Every archived cell must still be unconditioned and unamortised."""
    for name, cfg in CONFIGS.items():
        if name in AMORTISED_CELLS:
            continue
        assert cfg.composition is None, name
        assert cfg.model.condition_on_composition is False, name
