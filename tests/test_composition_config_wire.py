"""Config wiring for the amortised cells.

Pins the things that silently produce a wrong-but-running job: a cell that
asks for amortisation while building an unconditioned model (the composition
would be rejected at train time, but only after the job is queued), curriculum
stages that do not land on outer-cycle boundaries, and a control cell whose
value set has drifted away from the specialists it is meant to be compared to.
"""
from dataclasses import replace

import pytest
from experiments.constrained_soft_02.configs import CONFIGS
from experiments.dnfs_baseline_01.run import _build_model

from discrete_flow_sampler.targets.ising import IsingTarget

# The D=4 cell is the cheap end-to-end validation of the same machinery; the
# D=10 pair is the headline experiment and carries the surviving recipe.
VALIDATION_CELL = "S2_d4_camort_l50_letf"
NARROW_WINDOW_CELL = "S2_d4_camort_w15_l50_letf"
NULL_CONTROL_CELL = "S2_d4_cnull_l50_letf"
BUDGET_TWIN_CELL = "S2_d4_camort_50k_l50_letf"
ANNEALED_TWIN_CELL = "S2_d4_camort_50k_l50_letf_anneal"
OFFSET_ANNEAL_CELL = "S2_d4_camort_50k_l50_letf_anneal_offset"
D10_AMORTISED_CELLS = (
    "S2_d10_camort_l50_letf_ne128_anneal",
    "S2_d10_cgrid_l50_letf_ne128_anneal",
)
AMORTISED_CELLS = (
    VALIDATION_CELL, NARROW_WINDOW_CELL, NULL_CONTROL_CELL, BUDGET_TWIN_CELL,
    ANNEALED_TWIN_CELL, OFFSET_ANNEAL_CELL, *D10_AMORTISED_CELLS,
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


def test_validation_cell_differs_from_its_comparator_only_by_amortisation():
    """The D=4 cell is a controlled clone of the archived c=0.5 specialist.

    Strip the two amortisation knobs and the name, and what is left must be
    the specialist config byte-for-byte — otherwise a gap in the comparison
    could be some unnoticed hyperparameter drift rather than the cost of
    serving a whole range of compositions with one model.
    """
    validation = CONFIGS[VALIDATION_CELL]
    specialist = CONFIGS["S2_d4_c05_l50_letf"]

    stripped = replace(
        validation,
        name=specialist.name,
        model=replace(validation.model, condition_on_composition=False),
        composition=None,
    )
    assert stripped == specialist
    # And the widened window must reach the compositions it will be judged at:
    # D=4 has archived specialists at 0.30 and 0.50 only.
    final_half_width = validation.composition.curriculum[-1].half_width
    assert validation.composition.centre - final_half_width <= 0.30


def test_narrow_window_cell_differs_only_in_the_window():
    """The coverage-cost probe must vary one thing.

    Its whole purpose is to attribute a change in conditioning fidelity to the
    width of the training range, so anything else differing from the wide cell
    — steps, Euler budget, capacity — would make the comparison worthless.
    """
    wide = CONFIGS[VALIDATION_CELL]
    narrow = CONFIGS[NARROW_WINDOW_CELL]

    assert replace(narrow, name=wide.name, composition=wide.composition) == wide
    # Same schedule shape and boundaries, half the final width.
    assert [s.start_step for s in narrow.composition.curriculum] == [
        s.start_step for s in wide.composition.curriculum
    ]
    assert narrow.composition.curriculum[-1].half_width == pytest.approx(0.15)
    assert wide.composition.curriculum[-1].half_width == pytest.approx(0.30)
    # 0.30 and 0.80 fall outside the narrow training range on purpose: those
    # sweep rows measure extrapolation, and reading them as interpolation
    # would credit the model with coverage it never trained on.
    narrow_edge = narrow.composition.centre - narrow.composition.curriculum[-1].half_width
    assert narrow_edge > 0.30


def test_budget_twin_varies_only_the_training_budget():
    """5x the steps, nothing else — so a slope change is attributable.

    The curriculum boundaries move with the run length so the model spends the
    same fraction of training at each window width; that keeps the schedule
    the same experiment rather than a second variable.
    """
    wide = CONFIGS[VALIDATION_CELL]
    twin = CONFIGS[BUDGET_TWIN_CELL]

    assert twin.train.n_steps == 5 * wide.train.n_steps
    assert replace(
        twin,
        name=wide.name,
        train=replace(twin.train, n_steps=wide.train.n_steps),
        composition=wide.composition,
    ) == wide
    # Same widths, same fractions of the run.
    assert [s.half_width for s in twin.composition.curriculum] == [
        s.half_width for s in wide.composition.curriculum
    ]
    assert [s.start_step / twin.train.n_steps
            for s in twin.composition.curriculum] == [
        s.start_step / wide.train.n_steps
        for s in wide.composition.curriculum
    ]


def test_annealed_twin_varies_only_the_lambda_schedule():
    """The budget twin plus the surviving lambda anneal, nothing else.

    Fixed lambda=50 over 50k steps reproduced the from-scratch fragility
    (seed 42's Z2-breaking collapse, 2026-08-01), so this cell tests whether
    the anneal restores seed survival while keeping the budget-bought
    obedience. The final stage must land on the operating point lambda=50,
    or the cell samples a different soft target than every comparator.
    """
    twin = CONFIGS[BUDGET_TWIN_CELL]
    annealed = CONFIGS[ANNEALED_TWIN_CELL]

    assert replace(annealed, name=twin.name, lambda_curriculum=None) == twin
    stages = annealed.lambda_curriculum.stages
    assert [s.composition_penalty_strength for s in stages] == [10.0, 25.0, 50.0]
    assert (
        stages[-1].composition_penalty_strength
        == annealed.ising.composition_penalty_strength
    )
    # The lambda schedule steps on the same boundaries as the window widening,
    # so easy-to-hard moves together on both axes (the D=10 recipe's coupling).
    assert [s.start_step for s in stages] == [
        s.start_step for s in annealed.composition.curriculum
    ]


def test_offset_anneal_finishes_its_ramp_before_the_window_widens():
    """Same lambda ramp as the annealed twin, moved off the window boundaries.

    The annealed twin's training ESS collapses at exactly the steps where
    lambda rises, and those are also the steps where the draw window widens and
    the replay buffer is cleared -- three simultaneous shocks. This cell keeps
    the ramp identical in values but lands it entirely before the first
    widening, so a surviving run attributes the twin's deaths to the pile-up
    rather than to a lambda step as such. Nothing else may differ, or the
    attribution is lost.
    """
    annealed = CONFIGS[ANNEALED_TWIN_CELL]
    offset = CONFIGS[OFFSET_ANNEAL_CELL]

    assert (
        replace(offset, name=annealed.name, lambda_curriculum=None)
        == replace(annealed, lambda_curriculum=None)
    )
    stages = offset.lambda_curriculum.stages
    assert [s.composition_penalty_strength for s in stages] == [
        s.composition_penalty_strength for s in annealed.lambda_curriculum.stages
    ]
    # The whole point: the ramp is done before the window first widens, and it
    # still lands on the operating point every comparator samples.
    first_widening = offset.composition.curriculum[1].start_step
    assert max(s.start_step for s in stages) < first_widening
    assert (
        stages[-1].composition_penalty_strength
        == offset.ising.composition_penalty_strength
    )


def test_null_control_is_the_specialist_reached_through_the_amortised_path():
    """A zero-width window makes the amortised cell a specialist in disguise.

    Every c drawn is exactly the centre, so the target is the archived
    c=0.5 specialist's target — but it is reached through the adapter, the
    per-cycle draw, the per-state composition buffer, the buffered c_t
    baseline and the target binding. That makes it the one comparison where a
    discrepancy can only be the machinery, since the physics is held fixed.
    """
    null = CONFIGS[NULL_CONTROL_CELL]
    specialist = CONFIGS["S2_d4_c05_l50_letf"]

    assert null.composition.half_width == 0.0
    assert null.composition.curriculum is None
    assert null.composition.values is None
    assert null.composition.centre == specialist.ising.target_composition

    # Identical to the specialist once the amortisation knobs are stripped, so
    # a gap cannot be blamed on some other hyperparameter.
    stripped = replace(
        null,
        name=specialist.name,
        model=replace(null.model, condition_on_composition=False),
        composition=None,
    )
    assert stripped == specialist


@pytest.mark.parametrize("cell_name", D10_AMORTISED_CELLS)
def test_amortised_cell_inherits_the_surviving_recipe(cell_name):
    """λ anneal and ne128 are what made this leg train on 4/4 seeds.

    D=4 is deliberately excluded: its archived λ=50 comparator trained 4/4
    without the anneal, so the validation cell matches that recipe instead.
    """
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
