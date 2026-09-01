"""Config wiring for the amortised cells.

Pins the things that silently produce a wrong-but-running job: a cell that
asks for amortisation while building an unconditioned model (the composition
would be rejected at train time, but only after the job is queued), curriculum
stages that do not land on outer-cycle boundaries, and a control cell whose
value set has drifted away from the specialists it is meant to be compared to.
"""
from dataclasses import replace

import math

import pytest
import torch
from experiments.constrained_soft_02.configs import CONFIGS
from experiments.dnfs_baseline_01.run import _build_model

from discrete_flow_sampler.models.composition_conditioned import (
    CompositionConditioned,
)
from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
from discrete_flow_sampler.targets.ising import IsingTarget

# The D=4 cell is the cheap end-to-end validation of the same machinery; the
# D=10 pair is the headline experiment and carries the surviving recipe.
VALIDATION_CELL = "S2_d4_camort_l50_letf"
NARROW_WINDOW_CELL = "S2_d4_camort_w15_l50_letf"
NULL_CONTROL_CELL = "S2_d4_cnull_l50_letf"
# The re-priced machinery-cost pair: null control and specialist ceiling both
# at the final recipe (50k steps, clip 50, offset lambda anneal). They must
# stay identical in every field except the conditioning path, or the
# specialist-minus-null difference stops measuring the machinery.
FINAL_RECIPE_NULL_CELL = "S2_d4_cnull_50k_l50_letf_anneal_offset_clip50"
FINAL_RECIPE_SPECIALIST_CELL = "S2_d4_c05_50k_l50_letf_anneal_offset_clip50"
BUDGET_TWIN_CELL = "S2_d4_camort_50k_l50_letf"
ANNEALED_TWIN_CELL = "S2_d4_camort_50k_l50_letf_anneal"
OFFSET_ANNEAL_CELL = "S2_d4_camort_50k_l50_letf_anneal_offset"
# Uniform-from-start ablation of the widening curriculum (s64): a conditioned
# twin of the printed cell, so it carries a composition block like any other
# amortised cell (its own twin pin lives in test_configs.py).
FLAT_WINDOW_CELL = "S2_d4_camort_50k_l50_letf_anneal_offset_clip50_flatw30"
CLIP_CELLS = {
    "S2_d4_camort_50k_l50_letf_anneal_offset_clip50": 50.0,
    "S2_d4_camort_50k_l50_letf_anneal_offset_clip100": 100.0,
}
D10_BASE_AMORTISED_CELL = "S2_d10_camort_l50_letf_ne128_anneal"
D10_TRANSFER_CELL = "S2_d10_camort_l50_letf_ne128_anneal_offset_clip50"
D10_DEEP_BUFFER_CELL = "S2_d10_camort_l50_letf_ne128_anneal_offset_clip50_cyc8"
# Four arms probing neighbour log-ratio saturation, reaching the same
# unbinding threshold Delta* = clamp/(2*lambda) by two different routes:
# shrink the true ratio (lower lambda) or stop truncating it (raise the
# ceiling). Keeping both routes in one tuple is deliberate — they must stay
# identical in every respect except the one variable each moves.
D10_SATURATION_CELLS = (
    "S2_d10_camort_offset_clip50_lam10",
    "S2_d10_camort_offset_clip50_lam25",
    "S2_d10_camort_offset_clip50_clamp20",
    "S2_d10_camort_offset_clip50_clamp50",
)
# The staircase cell departs from the arms again (capped, gradual widening;
# periodic checkpoints), so like them it joins only the conditioning and
# specialist guards, not the surviving-recipe inheritance check.
D10_STAIRCASE_CELL = "S2_d10_camort_offset_clip50_lam10_hw20"
# The StableAdamW test of soft.tex 4.4's trust-region recommendation
# (prereg 2026-08-11-amort-stadamw-test.md): deliberately departs from the
# surviving recipe in exactly {optimiser, clip}, so like the arms above it
# joins only the conditioning and specialist guards.
D10_STADAMW_CELL = "S2_d10_camort_offset_cyc8_stadamw"
COMPOSITION_GAIN_CELL = (
    "S2_d8_camort_spine3_cgain_l50_letf_ne128_house_sc"
)
PAIRED_SPINE1_CELL = (
    "S2_d8_camort_spine1_pairgrad_l50_letf_ne128_house_sc"
)
PAIRED_SPECIALIST_CELL = (
    "S2_d8_c0500_pairgrad_l50_letf_ne128_house_sc"
)
D10_AMORTISED_CELLS = (
    D10_BASE_AMORTISED_CELL,
    "S2_d10_cgrid_l50_letf_ne128_anneal",
    D10_TRANSFER_CELL,
    D10_DEEP_BUFFER_CELL,
)
# Deliberately NOT in D10_AMORTISED_CELLS: that tuple drives
# `test_amortised_cell_inherits_the_surviving_recipe`, and these arms exist
# precisely to depart from that recipe. They are still amortised cells, so
# they join AMORTISED_CELLS for the conditioning and specialist guards.
AMORTISED_CELLS = (
    VALIDATION_CELL, NARROW_WINDOW_CELL, NULL_CONTROL_CELL, BUDGET_TWIN_CELL,
    ANNEALED_TWIN_CELL, OFFSET_ANNEAL_CELL, *CLIP_CELLS, *D10_AMORTISED_CELLS,
    *D10_SATURATION_CELLS, D10_STAIRCASE_CELL, D10_STADAMW_CELL,
    FINAL_RECIPE_NULL_CELL, FLAT_WINDOW_CELL,
    # Wave-3 house pair (s96): the conditioned cell and its zero-width null
    # on the house recipe (fixed lambda + channel; tests/
    # test_soft_house_configs.py pins their declared-diff sets).
    "S2_d4_camort_50k_l50_letf_house", "S2_d4_cnull_50k_l50_letf_house",
    # Matched-base cells (s101, plan 2026-08-31-soft-camort-matched-base):
    # spine-values draw, no staircase, base matched per cycle. Their own
    # lever pins live in tests/test_matched_base_amortisation.py.
    "S2_d4_camort_mb_50k_l50_letf_house",
    "S2_d8_camort_l50_letf_ne128_house",
    "S2_d8_camort_l50_letf_ne128_house_sc",
    # Draw-set ablation twins (s104): the 17-value draw back to the gate's
    # 3-value spine, one lever, both couplings (the sigma=0.1 twin is the
    # should-stay-healthy control).
    "S2_d8_camort_spine3_l50_letf_ne128_house",
    "S2_d8_camort_spine3_l50_letf_ne128_house_sc",
    # Collapse-mechanism twins (s106), sc only: single-value spine
    # (machinery-vs-mixture) and replay_buffer_cycles=1 (staleness lever).
    "S2_d8_camort_spine1_l50_letf_ne128_house_sc",
    "S2_d8_camort_spine3_rb1_l50_letf_ne128_house_sc",
    # Same A100 spine3 recipe with only a centred c-dependent correction to
    # the exact-field gain. Its parent is the archived dead 4-seed control.
    COMPOSITION_GAIN_CELL,
    PAIRED_SPINE1_CELL,
    # Sigma-ladder twin of the dead sigma_c camort cell (s108): the ladder
    # is its one lever; test_soft_house_configs.py pins it.
    "S2_d8_camort_l50_letf_ne128_house_sc_curr",
)
# The arms clone this cell, not D10_BASE_AMORTISED_CELL: it is the most
# advanced surviving-recipe D=10 run (offset lambda ramp, clip 50) and the
# one whose collapse is on disk as the control for this comparison.
SATURATION_CONTROL_CELL = D10_TRANSFER_CELL

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
    # The house cells come back wrapped in ExactFieldFlipModel, which
    # proxies condition_on_composition but not the embedder attribute —
    # the embedder lives on the wrapped leTF, so look through the wrapper.
    inner_model = getattr(model, "model", model)
    assert hasattr(inner_model, "comp_embedder")


def test_composition_gain_arm_is_one_lever_over_critical_spine3():
    parent = CONFIGS["S2_d8_camort_spine3_l50_letf_ne128_house_sc"]
    arm = CONFIGS[COMPOSITION_GAIN_CELL]

    assert arm.model.exact_field_composition_gain is True
    assert replace(
        arm,
        name=parent.name,
        model=replace(arm.model, exact_field_composition_gain=False),
    ) == parent

    target = IsingTarget(
        D=arm.ising.D,
        sigma=arm.ising.sigma,
        target_composition=arm.ising.target_composition,
        composition_penalty_strength=arm.ising.composition_penalty_strength,
    )
    model = _build_model(arm, target)
    assert model.composition_conditioned_gain is True
    assert model.composition_gain_constant.item() == 0.0
    assert model.composition_gain_slope.item() == 0.0


def test_pairgrad_arms_add_diagnostics_only_to_their_parents():
    pairs = (
        (
            PAIRED_SPECIALIST_CELL,
            "S2_d8_c0500_l50_letf_ne128_house_sc",
        ),
        (
            PAIRED_SPINE1_CELL,
            "S2_d8_camort_spine1_l50_letf_ne128_house_sc",
        ),
    )
    for arm_name, parent_name in pairs:
        arm = CONFIGS[arm_name]
        parent = CONFIGS[parent_name]
        assert arm.train.log_gradient_group_norms is True
        assert replace(
            arm,
            name=parent.name,
            train=replace(arm.train, log_gradient_group_norms=False),
        ) == parent


def test_pairgrad_specialist_and_spine1_share_every_initial_tensor():
    specialist_cfg = CONFIGS[PAIRED_SPECIALIST_CELL]
    spine1_cfg = CONFIGS[PAIRED_SPINE1_CELL]

    def build(cfg):
        target = IsingTarget(
            D=cfg.ising.D,
            sigma=cfg.ising.sigma,
            target_composition=cfg.ising.target_composition,
            composition_penalty_strength=cfg.ising.composition_penalty_strength,
            base_matches_composition=cfg.ising.base_matches_composition,
        )
        torch.manual_seed(42)
        # Compilation has no state-dict effect, but is irrelevant to this
        # construction invariant and expensive to repeat in a unit test.
        eager_cfg = replace(
            cfg, model=replace(cfg.model, compile_model=False)
        )
        model = _build_model(eager_cfg, target)
        return model.state_dict(), torch.get_rng_state().clone()

    specialist_state, specialist_rng = build(specialist_cfg)
    spine1_state, spine1_rng = build(spine1_cfg)
    assert torch.equal(specialist_rng, spine1_rng)
    extras = set(spine1_state) - set(specialist_state)
    assert extras == {
        "model.comp_embedder.mlp.0.weight",
        "model.comp_embedder.mlp.0.bias",
        "model.comp_embedder.mlp.2.weight",
        "model.comp_embedder.mlp.2.bias",
    }
    for name, value in specialist_state.items():
        assert torch.equal(value, spine1_state[name]), name


def test_pairgrad_step_zero_loss_and_shared_gradients_are_exact():
    specialist_cfg = CONFIGS[PAIRED_SPECIALIST_CELL]
    spine1_cfg = CONFIGS[PAIRED_SPINE1_CELL]

    def build(cfg):
        # D=2 makes the full neighbour residual cheap while preserving the
        # exact architecture and critical target formula under test.
        target = IsingTarget(
            D=2,
            sigma=cfg.ising.sigma,
            target_composition=0.5,
            composition_penalty_strength=(
                cfg.ising.composition_penalty_strength
            ),
            base_matches_composition=cfg.ising.base_matches_composition,
        )
        torch.manual_seed(42)
        eager_cfg = replace(
            cfg, model=replace(cfg.model, compile_model=False)
        )
        return _build_model(eager_cfg, target), target

    specialist, specialist_target = build(specialist_cfg)
    spine1, spine1_target = build(spine1_cfg)
    generator = torch.Generator().manual_seed(7)
    x = torch.randint(
        0, 2, (6, specialist_target.d), generator=generator
    ).float() * 2 - 1
    t = torch.rand(6, generator=generator)
    c_t = torch.randn(6, generator=generator)
    c = torch.full((6,), 0.5)

    specialist_loss = kolmogorov_loss(
        x, t, c_t, specialist, specialist_target
    )
    bound_spine1 = CompositionConditioned(spine1, c)
    with spine1_target.composition_batch(c):
        spine1_loss = kolmogorov_loss(
            x, t, c_t, bound_spine1, spine1_target
        )
    assert torch.equal(specialist_loss, spine1_loss)

    specialist_loss.backward()
    spine1_loss.backward()
    specialist_parameters = dict(specialist.named_parameters())
    spine1_parameters = dict(spine1.named_parameters())
    for name, parameter in specialist_parameters.items():
        assert torch.equal(parameter.grad, spine1_parameters[name].grad), name


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


@pytest.mark.parametrize("cell_name,max_norm", sorted(CLIP_CELLS.items()))
def test_clip_cells_vary_only_the_gradient_clip(cell_name, max_norm):
    """Same offset recipe, a tighter gradient clip, nothing else.

    Every death in the offset cell is preceded by the same optimiser
    signature: at the final window widening the pre-clip gradient norm jumps
    from ~30 to 10^3-10^5 and the clip fires on essentially every subsequent
    step, while the one surviving seed peaks at ~111 and stops clipping within
    2k steps. Clipping at 500 does not contain that, it *sustains* it -- a
    clipped step has magnitude exactly 500, about 17x a healthy step, taken in
    a direction estimated from importance weights that have just degenerated.
    So the update magnitude stops carrying information about the descent
    direction's quality precisely when it is least trustworthy.

    Two values rather than one because the healthy phase also spikes (the
    unconditioned specialist trains fine with 4% of steps above 500), so the
    tighter cell risks squashing informative tail gradients; 100 hedges that
    while still keeping a clipped step within ~4x a healthy one.

    Nothing else may differ, or a survival change cannot be attributed.
    """
    offset = CONFIGS[OFFSET_ANNEAL_CELL]
    clipped = CONFIGS[cell_name]

    assert clipped.train.grad_clip_max_norm == max_norm
    assert offset.train.grad_clip_max_norm > max_norm
    assert replace(
        clipped,
        name=offset.name,
        train=replace(
            clipped.train,
            grad_clip_max_norm=offset.train.grad_clip_max_norm,
        ),
    ) == offset


def test_deep_buffer_cell_varies_only_the_replay_buffer_depth():
    """Same D=10 recipe, twice the buffer depth, nothing else.

    The transfer cell carried both D=4 fixes and still ran away at the first
    widening: median gradient norm ~800 before step 10k and ~7.6e4 just after,
    escalating rather than falling back, with training ESS pinned at 1.0 for
    the remaining 40k steps. So a bounded step is not sufficient at 100 sites,
    and the remaining untested difference from the D=4 recipe is buffer depth.

    Depth matters here in a way it never did for a specialist. Batches are
    drawn uniformly across the buffer, so depth is the only mechanism mixing
    requested compositions *within* an update; and because the buffer holds
    states generated under the previous half-width, at a widening the model is
    scored on compositions its buffer has never visited. Doubling the depth
    halves the rate at which fresh compositions enter per optimiser step, so
    the buffer tracks the widened window before the loss charges for it.

    Nothing else may differ, or a survival change cannot be attributed.
    """
    transfer = CONFIGS[D10_TRANSFER_CELL]
    deep = CONFIGS[D10_DEEP_BUFFER_CELL]

    assert deep.train.replay_buffer_cycles == 2 * transfer.train.replay_buffer_cycles
    assert replace(
        deep,
        name=transfer.name,
        train=replace(
            deep.train,
            replay_buffer_cycles=transfer.train.replay_buffer_cycles,
        ),
    ) == transfer


def test_d10_transfer_cell_carries_exactly_the_two_d4_fixes():
    """The D=10 cell may differ from its parent only by the offset and the clip.

    Both interventions were established separately at D=4: finishing the
    penalty ramp before the first widening (so a moving target never shares a
    boundary with stretching coverage), and lowering the gradient clip so a
    saturated step is ~2x a healthy one rather than ~17x. Together they took
    the D=4 cell from 1/4 surviving seeds to 4/4.

    This run buys a single seed, so a third simultaneous change would make a
    failure unattributable -- notably replay_buffer_cycles, which stays at the
    parent's 4 even though D=4 uses 8 and buffer depth is what mixes
    compositions within a batch. That is a named next lever, not a silent one.
    """
    parent = CONFIGS[D10_BASE_AMORTISED_CELL]
    transfer = CONFIGS[D10_TRANSFER_CELL]

    assert transfer.train.grad_clip_max_norm == 50.0
    assert parent.train.grad_clip_max_norm > transfer.train.grad_clip_max_norm
    assert transfer.train.replay_buffer_cycles == parent.train.replay_buffer_cycles

    # The ramp visits the same strengths, but lands before coverage widens.
    stages = transfer.lambda_curriculum.stages
    assert [s.composition_penalty_strength for s in stages] == [
        s.composition_penalty_strength for s in parent.lambda_curriculum.stages
    ]
    assert max(s.start_step for s in stages) < transfer.composition.curriculum[1].start_step
    assert (
        stages[-1].composition_penalty_strength
        == transfer.ising.composition_penalty_strength
    )

    # Nothing else moved.
    assert replace(
        transfer,
        name=parent.name,
        train=replace(
            transfer.train,
            grad_clip_max_norm=parent.train.grad_clip_max_norm,
        ),
        lambda_curriculum=parent.lambda_curriculum,
    ) == parent


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


@pytest.mark.parametrize("cell_name", D10_SATURATION_CELLS)
def test_saturation_arms_differ_from_the_control_in_one_variable_only(cell_name):
    """Each arm must be the control cell with exactly one thing moved.

    The comparison is worthless if an arm also picked up a different seed,
    buffer depth, integration budget or window schedule: any of those would
    supply an alternative explanation for a survival difference, and the
    control is a single run so there is no seed spread to absorb it.
    """
    control = CONFIGS[SATURATION_CONTROL_CELL]
    arm = CONFIGS[cell_name]

    assert arm.train == control.train, "training knobs must be identical"
    assert arm.model == control.model
    assert arm.ctmc == control.ctmc
    assert arm.eval == control.eval
    assert arm.estimator == control.estimator
    assert arm.composition == control.composition, "window schedule is shared"

    # The one permitted axis: the terminal penalty strength, or the ceiling.
    moved = {
        field
        for field in ("composition_penalty_strength", "log_ratio_clamp")
        if getattr(arm.ising, field) != getattr(control.ising, field)
    }
    assert moved, f"{cell_name} moves nothing relative to the control"
    assert moved <= {"composition_penalty_strength", "log_ratio_clamp"}
    # Everything else about the target is held fixed.
    for field in ("D", "sigma", "bias", "target_composition", "base_composition"):
        assert getattr(arm.ising, field) == getattr(control.ising, field), field


@pytest.mark.parametrize("cell_name", D10_SATURATION_CELLS)
def test_saturation_arms_raise_delta_star_above_the_control(cell_name):
    """Every arm must actually widen the unbinding threshold.

    Delta* = clamp / (2 * lambda) is the composition error at which the
    ceiling starts truncating the neighbour ratio. The control sits at 0.05
    against a measured error of 0.078 — i.e. already saturating. An arm that
    did not raise Del* above the control would not be testing anything.
    """
    def delta_star(cfg):
        return cfg.ising.log_ratio_clamp / (
            2.0 * cfg.ising.composition_penalty_strength
        )

    control_star = delta_star(CONFIGS[SATURATION_CONTROL_CELL])
    assert control_star == pytest.approx(0.05)
    assert delta_star(CONFIGS[cell_name]) > control_star


@pytest.mark.parametrize("cell_name", D10_SATURATION_CELLS)
def test_saturation_arms_stay_inside_float32(cell_name):
    """exp(ceiling) multiplies the inflow term, so a ceiling above ~88
    overflows float32 and would fail for a reason unrelated to the
    hypothesis. Leave a wide margin rather than sit near the edge."""
    ceiling = CONFIGS[cell_name].ising.log_ratio_clamp
    assert math.exp(ceiling) < 1e30


def test_specialist_cells_are_untouched():
    """Every archived cell must still be unconditioned and unamortised."""
    for name, cfg in CONFIGS.items():
        if name in AMORTISED_CELLS:
            continue
        assert cfg.composition is None, name
        assert cfg.model.condition_on_composition is False, name


def test_staircase_cell_caps_and_paces_the_widening():
    """The staircase cell's three load-bearing choices, pinned.

    (1) Coverage is capped at half-width 0.20 — the requested composition
    range [0.3, 0.7] needs no more, and 0.30 is the width whose variance
    dose killed the flat-lambda arm. (2) Widening arrives in <= 0.05
    increments with >= 8k dwell, after a >= 20k proving phase at 0.05 —
    each increment is under half the dose the arm survived at 10k.
    (3) lambda is flat at 10 and checkpoints are periodic, so the staircase
    is the only moving schedule and eval can select a healthy state.
    """
    cfg = CONFIGS[D10_STAIRCASE_CELL]
    stages = cfg.composition.curriculum
    widths = [stage.half_width for stage in stages]
    starts = [stage.start_step for stage in stages]
    assert max(widths) == pytest.approx(0.20)
    assert all(
        later - earlier <= 0.05 + 1e-9
        for earlier, later in zip(widths, widths[1:])
    )
    assert all(
        later - earlier >= 8_000
        for earlier, later in zip(starts, starts[1:])
    )
    assert starts[1] >= 20_000
    assert cfg.lambda_curriculum is None
    assert cfg.ising.composition_penalty_strength == 10.0
    assert cfg.train.checkpoint_every == 2_500
    assert cfg.train.grad_clip_max_norm == 50.0
    assert cfg.ising.log_ratio_clamp is None or cfg.ising.log_ratio_clamp == 5.0


def test_final_recipe_machinery_pair_differs_only_in_conditioning():
    """The re-priced machinery-cost pair must isolate the conditioning path.

    Specialist-minus-null at c = 0.5 is quoted as the cost of the
    conditioning machinery at the final recipe (50k steps, clip 50, offset
    lambda anneal). That subtraction only measures the machinery if the two
    cells agree on every other field; any second difference becomes a
    confound riding inside the quoted number.
    """
    null_cfg = CONFIGS[FINAL_RECIPE_NULL_CELL]
    specialist_cfg = CONFIGS[FINAL_RECIPE_SPECIALIST_CELL]

    assert null_cfg.model.condition_on_composition is True
    assert specialist_cfg.model.condition_on_composition is False
    assert replace(null_cfg.model, condition_on_composition=False) == (
        specialist_cfg.model
    )

    assert null_cfg.composition.half_width == 0.0
    assert null_cfg.composition.centre == 0.5
    assert null_cfg.composition.curriculum is None
    assert specialist_cfg.composition is None

    for shared_field in (
        "ising", "train", "ctmc", "eval", "estimator", "lambda_curriculum"
    ):
        assert getattr(null_cfg, shared_field) == (
            getattr(specialist_cfg, shared_field)
        ), shared_field

    assert null_cfg.train.n_steps == 50_000
    assert null_cfg.train.grad_clip_max_norm == 50.0
