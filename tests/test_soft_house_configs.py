"""What correct looks like for the s95 8x8 soft house family, before launch.

The family (plan 2026-08-30-soft-chapter-revamp-efc) is built by
`soft_house_recipe` over the printed ne128 parent. These tests are the
recipe-parity audit in executable form:

1. DECLARED DIFFS ONLY. Every house cell differs from
   S2_d8_c03_l50_letf_ne128 in exactly the declared set — composition,
   sigma (the _sc half), the four recipe levers — and NOTHING else. A
   stray lever here would put a second undeclared change under every
   before/after claim the revamped chapter makes.
2. REPRESENTABILITY + MIRROR ECONOMY. Every trained c* is an integer site
   count at d=64, and no trained composition duplicates another under the
   Z2 mirror c -> 1-c (a trained 0.75 would re-buy 0.25's information).
3. SIGMA_C IS IMPORTED, NEVER RETYPED (the s58 migration rule): the _sc
   cells carry targets.ising.SIGMA_C to the last bit.
4. The nochan control gives back the channel flag ONLY — it exists to
   isolate the channel at sigma_c, so any other difference voids it.
5. The eager gate twin gives back compile_model ONLY — it exists to
   measure the compile loss gap, same logic.
"""
from dataclasses import asdict

from discrete_flow_sampler.targets.ising import SIGMA_C
from experiments.constrained_soft_02.configs import (
    CONFIGS, SOFT_HOUSE_WINDOWS)

PARENT = "S2_d8_c03_l50_letf_ne128"
D_SITES = 64
RECIPE_MODEL_DIFF = {"exact_field_channel", "compile_model"}
RECIPE_TRAIN_DIFF = {"c_t_from_rollout"}


def _diff(parent: dict, child: dict) -> set:
    return {key for key in parent if parent[key] != child[key]}


def test_house_cells_differ_from_parent_in_declared_set_only():
    parent = asdict(CONFIGS[PARENT])
    for c_target, c_tag in SOFT_HOUSE_WINDOWS:
        for sigma_suffix in ("", "_sc"):
            name = f"S2_d8_{c_tag}_l50_letf_ne128_house{sigma_suffix}"
            cell = asdict(CONFIGS[name])
            top = _diff(parent, cell)
            assert top == {"name", "ising", "model", "train", "ema_decay"}, (
                name, top)
            ising = _diff(parent["ising"], cell["ising"])
            expected_ising = {"target_composition"} | (
                {"sigma"} if sigma_suffix else set())
            assert ising == expected_ising, (name, ising)
            assert cell["ising"]["target_composition"] == c_target
            assert _diff(parent["model"], cell["model"]) == RECIPE_MODEL_DIFF
            assert _diff(parent["train"], cell["train"]) == RECIPE_TRAIN_DIFF
            assert cell["ema_decay"] == 0.9999


def test_trained_compositions_representable_and_mirror_disjoint():
    compositions = [c for c, _ in SOFT_HOUSE_WINDOWS]
    for c_target in compositions:
        sites = c_target * D_SITES
        assert sites == int(sites), (c_target, sites)
    mirrored = {1.0 - c for c in compositions}
    overlap = (mirrored & set(compositions)) - {0.5}
    assert not overlap, overlap


def test_sc_cells_carry_exact_sigma_c():
    for _, c_tag in SOFT_HOUSE_WINDOWS:
        cell = CONFIGS[f"S2_d8_{c_tag}_l50_letf_ne128_house_sc"]
        assert cell.ising.sigma == SIGMA_C


def test_sc_anneal_arm_adds_the_schedule_to_nochan_only():
    """The s99 three-fates trio at sigma_c: parent (nochan), anneal, channel.

    The anneal arm is the nochan control PLUS the chapter's declared
    lambda schedule (10/25/50 at 0/10k/20k) and NOTHING else -- it must
    differ from nochan by the schedule alone, or the deferred-vs-
    discharged comparison in fig:penalty-variance carries a second
    change. (It differs from the channel-on house cell by exactly two
    levers as a consequence: channel off, schedule on.)"""
    nochan = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"])
    anneal = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_anneal"])
    assert _diff(nochan, anneal) == {"name", "lambda_curriculum"}
    stages = anneal["lambda_curriculum"]["stages"]
    assert [(s["start_step"], s["composition_penalty_strength"])
            for s in stages] == [(0, 10.0), (10_000, 25.0), (20_000, 50.0)]


def test_matched_base_twins_give_back_base_composition_only():
    """The s99 matched-base wave: base_composition = c* at the off-centre
    windows, one declared lever against the run house twin — any second
    difference would put the uplift claim under two changes. No centre
    twin exists: Bernoulli(0.5) is already the matched base at c* = 0.5,
    so the centre rows anchor both columns unchanged."""
    for c_target, c_tag in SOFT_HOUSE_WINDOWS:
        for sigma_suffix in ("", "_sc"):
            mb_name = f"S2_d8_{c_tag}_l50_letf_ne128_house_mb{sigma_suffix}"
            if c_target == 0.50:
                assert mb_name not in CONFIGS, mb_name
                continue
            house = asdict(
                CONFIGS[f"S2_d8_{c_tag}_l50_letf_ne128_house{sigma_suffix}"])
            twin = asdict(CONFIGS[mb_name])
            assert _diff(house, twin) == {"name", "ising"}, mb_name
            assert _diff(house["ising"], twin["ising"]) == {
                "base_composition"}, mb_name
            assert twin["ising"]["base_composition"] == c_target


def test_nochan_control_gives_back_channel_flag_only():
    house = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"])
    control = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"])
    assert _diff(house, control) == {"name", "model"}
    assert _diff(house["model"], control["model"]) == {"exact_field_channel"}
    assert control["model"]["exact_field_channel"] is False


def test_subcritical_nochan_control_mirrors_the_sc_one():
    """s107 single-size completion: the {coupling} x {channel} 2x2 needs a
    subcritical nochan cell that is one declared lever off the run house
    centre cell, exactly as its sigma_c twin is off _house_sc."""
    house = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house"])
    control = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_nochan"])
    assert _diff(house, control) == {"name", "model"}
    assert _diff(house["model"], control["model"]) == {"exact_field_channel"}
    assert control["model"]["exact_field_channel"] is False


def test_lambda_twins_give_back_penalty_strength_only():
    """The 8x8 lambda-trade twins (both lambdas, both couplings) are the
    house centre cell with composition_penalty_strength moved and nothing
    else -- a second lever would confound the lambda trade the chapter
    reads off them."""
    for lam, lam_tag, sigma_suffix in (
            (10.0, "l10", ""), (10.0, "l10", "_sc"),
            (100.0, "l100", ""), (100.0, "l100", "_sc")):
        house = asdict(
            CONFIGS[f"S2_d8_c0500_l50_letf_ne128_house{sigma_suffix}"])
        twin = asdict(CONFIGS[
            f"S2_d8_c0500_{lam_tag}_letf_ne128_house{sigma_suffix}"])
        assert _diff(house, twin) == {"name", "ising"}, (lam_tag, sigma_suffix)
        assert _diff(house["ising"], twin["ising"]) == {
            "composition_penalty_strength"}
        assert twin["ising"]["composition_penalty_strength"] == lam


def test_eager_gate_twin_gives_back_compile_only():
    gate = asdict(CONFIGS["S2_d4_c05_l50_letf_house_gate"])
    eager = asdict(CONFIGS["S2_d4_c05_l50_letf_house_gate_eager"])
    assert _diff(gate, eager) == {"name", "model"}
    assert _diff(gate["model"], eager["model"]) == {"compile_model"}
    assert eager["model"]["compile_model"] is False


def test_amortised_house_cell_is_parent_plus_recipe_only():
    """Wave 3 (plan): the camort family re-run fixed-lambda + channel, with
    the anneal_offset_clip machinery retired. The cell must be the archived
    fixed-lambda 50k amortised parent plus the four recipe levers and
    NOTHING else — in particular no lambda_curriculum and no offset/clip
    fields, since the wave's claim is that the plain recipe replaces that
    whole confound family."""
    parent = asdict(CONFIGS["S2_d4_camort_50k_l50_letf"])
    cell = asdict(CONFIGS["S2_d4_camort_50k_l50_letf_house"])
    assert _diff(parent, cell) == {"name", "model", "train", "ema_decay"}
    assert _diff(parent["model"], cell["model"]) == RECIPE_MODEL_DIFF
    assert _diff(parent["train"], cell["train"]) == RECIPE_TRAIN_DIFF
    assert cell["ema_decay"] == 0.9999
    assert cell["lambda_curriculum"] is None
    assert cell["model"]["condition_on_composition"] is True


def test_amort_specialist_house_family_matched_recipe_and_windows():
    """The tab:amort-4x4 comparator rows re-run on the SAME recipe as the
    conditioned cell (the cnull block's own confound argument, applied
    forward), at the revamp's window set {0.25, 0.375, 0.50} — every one an
    integer site count at d=16, unlike the retired {0.30, 0.65, 0.80}. The
    c05 cell is the plain 10k specialist at the amortised 50k budget plus
    the recipe; the off-centre twins give back composition only."""
    parent = asdict(CONFIGS["S2_d4_c05_l50_letf"])
    base = asdict(CONFIGS["S2_d4_c0500_50k_l50_letf_house"])
    assert _diff(parent, base) == {"name", "model", "train", "ema_decay"}
    assert _diff(parent["model"], base["model"]) == RECIPE_MODEL_DIFF
    assert _diff(parent["train"], base["train"]) == (
        RECIPE_TRAIN_DIFF | {"n_steps"})
    assert base["train"]["n_steps"] == 50_000
    assert base["ema_decay"] == 0.9999
    for c_target, c_tag in ((0.25, "c0250"), (0.375, "c0375")):
        twin = asdict(CONFIGS[f"S2_d4_{c_tag}_50k_l50_letf_house"])
        assert _diff(base, twin) == {"name", "ising"}
        assert _diff(base["ising"], twin["ising"]) == {"target_composition"}
        assert twin["ising"]["target_composition"] == c_target
        assert (c_target * 16) == int(c_target * 16)


def test_amort_null_house_isolates_conditioning_machinery():
    """The null control prices the conditioning path against its matched
    specialist; both must share the house recipe or the price is the
    recipe gap instead. vs the c05 house specialist the null differs in
    the conditioning machinery alone: the model flag and the zero-width
    composition config that feeds it."""
    parent = asdict(CONFIGS["S2_d4_cnull_l50_letf"])
    null = asdict(CONFIGS["S2_d4_cnull_50k_l50_letf_house"])
    assert _diff(parent, null) == {"name", "model", "train", "ema_decay"}
    assert _diff(parent["model"], null["model"]) == RECIPE_MODEL_DIFF
    assert _diff(parent["train"], null["train"]) == (
        RECIPE_TRAIN_DIFF | {"n_steps"})
    assert null["train"]["n_steps"] == 50_000

    specialist = asdict(CONFIGS["S2_d4_c0500_50k_l50_letf_house"])
    machinery = _diff(specialist, null)
    assert machinery == {"name", "model", "composition"}, machinery
    assert _diff(specialist["model"], null["model"]) == {
        "condition_on_composition"}
    assert null["model"]["condition_on_composition"] is True
    assert null["composition"]["half_width"] == 0.0


def test_camort_sigma_ladder_twin_adds_the_ladder_only():
    """The dead sigma_c camort cell starts cold at sigma_c; the hard
    chapter's amortised sigma_c cell trains on the d64 sigma ladder. The
    twin must give back the ladder ALONE, end on the exact sigma_c, and
    align every stage with the outer cycle (the trainer rejects it
    otherwise)."""
    dead = asdict(CONFIGS["S2_d8_camort_l50_letf_ne128_house_sc"])
    twin = CONFIGS["S2_d8_camort_l50_letf_ne128_house_sc_curr"]
    assert _diff(dead, asdict(twin)) == {"name", "curriculum"}
    stages = twin.curriculum.stages
    assert stages[0].start_step == 0 and stages[0].sigma == 0.1
    assert stages[-1].sigma == SIGMA_C
    assert all(s.start_step % twin.train.inner_steps_per_outer == 0
               for s in stages)
    assert stages[-1].start_step < twin.train.n_steps


def test_specialist_sigma_ladder_twin_matches_the_camort_ladder():
    """The ladder camort cell trains where the cold one died, so its yield
    ratio needs a specialist on the SAME ladder: one lever off the house
    sigma_c specialist, and the identical stage tuple, or the ratio carries
    the ladder as a second difference."""
    specialist = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"])
    twin = CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_curr"]
    assert _diff(specialist, asdict(twin)) == {"name", "curriculum"}
    camort_ladder = CONFIGS["S2_d8_camort_l50_letf_ne128_house_sc_curr"]
    assert twin.curriculum == camort_ladder.curriculum


def test_d4_10k_table_family_is_the_50k_cell_at_the_4x4_budget():
    """tab:eval-soft-4x4 reads the 10k family: each subcritical cell is
    its _50k_ twin with n_steps alone moved to the cross-chapter 4x4
    budget, and each sigma_c cell moves sigma alone on top of that, to
    the exact SIGMA_C, cold (no ladder), like every soft sigma_c
    specialist."""
    for _, c_tag in SOFT_HOUSE_WINDOWS:
        budget_parent = asdict(CONFIGS[f"S2_d4_{c_tag}_50k_l50_letf_house"])
        cell = asdict(CONFIGS[f"S2_d4_{c_tag}_10k_l50_letf_house"])
        assert _diff(budget_parent, cell) == {"name", "train"}
        assert _diff(budget_parent["train"], cell["train"]) == {"n_steps"}
        assert cell["train"]["n_steps"] == 10_000
        sc = CONFIGS[f"S2_d4_{c_tag}_10k_l50_letf_house_sc"]
        assert _diff(cell, asdict(sc)) == {"name", "ising"}
        assert _diff(cell["ising"], asdict(sc.ising)) == {"sigma"}
        assert sc.ising.sigma == SIGMA_C
        assert sc.curriculum is None


def test_d4_10k_conditioned_twin_mirrors_the_8x8_construction():
    """The conditioned row of tab:eval-soft-4x4 sits at the table's own 10k
    budget and is built from the 10k centre specialist exactly as the 8x8
    conditioned cell is built from its specialist: matched base on, the
    conditioning flag on, the discrete spine draw -- and nothing else, so
    the row prices amortisation alone. Every spine value is an integer
    site count at d=16 (4/6/8 sites)."""
    for sigma_suffix in ("", "_sc"):
        spec = asdict(CONFIGS[f"S2_d4_c0500_10k_l50_letf_house{sigma_suffix}"])
        cell = asdict(CONFIGS[f"S2_d4_camort_10k_l50_letf_house{sigma_suffix}"])
        assert _diff(spec, cell) == {"name", "ising", "model", "composition"}
        assert _diff(spec["ising"], cell["ising"]) == {"base_matches_composition"}
        assert cell["ising"]["base_matches_composition"] is True
        assert _diff(spec["model"], cell["model"]) == {"condition_on_composition"}
        assert cell["model"]["condition_on_composition"] is True
        assert cell["train"]["n_steps"] == 10_000
        assert cell["composition"]["half_width"] == 0.0
        assert tuple(cell["composition"]["values"]) == (0.25, 0.375, 0.5)
        assert all(v * 16 == int(v * 16) for v in cell["composition"]["values"])
