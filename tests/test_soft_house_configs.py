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


def test_nochan_control_gives_back_channel_flag_only():
    house = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc"])
    control = asdict(CONFIGS["S2_d8_c0500_l50_letf_ne128_house_sc_nochan"])
    assert _diff(house, control) == {"name", "model"}
    assert _diff(house["model"], control["model"]) == {"exact_field_channel"}
    assert control["model"]["exact_field_channel"] is False


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
