"""Tests for the 20x20 fill.

This rung differs from every other house table in two ways, both places a
fill can go wrong silently.

Two couplings. The rung was single-coupling while only the sigma = 0.1 wave
existed. The sigma_c wave (tag 20260829-d400-sc, 12 cells, 100k steps) has its
own certified reference (kawasaki_ref_d400_sc, tau 18.9 sweeps), so the table
now carries the same s010/s220 pair as every rung below. The two waves have
different config names (50k flat vs 100k_curr) and different tags, so cells
are pinned per (row, coupling) rather than globbed from one template.

The reference FLOP bill is size-derived. `chain_trial_counts` converts sweeps
to swap proposals through `lattice_edge**2` and defaults to 16, having been
written for the d256 fill; left at its default here it would under-bill the
reference chain by (16/20)^2 = 0.64 -- a 36% error in the reference row's
FLOP/es, with nothing in the output to reveal it.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "kawasaki_ref_d400_s010"
REFERENCE_DIR_SC = REPO_ROOT / "results" / "kawasaki_ref_d400_sc"

L = 20
D_SITES = L * L


def test_rung_is_two_coupling_at_exact_sigma_c():
    """Both house couplings, with s220 at exact SIGMA_C, never legacy 0.223.

    Each coupling names its own reference directory: scoring sigma_c cells
    against the sigma = 0.1 pool would carry a d<nn>/dsigma systematic that
    no amount of sampling averages away."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        REFERENCE_DIRS,
        SIGMA,
        SIGMA_LABELS,
    )

    from discrete_flow_sampler.targets.ising import SIGMA_C

    assert SIGMA_LABELS == ("s010", "s220")
    assert SIGMA["s010"] == 0.1
    assert SIGMA["s220"] == SIGMA_C
    assert REFERENCE_DIRS["s010"] != REFERENCE_DIRS["s220"]


def test_reference_bill_uses_this_rung_s_lattice_not_the_d256_default():
    """The silent one: (16/20)^2 = 0.64 under-bill if the default is taken.

    Asserted against an independently recomputed value rather than against
    the function's own output at a different edge, so a change to the
    convention fails here rather than agreeing with itself.
    """
    from experiments.constrained_hard_03.analysis.house_table_16x16 import (
        chain_trial_counts,
    )
    from experiments.constrained_hard_03.analysis.house_table_20x20 import L as fill_L
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        reference_trial_counts,
    )

    assert fill_L == 20
    provenance = {
        "burn_in_sweeps": 1000,
        "sampling_sweeps_per_chain": 2000,
        "n_chains": 3,
    }
    expected = (1000 + 2000) * 20 * 20

    assert reference_trial_counts(provenance) == [expected] * 3
    # And it must not silently equal the d256 default.
    assert reference_trial_counts(provenance) != chain_trial_counts(provenance)


@pytest.mark.skipif(not REFERENCE_DIR.is_dir(), reason="d400 reference not generated")
def test_reference_is_certified_at_this_lattice_and_coupling():
    """A reference is a reference only for its own lattice and sigma.

    Both are recorded in the pool's provenance, and the directory name is
    not evidence of either -- `kawasaki_ref_d256_sc` is on record as
    mislabelled at 0.22305.
    """
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    certification = json.loads((REFERENCE_DIR / "certification.json").read_text())

    assert provenance["lattice_side"] == L
    assert provenance["n_sites"] == D_SITES
    assert abs(provenance["sigma"] - 0.1) < 1e-9
    assert certification["certified"] is True


@pytest.mark.skipif(
    not REFERENCE_DIR_SC.is_dir(), reason="d400 sigma_c reference not generated"
)
def test_sigma_c_reference_is_certified_at_exact_sigma_c():
    """The sigma_c pool must record exact SIGMA_C -- the d256 sc pool was
    mislabelled at 0.22305."""
    from discrete_flow_sampler.targets.ising import SIGMA_C

    provenance = json.loads((REFERENCE_DIR_SC / "provenance.json").read_text())
    certification = json.loads((REFERENCE_DIR_SC / "certification.json").read_text())

    assert provenance["lattice_side"] == L
    assert abs(provenance["sigma"] - SIGMA_C) < 1e-12
    assert certification["certified"] is True
    # Off d256 the mchammer anchor must be recorded absent (see the gate
    # test below), so certification rests on the internal checks.
    anchor = certification.get("external_nn_anchor")
    assert anchor is None or anchor.get("target") is None
    assert certification["hard_constraint"]["n_up_spins_exact"] == D_SITES // 2
    assert certification["hard_constraint"]["all_samples_on_manifold"] is True


def test_external_anchor_gates_on_lattice_side_as_well_as_sigma():
    """The mchammer nn anchor (0.578756) was measured at sigma_c and d256.

    The nn-correlation at criticality is D-dependent (finite-size effects
    peak at sigma_c), so a d400 chain at exact SIGMA_C must not be held to
    the d256 anchor: it could fail certification spuriously, or pass narrowly
    and record an external cross-check that was never valid. An earlier
    version of the check keyed on sigma alone."""
    from experiments.constrained_hard_03.generate_kawasaki_reference_d256 import (
        CERTIFICATION_NN_TARGET,
        external_nn_anchor,
    )

    from discrete_flow_sampler.targets.ising import SIGMA_C

    assert external_nn_anchor(SIGMA_C, 16)[0] == CERTIFICATION_NN_TARGET
    assert external_nn_anchor(SIGMA_C, 20) == (None, None)
    assert external_nn_anchor(0.1, 16) == (None, None)
    assert external_nn_anchor(0.1, 20) == (None, None)


@pytest.mark.skipif(not REFERENCE_DIR.is_dir(), reason="d400 reference not generated")
def test_reference_has_no_external_anchor_and_says_so():
    """Off sigma_c there is no mchammer anchor, and that must be recorded.

    The certification then rests on the internal checks alone (Gelman-Rubin,
    start-condition agreement). Silence here would let a reader assume an
    external cross-check that was never made.
    """
    certification = json.loads((REFERENCE_DIR / "certification.json").read_text())

    anchor = certification.get("external_nn_anchor")
    assert anchor in (None, "none off sigma_c") or anchor.get("target") is None
    assert certification["multi_chain_agreement"]["gelman_rubin_nn_correlation"] < 1.01


def test_every_arm_names_a_real_config_at_both_couplings():
    """A renamed arm would otherwise surface as a permanently blank cell."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        ARM_CONFIGS,
        ARMS,
        SIGMA_LABELS,
    )
    from experiments.constrained_hard_03.configs import CONFIGS

    assert set(ARM_CONFIGS) == set(ARMS)
    for row, per_sigma in ARM_CONFIGS.items():
        assert set(per_sigma) == set(SIGMA_LABELS), row
        for sigma_label, config_name in per_sigma.items():
            assert config_name in CONFIGS, config_name
            assert f"_{sigma_label}_" in config_name, config_name


def test_gfn_cells_name_registered_configs_at_both_couplings():
    """The GFN rows are pinned by name, like the swap arms; a renamed cell
    would otherwise print as a permanently blank row."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        GFN_ARMS,
        GFN_CELL_NAME,
        GFN_TAG,
        SIGMA_LABELS,
    )
    from experiments.constrained_hard_03.gfn_configs import GFN_CONFIGS

    assert set(GFN_TAG) == set(GFN_ARMS)
    for sigma_label in SIGMA_LABELS:
        for gfn_arm in GFN_ARMS:
            name = GFN_CELL_NAME[sigma_label].format(
                objective=gfn_arm.removeprefix("gfn_")
            )
            assert name in GFN_CONFIGS, name
            assert GFN_CONFIGS[name].D == L, name


def test_gfn_rows_stay_outside_the_bold_comparison():
    """The 8x8/16x16 rule carried up: a GFN cell holding the best number in
    a column must not take the bold, which marks the best swap cell. `best`
    runs over ARMS, which the GFN arms are not in."""
    from experiments.constrained_hard_03.analysis import house_table_20x20 as h20

    assert not set(h20.GFN_ARMS) & set(h20.ARMS)
    printed = [row[0] for row in h20.LATEX_ROWS if row]
    assert set(h20.GFN_ARMS) <= set(printed)

    def entry(ess, flops):
        return {
            "ESS": (ess, 0.001),
            "dMag": (0.05, 0.01),
            "dCorr": (0.05, 0.01),
            "EW2": (0.05, 0.01),
            "FLOP/es": (flops, 0.0),
        }

    table = {
        "thp2_w4_s220": entry(0.70, 1.0e11),
        "gfn_tb_s220": entry(0.74, 1.0e8),
    }
    body = h20.latex_table(table)
    swap_line = next(line for line in body.splitlines() if "$R=2$" in line)
    gfn_line = next(line for line in body.splitlines() if "trajectory balance" in line)
    assert "\\mathbf{0.700" in swap_line and "\\mathbf{1.0" in swap_line
    assert "\\mathbf" not in gfn_line
    fldb_line = next(line for line in body.splitlines() if "forward-looking" in line)
    assert fldb_line.count("--") == 10


def test_gfn_rows_take_the_caller_s_raw_bill_not_an_euler_grid():
    """A GFN cell prices one rollout plus one target eval; routed through
    the Euler-grid formula it would be billed n_euler times over."""
    import inspect

    from experiments.constrained_hard_03.analysis.house_table_20x20 import neural_cell

    assert "flops_raw" in inspect.signature(neural_cell).parameters
