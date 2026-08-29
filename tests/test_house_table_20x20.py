"""What correct looks like for the 20x20 fill, written before it.

This rung differs from every other house table in two ways, and both are
places a fill can go wrong silently rather than loudly.

ONE COUPLING, NOT TWO. The d400 wave ran at sigma = 0.1 only. Every other
house table carries a sigma = 0.1 / sigma_c pair, so the copied-from shape is
two columns, and a two-column d400 table would render an empty sigma_c half
that reads as "those runs have not landed yet" rather than "that experiment
was never run". The table must be single-coupling by construction.

THE REFERENCE FLOP BILL IS SIZE-DERIVED. `chain_trial_counts` converts sweeps
to swap proposals through `lattice_edge**2` and DEFAULTS TO 16, because it was
written for the d256 fill. Imported here and left at its default it would
under-bill the reference chain by (16/20)^2 = 0.64 -- a 36% error in the
reference row's FLOP/es, with nothing in the output to reveal it. The 16x16
module's docstring already promises the factor is "derived, not hard-coded, so
another rung built this way cannot be mis-billed"; this rung is the one that
tests the promise.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "kawasaki_ref_d400_s010"

L = 20
D_SITES = L * L


def test_rung_is_single_coupling():
    """sigma = 0.1 only: the d400 wave has no sigma_c cells and never had."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        SIGMA, SIGMA_LABELS)

    assert SIGMA_LABELS == ("s010",)
    assert SIGMA["s010"] == 0.1
    assert "s220" not in SIGMA


def test_reference_bill_uses_this_rung_s_lattice_not_the_d256_default():
    """The silent one: (16/20)^2 = 0.64 under-bill if the default is taken.

    Asserted against an independently recomputed value rather than against
    the function's own output at a different edge, so a change to the
    convention fails here rather than agreeing with itself.
    """
    from experiments.constrained_hard_03.analysis.house_table_16x16 import (
        chain_trial_counts)
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        L as fill_L, reference_trial_counts)

    assert fill_L == 20
    provenance = {"burn_in_sweeps": 1000, "sampling_sweeps_per_chain": 2000,
                  "n_chains": 3}
    expected = (1000 + 2000) * 20 * 20

    assert reference_trial_counts(provenance) == [expected] * 3
    # And it must NOT silently equal the d256 default.
    assert reference_trial_counts(provenance) != chain_trial_counts(provenance)


@pytest.mark.skipif(not REFERENCE_DIR.is_dir(),
                    reason="d400 reference not generated")
def test_reference_is_certified_at_this_lattice_and_coupling():
    """A reference is a reference only for its own lattice AND sigma.

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
    assert certification["hard_constraint"]["n_up_spins_exact"] == D_SITES // 2
    assert certification["hard_constraint"]["all_samples_on_manifold"] is True


@pytest.mark.skipif(not REFERENCE_DIR.is_dir(),
                    reason="d400 reference not generated")
def test_reference_has_no_external_anchor_and_says_so():
    """Off sigma_c there is no mchammer anchor, and that must be RECORDED.

    The certification then rests on the internal checks alone (Gelman-Rubin,
    start-condition agreement). Silence here would let a reader assume an
    external cross-check that was never made.
    """
    certification = json.loads((REFERENCE_DIR / "certification.json").read_text())

    anchor = certification.get("external_nn_anchor")
    assert anchor in (None, "none off sigma_c") or anchor.get("target") is None
    assert certification["multi_chain_agreement"][
        "gelman_rubin_nn_correlation"] < 1.01


def test_every_arm_names_a_real_config():
    """A renamed arm would otherwise surface as a permanently blank row."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import ARMS
    from experiments.constrained_hard_03.configs import CONFIGS

    for arm in ARMS:
        assert arm in CONFIGS, arm
