"""What correct looks like for the 24x24 fill, written before it.

The 24x24 rung (tag 20260903-d576-sc, 2026-09-03) is the d400 sigma_c R=3
bf16 cell moved to the lattice, plus one continuation of the radius knob
(R=4), three seeds each, ONE coupling. Two things can go wrong silently:

  * the fill imports the 20x20 module's lattice-bound helpers, which were
    written with L = 20 baked in; used at their defaults they would bill the
    reference at (20/24)^2 = 0.69 of its true proposals and score the
    magnetisation/correlation profiles on the wrong lattice. Every helper
    the 24x24 fill takes from 20x20 must therefore accept the lattice side
    explicitly, AND the 20x20 fill must be unchanged at its default.
  * the table is single-coupling by construction (no sigma = 0.1 wave was
    ever run at d576). A two-coupling body would print an empty half that
    reads as "not yet landed" rather than "never run".
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "kawasaki_ref_d576_sc"

L = 24
D_SITES = L * L


def test_rung_is_single_coupling_at_exact_sigma_c():
    from experiments.constrained_hard_03.analysis.house_table_24x24 import (
        REFERENCE_DIRS, SIGMA, SIGMA_LABELS)
    from discrete_flow_sampler.targets.ising import SIGMA_C

    assert SIGMA_LABELS == ("s220",)
    assert SIGMA["s220"] == SIGMA_C
    assert REFERENCE_DIRS == {"s220": REFERENCE_DIR}


def test_reference_bill_uses_this_rung_s_lattice():
    """(20/24)^2 = 0.69 under-bill if the 20x20 default rides through."""
    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        reference_trial_counts)
    from experiments.constrained_hard_03.analysis.house_table_24x24 import (
        L as fill_L)

    assert fill_L == 24
    provenance = {"burn_in_sweeps": 1000, "sampling_sweeps_per_chain": 2000,
                  "n_chains": 3}
    assert reference_trial_counts(provenance, lattice_side=24) == \
        [(1000 + 2000) * 24 * 24] * 3
    # The 20x20 fill's own bill is untouched by the refactor.
    assert reference_trial_counts(provenance) == [(1000 + 2000) * 400] * 3


def test_neural_cell_scores_profiles_on_the_fill_s_lattice(tmp_path):
    """The magnetisation/correlation profile errors take the lattice side to
    reshape flat states; a 24x24 state reshaped as 20x20 raises rather than
    silently scoring, so passing the side through is the whole fix."""
    import torch

    from experiments.constrained_hard_03.analysis.house_table_20x20 import (
        neural_cell)
    from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

    n_draws = 8
    states = torch.where(torch.rand(n_draws, D_SITES) < 0.5, 1.0, -1.0)
    run_dir = tmp_path / "run"
    (run_dir / "eval").mkdir(parents=True)
    torch.save(states, run_dir / "eval" / "samples.pt")
    torch.save(torch.zeros(n_draws), run_dir / "eval" / "log_weights.pt")
    (run_dir / "eval" / "metrics.json").write_text(
        json.dumps({"ess_fraction": 1.0}))
    target = IsingTarget(D=L, sigma=SIGMA_C)
    reference = states.clone()
    reference_energy = -target.log_prob(reference) / (2 * SIGMA_C * D_SITES)

    with pytest.raises(Exception):
        neural_cell(run_dir, target, reference, reference_energy,
                    per_forward=1.0, n_euler=1)
    cell = neural_cell(run_dir, target, reference, reference_energy,
                       per_forward=1.0, n_euler=1, lattice_side=L)
    assert cell["ESS"] == 1.0
    assert cell["dMag"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not REFERENCE_DIR.is_dir(),
                    reason="d576 sigma_c reference not generated")
def test_reference_is_certified_at_this_lattice_and_exact_sigma_c():
    from discrete_flow_sampler.targets.ising import SIGMA_C

    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    certification = json.loads((REFERENCE_DIR / "certification.json").read_text())

    assert provenance["lattice_side"] == L
    assert provenance["n_sites"] == D_SITES
    assert abs(provenance["sigma"] - SIGMA_C) < 1e-12
    assert certification["certified"] is True
    anchor = certification.get("external_nn_anchor")
    assert anchor is None or anchor.get("target") is None
    assert certification["hard_constraint"]["n_up_spins_exact"] == D_SITES // 2
    assert certification["hard_constraint"]["all_samples_on_manifold"] is True


def test_every_arm_names_a_real_d576_config():
    from experiments.constrained_hard_03.analysis.house_table_24x24 import (
        ARM_CONFIGS, ARMS, SIGMA_LABELS)
    from experiments.constrained_hard_03.configs import CONFIGS

    assert set(ARM_CONFIGS) == set(ARMS)
    for row, per_sigma in ARM_CONFIGS.items():
        assert set(per_sigma) == set(SIGMA_LABELS), row
        for config_name in per_sigma.values():
            assert config_name in CONFIGS, config_name
            assert "d576" in config_name and "_s220_" in config_name


def test_latex_body_is_single_coupling():
    """Five value columns per row, not ten: no empty sigma = 0.1 half."""
    from experiments.constrained_hard_03.analysis.house_table_24x24 import (
        latex_table)

    table = {
        "reference_s220": {"dMag": (0.01, 0.0), "dCorr": (0.02, 0.0),
                           "EW2": (0.001, 0.0), "FLOP/es": (1e6, 0.0)},
        "floor5000_s220": {"dMag": (0.1, 0.0), "dCorr": (0.1, 0.0),
                           "EW2": (0.004, 0.0)},
        "thp3_w5bf16_s220": {"ESS": (0.6, 0.05), "dMag": (0.2, 0.01),
                             "dCorr": (0.3, 0.01), "EW2": (0.02, 0.001),
                             "FLOP/es": (2e11, 0.0)},
        "thp4_w5bf16_s220": {"ESS": (0.7, 0.01), "dMag": (0.1, 0.01),
                             "dCorr": (0.2, 0.01), "EW2": (0.01, 0.001),
                             "FLOP/es": (3e11, 0.0)},
    }
    body = latex_table(table)
    data_rows = [line for line in body.splitlines() if line.strip().endswith(r"\\")]
    assert len(data_rows) == 4
    for line in data_rows:
        assert line.count("&") == 5, line
    assert r"\mathbf{0.700" in body      # ESS bold goes to R=4
    assert r"\mathbf{2.0\times10^{11}}" in body  # FLOP/es bold goes to R=3
