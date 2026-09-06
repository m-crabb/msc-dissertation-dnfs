"""Tests for the Kaufman / Ferdinand--Fisher exact finite-torus Ising solution.

Correctness is pinned three independent ways, none of which shares code with
the implementation under test:

1. Brute-force enumeration: for mn <= 16 sites the partition function is an
   exact sum over 2^(mn) states, computed here directly from the bond list.
   This validates the closed form at square AND non-square shapes, and on
   both sides of criticality (the gamma_0 = 2K + ln tanh K angle changes
   sign at K_c, and the sign handling of the Z4 product is the classic bug).
2. DNFS Table 2 (paper, App. E.1.2): the printed "Optimal Value" rows for
   the 10x10 lattice at sigma in {0.1, 0.22305} are independent published
   values of the same formula in the same units (F/D = -ln Z / (K mn),
   E/D = -d(ln Z)/dK / mn, S/D = K (E - F)/D with K = 2 sigma under the
   repo's double-counted convention).
3. Internal thermodynamic identity: S = K (E - F) must hold to numerical
   precision by construction, which pins the unit convention itself.
"""

import itertools
import math

import numpy as np
import pytest

from discrete_flow_sampler.targets.ising_exact import (
    ferdinand_fisher_per_site,
    log_partition_torus,
)


def _enumerate_log_partition(n_rows: int, n_cols: int, bond_coupling: float):
    """Exact log Z and per-site internal energy by summing over all states.

    Energy convention: beta*H = -K * sum_{<ij>} s_i s_j over each unordered
    torus edge ONCE (K = bond_coupling). Returns (log_Z, E_per_site) with
    E_per_site = -<sum_pairs s_i s_j> / (n_rows * n_cols), the DNFS Table 2
    unit (energy in units of J per site).
    """
    n_sites = n_rows * n_cols
    bonds = []
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            bonds.append((i, r * n_cols + (c + 1) % n_cols))
            bonds.append((i, ((r + 1) % n_rows) * n_cols + c))
    states = np.array(
        list(itertools.product((-1, 1), repeat=n_sites)), dtype=np.float64
    )
    pair_sum = np.zeros(len(states))
    for i, j in bonds:
        pair_sum += states[:, i] * states[:, j]
    # log Z = logsumexp(K * pair_sum) done stably
    scaled = bond_coupling * pair_sum
    shift = scaled.max()
    weights = np.exp(scaled - shift)
    log_z = shift + math.log(weights.sum())
    mean_pair_sum = float((weights * pair_sum).sum() / weights.sum())
    return log_z, -mean_pair_sum / n_sites


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (2, 3), (3, 5)])
@pytest.mark.parametrize("sigma", [0.1, 0.22305, 0.3])
def test_log_partition_matches_enumeration(shape, sigma):
    """Closed form == brute force, square and non-square, both phases.

    sigma = 0.1 is subcritical (gamma_0 < 0, Z4 negative), 0.22305 is just
    supercritical, 0.3 is deep in the ordered phase — the three sign regimes
    the gamma_0 angle can put the Z4 product in.
    """
    n_rows, n_cols = shape
    bond_coupling = 2.0 * sigma
    expected, _ = _enumerate_log_partition(n_rows, n_cols, bond_coupling)
    actual = log_partition_torus(n_rows, n_cols, bond_coupling)
    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("sigma", [0.1, 0.22305])
def test_internal_energy_matches_enumeration(sigma):
    """The derivative route reproduces the exact thermal average at 4x4."""
    _, expected_energy = _enumerate_log_partition(4, 4, 2.0 * sigma)
    values = ferdinand_fisher_per_site(D=4, sigma=sigma)
    assert values["internal_energy"] == pytest.approx(expected_energy, abs=1e-6)


def test_dnfs_table2_sigma01_row():
    """The 10x10 'Optimal Value' row of DNFS Table 2 at sigma = 0.1."""
    values = ferdinand_fisher_per_site(D=10, sigma=0.1)
    assert values["free_energy"] == pytest.approx(-3.6727, abs=1e-4)
    assert values["internal_energy"] == pytest.approx(-0.4282, abs=1e-4)
    assert values["entropy"] == pytest.approx(0.6489, abs=1e-4)


def test_dnfs_table2_critical_row_is_at_exact_criticality_not_022305():
    """FINDING: DNFS Table 2's 'Optimal Value' row labelled
    sigma = 0.22305 was computed at EXACT criticality sigma = ln(1+sqrt(2))/4
    = 0.220343, not at the labelled coupling. All three printed values match
    the exact-critical evaluation to printed precision and none matches the
    evaluation at 0.22305 (F -2.1165, E -1.5104, S 0.2704 there). Downstream
    consequence: the replication's sigma_c energy comparisons must use the
    exact values AT the operating coupling — against those, the measured
    E/D ~ -1.508 sits ~0.002 from truth, not the ~0.03 'replication gap'
    read against the paper's misplaced comparator."""
    exact_critical_sigma = math.log(1.0 + math.sqrt(2.0)) / 4.0
    at_critical = ferdinand_fisher_per_site(D=10, sigma=exact_critical_sigma)
    assert at_critical["free_energy"] == pytest.approx(-2.1242, abs=1e-4)
    assert at_critical["internal_energy"] == pytest.approx(-1.4763, abs=2e-4)
    assert at_critical["entropy"] == pytest.approx(0.2855, abs=1e-4)

    at_operating = ferdinand_fisher_per_site(D=10, sigma=0.22305)
    assert at_operating["free_energy"] == pytest.approx(-2.1165, abs=1e-4)
    assert at_operating["internal_energy"] == pytest.approx(-1.5104, abs=1e-4)
    assert at_operating["entropy"] == pytest.approx(0.2704, abs=1e-4)


def test_entropy_identity_pins_units():
    """S/D = K (E/D - F/D) with K = 2 sigma: the unit convention itself."""
    sigma = 0.22305
    values = ferdinand_fisher_per_site(D=10, sigma=sigma)
    identity = 2.0 * sigma * (values["internal_energy"] - values["free_energy"])
    assert values["entropy"] == pytest.approx(identity, abs=1e-9)
