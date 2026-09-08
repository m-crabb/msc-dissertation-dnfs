"""Tests for the slice thermodynamic-integration reference (slice_ti.py).

The on-slice free energy in the repo's convention (metrics.free_energy_lb_estimate,
paper Eq. 37) is F/d = -log Z_slice(sigma) / (2 sigma d). Thermodynamic
integration exploits d(log Z_slice)/d(sigma) = <x^T A x>_sigma:

    log Z_slice(sigma) = log C(d, N_plus) + integral_0^sigma <x^T A x>_s ds

so a quadrature over chain estimates of the mean double-counted pair energy
<x^T A x>_s yields an absolute reference in exactly the convention the
neural estimate -1.89703 was produced in. The failure mode that matters is
a silent convention mismatch (sign, factor 2 sigma d, single- vs
double-counted adjacency); these tests pin every link against the exact
4x4 enumeration the gate already trusts.
"""

import numpy as np
import pytest
from experiments.constrained_hard_03.gate_4x4 import on_slice_free_energy_reference
from experiments.constrained_hard_03.slice_ti import (
    composite_simpson,
    lattice_energy_double_counted,
    quadrature_error,
    slice_ti_free_energy_per_site,
    uniform_slice_mean_energy,
)

from discrete_flow_sampler.diagnostics.metrics import (
    conditional_pmf_at_composition,
    enumerate_states,
    exact_log_probs,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _slice_setup(lattice_side, sigma):
    target = FixedCompositionIsingTarget(
        D=lattice_side, sigma=sigma, target_composition=0.5
    )
    d = lattice_side * lattice_side
    all_states = enumerate_states(d)
    log_pi = exact_log_probs(target, all_states)
    slice_states, log_p_cond = conditional_pmf_at_composition(
        all_states, log_pi, target.n_plus_target
    )
    return target, slice_states.float(), log_p_cond


def exact_mean_energy(lattice_side, sigma):
    """<x^T A x> under the exact slice Boltzmann law — the TI integrand oracle."""
    target, slice_states, log_p_cond = _slice_setup(lattice_side, sigma)
    energies = (slice_states @ target.A @ slice_states.T).diagonal()
    return float((log_p_cond.exp() * energies).sum())


def test_simpson_exact_on_cubics():
    # Composite Simpson integrates cubics exactly regardless of interval count.
    grid = np.linspace(0.0, 0.223, 17)
    values = 3.0 * grid**3 - 2.0 * grid**2 + grid - 5.0
    exact = 3.0 / 4.0 * 0.223**4 - 2.0 / 3.0 * 0.223**3 + 0.223**2 / 2.0 - 5.0 * 0.223
    assert composite_simpson(grid, values) == pytest.approx(exact, abs=1e-12)


def test_simpson_rejects_odd_interval_count():
    grid = np.linspace(0.0, 1.0, 4)  # 3 intervals: Simpson needs an even count
    with pytest.raises(ValueError):
        composite_simpson(grid, np.ones(4))


def test_quadrature_error_matches_monte_carlo():
    # Var(sum w_i Y_i) = sum w_i^2 SE_i^2 for independent point estimates.
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 0.10, 9)
    ses = np.linspace(0.02, 0.08, 9)
    base = np.sin(grid * 30.0)
    draws = np.array(
        [composite_simpson(grid, base + rng.normal(0.0, ses)) for _ in range(20000)]
    )
    assert quadrature_error(grid, ses) == pytest.approx(draws.std(), rel=0.05)


def test_uniform_slice_mean_energy_formula():
    # At sigma = 0 the slice law is uniform and E[x_i x_j] = -1/(d-1) for
    # i != j (Var of the fixed total magnetisation is zero), so
    # <x^T A x>_0 = -sum_ij A_ij/(d-1) = -4d/(d-1) on the double-counted torus.
    assert uniform_slice_mean_energy(4) == pytest.approx(-64.0 / 15.0)
    assert uniform_slice_mean_energy(8) == pytest.approx(-256.0 / 63.0)
    # against the enumeration at 4x4 (sigma -> 0 limit taken as sigma = 0 exactly)
    target, slice_states, _ = _slice_setup(4, 0.1)
    energies = (slice_states @ target.A @ slice_states.T).diagonal()
    assert float(energies.mean()) == pytest.approx(-64.0 / 15.0, abs=1e-5)


def test_lattice_energy_matches_target_convention():
    # The chain-side energy computation (np.roll on int8 grids) must equal
    # x^T A x under the target's adjacency — the double-counting guard.
    target, slice_states, _ = _slice_setup(4, 0.223)
    states_np = slice_states[:100].numpy().astype(np.int8)
    chain_side = lattice_energy_double_counted(states_np, 4)
    oracle = (slice_states[:100] @ target.A @ slice_states[:100].T).diagonal()
    np.testing.assert_allclose(chain_side, oracle.numpy(), atol=1e-4)


@pytest.mark.parametrize("sigma_target", [0.10, 0.223])
def test_ti_on_exact_integrand_recovers_enumeration_reference(sigma_target):
    # End-to-end convention test: TI with the exact <x^T A x>_sigma curve
    # (enumeration integrand, zero statistical error) must reproduce
    # on_slice_free_energy_reference at 4x4 to quadrature precision.
    # This is the test that catches any silent sign / 2-sigma-d / adjacency
    # convention mismatch in one shot.
    intervals = 16
    grid = np.linspace(0.0, sigma_target, intervals + 1)
    integrand = np.array(
        [
            uniform_slice_mean_energy(4) if s == 0.0 else exact_mean_energy(4, s)
            for s in grid
        ]
    )
    f_ti = slice_ti_free_energy_per_site(grid, integrand, lattice_side=4, n_plus=8)
    target, slice_states, _ = _slice_setup(4, sigma_target)
    f_ref = float(on_slice_free_energy_reference(target, slice_states))
    assert f_ti == pytest.approx(f_ref, abs=2e-4)


def test_known_gate_value_guard():
    # The 4x4 gate's s010 reference (-2.9804 per site) is a known
    # point; the enumeration reference must still say so, or the convention
    # this module targets has drifted.
    target, slice_states, _ = _slice_setup(4, 0.10)
    f_ref = float(on_slice_free_energy_reference(target, slice_states))
    assert f_ref == pytest.approx(-2.98043, abs=2e-4)
