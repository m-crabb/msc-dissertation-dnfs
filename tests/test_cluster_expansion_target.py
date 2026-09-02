"""What correct looks like for the binary cluster-expansion target.

The target evaluates E(s) = J_0 + sum_k c_k sum_tuples prod s on a periodic
cell exported by `experiments/alloy_ce/export_binary_expansion.py`, and hands
the samplers closed-form flip and swap energy changes. Three things must hold:

1. It reproduces the fitting library's energies on the reference
   configurations stored in the JSON (icet for the square toy, CLEASE for the
   MetaDNS Cu-Au cell) -- the whole reason the export exists.
2. The closed-form swap change equals the brute-force difference obtained by
   materialising the swapped state, for every pair, on random states off and
   on the composition slice; likewise the single-flip change.
3. Handed the Ising torus as a pair-only expansion it is byte-for-byte the
   existing FixedCompositionIsingTarget: same log-density, same swap ratios,
   same slice constant. That pins the beta = 2 sigma convention so the
   free-energy estimator returns F/d in the expansion's own energy units.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import free_energy_lb_estimate
from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    ClusterExpansionTarget,
    FixedCompositionClusterExpansionTarget,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget, IsingTarget

DATA = Path(__file__).resolve().parents[1] / "data" / "ce"
SPECS = ["square_cuau_4x4.json", "cuau_fcc_2x2x4.json", "square_cuau_8x8.json"]
K_B_EV = 8.617333262e-5
BETA_500K = 1.0 / (K_B_EV * 500.0)


def _random_state(n_sites, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(n, n_sites, generator=g) < 0.5, 1.0, -1.0)


# --- 1. the oracle ------------------------------------------------------------


@pytest.mark.parametrize("name", SPECS)
def test_energy_matches_the_fitting_library(name):
    spec = BinaryExpansionSpec.from_json(DATA / name)
    ref = json.loads((DATA / name).read_text())["reference"]
    spins = torch.tensor(ref["spins"], dtype=torch.float64)
    energies = spec.energy(spins)
    assert torch.allclose(
        energies, torch.tensor(ref["energies"], dtype=torch.float64), atol=1e-9, rtol=0
    )


def test_log_prob_is_minus_beta_energy():
    spec = BinaryExpansionSpec.from_json(DATA / "cuau_fcc_2x2x4.json")
    target = ClusterExpansionTarget(spec, beta=BETA_500K)
    x = _random_state(spec.n_sites, 5)
    assert torch.allclose(target.log_prob(x), -BETA_500K * spec.energy(x).float(), atol=1e-4)
    assert target.d == spec.n_sites
    assert target.sigma == pytest.approx(BETA_500K / 2)


# --- 2. closed forms ----------------------------------------------------------


@pytest.mark.parametrize("name", SPECS[:2])
def test_swap_change_matches_materialised_swaps(name):
    spec = BinaryExpansionSpec.from_json(DATA / name)
    x = _random_state(spec.n_sites, 4, seed=3).double()
    closed = spec.swap_energy_change(x)  # (B, d, d)
    e0 = spec.energy(x)
    for i in range(spec.n_sites):
        for j in range(spec.n_sites):
            y = x.clone()
            y[:, i], y[:, j] = x[:, j], x[:, i]
            assert torch.allclose(closed[:, i, j], spec.energy(y) - e0, atol=1e-9)


def test_flip_change_matches_materialised_flips():
    spec = BinaryExpansionSpec.from_json(DATA / "cuau_fcc_2x2x4.json")
    x = _random_state(spec.n_sites, 4, seed=4).double()
    closed = spec.flip_energy_change(x)  # (B, d)
    e0 = spec.energy(x)
    for i in range(spec.n_sites):
        y = x.clone()
        y[:, i] = -x[:, i]
        assert torch.allclose(closed[:, i], spec.energy(y) - e0, atol=1e-9)


def test_fixed_composition_swap_log_ratio_is_the_generic_fallback():
    spec = BinaryExpansionSpec.from_json(DATA / "cuau_fcc_2x2x4.json")
    target = FixedCompositionClusterExpansionTarget(spec, beta=BETA_500K, target_composition=0.25)
    x = target.sample_base(6, device="cpu")
    target.assert_on_manifold(x)
    t = torch.tensor([0.0, 0.3, 0.7, 1.0, 0.5, 0.9])
    pairs = upper_tri_pairs(spec.n_sites, device="cpu")
    closed = target.swap_log_ratio(x, t, pairs)
    generic = IsingTarget.swap_log_ratio(target, x, t, pairs)
    assert torch.allclose(closed, generic, atol=1e-4)


def test_slice_base_and_constant():
    spec = BinaryExpansionSpec.from_json(DATA / "cuau_fcc_2x2x4.json")
    target = FixedCompositionClusterExpansionTarget(spec, beta=BETA_500K, target_composition=0.25)
    x = target.sample_base(200, device="cpu")
    assert torch.all(((x + 1) / 2).sum(1) == 4)
    assert target.base_log_eta(x)[0].item() == pytest.approx(-math.log(math.comb(16, 4)))
    with pytest.raises(ValueError):
        FixedCompositionClusterExpansionTarget(spec, beta=BETA_500K, target_composition=0.3)


# --- 3. Ising is the pair-only special case -----------------------------------


def test_ising_torus_as_an_expansion_is_the_existing_target():
    side, sigma, composition = 4, 0.3, 0.5
    ising = FixedCompositionIsingTarget(D=side, sigma=sigma, target_composition=composition)
    # log p = sigma x^T A x = 2 sigma sum_<ij> s_i s_j, so E = -sum_<ij> s_i s_j
    # (one unit of coupling per undirected edge) at beta = 2 sigma.
    edges = [(i, j) for i in range(side * side) for j in range(i + 1, side * side)
             if ising.A[i, j] > 0]
    spec = BinaryExpansionSpec(
        n_sites=side * side, constant=0.0,
        terms=[{"order": 2, "coefficient": -1.0, "tuples": edges}],
        nearest_neighbour_pairs=edges,
    )
    ce = FixedCompositionClusterExpansionTarget(spec, beta=2 * sigma, target_composition=composition)
    x = ising.sample_base(8, device="cpu")
    t = torch.rand(8)
    pairs = upper_tri_pairs(side * side, device="cpu")
    assert torch.allclose(ce.log_prob(x), ising.log_prob(x), atol=1e-5)
    assert torch.allclose(ce.log_p_tilde_t(x, t), ising.log_p_tilde_t(x, t), atol=1e-5)
    assert torch.allclose(ce.swap_log_ratio(x, t, pairs), ising.swap_log_ratio(x, t, pairs), atol=1e-5)
    assert torch.equal(ce.A, ising.A)
    # and the free-energy estimator convention carries over unchanged
    log_w = torch.randn(50)
    assert free_energy_lb_estimate(log_w, ce.sigma, ce.d) == pytest.approx(
        free_energy_lb_estimate(log_w, ising.sigma, ising.d)
    )


def test_set_sigma_rescales_the_temperature():
    spec = BinaryExpansionSpec.from_json(DATA / "square_cuau_4x4.json")
    target = ClusterExpansionTarget(spec, beta=10.0)
    x = _random_state(spec.n_sites, 3)
    before = target.log_prob(x)
    target.set_sigma(2.5)  # beta = 5
    assert torch.allclose(target.log_prob(x), before / 2, atol=1e-4)
