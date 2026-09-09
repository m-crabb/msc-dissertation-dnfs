"""The alloy reference chains agree with exact enumeration on the 16-site cell.

At 1200 K the free-composition Metropolis chain must reproduce the exact
composition marginal's mean and the mean energy; the Kawasaki chain at
x_Au = 0.25 must stay on the slice and reproduce the exact slice energy; and
the beta-ladder thermodynamic integration must return the exact canonical
free energy to within its own error. These are the gates that make the
64-site references trustworthy where nothing can be enumerated.
"""

import itertools
import math
from pathlib import Path

import pytest
import torch
from experiments.alloy_ce.probes.reference_chain import (
    K_B_EV,
    beta_ladder_free_energy,
    build_target,
    observables,
    run_chains,
)

from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

SPEC = BinaryExpansionSpec.from_json(
    Path(__file__).resolve().parents[1] / "data" / "ce" / "cuau_fcc_2x2x4.json"
)
STATES = torch.tensor(
    list(itertools.product([-1.0, 1.0], repeat=16)), dtype=torch.float64
)
ENERGY = SPEC.energy(STATES)
N_AU = ((STATES + 1) / 2).sum(1)


def _exact(temperature_K, composition=None):
    beta = 1.0 / (K_B_EV * temperature_K)
    mask = (
        N_AU == round(composition * 16)
        if composition is not None
        else torch.ones_like(N_AU, dtype=torch.bool)
    )
    log_w = -beta * ENERGY[mask]
    p = torch.softmax(log_w, 0)
    return {
        "energy_per_site": float((p * ENERGY[mask]).sum()) / 16,
        "composition": float((p * N_AU[mask]).sum()) / 16,
        "free_energy_per_site_eV": -float(torch.logsumexp(log_w, 0)) / beta / 16,
    }


def test_free_chain_matches_enumeration_at_1200K():
    target = build_target(SPEC, 1200.0)
    states, _ = run_chains(
        target,
        n_chains=64,
        burn_in_sweeps=200,
        n_records=100,
        thin_sweeps=5,
        seed=0,
        canonical=False,
    )
    obs = observables(target, states)
    exact = _exact(1200.0)
    for key in ("energy_per_site", "composition"):
        se = float(obs[key].std(unbiased=True) / math.sqrt(64))
        assert abs(float(obs[key].mean()) - exact[key]) < 4 * se + 1e-4, key


def test_kawasaki_chain_stays_on_the_slice_and_matches_enumeration():
    target = build_target(SPEC, 800.0, composition=0.25, canonical=True)
    states, _ = run_chains(
        target,
        n_chains=64,
        burn_in_sweeps=200,
        n_records=100,
        thin_sweeps=5,
        seed=1,
        canonical=True,
    )
    assert torch.all(((states + 1) / 2).sum(-1) == 4)
    obs = observables(target, states)
    exact = _exact(800.0, composition=0.25)
    se = float(obs["energy_per_site"].std(unbiased=True) / math.sqrt(64))
    assert (
        abs(float(obs["energy_per_site"].mean()) - exact["energy_per_site"])
        < 4 * se + 1e-4
    )


def test_beta_ladder_recovers_the_exact_canonical_free_energy():
    result = beta_ladder_free_energy(
        SPEC,
        800.0,
        composition=0.25,
        canonical=True,
        n_grid=9,
        n_chains=64,
        burn_in_sweeps=100,
        n_records=40,
        thin_sweeps=3,
        seed=2,
    )
    exact = _exact(800.0, composition=0.25)["free_energy_per_site_eV"]
    tolerance = 4 * result["se"] + abs(result["half_grid_shift"]) + 2e-4
    assert result["free_energy_per_site_eV"] == pytest.approx(exact, abs=tolerance)
