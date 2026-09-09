"""Pins for the mchammer Kawasaki demo harness (kawasaki_4x4.py): the
atom->site mapping and the CE embedding must reproduce the
FixedCompositionIsingTarget energy (in differences, so constant offsets
cancel), and a short chain must conserve composition at every snapshot.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("icet")

from experiments.constrained_hard_03.probes.kawasaki_4x4 import (
    ising_cluster_expansion,
    run_chain,
    site_index_map,
    spins_from_symbols,
)

from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _random_half_half_symbols(rng, n_sites):
    symbols = ["Au"] * (n_sites // 2) + ["Ag"] * (n_sites - n_sites // 2)
    rng.shuffle(symbols)
    return symbols


@pytest.mark.parametrize("sigma", [0.10, 0.223])
def test_ce_energy_differences_match_target_log_prob(sigma):
    """Delta E_CE == -Delta log p across random slice configurations, pinning
    both the Au/Ag -> +1/-1 site mapping and the ECI embedding."""
    D = 4
    prim, _, ce = ising_cluster_expansion(sigma)
    supercell = prim.repeat((D, D, 1))
    index_map = site_index_map(supercell, D)
    target = FixedCompositionIsingTarget(
        D=D, sigma=sigma, target_composition=0.5, bias=0.0, device="cpu"
    )
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(4):
        symbols = _random_half_half_symbols(rng, D * D)
        supercell.set_chemical_symbols(symbols)
        energy_ce = ce.predict(supercell) * len(supercell)
        spins = torch.tensor(spins_from_symbols(symbols, index_map)).float()
        log_p = target.log_prob(spins.unsqueeze(0)).item()
        rows.append((energy_ce, log_p))
    # abs=1e-5: target.log_prob evaluates in fp32 (~4e-8 error at |E|~2.4);
    # a mapping/ECI bug would show as O(8*sigma) energy-quantum discrepancies.
    for (e_a, lp_a), (e_b, lp_b) in zip(rows[:-1], rows[1:]):
        assert (e_a - e_b) == pytest.approx(-(lp_a - lp_b), abs=1e-5)


def test_short_chain_conserves_composition_and_spin_domain():
    # run_chain grew its wall-clock pair in 0bf793f (the house tables price the
    # MCMC row per effective sample, and an MC sweep has no NFE analogue).
    spins, mctrials, setup_seconds, run_seconds = run_chain(
        D=4, sigma=0.10, seed=7, n_trial_steps=200, snapshot_interval=10
    )
    assert setup_seconds > 0 and run_seconds > 0
    assert set(np.unique(spins)) <= {-1.0, 1.0}
    np.testing.assert_array_equal(spins.sum(axis=1), np.zeros(len(spins)))
    assert mctrials[-1] <= 200 and len(mctrials) == len(spins)
