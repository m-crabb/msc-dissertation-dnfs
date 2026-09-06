"""Synthetic lookup checks; no trained checkpoints or archived output needed."""

import itertools

import pytest
import torch
from experiments.constrained_hard_03 import analysis_local_field_regression as hard
from experiments.constrained_soft_02 import analysis_local_field_regression as soft

from discrete_flow_sampler.targets.ising import IsingTarget


def test_soft_lookup_separates_all_site_neighbour_patterns():
    adjacency = IsingTarget(D=4, sigma=0.1, device="cpu").A
    states, sites = [], []
    for site in range(16):
        neighbours = adjacency[site].nonzero().flatten()
        for spins in itertools.product([-1.0, 1.0], repeat=4):
            state = torch.full((16,), -1.0)
            state[neighbours] = torch.tensor(spins)
            states.append(state)
            sites.append(site)
    states = torch.stack(states)
    local, _ = soft.local_lookup_keys(states, adjacency)
    selected = local.reshape(-1, 16)[torch.arange(256), sites]
    assert selected.unique().numel() == 256


def test_soft_lookup_count_keeps_sites_distinct_and_excludes_hole():
    adjacency = IsingTarget(D=4, sigma=0.1, device="cpu").A
    states = torch.full((2, 16), -1.0)
    states[0, 4] = 1.0  # site 0's old local key: 0 * 16 + 2**4 = 16
    states[1, 3] = 1.0  # site 1's old local key: 1 * 16 + 0 = 16
    local, count = soft.local_lookup_keys(states, adjacency)
    assert local[0] != local[17]
    assert count[0] != count[17]  # both have one hole-excluded up-spin

    states[:, 0] *= -1
    local_flipped, count_flipped = soft.local_lookup_keys(states, adjacency)
    assert torch.equal(local.reshape(2, 16)[:, 0], local_flipped.reshape(2, 16)[:, 0])
    assert torch.equal(count.reshape(2, 16)[:, 0], count_flipped.reshape(2, 16)[:, 0])


@pytest.mark.parametrize("module", [soft, hard])
def test_lookup_r_squared_penalises_held_out_bias(module):
    # Fit predicts [0, 2], truth is [1, 3]: SSE=2, SST=2, R²=0.
    # Centring the residual would incorrectly return R²=1.
    target = torch.tensor([0.0, 2.0, 1.0, 3.0])
    keys = torch.tensor([0, 1, 0, 1])
    fit_mask = torch.tensor([True, True, False, False])
    result = module.lookup_r_squared(target, keys, fit_mask)
    assert result[0] == pytest.approx(0.0)
    assert torch.equal(result[-1], torch.ones(2))


@pytest.mark.parametrize("module", [soft, hard])
def test_lookup_unseen_cells_use_fit_mean_only(module):
    # The unseen-cell prediction is the fit mean 2, not the full-data mean 4.
    target = torch.tensor([1.0, 3.0, 5.0, 7.0])
    keys = torch.tensor([0, 0, 1, 2])
    fit_mask = torch.tensor([True, True, False, False])
    result = module.lookup_r_squared(target, keys, fit_mask)
    assert result[0] == pytest.approx(-16.0)  # 1 - (9 + 25) / 2
    assert torch.equal(result[-1], torch.tensor([3.0, 5.0]))


@pytest.mark.parametrize("module", [soft, hard])
def test_lookup_exact_local_signal_scores_one(module):
    target = torch.tensor([1.0, 3.0, 1.0, 3.0])
    keys = torch.tensor([0, 1, 0, 1])
    fit_mask = torch.tensor([True, True, False, False])
    result = module.lookup_r_squared(target, keys, fit_mask)
    assert result[0] == pytest.approx(1.0)
    assert torch.count_nonzero(result[-1]) == 0
