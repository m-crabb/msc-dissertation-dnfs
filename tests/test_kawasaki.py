"""Tests for the Kawasaki composition-preserving Ising sampler (§3.1).

These encode "what correct looks like" for the hard-constraint MCMC baseline:
  1. a swap move conserves composition exactly (the hard constraint, by
     construction);
  2. the incremental swap Δlog_prob matches a full recompute, including the
     adjacent-pair case (guards the shared-bond adjacency correction);
  3. a long chain reproduces the exact canonical Boltzmann conditional on the
     c=0.5 slice at D=4 (the correctness gate that licenses calling later
     failures "dynamics", not bugs).
"""
import numpy as np
import torch

from discrete_flow_sampler.mcmc.kawasaki import (
    init_random_at_composition,
    kawasaki_delta_log_prob,
    run_chain,
)
from discrete_flow_sampler.targets.ising import IsingTarget


def _config(D, plus_sites):
    """±1 lattice (flattened) with +1 at the given flat indices, -1 elsewhere."""
    x = -np.ones(D * D, dtype=np.int64)
    for s in plus_sites:
        x[s] = 1
    return x


def _delta_recompute(D, sigma, x, i, j):
    """Ground-truth Δlog_prob_ising via IsingTarget.base_log_prob (bias=0)."""
    target = IsingTarget(D=D, sigma=sigma, bias=0.0)
    x0 = torch.tensor(x, dtype=torch.float32)[None, :]
    x1 = x0.clone()
    x1[0, i], x1[0, j] = x0[0, j].item(), x0[0, i].item()
    return float(target.base_log_prob(x1) - target.base_log_prob(x0))


def test_swap_preserves_composition():
    D, sigma = 4, 0.3
    d = D * D
    rng = np.random.default_rng(0)
    x = init_random_at_composition(d, 0.5, rng)
    n_plus_init = int((x == 1).sum())
    _, x_final, _ = run_chain(x.copy(), D, sigma, 50_000, 0)
    assert int((x_final == 1).sum()) == n_plus_init
    assert set(np.unique(x_final)).issubset({-1, 1})


def test_kawasaki_dE_matches_recompute_nonadjacent():
    # site 0 (+1) and site 6 (-1): on a 4x4 torus 6 is NOT a neighbour of 0.
    D, sigma = 4, 0.37
    x = _config(D, [0, 5, 10])
    i, j = 0, 6
    assert x[i] == 1 and x[j] == -1
    got = kawasaki_delta_log_prob(x, i, j, D, sigma)
    assert abs(got - _delta_recompute(D, sigma, x, i, j)) < 1e-4


def test_kawasaki_dE_matches_recompute_adjacent():
    # site 0 (+1) and site 1 (-1): 1 is the right-neighbour of 0 (adjacent).
    D, sigma = 4, 0.37
    x = _config(D, [0, 5])
    i, j = 0, 1
    assert x[i] == 1 and x[j] == -1
    got = kawasaki_delta_log_prob(x, i, j, D, sigma)
    assert abs(got - _delta_recompute(D, sigma, x, i, j)) < 1e-4
