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


def test_kawasaki_stationary_matches_exact_enum_d4():
    """A long chain reproduces the exact canonical conditional on the c=0.5
    slice. Compared over discrete energy LEVELS (not per-state), so it is not
    subject to the finite-N TVD floor (project_tvd_floor_at_low_n)."""
    from discrete_flow_sampler.diagnostics.metrics import (
        conditional_pmf_at_composition,
        enumerate_states,
        exact_log_probs,
    )

    D, sigma = 4, 0.37
    d = D * D
    target = IsingTarget(D=D, sigma=sigma, bias=0.0)

    states = enumerate_states(d)                        # (65536, 16) ±1
    log_pi = exact_log_probs(target, states)
    slice_states, log_pi_cond = conditional_pmf_at_composition(states, log_pi, d // 2)
    # Exact slice energies in float64 to match the chain's float64 accumulator
    # (base_log_prob is float32; σ·xᵀAx differs in low digits, breaking the
    # discrete-level binning below). A is integer-valued so this is exact.
    slice_f = slice_states.double()
    slice_energy = (sigma * (slice_f @ target.A.double() * slice_f).sum(-1)).numpy()
    p_cond = np.exp(log_pi_cond.numpy())

    levels = np.unique(np.round(slice_energy, 6))
    rounded = np.round(slice_energy, 6)
    exact_level_p = np.array([p_cond[rounded == lv].sum() for lv in levels])
    exact_mean_E = float((p_cond * slice_energy).sum())

    rng = np.random.default_rng(7)
    x = init_random_at_composition(d, 0.5, rng)
    energy_trace, _, _ = run_chain(x, D, sigma, 2_000_000, 7)
    samples = energy_trace[100_000::5]                  # burn-in + thin

    emp = np.round(samples, 6)
    emp_level_p = np.array([(emp == lv).mean() for lv in levels])

    assert abs(samples.mean() - exact_mean_E) < 0.15
    assert np.max(np.abs(emp_level_p - exact_level_p)) < 0.02
