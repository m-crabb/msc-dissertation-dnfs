import torch
import pytest

from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states, exact_log_probs,
)


@pytest.fixture
def target_d4():
    return IsingTarget(D=4, sigma=0.1)


def test_output_shape(target_d4):
    samples = gibbs_sample(target_d4, n_chains=8, n_sweeps=10)
    assert samples.shape == (8, 16)
    assert torch.all((samples == 1) | (samples == -1))


def test_z2_invariant_at_zero_bias(target_d4):
    """At zero bias and finite β, magnetisation distribution is symmetric
    around 0. Run many short chains, M = sum(x), assert mean(M) ≈ 0."""
    torch.manual_seed(0)
    samples = gibbs_sample(target_d4, n_chains=2000, n_sweeps=50)
    magnetisation = samples.sum(dim=-1).float()
    assert magnetisation.mean().abs().item() < 0.5


def test_matches_exact_on_small_lattice():
    """Long Gibbs run on D=2 (4 sites, 16 states, periodic torus) must
    match the exact enumerated distribution to within total-variation
    distance < 0.02. D=2 keeps the state space small enough that
    1024-scale sample budgets can validate to a tight threshold; for D=4
    the budget needed for TVD<0.02 is ~|states|/ε² ≈ 2·10⁸, infeasible.

    TVD is computed inline (`½ Σ |p_emp - p_exact|`) rather than via a
    diagnostics import — it is used here only as a unit-test correctness
    check for the Gibbs sampler, not as a paper-headline metric.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    states = enumerate_states(D=4)             # 2^4 = 16 states
    log_p_exact = exact_log_probs(target, states.float())
    p_exact = log_p_exact.exp()

    samples = gibbs_sample(target, n_chains=8192, n_sweeps=200)

    state_to_idx = {tuple(s.tolist()): i for i, s in enumerate(states)}
    counts = torch.zeros(len(states))
    for s in samples:
        counts[state_to_idx[tuple(s.long().tolist())]] += 1
    p_emp = counts / counts.sum()

    tv_distance = 0.5 * (p_emp - p_exact).abs().sum().item()
    assert tv_distance < 0.02
