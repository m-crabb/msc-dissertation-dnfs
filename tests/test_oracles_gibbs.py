import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    exact_log_probs,
)
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget


@pytest.fixture
def target_d4():
    return IsingTarget(D=4, sigma=0.1)


def test_output_shape(target_d4):
    samples = gibbs_sample(target_d4, n_chains=8, n_sweeps=10)
    assert samples.shape == (8, 16)
    assert torch.all((samples == 1) | (samples == -1))


def test_record_energy_trace_shape(target_d4):
    """When `record_energy_every=K`, returns (final_spins, energy_trace) with
    trace shape (n_sweeps // K + 1, n_chains) — one record at t=0 plus one per
    Kth sweep. Used as the mixing diagnostic for the D=10 long-chain reference.
    """
    spins, trace = gibbs_sample(
        target_d4,
        n_chains=8,
        n_sweeps=20,
        record_energy_every=5,
    )
    assert spins.shape == (8, 16)
    assert trace.shape == (20 // 5 + 1, 8)
    assert torch.isfinite(trace).all()


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
    states = enumerate_states(D=4)  # 2^4 = 16 states
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


def test_constrained_matches_exact_on_small_lattice():
    """Active composition penalty must still sample the exact target.

    This pins the full constrained conditional, including the discrete +1/d
    correction, rather than only testing that the mean composition moves in the
    right direction.
    """
    torch.manual_seed(0)
    target = IsingTarget(
        D=2,
        sigma=0.1,
        target_composition=0.3,
        composition_penalty_strength=5.0,
    )
    states = enumerate_states(D=4)
    log_p_exact = exact_log_probs(target, states.float())
    p_exact = log_p_exact.exp()

    samples = gibbs_sample(target, n_chains=16384, n_sweeps=500)

    state_to_idx = {tuple(s.tolist()): i for i, s in enumerate(states)}
    counts = torch.zeros(len(states))
    for s in samples:
        counts[state_to_idx[tuple(s.long().tolist())]] += 1
    p_emp = counts / counts.sum()

    tv_distance = 0.5 * (p_emp - p_exact).abs().sum().item()
    assert tv_distance < 0.02


def test_constrained_chain_converges_to_target_composition():
    """Penalty-aware heat-bath: a chain on the soft-constrained target must
    concentrate composition near c_target. The unconstrained Ising at σ=0.1 is
    Z₂-symmetric ⇒ ⟨c₊⟩ = 0.5, so a penalty-blind sampler stays near 0.5 and
    fails this. With λ=50, d=16 the constrained distribution is tightly peaked
    (run composition_std ≈ 0.023), so the 2000-chain mean lands well within
    0.03 of c_target = 0.3.
    """
    torch.manual_seed(0)
    target = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.3,
        composition_penalty_strength=50.0,
    )
    samples = gibbs_sample(target, n_chains=2000, n_sweeps=300)
    composition = ((samples + 1.0) * 0.5).mean(dim=-1)
    assert abs(composition.mean().item() - 0.3) < 0.03


def test_penalty_inactive_matches_unconstrained():
    """strength=0 must reproduce the unconstrained sampler byte-for-byte:
    same global-RNG stream, same log-odds. Pins that the Ising path is
    untouched by the penalty branch.
    """
    torch.manual_seed(0)
    plain = IsingTarget(D=4, sigma=0.1)
    out_plain = gibbs_sample(plain, n_chains=64, n_sweeps=20)

    torch.manual_seed(0)
    inactive = IsingTarget(
        D=4,
        sigma=0.1,
        target_composition=0.3,
        composition_penalty_strength=0.0,
    )
    out_inactive = gibbs_sample(inactive, n_chains=64, n_sweeps=20)

    assert torch.equal(out_plain, out_inactive)
