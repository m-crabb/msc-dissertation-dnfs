"""Tests for the ∂_t log Z_t estimator(s).

Stage 1 has one estimator: `naive_mc`, a plain Monte Carlo average of the
integrand ∂_t log p̃_t(x) under x ~ p_t.

The test pins unbiasedness on D=2 (16-state Ising), where we can:
    1) enumerate p_t exactly,
    2) draw a large multinomial batch from it,
    3) compare the estimator to the analytic ground truth
       ∂_t log Z_t = E_{p_t}[∂_t log p̃_t(X)].

We do NOT test variance here -- the whole point of Stage 1 is that variance
is high; that's the failure mode Stage 2 will address.
"""
import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.samplers.log_z_estimators import naive_mc
from discrete_flow_sampler.targets.ising import IsingTarget


def _exact_dt_log_Zt(target: IsingTarget, t_value: float) -> float:
    """Analytic ∂_t log Z_t for tiny problems via full enumeration.

    Uses the identity (see file docstring of log_z_estimators.py):
        ∂_t log Z_t = E_{p_t}[∂_t log p̃_t(X)].
    """
    n_sites = target.D * target.D
    states = enumerate_states(n_sites).float()
    n_states = states.shape[0]

    t_per_state = torch.full((n_states,), t_value)
    log_p_tilde = target.log_p_tilde_t(states, t_per_state)
    p_t = torch.softmax(log_p_tilde, dim=0)

    integrand = target.dt_log_p_tilde_t(states, t_per_state)
    return (p_t * integrand).sum().item()


def test_naive_mc_unbiased_on_d2():
    """With ~10k samples drawn from p_t exactly, the MC estimate should
    match the analytic value to within 0.05 on the 16-state Ising at t=0.5.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)  # 4 sites, 16 states
    t_value = 0.5
    n_sites = target.D * target.D  # 4
    n_states = 2 ** n_sites          # 16
    n_samples = 10_000

    # Build p_t over the full state space and draw an exact multinomial batch.
    states = enumerate_states(n_sites).float()
    log_p_tilde = target.log_p_tilde_t(states, torch.full((n_states,), t_value))
    p_t = torch.softmax(log_p_tilde, dim=0)
    sample_idx = torch.multinomial(p_t, num_samples=n_samples, replacement=True)
    x_batch = states[sample_idx]  # (n_samples, n_sites)

    t_batch = torch.full((n_samples,), t_value)
    estimate = naive_mc(t_batch, x_batch, target, model=None)
    truth = _exact_dt_log_Zt(target, t_value)

    assert abs(estimate.item() - truth) < 5e-2, (
        f"naive_mc deviated from analytic dt_log_Zt: "
        f"estimate={estimate.item():.4f}, truth={truth:.4f}"
    )


def test_naive_mc_returns_scalar():
    """Output must be a 0-dim tensor; downstream loss code will treat it
    as a precomputed scalar constant."""
    torch.manual_seed(1)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    x_batch = torch.randint(0, 2, (32, n_sites)).float() * 2 - 1
    t_batch = torch.full((32,), 0.3)
    out = naive_mc(t_batch, x_batch, target, model=None)
    assert out.dim() == 0, f"expected scalar, got shape {tuple(out.shape)}"
