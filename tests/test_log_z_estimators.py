"""Tests for the ∂_t log Z_t estimator(s).

Stage 1: `naive_mc`, a plain MC average of `∂_t log p̃_t(x)` under x ~ p_t.
Stage 2: `control_variate`, the Eq. 8 estimator that subtracts the
Kolmogorov-derived control statistic.

Both estimators conform to the LogZEstimator protocol returning
`(estimate, modified_integrand)`. The protocol's modified_integrand is the
per-state vector that was averaged to produce the scalar estimate; in
Stage 1 it equals `∂_t log p̃_t` itself, in Stage 2 it diverges below by
the variance-reduction factor.

D=2 (16-state Ising) is small enough for exact enumeration of p_t, so
unbiasedness can be tested against the analytic ground truth
`∂_t log Z_t = E_{p_t}[∂_t log p̃_t(X)]`.
"""
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers.log_z_estimators import naive_mc
from discrete_flow_sampler.targets.ising import IsingTarget


def _exact_dt_log_Zt(target: IsingTarget, t_value: float) -> float:
    """Analytic ∂_t log Z_t for tiny problems via full enumeration."""
    n_sites = target.D * target.D
    states = enumerate_states(n_sites).float()
    n_states = states.shape[0]

    t_per_state = torch.full((n_states,), t_value)
    log_p_tilde = target.log_p_tilde_t(states, t_per_state)
    p_t = torch.softmax(log_p_tilde, dim=0)

    integrand = target.dt_log_p_tilde_t(states, t_per_state)
    return (p_t * integrand).sum().item()


def _draw_exact_pt_batch(target, t_value, n_samples):
    """Draw an exact multinomial batch from p_t for a small enumerable target.

    Reused by every estimator unbiasedness test — same batch construction,
    different estimator under test.
    """
    n_sites = target.D * target.D
    n_states = 2 ** n_sites
    states = enumerate_states(n_sites).float()
    log_p_tilde = target.log_p_tilde_t(states, torch.full((n_states,), t_value))
    p_t = torch.softmax(log_p_tilde, dim=0)
    sample_idx = torch.multinomial(p_t, num_samples=n_samples, replacement=True)
    return states[sample_idx], p_t, states


def test_naive_mc_unbiased_on_d2():
    """With ~10k samples drawn from p_t exactly, the MC estimate should
    match the analytic value to within 0.05 on the 16-state Ising at t=0.5.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    t_value = 0.5
    n_samples = 10_000
    x_batch, _, _ = _draw_exact_pt_batch(target, t_value, n_samples)
    t_batch = torch.full((n_samples,), t_value)

    estimate, _ = naive_mc(t_batch, x_batch, target, model=None)
    truth = _exact_dt_log_Zt(target, t_value)

    assert abs(estimate.item() - truth) < 5e-2, (
        f"naive_mc deviated from analytic dt_log_Zt: "
        f"estimate={estimate.item():.4f}, truth={truth:.4f}"
    )


def test_naive_mc_returns_estimate_and_integrand():
    """Output must be (scalar, (B,)) — the LogZEstimator protocol shape.
    Training.py unpacks both: scalar feeds the loss, integrand vector feeds
    the var_estimator_integrand mechanism column."""
    torch.manual_seed(1)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    x_batch = torch.randint(0, 2, (32, n_sites)).float() * 2 - 1
    t_batch = torch.full((32,), 0.3)
    estimate, modified_integrand = naive_mc(
        t_batch, x_batch, target, model=None
    )
    assert estimate.dim() == 0, (
        f"expected scalar estimate, got shape {tuple(estimate.shape)}"
    )
    assert modified_integrand.shape == (32,), (
        f"expected (32,) modified_integrand, "
        f"got {tuple(modified_integrand.shape)}"
    )


def test_control_variate_unbiased_on_d2():
    """Mirror of test_naive_mc_unbiased_on_d2 for control_variate (Eq. 8).

    The unbiasedness identity holds for ANY R_t — proof: the control
    statistic Σ_y R_t(x,y) p_t(y)/p_t(x) equals ∂_t log p_t(x) under
    Kolmogorov (paper Eq. 4, using Σ_y R_t(y,x) = 0). Subtracting it from
    ∂_t log p̃_t(x) leaves ∂_t log Z_t for every x. So a fresh-init small
    MLP suffices to test this — a buggy implementation will fail
    regardless of R_t quality.
    """
    from discrete_flow_sampler.samplers.log_z_estimators import (
        control_variate,
    )

    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    t_value = 0.5
    n_samples = 10_000
    n_sites = target.D * target.D
    x_batch, _, _ = _draw_exact_pt_batch(target, t_value, n_samples)
    t_batch = torch.full((n_samples,), t_value)

    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    model.eval()
    with torch.no_grad():
        estimate, _ = control_variate(t_batch, x_batch, target, model)
    truth = _exact_dt_log_Zt(target, t_value)

    assert abs(estimate.item() - truth) < 5e-2, (
        f"control_variate biased: estimate={estimate.item():.4f}, "
        f"truth={truth:.4f}"
    )


def test_control_variate_reduces_variance_vs_naive_mc():
    """K=200 paired replicates: same R_t, different batches each replicate.
    Empirical Var of the scalar estimate must satisfy
        Var[CV] < 0.5 * Var[naive_mc].
    The 0.5 factor is the sanity floor from Stage 2 plan §0.3 (paper
    Figure 1 suggests order-of-magnitude reduction in practice). The
    paired (CV, naive) variances feed the dissertation mechanism figure.
    """
    from discrete_flow_sampler.samplers.log_z_estimators import (
        control_variate,
    )

    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    t_value = 0.5
    n_samples = 256
    K = 200
    n_sites = target.D * target.D

    n_states = 2 ** n_sites
    states = enumerate_states(n_sites).float()
    log_p_tilde = target.log_p_tilde_t(states, torch.full((n_states,), t_value))
    p_t = torch.softmax(log_p_tilde, dim=0)

    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    model.eval()

    naive_estimates = []
    cv_estimates = []
    for replicate in range(K):
        torch.manual_seed(replicate + 100)
        sample_idx = torch.multinomial(
            p_t, num_samples=n_samples, replacement=True
        )
        x_batch = states[sample_idx]
        t_batch = torch.full((n_samples,), t_value)
        with torch.no_grad():
            naive_est, _ = naive_mc(t_batch, x_batch, target, model)
            cv_est, _ = control_variate(t_batch, x_batch, target, model)
        naive_estimates.append(naive_est.item())
        cv_estimates.append(cv_est.item())

    naive_var = torch.tensor(naive_estimates).var().item()
    cv_var = torch.tensor(cv_estimates).var().item()

    assert cv_var < 0.5 * naive_var, (
        f"control_variate did not reduce variance below the 0.5 floor: "
        f"naive_var={naive_var:.4f}, cv_var={cv_var:.4f}, "
        f"ratio={cv_var / naive_var:.3f}"
    )
