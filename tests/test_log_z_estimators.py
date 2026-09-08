"""Tests for `compute_c_t_grid` -- the outer-step ∂_t log Z_t estimator.

Paper reference: Algorithm 1 line 4 (App. C.1). Per outer step we compute
a scalar c_t per time-grid point t_k by averaging the per-state integrand
over M outer-batch samples. Two modes:

  - naive_mc:        integrand = ∂_t log p̃_t(x) only (target-only). For
                     stage_0 / stage_1 ablations.
  - control_variate: integrand = ξ_t(x; R_t) per paper Eq. 8 (target term
                     minus inflow/outflow correction). For paper-faithful
                     stage_0_cv / stage_2 runs.

Both modes share the outer-step contract: torch.no_grad always, output
detached from the autograd graph (so the inner-loop loss `(ξ_θ − c_t)²`
treats c_t as a frozen target -- the stop-gradient `c_sg` of paper line 4).

D=2 (16-state Ising) is small enough for exact enumeration of p_t, so
unbiasedness is checked against the analytic ground truth
`∂_t log Z_t = E_{p_t}[∂_t log p̃_t(X)]`.
"""

import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.mlp import MLPRateMatrix
from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid
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
    """Draw an exact multinomial batch from p_t for a small enumerable target."""
    n_sites = target.D * target.D
    n_states = 2**n_sites
    states = enumerate_states(n_sites).float()
    log_p_tilde = target.log_p_tilde_t(states, torch.full((n_states,), t_value))
    p_t = torch.softmax(log_p_tilde, dim=0)
    sample_idx = torch.multinomial(p_t, num_samples=n_samples, replacement=True)
    return states[sample_idx], p_t, states


def test_compute_c_t_grid_shapes():
    """`compute_c_t_grid` returns (c_t_grid, integrand_per_t) of shapes
    (T,) and (T, M) respectively. T = len(t_grid), M = number of outer-
    batch samples per time slot."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    n_grid = 5
    outer_batch = 16

    t_grid = torch.linspace(0.0, 1.0, n_grid)
    x_traj = torch.randint(0, 2, (n_grid, outer_batch, n_sites)).float() * 2 - 1
    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)

    for mode in ("naive_mc", "control_variate"):
        c_t_grid, integrand = compute_c_t_grid(
            t_grid,
            x_traj,
            target,
            model,
            mode=mode,
        )
        assert c_t_grid.shape == (n_grid,), (
            f"mode={mode}: expected (T,), got {tuple(c_t_grid.shape)}"
        )
        assert integrand.shape == (n_grid, outer_batch), (
            f"mode={mode}: expected (T, M), got {tuple(integrand.shape)}"
        )


def test_compute_c_t_grid_naive_mc_unbiased_on_d2():
    """Single time slot at t=0.5 with 10k exact-p_t samples: the per-slot
    naive_mc c_t should match the analytic ∂_t log Z_t to within 0.05 on
    the 16-state Ising. This pins the average-over-M reduction to the
    correct dimension and the integrand formula."""
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    t_value = 0.5
    n_samples = 10_000

    x_batch, _, _ = _draw_exact_pt_batch(target, t_value, n_samples)
    t_grid = torch.tensor([t_value])
    x_traj = x_batch.unsqueeze(0)  # (T=1, M=n_samples, D)

    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    c_t_grid, _ = compute_c_t_grid(
        t_grid,
        x_traj,
        target,
        model,
        mode="naive_mc",
    )
    truth = _exact_dt_log_Zt(target, t_value)

    assert abs(c_t_grid[0].item() - truth) < 5e-2, (
        f"naive_mc c_t deviated from analytic dt_log_Zt: "
        f"c_t={c_t_grid[0].item():.4f}, truth={truth:.4f}"
    )


def test_compute_c_t_grid_control_variate_unbiased_on_d2():
    """Mirror of naive_mc unbiasedness test for control_variate.

    The unbiasedness identity holds for any R_t under the rate-matrix
    algebra Σ_y R(y,x) = 0 (which our per-site-flip parameterisation
    enforces by construction): subtracting the control statistic
    Σ_y R(x,y) p_t(y)/p_t(x) = ∂_t log p_t(x) from ∂_t log p̃_t(x) leaves
    ∂_t log Z_t for every x. So a fresh-init random MLP suffices.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    t_value = 0.5
    n_samples = 10_000

    x_batch, _, _ = _draw_exact_pt_batch(target, t_value, n_samples)
    t_grid = torch.tensor([t_value])
    x_traj = x_batch.unsqueeze(0)

    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    c_t_grid, _ = compute_c_t_grid(
        t_grid,
        x_traj,
        target,
        model,
        mode="control_variate",
    )
    truth = _exact_dt_log_Zt(target, t_value)

    assert abs(c_t_grid[0].item() - truth) < 5e-2, (
        f"control_variate c_t biased: c_t={c_t_grid[0].item():.4f}, truth={truth:.4f}"
    )


def test_compute_c_t_grid_outputs_are_detached():
    """Stop-gradient property: paper Algorithm 1 line 4 computes c_t with
    R_t^{θ_sg}, the stop-grad model. We enforce this by always wrapping
    the computation in torch.no_grad, so output tensors don't carry grad
    state -- the inner-step loss `(ξ_θ − c_t)²` then has gradient flow
    only through ξ_θ, which is the uncentred form the paper derives.
    """
    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    # Sanity: model parameters are trainable.
    assert any(p.requires_grad for p in model.parameters())

    t_grid = torch.linspace(0.0, 1.0, 4)
    x_traj = torch.randint(0, 2, (4, 8, n_sites)).float() * 2 - 1

    for mode in ("naive_mc", "control_variate"):
        c_t_grid, integrand = compute_c_t_grid(
            t_grid,
            x_traj,
            target,
            model,
            mode=mode,
        )
        assert not c_t_grid.requires_grad, (
            f"mode={mode}: c_t_grid must be detached from autograd graph"
        )
        assert not integrand.requires_grad, (
            f"mode={mode}: integrand must be detached from autograd graph"
        )


def test_compute_c_t_grid_unknown_mode_raises():
    """Unknown mode is a programming error -- fail loud rather than
    silently default."""
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    model = MLPRateMatrix(d=n_sites, hidden_dim=8, n_layers=1)
    t_grid = torch.tensor([0.5])
    x_traj = torch.randint(0, 2, (1, 4, n_sites)).float() * 2 - 1

    with pytest.raises(ValueError, match="mode"):
        compute_c_t_grid(t_grid, x_traj, target, model, mode="bogus")


def test_control_variate_reduces_variance_vs_naive_mc():
    """K=200 paired replicates after brief naive_mc pretraining.

    The variance-reduction identity (paper Eq. 8) holds asymptotically as
    R_t approaches Kolmogorov-satisfying; with a fresh-init random MLP
    the control statistic is uncorrelated with the integrand and adds
    variance instead. So we train ~1000 inner-step-equivalents at D=2
    using `compute_c_t_grid` in single-slot naive_mc mode + the existing
    kolmogorov_loss to bring R_t into the regime where the CV variance-
    floor (ratio < 0.5 vs naive) actually applies. This test is the
    unit-test analog of the dissertation's Stage 2 mechanism figure.
    """
    from discrete_flow_sampler.samplers.ctmc import sample_ctmc
    from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss

    torch.manual_seed(0)
    target = IsingTarget(D=2, sigma=0.1)
    n_sites = target.D * target.D
    t_value = 0.5

    # Brief training so R_t is non-random.
    model = MLPRateMatrix(d=n_sites, hidden_dim=32, n_layers=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_train_steps, train_batch_size, n_euler_steps = 1_000, 64, 30
    for _ in range(n_train_steps):
        t_step = torch.rand(1).item()
        time_grid = torch.linspace(0.0, t_step, n_euler_steps)
        x_init = torch.randint(0, 2, (train_batch_size, n_sites)).float() * 2 - 1
        with torch.no_grad():
            x_train = sample_ctmc(model, x_init, time_grid)
        # Single-slot c_t under naive_mc: the integrand is purely target,
        # so c_t is constant in θ -- we only need the kolmogorov_loss to
        # drive ∂_θ. Mimics the paper's outer-step structure degenerately.
        t_grid_step = torch.tensor([t_step])
        x_traj_step = x_train.unsqueeze(0)
        c_t_grid, _ = compute_c_t_grid(
            t_grid_step,
            x_traj_step,
            target,
            model,
            mode="naive_mc",
        )
        c_t_per_state = c_t_grid[0].expand(train_batch_size)  # (B,)
        t_batch_train = torch.full((train_batch_size,), t_step)
        loss_val = kolmogorov_loss(
            x_train,
            t_batch_train,
            c_t_per_state,
            model,
            target,
        )
        optimiser.zero_grad()
        loss_val.backward()
        optimiser.step()
    model.eval()

    # Variance comparison at the trained R_t. K replicates, each is a
    # single-time-slot c_t computation.
    n_states = 2**n_sites
    states = enumerate_states(n_sites).float()
    log_p_tilde = target.log_p_tilde_t(states, torch.full((n_states,), t_value))
    p_t = torch.softmax(log_p_tilde, dim=0)

    n_replicates, n_samples = 200, 256
    naive_estimates, cv_estimates = [], []
    for replicate in range(n_replicates):
        torch.manual_seed(replicate + 100)
        sample_idx = torch.multinomial(
            p_t,
            num_samples=n_samples,
            replacement=True,
        )
        x_batch = states[sample_idx]
        t_grid_one = torch.tensor([t_value])
        x_traj_one = x_batch.unsqueeze(0)
        naive_c_t, _ = compute_c_t_grid(
            t_grid_one,
            x_traj_one,
            target,
            model,
            mode="naive_mc",
        )
        cv_c_t, _ = compute_c_t_grid(
            t_grid_one,
            x_traj_one,
            target,
            model,
            mode="control_variate",
        )
        naive_estimates.append(naive_c_t[0].item())
        cv_estimates.append(cv_c_t[0].item())

    naive_var = torch.tensor(naive_estimates).var().item()
    cv_var = torch.tensor(cv_estimates).var().item()

    assert cv_var < 0.5 * naive_var, (
        f"control_variate did not reduce variance below the 0.5 floor: "
        f"naive_var={naive_var:.4f}, cv_var={cv_var:.4f}, "
        f"ratio={cv_var / naive_var:.3f}"
    )
