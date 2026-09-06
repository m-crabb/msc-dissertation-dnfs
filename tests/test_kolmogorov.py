"""Tests for the Kolmogorov residual + loss (paper Eq. 7).

This is the highest-value test in the suite: with an analytic rate matrix
that satisfies Kolmogorov by construction, the residual must be zero to
floating-point precision. Catches a huge class of stage-1 bugs in the
residual formula, the neighbour construction, the ratio sign, and the
∂_t log Z_t handling -- all without needing a working trainer.

Tests pinned here:
    1) Trivial uniform target with R = 0 -> residual = 0 (floating-point).
    2) Random rate matrix on a real Ising target -> residual non-zero.
    3) loss(...) == residual(...).pow(2).mean().
"""

import torch

from discrete_flow_sampler.samplers.kolmogorov import loss, residual_general
from discrete_flow_sampler.targets.ising import IsingTarget


class ZeroRateModel:
    """R == 0 everywhere. Used for the trivial Kolmogorov-satisfying setup."""

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).float()


class TrivialUniformTarget:
    """rho == eta == uniform. log p_tilde_t and dt_log_p_tilde_t are zero
    on every state, for every t. Combined with R = 0, this is the only
    setting where Kolmogorov is trivially satisfied without doing real
    work -- so it isolates the residual formula from target/model bugs."""

    def __init__(self, D: int = 2):
        self.D = D
        self.d = D * D

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])

    def log_p_tilde_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])

    def dt_log_p_tilde_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])


def test_residual_zero_for_trivial_transport():
    """rho = eta, R = 0 -> dt_log_p_t = 0, all rate-matrix terms 0
    -> residual must be 0 to floating-point precision."""
    target = TrivialUniformTarget(D=2)
    model = ZeroRateModel()
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    t = torch.tensor([0.5, 0.7])
    dt_log_Zt = torch.tensor(0.0)  # uniform target -> log Z_t constant in t

    got = residual_general(x, t, dt_log_Zt, model, target)
    torch.testing.assert_close(got, torch.zeros(2), atol=1e-6, rtol=0)


def test_residual_nonzero_for_random_model():
    """Sanity: a random rate matrix on a real Ising target should NOT
    satisfy Kolmogorov -- residual must be non-zero. This guards against
    a vacuous `test_residual_zero_*` (e.g. residual that always returns 0)."""
    target = IsingTarget(D=2, sigma=0.1)

    class RandomModel:
        def __call__(self, x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            # Deterministic-but-non-trivial rate output.
            torch.manual_seed(0)
            return torch.rand_like(x_in).float()

    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    t = torch.tensor([0.5])
    dt_log_Zt = torch.tensor(0.1)

    got = residual_general(x, t, dt_log_Zt, RandomModel(), target)
    assert got.abs().item() > 1e-3, (
        f"random rate matrix unexpectedly satisfied Kolmogorov: residual={got}"
    )


def test_loss_is_mean_squared_residual():
    """loss is just mean of residual squared. Pinning this means downstream
    code can trust either name and get the same answer."""
    target = TrivialUniformTarget(D=2)
    model = ZeroRateModel()
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    t = torch.tensor([0.5, 0.7])
    dt_log_Zt = torch.tensor(0.0)

    expected = residual_general(x, t, dt_log_Zt, model, target).pow(2).mean()
    got = loss(x, t, dt_log_Zt, model, target)
    torch.testing.assert_close(got, expected)


from discrete_flow_sampler.samplers.kolmogorov import residual_lenet


class AnalyticLERateModel:
    """Locally equivariant 'rate matrix' that returns G == 0 everywhere.

    With G == 0, both [G]_+ and [-G]_+ are zero, so on a uniform target
    (TrivialUniformTarget where ∂_t log p_t = 0 and p_t(y)/p_t(x) = 1)
    Eq. (10) reduces to dt_log_p_t(x) - dt_log_Zt = 0. This isolates the
    Eq. (10) wiring from the model: a non-trivial leMLP is exercised in
    the next test below."""

    is_locally_equivariant: bool = True
    vocab_size: int = 2

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*x.shape, 2), dtype=x.dtype, device=x.device)


def test_residual_lenet_zero_for_trivial_transport():
    """Eq. (10) form on uniform target with zero G: residual must be 0."""
    target = TrivialUniformTarget(D=2)
    model = AnalyticLERateModel()
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    t = torch.tensor([0.5, 0.7])
    dt_log_Zt = torch.tensor(0.0)

    got = residual_lenet(x, t, dt_log_Zt, model, target)
    torch.testing.assert_close(got, torch.zeros(2), atol=1e-6, rtol=0)


def test_residual_lenet_nonzero_for_random_lemlp():
    """A randomly initialised LeMLPRateMatrix on a real Ising target
    should NOT satisfy Kolmogorov — guards against a vacuous
    `test_residual_lenet_zero_*`."""
    from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix

    target = IsingTarget(D=2, sigma=0.1)
    torch.manual_seed(0)
    model = LeMLPRateMatrix(d=4, vocab_size=2, hidden_dim=16, n_summands=2)

    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    t = torch.tensor([0.5])
    dt_log_Zt = torch.tensor(0.1)
    got = residual_lenet(x, t, dt_log_Zt, model, target)
    assert got.abs().item() > 1e-3, (
        f"random leMLP unexpectedly satisfied Kolmogorov: residual={got}"
    )


def test_loss_dispatches_on_is_locally_equivariant():
    """`loss(...)` must call residual_lenet when model.is_locally_equivariant
    is True, residual_general otherwise."""
    target = TrivialUniformTarget(D=2)
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    t = torch.tensor([0.5])
    dt_log_Zt = torch.tensor(0.0)

    le_model = AnalyticLERateModel()
    le_loss = loss(x, t, dt_log_Zt, le_model, target)

    non_le_model = ZeroRateModel()
    non_le_loss = loss(x, t, dt_log_Zt, non_le_model, target)

    expected_le = residual_lenet(x, t, dt_log_Zt, le_model, target).pow(2).mean()
    expected_non_le = (
        residual_general(x, t, dt_log_Zt, non_le_model, target).pow(2).mean()
    )

    torch.testing.assert_close(le_loss, expected_le)
    torch.testing.assert_close(non_le_loss, expected_non_le)
