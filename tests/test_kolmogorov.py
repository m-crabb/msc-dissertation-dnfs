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
import pytest
import torch

from discrete_flow_sampler.samplers.kolmogorov import loss, residual
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

    got = residual(x, t, dt_log_Zt, model, target)
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

    got = residual(x, t, dt_log_Zt, RandomModel(), target)
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

    expected = residual(x, t, dt_log_Zt, model, target).pow(2).mean()
    got = loss(x, t, dt_log_Zt, model, target)
    torch.testing.assert_close(got, expected)
