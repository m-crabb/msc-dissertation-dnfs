"""Ising target distribution. Paper Eq. (11): p(x) ∝ exp(x^T J x + b · Σx). """

import math

import torch
from torch import Tensor


class IsingTarget:
    """Periodic-boundary DxD Ising lattice with annealing path.

    Target distribution (paper Eq. 11):

        p(x) ∝ exp( x^T J x + bias · Σ_i x_i ),    x ∈ {-1, +1}^d, d = D²

    where J = sigma · A_D and A_D is the adjacency matrix of the DxD grid
    with periodic boundaries (the lattice is a torus, no edge effects).
    The convention is that A_D is symmetric with zeros on the diagonal,
    each nearest-neighbour pair {i, j} contributing entry sigma in BOTH
    A_D[i, j] and A_D[j, i] — so the quadratic form x^T J x picks up
    each edge twice. This matches the paper's igraph-based construction.

    Annealing path (paper Eq. 4):

        log p̃_t(x) = (1 - t) · log η(x) + t · log p(x)

    with η = uniform on {-1, +1}^d (so log η ≡ -d · log 2, constant in x)
    and log p(x) = log_prob(x) - log Z. Working with the unnormalised
    log_prob is fine because the constant log Z cancels in derivatives.

    Because the path is linear in log, the time-derivative is path-position
    independent:

        ∂_t log p̃_t(x) = log p(x) - log η(x)         [t-independent]

    This is the term DNFS estimates the expectation of (under p̃_t) to form
    the ∂_t log Z_t signal in the Kolmogorov-residual training loss.
    """

    def __init__(
        self,
        D: int,
        sigma: float,
        bias: float = 0.0,
        device: torch.device | str = "cpu",
    ):
        self.D = D
        self.d = D * D
        self.sigma = sigma
        self.bias = bias
        self.device = torch.device(device)

        A = torch.zeros((self.d, self.d), device=self.device)

        for r in range(self.D):
            for c in range(self.D):
                i = r * self.D + c                          # (r, c)             -> flat
                right = r * self.D + (c + 1) % self.D       # (r, (c+1) % D)     -> flat
                down = ((r + 1) % self.D) * self.D + c      # ((r+1) % D, c)     -> flat
                A[i, right] = 1.0
                A[i, down] = 1.0

        # symmetrise so the matrix is symmetric (undirected edges)
        A = A + A.T
        self.A = A                    # kept for `set_sigma` rescaling
        self.J = self.sigma * A

    def set_sigma(self, sigma: float) -> None:
        """Mutate σ in place; rescales J = σ · A.

        Used for MDNS-style temperature warm-up (App D.2.4): train at an
        easier σ first, then swap to the harder target σ without rebuilding
        the model or optimizer state.
        """
        self.sigma = sigma
        self.J = sigma * self.A

    def log_prob(self, x: Tensor) -> Tensor:
        """Un-normalised target log-density.

        x: (B, d) float tensor with entries in {-1, +1}.
        Returns: (B,) tensor.

            log_prob(x) = x^T J x  +  bias · Σ_i x_i

        Note: omits the log-partition-function constant log Z; this is the
        un-normalised log p.
        """
        return (x @ self.J * x).sum(dim=-1)  + (self.bias * x.sum(dim=1))

    def log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Annealing-path log-density at time t (paper Eq. 4).

        x: (B, d) float tensor.
        t: (B,) tensor with entries in [0, 1].
        Returns: (B,) tensor.

            log p̃_t(x) = (1 - t) · log η(x) + t · log p(x)
                       = (1 - t) · (-d · log 2) + t · log_prob(x)

        At t=0: returns the constant -d · log 2 (uniform prior).
        At t=1: returns log_prob(x) (full target).
        """
        return (1 - t) * (-self.d * math.log(2)) + (t * self.log_prob(x))

    def dt_log_p_tilde_t(self, x: Tensor, t: Tensor) -> Tensor:
        """Time-derivative of the annealing log-density. t-independent.

        x: (B, d) float tensor.
        t: (B,) tensor (unused; kept in signature to match log_p_tilde_t).
        Returns: (B,) tensor.

            ∂_t log p̃_t(x) = log p(x) - log η(x)
                            = log_prob(x) - (-d · log 2)
                            = log_prob(x) + d · log 2

        This expression has no t-dependence - that's the consequence of
        choosing a linear-in-log annealing path.
        """
        return self.log_prob(x) + (self.d * math.log(2))
