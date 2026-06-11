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
        target_composition: float | None = None,
        composition_penalty_strength: float = 0.0,
    ):
        if target_composition is not None and not 0.0 <= target_composition <= 1.0:
            raise ValueError(
                "target_composition must be in [0, 1], "
                f"got {target_composition}"
            )
        if composition_penalty_strength < 0.0:
            raise ValueError(
                "composition_penalty_strength must be non-negative, "
                f"got {composition_penalty_strength}"
            )
        if composition_penalty_strength > 0.0 and target_composition is None:
            raise ValueError(
                "target_composition must be set when "
                "composition_penalty_strength is nonzero"
            )

        self.D = D
        self.d = D * D
        self.sigma = sigma
        self.bias = bias
        self.device = torch.device(device)
        self.target_composition = target_composition
        self.composition_penalty_strength = composition_penalty_strength

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

        Used by temperature curricula to move through easier intermediate
        targets without rebuilding model or optimizer state.
        """
        self.sigma = sigma
        self.J = sigma * self.A

    def set_composition_penalty_strength(self, strength: float) -> None:
        """Mutate λ in place.

        Used by λ-annealing curricula to tighten the soft composition
        constraint without rebuilding model or optimizer state.
        """
        if strength < 0.0:
            raise ValueError(
                "composition_penalty_strength must be non-negative, "
                f"got {strength}"
            )
        if strength > 0.0 and self.target_composition is None:
            raise ValueError(
                "target_composition must be set when "
                "composition_penalty_strength is nonzero"
            )
        self.composition_penalty_strength = strength

    def composition_fraction(self, x: Tensor) -> Tensor:
        """Fraction of +1 spins in each state, shape (B,)."""
        return ((x + 1.0) * 0.5).mean(dim=-1)

    def base_log_prob(self, x: Tensor) -> Tensor:
        """Unnormalised Ising log-density before optional soft constraints.

        x: (B, d) float tensor with entries in {-1, +1}.
        Returns: (B,) tensor.

            log_prob(x) = x^T J x  +  bias · Σ_i x_i

        Note: omits the log-partition-function constant log Z; this is the
        un-normalised log p.
        """
        return (x @ self.J * x).sum(dim=-1)  + (self.bias * x.sum(dim=1))

    def composition_penalty(self, x: Tensor) -> Tensor:
        """Extensive soft-composition penalty, shape (B,).

        The form mirrors VCSGC-style concentration control by scaling the
        squared composition deviation by the number of sites:

            λ · d · (c_+(x) - c_target)^2

        It is subtracted from `log_prob`, equivalently added to the target
        energy.
        """
        if self.target_composition is None or self.composition_penalty_strength == 0.0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        diff = self.composition_fraction(x) - self.target_composition
        return self.composition_penalty_strength * self.d * diff.pow(2)

    def log_prob(self, x: Tensor) -> Tensor:
        """Un-normalised target log-density.

        x: (B, d) float tensor with entries in {-1, +1}.
        Returns: (B,) tensor.

            log_prob(x) = base_log_prob(x) - composition_penalty(x)

        With no soft-composition constraint, this reduces to the base Ising
        log-density.
        """
        return self.base_log_prob(x) - self.composition_penalty(x)

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
