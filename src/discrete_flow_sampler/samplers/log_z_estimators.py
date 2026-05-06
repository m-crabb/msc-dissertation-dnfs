"""Estimators for ∂_t log Z_t along the annealing path.

Background (paper Sec. 3, Eq. (5)/(6)):
    The annealing path uses unnormalised density
        p̃_t(x) = exp((1 - t) log eta(x) + t log rho(x)),
    so its normaliser is Z_t = sum_x p̃_t(x) and the marginal is
    p_t(x) = p̃_t(x) / Z_t. The Kolmogorov residual loss (Task 9) needs

        ∂_t log p_t(x) = ∂_t log p̃_t(x) - ∂_t log Z_t,

    and the second term is a single scalar -- the same number for every x
    -- but it depends on the full distribution, so it has to be estimated.

The clean identity (used by all estimators here):

        ∂_t log Z_t
        = (1 / Z_t) * sum_x p̃_t(x) * ∂_t log p̃_t(x)
        = E_{p_t}[ ∂_t log p̃_t(X) ].

Stage 1 swap point: `naive_mc` -- average the integrand over a batch of
x ~ p_t. Simple, unbiased, but high-variance at large D.

Stage 2 swap point (added later): control-variate estimator (Eq. 8), which
subtracts a model-dependent baseline to reduce variance.

Signature convention:
    Every estimator takes (t, x_batch, target, model) -> 0-dim Tensor.
    Keeping the signature uniform lets the training loop swap estimators by
    wiring a different function into the same call site -- no branching at
    the call site.
"""
from __future__ import annotations

from typing import Callable

from torch import Tensor


# Type alias used by the training loop to declare its dependency.
# The forward references are intentional: importing Target/RateMatrix at
# module load time would create a cycle through samplers.
LogZEstimator = Callable[[Tensor, Tensor, "Target", "RateMatrix"], Tensor]  # noqa: F821


def naive_mc(t: Tensor, x_batch: Tensor, target, model) -> Tensor:
    """Stage 1 estimator: plain Monte Carlo of the integrand ∂_t log p̃_t(x).

    Implements
            ∂_t log Z_t  ~=  (1 / K) * sum_k ∂_t log p̃_t( x^(k) ),
    with x^(k) drawn approximately from p_t. In practice the caller passes
    states from the current model's CTMC trajectory at time t.

    Why `model` is unused here:
        The naive estimator does not consult the learned rate matrix at all
        -- the integrand is purely a function of the target. The argument
        is kept so that this function and the Stage 2 control-variate
        estimator are interchangeable at the call site.

    Args:
        t: (B,) in [0, 1]. For typical use, all entries equal: one t per
            training step. Shape just has to broadcast with `x_batch` along
            the leading (batch) dimension.
        x_batch: (B, d) in {-1, +1}. Approximate samples from p_t.
        target: an IsingTarget (or any object exposing
            `dt_log_p_tilde_t(x, t) -> (B,)`).
        model: unused; present for signature uniformity with future estimators.

    Returns:
        0-dim Tensor: the scalar estimate of ∂_t log Z_t at (the implied) t.
    """
    # Per-state values of the integrand ∂_t log p̃_t(x). For the Ising target
    # this collapses to (log rho - log eta) evaluated at each x in the batch,
    # because p̃_t is exp-linear in t. Shape: (B,).
    dt_log_p_tilde_per_state = target.dt_log_p_tilde_t(x_batch, t)

    # Plain MC average over the batch -> a 0-dim Tensor (scalar). torch.mean
    # of a (B,) tensor with no `dim` argument reduces to scalar, which is
    # exactly the contract the docstring promises.
    return dt_log_p_tilde_per_state.mean()
