"""Estimators for ∂_t log Z_t along the annealing path.

Background (paper Sec. 3, Eq. (5)/(6)):
    The annealing path uses unnormalised density
        p̃_t(x) = exp((1 - t) log eta(x) + t log rho(x)),
    so its normaliser is Z_t = sum_x p̃_t(x) and the marginal is
    p_t(x) = p̃_t(x) / Z_t. The Kolmogorov residual loss needs

        ∂_t log p_t(x) = ∂_t log p̃_t(x) - ∂_t log Z_t,

    and the second term is a single scalar -- the same number for every x
    -- but it depends on the full distribution, so it has to be estimated.

The clean identity:
        ∂_t log Z_t = E_{p_t}[ ∂_t log p̃_t(X) ].

Stage 1 swap point: `naive_mc` -- average the integrand over a batch of
x ~ p_t. Simple, unbiased, but high-variance at large D.

Stage 2 swap point: `control_variate` (Eq. 8), which subtracts a Kolmogorov-
derived baseline `Σ_y R_t(x,y) p_t(y)/p_t(x)` that, when R_t exactly
generates p_t, makes the per-state integrand identically equal to the
constant ∂_t log Z_t (zero variance). Off-optimum the variance is reduced
in proportion to how well R_t satisfies the Kolmogorov equation.

Signature convention:
    Every estimator takes (t, x_batch, target, model) and returns a 2-tuple
        (estimate, modified_integrand)
    where `estimate` is the 0-dim scalar estimate of ∂_t log Z_t and
    `modified_integrand` is the (B,) per-state vector that was averaged to
    produce the estimate. Surfacing the modified integrand lets the
    training loop log Var[modified_integrand] as the mechanism diagnostic
    of §0.3 in the Stage 2 plan: in Stage 1 (naive_mc) the modified
    integrand is just ∂_t log p̃_t and its variance equals var_dt_log_p_tilde
    by construction; in Stage 2 (control_variate) it diverges below, with
    the ratio quantifying the variance-reduction factor.
"""
from typing import Callable

from torch import Tensor


# Type alias used by the training loop to declare its dependency.
# The forward references are intentional: importing Target/RateMatrix at
# module load time would create a cycle through samplers.
LogZEstimator = Callable[
    [Tensor, Tensor, "Target", "RateMatrix"], tuple[Tensor, Tensor]    # noqa: F821
]


def naive_mc(
    t: Tensor, x_batch: Tensor, target, model
) -> tuple[Tensor, Tensor]:
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
            training step.
        x_batch: (B, d) in {-1, +1}. Approximate samples from p_t.
        target: an IsingTarget (or any object exposing
            `dt_log_p_tilde_t(x, t) -> (B,)`).
        model: unused; present for signature uniformity.

    Returns:
        (estimate, modified_integrand):
            estimate: 0-dim Tensor, scalar estimate of ∂_t log Z_t.
            modified_integrand: (B,) Tensor of `∂_t log p̃_t(x)` per state.
                For naive MC this IS the integrand; in Stage 2's control
                variate it diverges below by the control-variate-modified
                value.
    """
    # Per-state values of the integrand ∂_t log p̃_t(x). Shape (B,).
    modified_integrand = target.dt_log_p_tilde_t(x_batch, t)
    return modified_integrand.mean(), modified_integrand
