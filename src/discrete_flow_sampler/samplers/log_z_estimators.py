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


def control_variate(
    t: Tensor, x_batch: Tensor, target, model
) -> tuple[Tensor, Tensor]:
    """Stage 2 estimator: control-variate `∂_t log Z_t` (paper Eq. 8).

    Subtracts a Kolmogorov-derived control statistic from the naive
    integrand:

        ξ_t(x; R_t) ≜ ∂_t log p̃_t(x) − Σ_y R_t(x, y) · p_t(y)/p_t(x)
        ∂_t log Z_t ≈ (1 / K) Σ_k ξ_t(x_k; R_t),    x_k ~ p_t.

    Why this works (Stage 2 plan §0.4 derivation, paraphrased):
        Under the rate-matrix algebra `Σ_y R_t(y, x) = 0`, the Kolmogorov
        forward equation gives
            ∂_t log p_t(x) = Σ_y R_t(x, y) p_t(y)/p_t(x).
        With `log p̃_t = log p_t + log Z_t` we get
            ∂_t log p̃_t(x) − Σ_y R_t(x, y) p_t(y)/p_t(x)
                = ∂_t log p_t(x) + ∂_t log Z_t − ∂_t log p_t(x)
                = ∂_t log Z_t,
        identically in x, when R_t exactly generates p_t. Off-optimum the
        residual is exactly the Kolmogorov residual the loss minimises, so
        the variance reduction tightens as training progresses.

    Decomposition under the single-spin-flip restriction (paper Eq. 6 —
    R_t(y, x) = 0 for y ∉ N(x)):

        Σ_y R_t(x, y) p_t(y)/p_t(x) splits into
            y = x:           R_t(x, x) · 1 = −outflow_sum(x)
            y ∈ N(x) \\ {x}:  Σ_i R_t(x, y_i^flip) · p_t(y_i^flip)/p_t(x)
                              =: inflow_sum(x)
        Combined:  Σ_y R_t · p_t(y)/p_t(x) = inflow_sum − outflow_sum.

    Therefore:
        ξ_t(x; R_t) = ∂_t log p̃_t(x) − (inflow_sum − outflow_sum)
                    = ∂_t log p̃_t(x) + outflow_sum − inflow_sum

    where
        outflow_sum = Σ_i model(x, t)[i]   (per-site flip rates at x)
        inflow_sum  = Σ_i model(x_flip_i, t)[i] · p_t(x_flip_i)/p_t(x)
                    = Σ_i model(x_flip_i, t)[i]
                          · exp(log_p̃_t(x_flip_i) − log_p̃_t(x))
                    (Z_t cancels in the ratio).

    Reference implementation pattern: `samplers/ctmc.py` (lines ~140-186)
    computes the same `ξ_t` form for the eval-time IS log-weight integrand;
    the only differences for training are (a) gradients must flow through
    R_t (do NOT detach), and (b) the function returns the scalar mean and
    the (B,) per-state vector rather than accumulating along a trajectory.

    Args:
        t: (B,) in [0, 1]. Per-step single value, broadcast across batch.
        x_batch: (B, d) in {-1, +1}. Approximate samples from p_t.
        target: exposes `log_p_tilde_t(x, t) -> (B,)` and
            `dt_log_p_tilde_t(x, t) -> (B,)`.
        model: rate matrix returning per-site flip rates of shape (B, d).

    Returns:
        (estimate, modified_integrand):
            estimate: 0-dim Tensor — scalar estimate of ∂_t log Z_t.
            modified_integrand: (B,) Tensor of ξ_t per state. Var of this
                vector is the mechanism column for Stage 2 plan §0.3
                (gate: ratio < 0.5 vs Stage 1's `var_dt_log_p_tilde`).
    """
    raise NotImplementedError(
        "control_variate body is research-bearing (Stage 2 plan Task 2 "
        "step 3 — user implements). Math fully spelled out above; the "
        "samplers/ctmc.py inflow/outflow decomposition gives the shape "
        "and indexing conventions to mirror."
    )


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
