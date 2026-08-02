"""Kolmogorov forward-equation residual and squared-residual loss.

Paper reference: Eq. (3) (forward equation), Eq. (4) (log form), Eq. (7)
(general residual under the single-site-flip parameterisation, Eq. 6), in
`dnfs.pdf`. The locally-equivariant specialisation Eq. (10) lives in
`residual_lenet` below; it is *not* the same as Eq. (7).

Derivation
----------
The Kolmogorov forward equation in log form (paper Eq. 4) says that, if R
generates the marginal p_t,

    dt log p_t(x)  =  Σ_{y != x}  R_t(x, y) * p_t(y) / p_t(x)  -  R_t(y, x).

Move everything to one side: a rate matrix R^θ satisfies the equation iff

    delta_t(x; R^θ)  ==  0  for all x, t,

with

    delta_t(x; R^θ)
        = dt log p_t(x)
          + Σ_{i: y_i != x_i} [ R^θ(y_i, i | x)  -  R^θ(x_i, i | y) * p_t(y) / p_t(x) ]   (Eq. 7)

where we have specialised to single-site flips only (Eq. 6): the Σ_{y != x}
collapses to Σ_i over the d single-flip neighbours y(i). Eq. (7) does NOT
require the one-way `R = [G]_+` parameterisation of Prop. 1 — that further
specialisation is what produces Eq. (10) and `residual_lenet`. Any
parameterisation that emits per-site flip rates (e.g. `MLPRateMatrix`)
satisfies Eq. (7)'s premises; the cost is empirical estimator variance,
which the paper attacks with the control-variate estimator (Eq. 8).

Implementation notes
--------------------
- ∂_t log p_t(x) = ∂_t log p̃_t(x) − ∂_t log Z_t. The caller pre-computes
  ∂_t log Z_t (via `samplers.log_z_estimators`) and passes it in. This
  separates the "average over a batch" logic (in the estimator) from the
  per-x residual computation here.

- The Z_t in p_t(y)/p_t(x) cancels: it equals exp(log p̃_t(y) − log p̃_t(x)).
  So the residual never needs Z_t directly -- only its t-derivative.

- For each x in the batch, computing the residual requires evaluating the
  *model* on x AND on each of its d single-flip neighbours -- d + 1 forward
  passes' worth of model inputs per x. Vectorised here as one
  (B*d, d) batched call. This is exactly the cost cliff that motivates
  the locally equivariant Transformer in Stage 3.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._neighbours import (
    DEFAULT_LOG_RATIO_CLAMP,
    _log_p_tilde_at_neighbours,
    log_ratio_clamp,
)

__all__ = ["DEFAULT_LOG_RATIO_CLAMP", "residual_general", "residual_lenet", "loss"]


def _flip_signs_matrix(n_sites: int, *, device, dtype) -> Tensor:
    """Return the (d, d) matrix with -1 on the diagonal and +1 elsewhere.

    Multiplying state.unsqueeze(1) (shape (B, 1, d)) by this matrix's
    unsqueezed form (shape (1, d, d)) builds the (B, d, d) tensor whose
    [b, i, :] row is state[b, :] with site i flipped. Cheaper and clearer
    than a Python loop or in-place index_put_.
    """
    return 1.0 - 2.0 * torch.eye(n_sites, device=device, dtype=dtype)


def residual_general(
    x: Tensor,
    t: Tensor,
    dt_log_Zt: Tensor,
    model,
    target,
) -> Tensor:
    """Per-state Kolmogorov residual delta_t(x) (Eq. 7).

    Args:
        x: (B, d) state tensor in {-1, +1}.
        t: (B,) time tensor in [0, 1].
        dt_log_Zt: 0-dim or (B,) Tensor. Pre-computed estimate of
            ∂_t log Z_t at the relevant t -- a per-sample lookup from
            the outer-step c_t cache (paper Algorithm 1 line 7-9), or
            a scalar if all states share a t.
        model: a `RateMatrix` -- callable (state, time) -> (B, d) rates >= 0.
        target: an IsingTarget (or any object exposing `log_p_tilde_t` and
            `dt_log_p_tilde_t`).

    Returns:
        (B,) Tensor of residuals. delta_t(x_b) for each x_b in the batch.
        At an R^θ that exactly generates p_t, residual is identically zero.
    """
    batch_size, n_sites = x.shape

    # Pre-compute the off-diagonal flip matrix.
    flip_signs = _flip_signs_matrix(n_sites, device=x.device, dtype=x.dtype)

    # (B, d, d) of single-site flips: flip_neighbours[b, i, :] is x[b, :]
    # with only site i flipped. Then flatten to (B*d, d) so the model and
    # target can be evaluated on all neighbours in a single batched call.
    flip_neighbours = x.unsqueeze(1) * flip_signs.unsqueeze(0)  # (B, d, d)
    flat_neighbours = flip_neighbours.reshape(batch_size * n_sites, n_sites)
    t_per_neighbour = t.repeat_interleave(n_sites)  # (B*d,)

    # Forward (outflow) rates at x: model(x, t)[b, i] = R_t(x_flip_i, i | x).
    forward_rates = model(x, t)  # (B, d)

    # Reverse (inflow) rates from each flipped neighbour: at state x_flip_i,
    # the rate of flipping site i back to x is the diagonal of model evaluated
    # at the (B*d, d) neighbour batch -- entry [b, i, i] in the (B, d, d)
    # reshape. That's R_t(x_i, i | x_flip_i).
    rates_at_neighbours = model(flat_neighbours, t_per_neighbour).reshape(
        batch_size, n_sites, n_sites
    )
    reverse_rates = rates_at_neighbours.diagonal(dim1=1, dim2=2)  # (B, d)

    # p_t(x_flip_i) / p_t(x). Z_t cancels in the ratio, so we only need
    # log p̃_t differences.
    log_p_tilde_at_x = target.log_p_tilde_t(x, t)  # (B,)
    log_p_tilde_at_flips = target.log_p_tilde_t(
        flat_neighbours, t_per_neighbour
    ).reshape(batch_size, n_sites)
    log_neighbour_ratio = log_p_tilde_at_flips - log_p_tilde_at_x.unsqueeze(-1)
    neighbour_ratio = log_neighbour_ratio.exp()  # (B, d)

    # ∂_t log p_t(x) = ∂_t log p̃_t(x) − ∂_t log Z_t. The Z_t-derivative is
    # a single scalar that broadcasts over the batch.
    dt_log_pt_x = target.dt_log_p_tilde_t(x, t) - dt_log_Zt  # (B,)

    # Per-site bracket from Eq. 7: forward rate - reverse rate * ratio.
    site_terms = forward_rates - reverse_rates * neighbour_ratio  # (B, d)

    return dt_log_pt_x + site_terms.sum(dim=-1)


def residual_lenet(
    x: Tensor,
    t: Tensor,
    dt_log_Zt: Tensor,
    model,
    target,
) -> Tensor:
    """Per-state Kolmogorov residual for locally equivariant models — paper Eq. (10).

    Math (DNFS App. B.2, third line of the derivation after Definition 3):
        δ_t(x; R^θ) = ∂_t log p_t(x)
                    + Σ_{i, y_i ≠ x_i} [G(y_i, i | x)]_+
                                       - [-G(y_i, i | x)]_+ · p_t(y_i)/p_t(x)

    Why this collapses to a single forward pass:
        Local equivariance (Eq. 20) gives G(x_i, i | y_i) = -G(y_i, i | x).
        So the reverse rate at the flipped neighbour, R^θ(x_i, i | y_i)
        = [G(x_i, i | y_i)]_+ = [-G(y_i, i | x)]_+, is computable from the
        same G(·, i | x) tensor — no second model call. This is the
        O(|N|) → O(1) reduction the paper attributes to leNets.

    Args:
        x: (B, D) state in {-1, +1}.
        t: (B,) in [0, 1].
        dt_log_Zt: scalar Tensor — pre-computed estimate of ∂_t log Z_t.
        model: locally equivariant — `model(x, t)` returns G of shape
            (B, D, vocab_size). The τ = x_i slot is zero by construction.
        target: exposes `log_p_tilde_t(x, t) -> (B,)` and
            `dt_log_p_tilde_t(x, t) -> (B,)`.

    Returns:
        (B,) per-state residual.
    """
    batch_size, n_sites = x.shape
    vocab_size = model.vocab_size

    # G_t shape (B, D, S). The τ = x_i slot is already 0 by the leMLP scatter.
    G_t = model(x, t)
    G_plus     = F.relu(G_t)
    neg_G_plus = F.relu(-G_t)

    log_p_neighbours = _log_p_tilde_at_neighbours(x, t, target, vocab_size)
    log_p_x = target.log_p_tilde_t(x, t)
    log_ratio = log_p_neighbours - log_p_x[:, None, None]
    log_ratio = log_ratio.clamp(max=log_ratio_clamp(target))
    site_terms = (G_plus - neg_G_plus * log_ratio.exp()).sum(dim=(-2, -1))
    dt_log_pt_x = target.dt_log_p_tilde_t(x, t) - dt_log_Zt
    return dt_log_pt_x + site_terms


def loss(
    x: Tensor,
    t: Tensor,
    dt_log_Zt: Tensor,
    model,
    target,
) -> Tensor:
    """Mean-squared Kolmogorov residual over the batch. Scalar Tensor.

    Dispatches on `model.is_locally_equivariant`:
      - True  -> residual_lenet (Eq. 10, single forward pass).
      - False -> residual_general (Eq. 7, two forward passes).

    The training loop calls this single entry point regardless of stage;
    only the model's class attribute decides the residual form.
    """
    if getattr(model, "is_locally_equivariant", False):
        residual = residual_lenet(x, t, dt_log_Zt, model, target)
    else:
        residual = residual_general(x, t, dt_log_Zt, model, target)
    residual = residual.nan_to_num(posinf=1.0, neginf=-1.0, nan=0.0)
    return residual.pow(2).mean()
