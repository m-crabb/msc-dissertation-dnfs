"""Outer-step ∂_t log Z_t estimator for paper Algorithm 1 line 4.

The Kolmogorov residual loss needs

    ∂_t log p_t(x) = ∂_t log p̃_t(x) − ∂_t log Z_t,

and the second term is one scalar for every x. Algorithm 1 amortises it over
a replay buffer: at the start of each outer cycle, evaluate it once per
time-grid point under the stop-gradient model, cache the K+1 scalars, and
reuse them across `inner_steps_per_outer` gradient updates.

Two modes, both giving a per-slot scalar c_t under `torch.no_grad`:

  - naive_mc:        c_t = mean_m ∂_t log p̃_t(x_t^{(m)}). Target-only,
                     θ-independent; the stage_0 / stage_1 ablation against
                     control_variate (not in the paper's Algorithm 1).
  - control_variate: c_t = mean_m ξ_t(x_t^{(m)}; R_t^{sg}), paper Eq. 8 and
                     Algorithm 1 line 4 verbatim. ξ_t subtracts the control
                     statistic Σ_y R(x,y) p_t(y)/p_t(x), driving c_t to zero
                     variance as R approaches Kolmogorov-satisfying.

The inner-step loss is `(ξ_θ(x) − c_t)²` with c_t looked up by the sample's
t-index. c_t is detached, so gradient flows only through ξ_θ: the uncentred
gradient form of §C.1, whose fixed points are the R satisfying Kolmogorov
forward exactly rather than merely making ξ_θ flat in x.
"""

from typing import Literal

import torch
from torch import Tensor

from discrete_flow_sampler.samplers.ctmc import compute_xi_t


def compute_c_t_grid(
    t_grid: Tensor,
    x_traj: Tensor,
    target,
    model,
    *,
    mode: Literal["naive_mc", "control_variate"],
) -> tuple[Tensor, Tensor]:
    """Per-time-slot c_t for paper Algorithm 1 line 4.

    For each t_k in `t_grid`, average a per-state integrand over the M
    outer-batch samples cached at that time slot.

    Args:
        t_grid: (T,) time-grid points {t_k}_{k=0}^{T-1} (paper's K+1 = T).
        x_traj: (T, M, d) buffer of outer-batch states -- x_traj[k] holds
            the M states sampled at time t_k under the stop-gradient
            model. Produced by `sample_ctmc(..., return_all_states=True)`.
        target: exposes `log_p_tilde_t` and `dt_log_p_tilde_t`.
        model: rate matrix; consulted only in `control_variate` mode.
        mode: integrand selector -- "naive_mc" or "control_variate".

    Returns:
        c_t_grid: (T,) -- scalar c_t per time-grid point. Detached.
        integrand_per_t: (T, M) -- the per-state integrands averaged,
            detached; the training loop logs their variance as
            `var_estimator_integrand` (naive vs CV comparison).
    """
    if mode not in ("naive_mc", "control_variate"):
        raise ValueError(
            f"Unknown mode {mode!r}; expected 'naive_mc' or 'control_variate'"
        )

    n_grid, outer_batch, _ = x_traj.shape
    integrand_per_t = torch.empty(
        (n_grid, outer_batch),
        dtype=x_traj.dtype,
        device=x_traj.device,
    )

    # Per-slot loop keeps peak memory at M (not T*M) and matches the
    # per-slot semantics of Algorithm 1 line 4.
    with torch.no_grad():
        for k in range(n_grid):
            x_k = x_traj[k]
            t_k = t_grid[k].expand(outer_batch)
            if mode == "naive_mc":
                integrand_per_t[k] = target.dt_log_p_tilde_t(x_k, t_k)
            else:
                integrand_per_t[k] = compute_xi_t(x_k, t_k, model, target)

    c_t_grid = integrand_per_t.mean(dim=-1)
    return c_t_grid, integrand_per_t
