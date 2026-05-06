"""Continuous-Time Markov Chain (CTMC) Euler-step trajectory sampler.

Paper reference: Eq. (2) (forward Euler step) and Eq. (8)/(13) (importance
weight + ESS), in `dnfs.pdf`.

Forward simulation
------------------
Given a learned rate matrix R_t(x, i) (per-site flip rate), a single Euler
step of size dt is

    Pr[ flip site i in step | x ]  =  clip( R_t(x, i) * dt , 0, 1 ).

For binary spins, sites are conditionally independent within one Euler
step (the joint-state Categorical factorises across sites), so we can
sample all per-site flips with a single uniform draw per site.

Importance log-weight
---------------------
When `return_log_weights=True`, we accumulate

    w(t + dt) = w(t) + xi_t( x_t ; R_t ) * dt,

where xi_t is the local IS integrand. The exact form is the design choice:
    a) Textbook ("full") xi_t (paper Eq. analogous to (8)):
            xi_t(x) = dt_log_p_tilde_t(x)
                     - sum_i R_t(x, i) * ( p_t(x_flip_i) / p_t(x) - 1 )
       Needs the target (for log p_tilde at flipped neighbours -- the
       ratio collapses to exp(log_p_tilde(x_flip) - log_p_tilde(x))).
    b) Simpler model-only variant the paper uses for ESS: see Eq. (13).
The user picks one (driven by what Eq. 8 / 13 in the paper specifies).

Caller contract
---------------
- `model(x, t_per_batch)` must return (B, d) non-negative rates.
- `ts` is a 1-D monotonically increasing time grid; dt is read from
  consecutive entries (NOT assumed uniform -- compute dt = ts[i+1] - ts[i]).
- When `return_log_weights=True`, `target` must be supplied. We do NOT
  silently fall back to a model-only weight, because that would change
  semantics under the same flag.

Research-bearing; user implements body.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sample_ctmc(
    model,
    x0: Tensor,
    ts: Tensor,
    *,
    return_log_weights: bool = False,
    target=None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Simulate a CTMC trajectory by Euler stepping along `ts`.

    Args:
        model: a `RateMatrix` -- callable (x, t_per_batch) -> (B, d) rates,
            non-negative.
        x0: (B, d) initial state in {-1, +1}. The simulation starts at
            time `ts[0]` from this state.
        ts: (T,) monotonically increasing time grid covering the desired
            range (typically [0, 1]). dt = ts[i+1] - ts[i] (per step).
        return_log_weights: if True, also return per-trajectory IS log-
            weights accumulated along the path.
        target: required when `return_log_weights=True`; used to evaluate
            `dt_log_p_tilde_t` (and, depending on xi_t form, `log_p_tilde_t`
            at flipped neighbours).

    Returns:
        x_final: (B, d). The state at time `ts[-1]`.
        OR (when return_log_weights=True):
        (x_final, log_w) where log_w has shape (B,).

    Implementation skeleton (USER fills body):
        x = x0.clone()
        log_w = torch.zeros(x.shape[0]) if return_log_weights else None
        batch_size = x.shape[0]

        for step in range(len(ts) - 1):
            t_curr = ts[step]
            dt = ts[step + 1] - ts[step]
            t_per_batch = t_curr.expand(batch_size)

            rates = model(x, t_per_batch)                  # (B, d), >= 0

            # Per-site Euler flip probabilities, clipped for numerical safety.
            flip_prob = (rates * dt).clamp(0.0, 1.0)        # (B, d)
            uniforms = torch.rand_like(flip_prob)
            x = torch.where(uniforms < flip_prob, -x, x)

            if return_log_weights:
                # TODO(human): pick xi_t form (paper Eq. 8 / 13) and compute it.
                #   xi_t : (B,) tensor.
                # See the file docstring for the two candidate forms (textbook
                # vs. model-only). The first needs `target.log_p_tilde_t` at
                # x and at each single-site flip neighbour; the second is
                # simpler but has different variance properties.
                xi_t = ...  # USER
                log_w = log_w + xi_t * dt

        return (x, log_w) if return_log_weights else x

    Edge cases worth thinking about while you implement:
    - `flip_prob > 1` if `dt` is too coarse for the rate. The clamp masks
      it but it indicates a stiff regime where Euler is a bad approximation.
      The plan's training schedule keeps dt small enough that this is rare.
    - Vectorise the xi_t computation over the batch; never write a Python
      loop over batch elements.
    """
    if return_log_weights and target is None:
        raise ValueError(
            "sample_ctmc(return_log_weights=True) requires `target` to be "
            "provided so xi_t can be evaluated along the trajectory."
        )

    state = x0.clone()
    batch_size, n_sites = state.shape

    log_weights = (
        torch.zeros(batch_size, dtype=state.dtype, device=state.device)
        if return_log_weights
        else None
    )

    # flip_signs[i, j] = -1 if i == j else +1. Used to construct, for every
    # site i, the state with site i flipped: state * flip_signs[i] flips
    # only the i-th coordinate of state. Pre-built once outside the loop.
    flip_signs = 1.0 - 2.0 * torch.eye(
        n_sites, device=state.device, dtype=state.dtype
    )  # (d, d)

    for step in range(len(ts) - 1):
        t_curr = ts[step]
        step_dt = ts[step + 1] - ts[step]
        t_per_batch = t_curr.expand(batch_size)  # (B,)

        # Rates at the current state x_t. These are R_t(x_flip_i, x) -- the
        # rate of leaving x by flipping site i. Shape (B, d).
        outflow_rates = model(state, t_per_batch)

        if return_log_weights:
            # Build the (B, d, d) tensor of single-site flips:
            # flip_neighbours[b, i, :] is state[b, :] with site i flipped.
            # Implemented as a broadcasted multiplication by flip_signs;
            # avoids a Python loop over sites.
            flip_neighbours = state.unsqueeze(1) * flip_signs.unsqueeze(0)
            flat_neighbours = flip_neighbours.reshape(
                batch_size * n_sites, n_sites
            )
            t_per_neighbour = t_per_batch.repeat_interleave(n_sites)  # (B*d,)

            # Evaluate model at every flipped neighbour. We only need the
            # diagonal entry: at state x_flip_i, the rate of flipping site i
            # back to x. That's R_t(x_flip_i -> x) in paper notation, i.e.
            # R_t(x, x_flip_i) under Eq. 1's "first index = destination" rule.
            rates_at_flipped = model(flat_neighbours, t_per_neighbour).reshape(
                batch_size, n_sites, n_sites
            )
            return_rates = rates_at_flipped.diagonal(dim1=1, dim2=2)  # (B, d)

            # Target log p̃_t at x and at every flipped neighbour. The Z_t
            # in p_t(y)/p_t(x) cancels because we only ever take the ratio.
            log_p_tilde_at_state = target.log_p_tilde_t(state, t_per_batch)
            log_p_tilde_at_flips = target.log_p_tilde_t(
                flat_neighbours, t_per_neighbour
            ).reshape(batch_size, n_sites)
            log_ratio = log_p_tilde_at_flips - log_p_tilde_at_state.unsqueeze(-1)
            neighbour_ratio = log_ratio.exp()  # (B, d) -- p_t(x_flip_i)/p_t(x)

            dt_log_p_tilde_at_state = target.dt_log_p_tilde_t(state, t_per_batch)

            # Decompose Σ_y R_t(x, y) p_t(y)/p_t(x) for the one-way binary
            # rate matrix (Eq. 6, Prop. 1). Two contributions survive:
            #   y = x_flip_i   ->   inflow_sum  =  Σ_i return_rates[b,i] * ratio[b,i]
            #   y = x          ->   diagonal R_t(x,x) = -Σ_i outflow_rates[b,i]
            # Sum_y = inflow_sum - outflow_sum  (note the sign on the diagonal).
            inflow_sum = (return_rates * neighbour_ratio).sum(dim=-1)  # (B,)
            outflow_sum = outflow_rates.sum(dim=-1)  # (B,)

            # ξ_t per Eq. 8, evaluated at x_t (left endpoint of the Euler
            # interval -- standard for forward Euler).
            xi_t = (
                dt_log_p_tilde_at_state
                - inflow_sum
                + outflow_sum
            )
            log_weights = log_weights + xi_t * step_dt

        # Forward Euler flip step (Eq. 2): each site flips independently with
        # probability outflow_rates * dt, clipped to [0, 1] for numerical
        # safety when dt is too coarse. The clamp signals stiffness rather
        # than masking it -- check post-hoc if many flip_prob entries hit 1.
        flip_prob = (outflow_rates * step_dt).clamp(0.0, 1.0)
        uniforms = torch.rand_like(flip_prob)
        state = torch.where(uniforms < flip_prob, -state, state)

    if return_log_weights:
        return state, log_weights
    return state
