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
"""
import torch
from torch import Tensor


def compute_xi_t(
    state: Tensor,
    t: Tensor,
    model,
    target,
    outflow_rates: Tensor | None = None,
) -> Tensor:
    """Per-state IS integrand ξ_t(x; R_t) for ∂_t log Z_t (paper Eq. 8).

    Same quantity used in two places:
      • `sample_ctmc(..., return_log_weights=True)` — accumulated along the
        Euler trajectory to produce IS log-weights for ESS / F/D / E/D eval.
      • `samplers.log_z_estimators.control_variate` — averaged at each
        training step as the gradient estimator for ∂_t log Z_t (Stage 2).

    Math (single-spin-flip restriction, paper Eq. 6):
        ξ_t(x; R_t) = ∂_t log p̃_t(x) − Σ_y R_t(x, y) p_t(y)/p_t(x)
                    = ∂_t log p̃_t(x) + outflow_sum(x) − inflow_sum(x)

    where outflow_sum(x) = Σ_i model(x, t)[i] and
          inflow_sum(x)  = Σ_i model(x_flip_i, t)[i] · p_t(x_flip_i)/p_t(x).
    The Z_t in the ratio cancels:
          p_t(x_flip_i)/p_t(x) = exp(log_p̃_t(x_flip_i) − log_p̃_t(x)).

    `outflow_rates` is optional — `sample_ctmc` already computes
    `model(state, t)` for the Euler step and passes it through to avoid the
    duplicate forward pass; `control_variate` omits it and lets the helper
    compute it. Gradients flow through `model(...)` when called outside
    `torch.no_grad`.
    """
    batch_size, n_sites = state.shape

    if outflow_rates is None:
        outflow_rates = model(state, t)
    outflow_sum = outflow_rates.sum(dim=-1)                            # (B,)

    flip_signs = 1.0 - 2.0 * torch.eye(
        n_sites, device=state.device, dtype=state.dtype
    )
    flip_neighbours = state.unsqueeze(1) * flip_signs.unsqueeze(0)     # (B, d, d)
    flat_neighbours = flip_neighbours.reshape(batch_size * n_sites, n_sites)
    t_per_neighbour = t.repeat_interleave(n_sites)                     # (B*d,)

    # Rate of returning to x from each flipped neighbour, i.e.
    # R_t(x, x_flip_i) under the paper's first-index-is-destination
    # convention. Equals the i-th model output evaluated AT x_flip_i.
    rates_at_flipped = model(flat_neighbours, t_per_neighbour).reshape(
        batch_size, n_sites, n_sites
    )
    return_rates = rates_at_flipped.diagonal(dim1=1, dim2=2)           # (B, d)

    log_p_tilde_at_state = target.log_p_tilde_t(state, t)
    log_p_tilde_at_flips = target.log_p_tilde_t(
        flat_neighbours, t_per_neighbour
    ).reshape(batch_size, n_sites)
    neighbour_ratio = (
        log_p_tilde_at_flips - log_p_tilde_at_state.unsqueeze(-1)
    ).exp()                                                            # (B, d)

    inflow_sum = (return_rates * neighbour_ratio).sum(dim=-1)          # (B,)
    dt_log_p_tilde_at_state = target.dt_log_p_tilde_t(state, t)        # (B,)

    return dt_log_p_tilde_at_state + outflow_sum - inflow_sum


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

    for step in range(len(ts) - 1):
        t_curr = ts[step]
        step_dt = ts[step + 1] - ts[step]
        t_per_batch = t_curr.expand(batch_size)  # (B,)

        # Rates at the current state x_t. These are R_t(x_flip_i, x) -- the
        # rate of leaving x by flipping site i. Shape (B, d). Used both for
        # the Euler step below and (when integrating IS log-weights)
        # threaded into compute_xi_t to avoid a duplicate forward pass.
        outflow_rates = model(state, t_per_batch)

        if return_log_weights:
            # ξ_t per paper Eq. 8 evaluated at x_t (left endpoint of the
            # Euler interval -- standard forward Euler). compute_xi_t
            # encapsulates the inflow/outflow decomposition; same helper is
            # called by samplers.log_z_estimators.control_variate at training
            # time for Stage 2.
            xi_t = compute_xi_t(
                state, t_per_batch, model, target,
                outflow_rates=outflow_rates,
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
