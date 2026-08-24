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
The caller picks one (driven by what Eq. 8 / 13 in the paper specifies).

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
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._neighbours import (
    _log_p_tilde_at_neighbours,
    log_ratio_clamp,
)
from discrete_flow_sampler.samplers.resampling import (
    ResamplingConfig,
    ResamplingStats,
    resample_if_needed,
)


def _compute_xi_t_general(
    state: Tensor,
    t: Tensor,
    model,
    target,
    outflow_rates: Tensor | None = None,
) -> Tensor:
    """ξ_t for a non-LE model — two forward passes (paper Eq. 8)."""
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


def xi_t_lenet_from_scores(
    G_t: Tensor,
    state: Tensor,
    t: Tensor,
    target,
) -> Tensor:
    """ξ_t for a locally equivariant model from an already-computed G_t.

    Local equivariance (paper Eq. 20) means G(x_i, i | y_i) = -G(y_i, i | x),
    so the forward rate is [G]_+ and the return rate at each flipped
    neighbour is [-G]_+ of the SAME tensor — ξ_t is a single-forward
    quantity. The Euler step evaluates the model at exactly the (state, t)
    this integrand needs, so `sample_ctmc` passes the step's G_t here
    instead of re-running the model (the swap sampler makes the same move
    via `xi_t_swap_from_scores`); bit-identical because it is the same
    deterministic forward on the same tensors.
    """
    vocab_size = G_t.shape[-1]
    G_plus     = F.relu(G_t)
    neg_G_plus = F.relu(-G_t)

    log_p_neighbours = _log_p_tilde_at_neighbours(state, t, target, vocab_size)
    log_p_x = target.log_p_tilde_t(state, t)
    log_ratio = log_p_neighbours - log_p_x[:, None, None]
    log_ratio = log_ratio.clamp(max=log_ratio_clamp(target))
    neighbour_ratio = log_ratio.exp()                                   # (B, D, S)

    outflow_sum = G_plus.sum(dim=(-2, -1))                             # (B,)
    inflow_sum  = (neg_G_plus * neighbour_ratio).sum(dim=(-2, -1))     # (B,)
    dt_log_p_tilde_at_state = target.dt_log_p_tilde_t(state, t)        # (B,)

    return dt_log_p_tilde_at_state + outflow_sum - inflow_sum


def _compute_xi_t_lenet(
    state: Tensor,
    t: Tensor,
    model,
    target,
) -> Tensor:
    """ξ_t for a locally equivariant model — single forward pass, then
    delegates to `xi_t_lenet_from_scores` (which callers holding the Euler
    step's G_t use directly to skip this forward)."""
    return xi_t_lenet_from_scores(model(state, t), state, t, target)


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
      • `samplers.log_z_estimators.compute_c_t_grid` (control_variate
        mode) — averaged per time slot in the outer step as the c_t
        target for the inner-loop loss (paper Algorithm 1 line 4).

    Dispatches on `model.is_locally_equivariant`:
      - True  -> `_compute_xi_t_lenet`  (single forward pass, paper Eq. 8 LE form).
      - False -> `_compute_xi_t_general` (two forward passes, paper Eq. 8 general form).

    `outflow_rates` is the non-LE-only passthrough optimisation: in that
    branch `sample_ctmc` already computed `model(state, t)` for the Euler
    step and feeds it through to skip the duplicate forward pass.
    Outer-step c_t computation omits it and lets the helper recompute.
    The LE branch ignores the argument — its reuse path is
    `xi_t_lenet_from_scores`, fed the Euler step's pre-relu G_t directly
    by `sample_ctmc`.
    """
    if getattr(model, "is_locally_equivariant", False):
        return _compute_xi_t_lenet(state, t, model, target)
    return _compute_xi_t_general(state, t, model, target, outflow_rates)


def _euler_step(model, state: Tensor, t_per_batch: Tensor, step_dt: Tensor):
    """Take one forward-Euler CTMC step. Returns (new_state, step_scores).

    The second return is what ξ_t needs to avoid a duplicate forward pass
    at the same state: the non-LE path returns the (B, D) per-site flip
    rates (threaded into `compute_xi_t` as `outflow_rates`); the LE path
    returns the pre-relu (B, D, S) scores G_t, from which both [G]_+ and
    the reverse rate [-G]_+ are recoverable (`xi_t_lenet_from_scores`).
    """
    if getattr(model, "is_locally_equivariant", False):
        return _euler_step_lenet(model, state, t_per_batch, step_dt)
    return _euler_step_general(model, state, t_per_batch, step_dt)


def _euler_step_general(model, state: Tensor, t_per_batch: Tensor, step_dt: Tensor):
    """Existing binary-flip Euler step (single rate per site).

    Forward Euler flip step (paper Eq. 2): each site flips independently with
    probability outflow_rates * dt, clipped to [0, 1] for numerical safety
    when dt is too coarse. The clamp signals stiffness rather than masking
    it -- check post-hoc if many flip_prob entries hit 1.

    Returns (new_state, outflow_rates) where outflow_rates is (B, D).
    The caller re-uses outflow_rates to avoid a duplicate forward pass
    inside compute_xi_t.
    """
    outflow_rates = model(state, t_per_batch)
    flip_prob = (outflow_rates * step_dt).clamp(0.0, 1.0)
    uniforms = torch.rand_like(flip_prob)
    new_state = torch.where(uniforms < flip_prob, -state, state)
    return new_state, outflow_rates


def _euler_step_lenet(model, state: Tensor, t_per_batch: Tensor, step_dt: Tensor):
    """LE-path Euler step: per-site categorical over S values.

    Math (paper Eq. 2 specialised to R_t = [G]_+ under local equivariance):
        Pr[site i -> τ in step] = [G(τ, i | x)]_+ * dt   for τ ≠ x_i,
        Pr[site i stays]        = 1 - Σ_{τ ≠ x_i} [G(τ, i | x)]_+ * dt.

    For binary this collapses back to the non-LE single-flip-prob form,
    but we write the general categorical so the same code path serves
    S > 2 (alloy extension; future cells under `constrained_soft_02`).

    Implementation: build a (B, D, S+1) per-site categorical with the
    last slot = "stay", sample once via torch.multinomial on the
    flattened (B*D, S+1) tensor, then map indices back to spin values
    in {-1, +1} using spin_of_idx = 2*idx - 1 (with the stay-index
    routed to the original state value).

    Returns (new_state, G_t) — the PRE-relu (B, D, S) scores, not the
    rates: [G]_+ is recoverable from G_t but not vice versa, and ξ_t's
    reverse rate needs [-G]_+ of the same tensor (see
    `xi_t_lenet_from_scores`).
    """
    G_t = model(state, t_per_batch)
    R_t = F.relu(G_t)                                      # (B, D, S)

    step_probs = (R_t * step_dt).clamp(0.0, 1.0)           # (B, D, S)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)   # (B, D)
    cat_probs = torch.cat([step_probs, stay_prob.unsqueeze(-1)], dim=-1)
    # cat_probs: (B, D, S+1); last slot is stay.

    batch_size, n_sites = state.shape
    vocab_size = G_t.shape[-1]
    sampled_idx = torch.multinomial(
        cat_probs.reshape(batch_size * n_sites, vocab_size + 1),
        num_samples=1,
    ).reshape(batch_size, n_sites)

    spin_of_idx = (
        2.0 * torch.arange(vocab_size, device=state.device, dtype=state.dtype) - 1.0
    )
    stay_mask = sampled_idx == vocab_size
    sampled_spin = torch.where(
        stay_mask,
        state,
        spin_of_idx[sampled_idx.clamp(max=vocab_size - 1)],
    )
    return sampled_spin, G_t


def sample_ctmc(
    model,
    x0: Tensor,
    ts: Tensor,
    *,
    return_log_weights: bool = False,
    return_all_states: bool = False,
    target=None,
    resampling: ResamplingConfig | None = None,
    return_cv_integrand: bool = False,
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
            weights accumulated along the path. Mutually exclusive with
            `return_all_states`.
        return_all_states: if True, return the full (T, B, d) trajectory
            -- traj[0] == x0, traj[k] == state after k Euler steps. Used
            by the Algorithm 1 outer step to build the replay buffer.
            Mutually exclusive with `return_log_weights`.
        target: required when `return_log_weights=True`; used to evaluate
            `dt_log_p_tilde_t` (and, depending on xi_t form, `log_p_tilde_t`
            at flipped neighbours).
        resampling: optional `ResamplingConfig` enabling adaptive systematic
            resampling of the particle batch when interim ESS < τ·B
            (`samplers.resampling`). Needs `target`, plus one of
            `return_log_weights` (the eval mode) or `return_all_states` (the
            training-rollout mode — see `swap_ctmc.sample_swap_ctmc` for the
            full argument; the two samplers keep one contract). Applies
            unchanged to soft-tilted targets — resampling only touches
            (state, log_weights). The ξ_t the trigger reads per step reuses
            the Euler step's own forward for BOTH model kinds (non-LE: the
            outflow rates pass through; LE: the step returns the pre-relu
            G_t and `xi_t_lenet_from_scores` recovers both rate signs).
        return_cv_integrand: optimisation B1-flip (decided 2026-08-24;
            requires `return_all_states=True`, `target`, resampling OFF).
            Additionally returns the (T, B) per-slot CV integrand ξ_t
            (paper Eq. 8) accumulated from each Euler step's own forward
            — only the final slot needs one fresh model call — so the
            outer step can build the c_t grid without re-running the
            model on states it just visited. BIT-IDENTICAL to
            `log_z_estimators.compute_c_t_grid` (control_variate) on the
            returned trajectory: same tensors, same arithmetic, no RNG
            consumed (tests/test_cv_integrand_reuse.py). Return becomes
            (trajectory, cv_integrand).

    Returns:
        x_final: (B, d). The state at time `ts[-1]`.
        OR (when return_log_weights=True):
        (x_final, log_w) where log_w has shape (B,).
        OR (when return_all_states=True):
        (T, B, d) trajectory tensor.
        OR (when resampling is not None and return_log_weights=True):
        (x_final, log_w, ResamplingStats) — final-segment log_w plus the
        banked log-Z increments; combine via `smc_log_z_estimate`.
        OR (when resampling is not None and return_all_states=True):
        (trajectory, ResamplingStats) — no weights, deliberately: after the
        resets they are a per-segment residue, not the path's IS weights.
    """
    if return_log_weights and target is None:
        raise ValueError(
            "sample_ctmc(return_log_weights=True) requires `target` to be "
            "provided so xi_t can be evaluated along the trajectory."
        )
    if return_log_weights and return_all_states:
        raise ValueError(
            "sample_ctmc: return_all_states and return_log_weights are "
            "mutually exclusive -- the buffer-build path does not need IS "
            "weights, and the eval path does not need every intermediate "
            "state."
        )
    if resampling is not None and not (return_log_weights or return_all_states):
        raise ValueError(
            "sample_ctmc(resampling=...) requires return_log_weights=True or "
            "return_all_states=True -- the trigger lives on the weights, and "
            "there is nothing to hand back from a bare final state."
        )
    if resampling is not None and target is None:
        raise ValueError(
            "sample_ctmc(resampling=...) requires `target`: the ESS trigger "
            "reads log-weights, which are accumulated from xi_t."
        )
    if return_cv_integrand and not return_all_states:
        raise ValueError(
            "sample_ctmc(return_cv_integrand=True) requires "
            "return_all_states=True: the integrand slots are the "
            "trajectory's grid slots."
        )
    if return_cv_integrand and target is None:
        raise ValueError(
            "sample_ctmc(return_cv_integrand=True) requires `target`: "
            "xi_t reads the annealing density."
        )
    if return_cv_integrand and resampling is not None:
        raise ValueError(
            "sample_ctmc(return_cv_integrand=True) requires resampling "
            "OFF: the reuse is certified only for the plain buffer rollout."
        )

    state = x0.clone()
    batch_size, n_sites = state.shape

    # The trigger needs weights even when the caller does not want them back.
    accumulate_log_weights = return_log_weights or resampling is not None
    log_weights = (
        torch.zeros(batch_size, dtype=state.dtype, device=state.device)
        if accumulate_log_weights
        else None
    )
    smc_stats = (
        ResamplingStats(
            log_z_increment=torch.zeros((), dtype=state.dtype, device=state.device)
        )
        if resampling is not None
        else None
    )
    if return_all_states:
        trajectory = torch.empty(
            (len(ts), batch_size, n_sites),
            dtype=state.dtype, device=state.device,
        )
        trajectory[0] = state
    cv_integrand = (
        torch.empty(
            (len(ts), batch_size), dtype=state.dtype, device=state.device
        )
        if return_cv_integrand
        else None
    )
    model_is_locally_equivariant = getattr(
        model, "is_locally_equivariant", False
    )

    for step in range(len(ts) - 1):
        t_curr = ts[step]
        step_dt = ts[step + 1] - ts[step]
        t_per_batch = t_curr.expand(batch_size)

        new_state, step_scores = _euler_step(model, state, t_per_batch, step_dt)

        if accumulate_log_weights or return_cv_integrand:
            # ξ_t per paper Eq. 8 evaluated at x_t (left endpoint of the
            # Euler interval -- standard forward Euler), reusing the Euler
            # step's own forward: the non-LE step returns the (B, D) rate
            # vector compute_xi_t accepts as a passthrough; the LE step
            # returns the pre-relu G_t, from which xi_t_lenet_from_scores
            # recovers both [G]_+ and the reverse rate [-G]_+.
            if model_is_locally_equivariant:
                xi_t = xi_t_lenet_from_scores(
                    step_scores, state, t_per_batch, target
                )
            else:
                xi_t = compute_xi_t(
                    state, t_per_batch, model, target,
                    outflow_rates=step_scores,
                )
            if accumulate_log_weights:
                log_weights = log_weights + xi_t * step_dt
            if return_cv_integrand:
                # Slot k of the CV grid IS this ξ_t: `state` here equals
                # trajectory[step] and t_per_batch equals t_grid[step].
                cv_integrand[step] = xi_t

        state = new_state
        # Checkpoint AFTER the state advance: the particle carrying log w(t+dt)
        # is x_{t+dt}, so that is the row set resampling duplicates/kills.
        if resampling is not None and step % resampling.check_every == 0:
            state, log_weights, log_z_increment, fired = resample_if_needed(
                state, log_weights, resampling.ess_threshold_fraction
            )
            if fired:
                smc_stats.log_z_increment = (
                    smc_stats.log_z_increment + log_z_increment
                )
                smc_stats.n_events += 1
                smc_stats.event_steps.append(step)
        # Recorded after the checkpoint so the slice is the ensemble that
        # continues. Unchanged when resampling is off: `state` is returned
        # untouched by a checkpoint that does not fire.
        if return_all_states:
            trajectory[step + 1] = state

    if return_cv_integrand:
        # The loop covered slots 0..T-2 (each step's forward is at the
        # slot it STARTED from); the final state never gets an Euler step,
        # so its slot is the one fresh model call of the whole grid.
        cv_integrand[-1] = compute_xi_t(
            state, ts[-1].expand(batch_size), model, target
        )
        return trajectory, cv_integrand
    if return_all_states and resampling is not None:
        return trajectory, smc_stats
    if resampling is not None:
        return state, log_weights, smc_stats
    if return_log_weights:
        return state, log_weights
    if return_all_states:
        return trajectory
    return state
