"""Swap-move CTMC: IS integrand ξ_t, one-event Euler step, sampler, c_t grid.

Mirrors ctmc.py / log_z_estimators.py for the swap move set. ξ_t and the Euler
step reuse the merged swap readout head; the residual/ξ_t read one i<j
representative per unordered pair so the single-pass reverse rate is exact.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._swap_neighbours import (
    SWAP_LOG_RATIO_CLAMP,
    gather_pair_scores,
    upper_tri_pairs,
)


def compute_xi_t_swap(x: Tensor, t: Tensor, head, target) -> Tensor:
    """Per-state swap IS integrand ξ_t (Eq. 8, swap form). (B,).

    E_{p_t}[ξ_t] = ∂_t log Z_t for any valid rate matrix (the outflow/inflow
    expectations cancel by a dummy-index relabel), so this is the swap c_t
    integrand and the eval IS-weight integrand.
    """
    pairs = upper_tri_pairs(x.shape[1], x.device)
    pair_scores = gather_pair_scores(head(x, t), pairs)  # (B, P), i<j
    return xi_t_swap_from_scores(pair_scores, x, t, target, pairs)


def xi_t_swap_from_scores(
    pair_scores: Tensor, x: Tensor, t: Tensor, target, pairs: Tensor
) -> Tensor:
    """ξ_t from already-gathered pair scores G[i,j], i<j. (B,).

    The head forward dominates eval wall-clock; the Euler step evaluates the
    head on the same (state, t) this integrand needs, so the eval loop passes
    the step's scores here instead of calling the head a second time.
    """
    G_plus = F.relu(pair_scores)
    neg_G_plus = F.relu(-pair_scores)
    log_ratio = target.swap_log_ratio(x, t, pairs).clamp(max=SWAP_LOG_RATIO_CLAMP)
    outflow = G_plus.sum(dim=-1)  # (B,)
    inflow = (neg_G_plus * log_ratio.exp()).sum(dim=-1)  # (B,)
    return target.dt_log_p_tilde_t(x, t) + outflow - inflow


def _euler_step_swap(head, state: Tensor, t_per_batch: Tensor, step_dt: Tensor):
    """One-event swap Euler step. Returns (new_state, pair_scores) (B, n_pairs).

    Single global categorical over the i<j pairs plus a stay slot: at most one
    composition-preserving swap fires per step. Edges sharing a vertex conflict,
    so per-edge independent firing (single-site tau-leaping) is invalid here.
    Correct to O(dt^2). When Λ·dt > 1 the stay slot clamps to 0 and
    torch.multinomial renormalises the pair probabilities, so exactly one swap
    fires that step; logging the Λ·dt>1 clip fraction is the caller's
    responsibility (the train driver).

    The second return is the RAW gathered head output G[i,j] (relu applied
    internally where rates are needed), so the eval loop can reuse this one
    head call for the ξ_t integrand.
    """
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)  # (P, 2)
    n_pairs = pairs.shape[0]
    pair_scores = gather_pair_scores(head(state, t_per_batch), pairs)  # (B, P)
    forward_rates = F.relu(pair_scores)
    step_probs = (forward_rates * step_dt).clamp(0.0, 1.0)  # (B, P)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)  # (B,)
    categorical = torch.cat([step_probs, stay_prob[:, None]], dim=-1)  # (B, P+1)
    choice = torch.multinomial(categorical, num_samples=1).squeeze(-1)  # (B,)

    fired = choice < n_pairs
    chosen = pairs[choice.clamp(max=n_pairs - 1)]  # (B, 2); stay rows dummy
    site_i = torch.where(fired, chosen[:, 0], chosen.new_zeros(()))
    site_j = torch.where(fired, chosen[:, 1], chosen.new_zeros(()))
    # Branch-free swap-or-identity permutation per row: stay rows map site
    # 0 -> 0 (a no-op), so no `.any()`/`nonzero()` host-device sync.
    perm = torch.arange(d, device=state.device).expand(batch_size, d).clone()
    perm.scatter_(1, site_i[:, None], site_j[:, None])
    perm.scatter_(1, site_j[:, None], site_i[:, None])
    return state.gather(1, perm), pair_scores


def _vertex_disjoint_matching(proposed, priority, pairs, d, max_rounds=8):
    """Extract a vertex-disjoint subset (matching) of the `proposed` pairs.

    Luby-style rounds: a pair is a winner iff it holds the highest priority
    among all still-active pairs at BOTH its endpoints; winners are then
    vertex-disjoint by construction (distinct priorities ⇒ a vertex is the max
    for ≤1 pair). Winning vertices are retired and pairs touching them
    deactivated, then repeat. Driving toward a MAXIMAL matching drops only
    genuinely unresolvable conflicts, keeping the per-pair firing rate close to
    the proposal rate (the property the IS weight relies on). Returns a (B, P)
    bool mask ⊆ `proposed`.
    """
    batch_size, n_pairs = proposed.shape
    idx_i = pairs[:, 0].view(1, n_pairs).expand(batch_size, n_pairs)
    idx_j = pairs[:, 1].view(1, n_pairs).expand(batch_size, n_pairs)
    neg_inf = torch.finfo(priority.dtype).min
    active = proposed.clone()
    accepted = torch.zeros_like(proposed)
    for _ in range(max_rounds):
        if not active.any():
            break
        pr = torch.where(active, priority, neg_inf)  # (B, P)
        vmax = torch.full(
            (batch_size, d), neg_inf, dtype=priority.dtype, device=priority.device
        )
        vmax.scatter_reduce_(1, idx_i, pr, reduce="amax", include_self=True)
        vmax.scatter_reduce_(1, idx_j, pr, reduce="amax", include_self=True)
        win = active & (pr == vmax.gather(1, idx_i)) & (pr == vmax.gather(1, idx_j))
        accepted |= win
        used = torch.zeros(
            (batch_size, d), dtype=priority.dtype, device=priority.device
        )
        win_f = win.to(priority.dtype)
        used.scatter_reduce_(1, idx_i, win_f, reduce="amax", include_self=True)
        used.scatter_reduce_(1, idx_j, win_f, reduce="amax", include_self=True)
        endpoint_used = (used.gather(1, idx_i) > 0) | (used.gather(1, idx_j) > 0)
        active = active & ~win & ~endpoint_used
    return accepted


def _apply_swaps(state: Tensor, accepted: Tensor, pairs: Tensor) -> Tensor:
    """Apply a matching of swaps simultaneously via a site permutation.

    `accepted` is vertex-disjoint, so each (row, site) is reassigned at most
    once — the permutation has no collisions and every row stays on the slice.
    Non-accepted pairs scatter into a dummy slot d that is dropped before the
    gather, so no `nonzero()` host-device sync is needed.
    """
    batch_size, d = state.shape
    site_i = pairs[:, 0].unsqueeze(0).expand(batch_size, -1)
    site_j = pairs[:, 1].unsqueeze(0).expand(batch_size, -1)
    dummy = torch.full_like(site_i, d)
    perm = (
        torch.arange(d + 1, device=state.device).expand(batch_size, d + 1).clone()
    )
    perm.scatter_(
        1, torch.where(accepted, site_i, dummy), torch.where(accepted, site_j, dummy)
    )
    perm.scatter_(
        1, torch.where(accepted, site_j, dummy), torch.where(accepted, site_i, dummy)
    )
    return state.gather(1, perm[:, :d])


def _euler_step_swap_matching(head, state: Tensor, t_per_batch: Tensor, step_dt):
    """Multi-event swap Euler step: fire a vertex-disjoint matching of pairs.

    Thin every pair by its firing probability rate·dt, then keep a random
    vertex-disjoint matching of the proposals (§B). Returns (new_state,
    pair_scores) to match `_euler_step_swap`'s contract, so `sample_swap_ctmc`
    can swap the two step kinds. Correct to O(dt): proposal conflicts are O(dt²)
    as dt → 0, so this collapses to the one-event step in that limit. The caller
    controls dt to hold the expected events per site per step ≤ 0.1 (pre-reg §6).
    """
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)
    pair_scores = gather_pair_scores(head(state, t_per_batch), pairs)  # (B, P)
    fire_prob = (F.relu(pair_scores) * step_dt).clamp(0.0, 1.0)
    proposed = torch.bernoulli(fire_prob).bool()
    priority = torch.rand(batch_size, pairs.shape[0], device=state.device)
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    return _apply_swaps(state, accepted, pairs), pair_scores


def sample_swap_ctmc(
    head,
    x0: Tensor,
    ts: Tensor,
    *,
    return_log_weights: bool = False,
    return_all_states: bool = False,
    target=None,
    multi_event: bool = False,
):
    """Swap-CTMC trajectory sampler. Same contract as `ctmc.sample_ctmc`.

    Composition-preserving: every state stays on the fixed-N slice. IS weights
    accumulate ξ_t·dt at the left endpoint (Eq. 8, swap form).

    `multi_event=True` uses the vertex-disjoint-matching step (many swaps/step,
    O(d) trajectory length at scale); the default one-event step fires ≤1
    swap/step (O(d²) steps at the critical coupling — followups §B).
    """
    if return_log_weights and target is None:
        raise ValueError("sample_swap_ctmc(return_log_weights=True) requires `target`.")
    if return_log_weights and return_all_states:
        raise ValueError(
            "return_all_states and return_log_weights are mutually exclusive."
        )

    state = x0.clone()
    batch_size, d = state.shape
    log_weights = (
        torch.zeros(batch_size, dtype=state.dtype, device=state.device)
        if return_log_weights
        else None
    )
    if return_all_states:
        trajectory = torch.empty(
            (len(ts), batch_size, d), dtype=state.dtype, device=state.device
        )
        trajectory[0] = state

    step_fn = _euler_step_swap_matching if multi_event else _euler_step_swap
    pairs = upper_tri_pairs(d, x0.device)
    dts = ts[1:] - ts[:-1]
    for step in range(len(ts) - 1):
        step_dt = dts[step]
        t_per_batch = ts[step].expand(batch_size)
        new_state, step_pair_scores = step_fn(head, state, t_per_batch, step_dt)
        if return_log_weights:
            # ξ_t at the left endpoint reads the same head(state, t) the step
            # just computed; reusing its scores halves the head calls per step.
            xi_t = xi_t_swap_from_scores(
                step_pair_scores, state, t_per_batch, target, pairs
            )
            log_weights = log_weights + xi_t * step_dt
        state = new_state
        if return_all_states:
            trajectory[step + 1] = state

    if return_log_weights:
        return state, log_weights
    if return_all_states:
        return trajectory
    return state


def compute_c_t_grid_swap(t_grid: Tensor, x_traj: Tensor, target, head, *, mode):
    """Per-time-slot c_t for the swap loss (mirror of compute_c_t_grid).

    mode='naive_mc'        -> c_t = mean_m ∂_t log p̃_t(x_t^{(m)})
    mode='control_variate' -> c_t = mean_m ξ_t^swap(x_t^{(m)})   (Eq. 8, swap form)
    """
    if mode not in ("naive_mc", "control_variate"):
        raise ValueError(f"Unknown mode {mode!r}")
    n_grid, outer_batch, _ = x_traj.shape
    integrand_per_t = torch.empty(
        (n_grid, outer_batch), dtype=x_traj.dtype, device=x_traj.device
    )
    with torch.no_grad():
        for k in range(n_grid):
            x_k = x_traj[k]
            t_k = t_grid[k].expand(outer_batch)
            if mode == "naive_mc":
                integrand_per_t[k] = target.dt_log_p_tilde_t(x_k, t_k)
            else:
                integrand_per_t[k] = compute_xi_t_swap(x_k, t_k, head, target)
    return integrand_per_t.mean(dim=-1), integrand_per_t
