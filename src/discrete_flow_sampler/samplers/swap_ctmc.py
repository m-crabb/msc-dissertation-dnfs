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
    _log_p_tilde_at_swap_neighbours,
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
    G_edge = gather_pair_scores(head(x, t), pairs)          # (B, P), i<j
    G_plus = F.relu(G_edge)
    neg_G_plus = F.relu(-G_edge)
    log_p_neighbours = _log_p_tilde_at_swap_neighbours(x, t, target)
    log_p_x = target.log_p_tilde_t(x, t)
    log_ratio = (log_p_neighbours - log_p_x[:, None]).clamp(max=SWAP_LOG_RATIO_CLAMP)
    outflow = G_plus.sum(dim=-1)                            # (B,)
    inflow = (neg_G_plus * log_ratio.exp()).sum(dim=-1)     # (B,)
    return target.dt_log_p_tilde_t(x, t) + outflow - inflow


def _euler_step_swap(head, state: Tensor, t_per_batch: Tensor, step_dt: Tensor):
    """One-event swap Euler step. Returns (new_state, forward_rates) (B, n_pairs).

    Single global categorical over the i<j pairs plus a stay slot: at most one
    composition-preserving swap fires per step. Edges sharing a vertex conflict,
    so per-edge independent firing (single-site tau-leaping) is invalid here.
    Correct to O(dt^2). When Λ·dt > 1 the stay slot clamps to 0 and
    torch.multinomial renormalises the pair probabilities, so exactly one swap
    fires that step; logging the Λ·dt>1 clip fraction is the caller's
    responsibility (the train driver).
    """
    batch_size, d = state.shape
    pairs = upper_tri_pairs(d, state.device)                       # (P, 2)
    n_pairs = pairs.shape[0]
    forward_rates = F.relu(gather_pair_scores(head(state, t_per_batch), pairs))
    step_probs = (forward_rates * step_dt).clamp(0.0, 1.0)         # (B, P)
    stay_prob = (1.0 - step_probs.sum(dim=-1)).clamp(0.0, 1.0)     # (B,)
    categorical = torch.cat([step_probs, stay_prob[:, None]], dim=-1)  # (B, P+1)
    choice = torch.multinomial(categorical, num_samples=1).squeeze(-1)  # (B,)

    new_state = state.clone()
    fired = choice < n_pairs
    if fired.any():
        rows = torch.nonzero(fired, as_tuple=False).squeeze(-1)
        chosen = pairs[choice[rows]]                               # (K, 2)
        site_i, site_j = chosen[:, 0], chosen[:, 1]
        spin_i = new_state[rows, site_i].clone()
        new_state[rows, site_i] = new_state[rows, site_j]
        new_state[rows, site_j] = spin_i
    return new_state, forward_rates


def sample_swap_ctmc(
    head, x0: Tensor, ts: Tensor, *,
    return_log_weights: bool = False,
    return_all_states: bool = False,
    target=None,
):
    """Swap-CTMC trajectory sampler. Same contract as `ctmc.sample_ctmc`.

    Composition-preserving: every state stays on the fixed-N slice. IS weights
    accumulate ξ_t·dt at the left endpoint (Eq. 8, swap form).
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
        if return_log_weights else None
    )
    if return_all_states:
        trajectory = torch.empty(
            (len(ts), batch_size, d), dtype=state.dtype, device=state.device
        )
        trajectory[0] = state

    for step in range(len(ts) - 1):
        step_dt = ts[step + 1] - ts[step]
        t_per_batch = ts[step].expand(batch_size)
        new_state, _ = _euler_step_swap(head, state, t_per_batch, step_dt)
        if return_log_weights:
            xi_t = compute_xi_t_swap(state, t_per_batch, head, target)
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
