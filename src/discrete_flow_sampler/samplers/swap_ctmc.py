"""Swap-move CTMC: IS integrand ξ_t, one-event Euler step, sampler, c_t grid.

Mirrors ctmc.py / log_z_estimators.py for the swap move set. ξ_t and the Euler
step reuse the merged swap readout head; the residual/ξ_t read one i<j
representative per unordered pair so the single-pass reverse rate is exact.
"""
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._swap_neighbours import (
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
    log_ratio = (log_p_neighbours - log_p_x[:, None]).clamp(max=5.0)
    outflow = G_plus.sum(dim=-1)                            # (B,)
    inflow = (neg_G_plus * log_ratio.exp()).sum(dim=-1)     # (B,)
    return target.dt_log_p_tilde_t(x, t) + outflow - inflow
