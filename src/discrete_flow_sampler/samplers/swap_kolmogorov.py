"""Swap-move Kolmogorov residual and squared-residual loss.

The single-site residual (Eq. 10) sums over d single-flip neighbours; the swap
residual sums over the i<j opposite-spin pairs, with the reverse rate read off
the SAME tensor via the readout's exact antisymmetry
    R(y_ij -> x) = [G_swap(i,j | y_ij)]_+ = [-G_swap(i,j | x)]_+ .
Same-spin pairs vanish for free (G_swap = 0, y_ij = x), so summing all i<j is
correct without masking.
"""
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.samplers._swap_neighbours import (
    SWAP_LOG_RATIO_CLAMP,
    _log_p_tilde_at_swap_neighbours,
    gather_pair_scores,
    upper_tri_pairs,
)


def residual_swap(x: Tensor, t: Tensor, dt_log_Zt, head, target) -> Tensor:
    """Per-state swap Kolmogorov residual δ_t(x) (Eq. 10, swap form). (B,)."""
    pairs = upper_tri_pairs(x.shape[1], x.device)
    G_edge = gather_pair_scores(head(x, t), pairs)          # (B, P), i<j
    G_plus = F.relu(G_edge)
    neg_G_plus = F.relu(-G_edge)
    log_p_neighbours = _log_p_tilde_at_swap_neighbours(x, t, target)
    log_p_x = target.log_p_tilde_t(x, t)
    log_ratio = (log_p_neighbours - log_p_x[:, None]).clamp(max=SWAP_LOG_RATIO_CLAMP)
    site_terms = (G_plus - neg_G_plus * log_ratio.exp()).sum(dim=-1)   # (B,)
    dt_log_pt_x = target.dt_log_p_tilde_t(x, t) - dt_log_Zt
    return dt_log_pt_x + site_terms


def loss_swap(x: Tensor, t: Tensor, dt_log_Zt, head, target) -> Tensor:
    """Mean-squared swap residual over the batch. Scalar."""
    residual = residual_swap(x, t, dt_log_Zt, head, target)
    residual = residual.nan_to_num(posinf=1.0, neginf=-1.0, nan=0.0)
    return residual.pow(2).mean()
