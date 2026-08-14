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
    gather_pair_scores,
    upper_tri_pairs,
)


def residual_swap(x: Tensor, t: Tensor, dt_log_Zt, head, target) -> Tensor:
    """Per-state swap Kolmogorov residual δ_t(x) (Eq. 10, swap form). (B,)."""
    pairs = upper_tri_pairs(x.shape[1], x.device)
    G_edge = gather_pair_scores(head(x, t), pairs)          # (B, P), i<j
    G_plus = F.relu(G_edge)
    neg_G_plus = F.relu(-G_edge)
    log_ratio = target.swap_log_ratio(x, t, pairs).clamp(max=SWAP_LOG_RATIO_CLAMP)
    site_terms = (G_plus - neg_G_plus * log_ratio.exp()).sum(dim=-1)   # (B,)
    dt_log_pt_x = target.dt_log_p_tilde_t(x, t) - dt_log_Zt
    return dt_log_pt_x + site_terms


def loss_swap(x: Tensor, t: Tensor, dt_log_Zt, head, target, *,
              return_residual: bool = False):
    """Mean-squared swap residual over the batch. Scalar.

    `return_residual=True` additionally hands back the SANITISED per-state
    residual the loss squares, so a caller can measure its mean without a
    second forward pass. Default stays scalar-only: `profile_swap.py` and the
    archived tests call this positionally and must not see a tuple.
    """
    residual = residual_swap(x, t, dt_log_Zt, head, target)
    residual = residual.nan_to_num(posinf=1.0, neginf=-1.0, nan=0.0)
    loss = residual.pow(2).mean()
    return (loss, residual) if return_residual else loss


def c_t_offset_rms(residual_sum: Tensor, residual_count: Tensor) -> float:
    """RMS over time slots of Δ_t ≜ E_q[ξ_t] − c_t. Scalar float.

    WHAT THIS MEASURES. With c_t detached (the paper's stop-gradient
    treatment), one slot's objective decomposes exactly as

        E_q[(ξ_t − c_t)^2] = Var_q[ξ_t] + Δ_t^2 ,   Δ_t = E_q[ξ_t] − c_t ,

    so c_t's VALUE never reaches the gradient — only Δ_t does, and it arrives
    as a rank-one term 2·Δ_t·E_q[∇_θ ξ_t] that shifts ξ uniformly instead of
    narrowing it. At Δ_t = 0 the detached gradient equals the exact variance
    gradient identically. The trainer estimates c_t from the current cycle's
    fresh rollout but averages the loss over a replay buffer holding several
    past models' states, so Δ_t ≠ 0 by construction.

    Since −E_q[residual] is exactly Δ_t, the caller accumulates residuals into
    per-slot (sum, count) tensors across one outer cycle — the window over
    which c_t is held fixed — and this reduces them.

    WHY PER-SLOT RMS, not the signed mean of a mixed batch. The objective pays
    Δ_t^2 in every slot independently, so offsets of opposite sign across
    slots ADD to the damage while cancelling in a signed average: a batch mean
    would report 0.0 on a maximally mismatched run. The sign is discarded for
    the same reason a variance discards it.

    Slots no inner batch drew are excluded rather than counted as zero, which
    would dilute the RMS toward a falsely healthy reading. Returns NaN when no
    slot has been visited, matching the trainer's convention for a diagnostic
    whose window has not opened yet.
    """
    seen = residual_count > 0
    if not bool(seen.any()):
        return float("nan")
    delta_per_slot = residual_sum[seen] / residual_count[seen]
    return delta_per_slot.pow(2).mean().sqrt().item()
