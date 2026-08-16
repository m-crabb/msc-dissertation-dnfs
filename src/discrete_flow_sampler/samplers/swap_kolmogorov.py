"""Swap-move Kolmogorov residual and squared-residual loss.

The single-site residual (Eq. 10) sums over d single-flip neighbours; the swap
residual sums over the i<j opposite-spin pairs, with the reverse rate read off
the SAME tensor via the readout's exact antisymmetry
    R(y_ij -> x) = [G_swap(i,j | y_ij)]_+ = [-G_swap(i,j | x)]_+ .
Same-spin pairs vanish for free (G_swap = 0, y_ij = x), so summing all i<j is
correct without masking.
"""
import torch
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


def loss_swap_backward_microbatched(
    x: Tensor, t: Tensor, dt_log_Zt, head, target, *,
    microbatch_size: int | None,
) -> tuple[Tensor, Tensor]:
    """`loss_swap(...).backward()` with the retained graph bounded to
    `microbatch_size` rows. Returns (detached full-batch loss, detached
    per-state residual); gradients are left ACCUMULATED in `head`'s .grad
    buffers, so the caller must zero_grad() first.

    WHY THIS IS NOT A RECIPE CHANGE. The loss is a per-row mean — the
    residual (Eq. 10, swap form) is computed row-wise, `dt_log_Zt` is a
    per-row gather from a grid held fixed across the cycle, and the
    nan_to_num sanitiser is row-wise — so the batch mean decomposes exactly
    as mean_N = sum_k (n_k/N) * mean_slice_k, and autograd's linearity
    carries that identity to the gradient: backwarding each slice's
    weighted loss sums to the single-backward gradient, bit-for-bit up to
    float summation order. Clipping and the optimiser step then see the
    same total gradient (tests/test_loss_microbatch_parity.py). This would
    NOT hold for a batch-coupled objective (self-normalised weights, batch
    statistics); anything of that kind added to loss_swap breaks the
    parity test before it breaks a run.

    WHAT IT BUYS. Peak training memory is the retained autograd graph,
    linear in batch rows (for mask_one, each row additionally rides its d
    stacked anchor passes). Slicing frees each slice's graph at its own
    backward, so the peak drops by ~N/microbatch_size at unchanged total
    FLOPs — unlike activation checkpointing, which pays a recompute
    forward. This is what makes the two measured A100-80GB OOM arms of the
    16x16 screen (masked-attention h128; mask_one) launchable.

    `microbatch_size=None` (or >= the batch) is the archived single-
    backward path, op-for-op — the default every queued or archived cell
    runs, pinned bit-exact by the parity test.
    """
    batch_size = x.shape[0]
    if microbatch_size is None or microbatch_size >= batch_size:
        loss, residual = loss_swap(
            x, t, dt_log_Zt, head, target, return_residual=True
        )
        loss.backward()
        return loss.detach(), residual.detach()

    c_t_is_per_row = torch.is_tensor(dt_log_Zt) and dt_log_Zt.ndim >= 1
    loss_total = torch.zeros((), device=x.device)
    residual_slices = []
    for start in range(0, batch_size, microbatch_size):
        rows = slice(start, start + microbatch_size)
        c_t_rows = dt_log_Zt[rows] if c_t_is_per_row else dt_log_Zt
        slice_loss, slice_residual = loss_swap(
            x[rows], t[rows], c_t_rows, head, target, return_residual=True
        )
        slice_weight = slice_residual.shape[0] / batch_size
        (slice_loss * slice_weight).backward()
        loss_total += slice_loss.detach() * slice_weight
        residual_slices.append(slice_residual.detach())
    return loss_total, torch.cat(residual_slices)


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
