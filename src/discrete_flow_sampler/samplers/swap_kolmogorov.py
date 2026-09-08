"""Swap-move Kolmogorov residual and squared-residual loss.

The single-site residual (Eq. 10) sums over d single-flip neighbours; the swap
residual sums over the i<j opposite-spin pairs, with the reverse rate read off
the same tensor via the readout's exact antisymmetry
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
    G_edge = gather_pair_scores(head(x, t), pairs)  # (B, P), i<j
    G_plus = F.relu(G_edge)
    neg_G_plus = F.relu(-G_edge)
    log_ratio = target.swap_log_ratio(x, t, pairs).clamp(max=SWAP_LOG_RATIO_CLAMP)
    site_terms = (G_plus - neg_G_plus * log_ratio.exp()).sum(dim=-1)  # (B,)
    dt_log_pt_x = target.dt_log_p_tilde_t(x, t) - dt_log_Zt
    return dt_log_pt_x + site_terms


def loss_swap(
    x: Tensor, t: Tensor, dt_log_Zt, head, target, *, return_residual: bool = False
):
    """Mean-squared swap residual over the batch. Scalar.

    `return_residual=True` additionally hands back the sanitised per-state
    residual the loss squares, so a caller can measure its mean without a
    second forward pass. Default stays scalar-only: `profile_swap.py` and the
    archived tests call this positionally and must not see a tuple.
    """
    residual = residual_swap(x, t, dt_log_Zt, head, target)
    residual = residual.nan_to_num(posinf=1.0, neginf=-1.0, nan=0.0)
    loss = residual.pow(2).mean()
    return (loss, residual) if return_residual else loss


def loss_swap_backward_microbatched(
    x: Tensor,
    t: Tensor,
    dt_log_Zt,
    head,
    target,
    *,
    microbatch_size: int | None,
    slice_grad_sqnorms_out: list | None = None,
) -> tuple[Tensor, Tensor]:
    """`loss_swap(...).backward()` with the retained graph bounded to
    `microbatch_size` rows. Returns (detached full-batch loss, detached
    per-state residual); gradients are left accumulated in `head`'s .grad
    buffers, so the caller must zero_grad() first.

    The loss is a per-row mean (row-wise residual, per-row `dt_log_Zt`
    gather, row-wise nan_to_num), so mean_N = sum_k (n_k/N) * mean_slice_k
    and autograd's linearity carries this to the gradient: backwarding each
    slice's weighted loss sums to the single-backward gradient up to float
    summation order (tests/test_loss_microbatch_parity.py). A batch-coupled
    objective (self-normalised weights, batch statistics) would break this.
    Peak memory drops by ~N/microbatch_size at unchanged FLOPs, with no
    recompute forward. `microbatch_size=None` (or >= the batch) is the
    archived single-backward path, op-for-op.

    `slice_grad_sqnorms_out`, when a list, collects one `(rows, sqnorm)` pair
    per slice, `sqnorm` the squared norm of that slice's unweighted gradient
    (the E|g_b|^2 ingredient of the McCandlish gradient-noise-scale
    estimator, diagnostics.metrics.gradient_noise_scale_components; |g_N| is
    the trainer's pre-clip grad_norm). The accumulated-grad increment after
    slice k is (n_k/N) * g_slice_k, so the unweighted gradient is recovered
    by rescaling with N/n_k. Collection only reads `.grad` between slice
    backwards; the single-backward path never touches the list, so callers
    log NaN from an empty list.
    """
    batch_size = x.shape[0]
    if microbatch_size is None or microbatch_size >= batch_size:
        loss, residual = loss_swap(x, t, dt_log_Zt, head, target, return_residual=True)
        loss.backward()
        return loss.detach(), residual.detach()

    c_t_is_per_row = torch.is_tensor(dt_log_Zt) and dt_log_Zt.ndim >= 1
    loss_total = torch.zeros((), device=x.device)
    residual_slices = []
    previous_grads: list[Tensor | None] | None = None
    if slice_grad_sqnorms_out is not None:
        previous_grads = [
            None if p.grad is None else p.grad.detach().clone()
            for p in head.parameters()
        ]
    for start in range(0, batch_size, microbatch_size):
        rows = slice(start, start + microbatch_size)
        c_t_rows = dt_log_Zt[rows] if c_t_is_per_row else dt_log_Zt
        slice_loss, slice_residual = loss_swap(
            x[rows], t[rows], c_t_rows, head, target, return_residual=True
        )
        slice_rows = slice_residual.shape[0]
        slice_weight = slice_rows / batch_size
        (slice_loss * slice_weight).backward()
        loss_total += slice_loss.detach() * slice_weight
        residual_slices.append(slice_residual.detach())
        if slice_grad_sqnorms_out is not None:
            increment_sqnorm = 0.0
            for param_index, param in enumerate(head.parameters()):
                if param.grad is None:
                    continue
                grad_now = param.grad.detach()
                previous = previous_grads[param_index]
                increment = grad_now if previous is None else grad_now - previous
                increment_sqnorm += float(increment.pow(2).sum())
                previous_grads[param_index] = grad_now.clone()
            slice_grad_sqnorms_out.append(
                (slice_rows, increment_sqnorm / slice_weight**2)
            )
    return loss_total, torch.cat(residual_slices)


def c_t_offset_rms(residual_sum: Tensor, residual_count: Tensor) -> float:
    """RMS over time slots of Δ_t ≜ E_q[ξ_t] − c_t. Scalar float.

    With c_t detached (the paper's stop-gradient treatment), one slot's
    objective decomposes as

        E_q[(ξ_t − c_t)^2] = Var_q[ξ_t] + Δ_t^2 ,   Δ_t = E_q[ξ_t] − c_t ,

    so only Δ_t reaches the gradient, as a rank-one term 2·Δ_t·E_q[∇_θ ξ_t]
    that shifts ξ uniformly; at Δ_t = 0 the detached gradient equals the
    exact variance gradient. The trainer estimates c_t from the current
    cycle's rollout but averages the loss over a replay buffer of past
    models' states, so Δ_t ≠ 0 by construction. Since −E_q[residual] = Δ_t,
    the caller accumulates residuals into per-slot (sum, count) tensors over
    one outer cycle (the window over which c_t is fixed) and this reduces
    them. Per-slot RMS rather than a signed batch mean because Δ_t^2 is paid
    in every slot independently and opposite-sign offsets would cancel in a
    mean. Unvisited slots are excluded rather than counted as zero; returns
    NaN when no slot has been visited (the trainer's convention for an
    unopened window).
    """
    seen = residual_count > 0
    if not bool(seen.any()):
        return float("nan")
    delta_per_slot = residual_sum[seen] / residual_count[seen]
    return delta_per_slot.pow(2).mean().sqrt().item()
