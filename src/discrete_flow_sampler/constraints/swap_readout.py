"""Paired-swap antisymmetric rate readout (swap-DNFS, instantiation A).

A composition-preserving swap exchanges the spins at sites (i, j). The swap
rate is read off a context hollow in both sites against the token difference:

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

Doubly-hollow H_ij gives exact state-swap antisymmetry
    G_swap(i, j | x) = -G_swap(i, j | Swap2(x, i, j))
and trivial-swap vanishing (x_i = x_j => G_swap = 0) at random init. Two
heads share the readout and differ in how H_ij is built:

  * DoublyHollowSwapHead: mask both i and j (O(d^2) passes; correctness gate).
  * LeTFMaskOneSwapHead: mask anchor i, reuse the single-site leTF hollowness
    at j; the d anchor passes are batched into one stacked pass.

Both read H_ij at the second index j and agree numerically, since the leTF
readout at j already ignores j's own input. The heads are not label-symmetric
(H_ij != H_ji), so the swap residual must order each pair by site index
(i<j), never by spin.
"""

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.letf import LeTFRateMatrix


def swap2(x: Tensor, i: int, j: int) -> Tensor:
    """Return x with the contents of sites i and j exchanged. Involution."""
    y = x.clone()
    col_i = y[:, i].clone()
    y[:, i] = y[:, j]
    y[:, j] = col_i
    return y


def _masked_body(
    model: LeTFRateMatrix, x: Tensor, t: Tensor, mask_sites: Iterable[int]
) -> Tensor:
    """Run the leTF body with `mask_sites` content-free (embedding zeroed).

    Replicates LeTFRateMatrix.compute_body plus the output_norm + time line.
    Returns H of shape (B, d, hidden_dim); H[:, k, :] depends neither on x_k
    (leTF hollowness) nor on any masked site's value (the zeroing is an
    unconditional override).
    """
    x_idx = ((x + 1) / 2).long()
    x_emb = model.token_embedder(x_idx).clone()  # (B, d, h)
    for s in mask_sites:
        x_emb[:, s, :] = 0.0  # content-free override
    cond_t = model.time_embedder(t).unsqueeze(1)  # (B, 1, h)
    fwd_x = model.fwd_stack(torch.cat([cond_t, x_emb], dim=1))
    bwd_x = model.bwd_stack(torch.cat([cond_t, x_emb.flip(1)], dim=1)).flip(1)
    H = model.attention_readout(fwd_x, bwd_x, cond_t)  # (B, d, h)
    H = model.output_norm(H) + model.time_embedder(t).unsqueeze(1)
    return H


def _keep_masked_bodies(
    model: LeTFRateMatrix, x: Tensor, t: Tensor, keep: Tensor
) -> Tensor:
    """Batched `_masked_body` over a stack of keep-masks in one stacked pass.

    `keep` is (n_masks, d), 1.0 where the token embedding survives and 0.0
    where the site is forced content-free. The masked passes are independent,
    so they ride the model batch dimension as (n_masks*B, d, h); every kernel
    reduces over non-batch dims, so the result is bit-exact with the loop.
    Masking at the input means H never depends on a masked token at any
    depth, unlike the two-hop leak of the one-pass heads (interval_swap_head.py).

    Returns (n_masks, B, d, h): [a, :, j, :] is the body under mask a read at
    site j, blind to every site zeroed by keep[a] and hollow in x_j.
    """
    x_idx = ((x + 1) / 2).long()
    x_emb = model.token_embedder(x_idx)  # (B, d, h)
    batch, d, hidden = x_emb.shape
    n_masks = keep.shape[0]

    emb = x_emb.unsqueeze(0) * keep[:, None, :, None]  # (A, B, d, h)
    emb = emb.reshape(n_masks * batch, d, hidden)

    cond_t = model.time_embedder(t).unsqueeze(1)  # (B, 1, h)
    cond_t = cond_t.expand(n_masks, batch, 1, hidden).reshape(-1, 1, hidden)

    fwd_x = model.fwd_stack(torch.cat([cond_t, emb], dim=1))
    bwd_x = model.bwd_stack(torch.cat([cond_t, emb.flip(1)], dim=1)).flip(1)
    H = model.attention_readout(fwd_x, bwd_x, cond_t)  # (A*B, d, h)
    H = model.output_norm(H) + cond_t
    return H.reshape(n_masks, batch, d, hidden)


def _anchor_masked_bodies(
    model: LeTFRateMatrix, x: Tensor, t: Tensor, anchor_sites: Tensor
) -> Tensor:
    """Single-anchor case of `_keep_masked_bodies`: keep-mask a zeroes only
    site anchor_sites[a].

    Returns (n_anchors, B, d, h): [a, :, j, :] = H_ij for anchor i = anchor_sites[a].
    """
    keep = x.new_ones(anchor_sites.shape[0], model.d)
    keep[torch.arange(anchor_sites.shape[0], device=x.device), anchor_sites] = 0.0
    return _keep_masked_bodies(model, x, t, keep)


class DoublyHollowSwapHead(nn.Module):
    """Brute-force doubly-hollow swap head: mask both sites. Gate-only, O(d^2).

    For each unordered pair {i, j} one masked pass gives both entries:
    H[:, j] against omega_i - omega_j is G[i, j], H[:, i] against
    omega_j - omega_i is G[j, i]. Diagonal and same-spin pairs vanish because
    the token difference is zero. tests/test_swap_head_vectorised.py pins
    mask_one to this oracle at exact equality.
    """

    def __init__(self, backbone: LeTFRateMatrix):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        om = m.omega(x_idx)  # (B, d, h)
        batch, d = x.shape
        G = x.new_zeros(batch, d, d)
        for i in range(d):
            for j in range(i + 1, d):
                H = _masked_body(m, x, t, (i, j))  # (B, d, h)
                # Both differences written out rather than one negated: at a
                # same-spin pair the negation would produce -0.0 where the
                # subtraction produces +0.0, which is equal but not identical.
                G[:, i, j] = (H[:, j, :] * (om[:, i, :] - om[:, j, :])).sum(-1)
                G[:, j, i] = (H[:, i, :] * (om[:, j, :] - om[:, i, :])).sum(-1)
        return G


class LeTFMaskOneSwapHead(nn.Module):
    """Mask-one swap head: mask anchor i, reuse single-site leTF hollowness.

    For each anchor i one masked pass returns H_ij = H[:, j, :] for all
    j != i, blind to x_i and hollow in x_j; the readout against
    omega_{x_i} - omega_{x_j} gives the row G_swap(i, :). Not label-symmetric.

    forward batches the d anchor passes into the model batch dimension (the
    sequential loop took 56 s/forward at d=256). `anchor_chunk_size` caps
    anchors per stacked pass, since the readout attention buffer
    (n_anchors*B, n_heads, d, 2d) stops fitting memory at large d; None = all
    d anchors at once. forward_looped is the sequential test oracle.
    """

    def __init__(self, backbone: LeTFRateMatrix, anchor_chunk_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d
        self.anchor_chunk_size = anchor_chunk_size

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        om = m.omega(x_idx)  # (B, d, h)
        batch, d = x.shape
        chunk = self.anchor_chunk_size or d
        rows = []
        for start in range(0, d, chunk):
            anchors = torch.arange(start, min(start + chunk, d), device=x.device)
            H = _anchor_masked_bodies(m, x, t, anchors)  # (A, B, d, h)
            # diff[a, :, j, :] = om_{x_i} - om_{x_j} for anchor i = anchors[a]
            diff = om[:, anchors, :].permute(1, 0, 2).unsqueeze(2) - om.unsqueeze(0)
            rows.append((H * diff).sum(-1))  # (A, B, d); diagonal j==i -> 0
        return torch.cat(rows, dim=0).permute(1, 0, 2)  # (B, d, d)

    def forward_looped(self, x: Tensor, t: Tensor) -> Tensor:
        """Sequential oracle: one masked pass per anchor."""
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        om = m.omega(x_idx)  # (B, d, h)
        batch, d = x.shape
        G = x.new_zeros(batch, d, d)
        for i in range(d):
            H = _masked_body(m, x, t, (i,))  # (B, d, h); H[:, j, :] = H_ij
            diff = om[:, i : i + 1, :] - om  # (B, d, h): [:, j, :] = om_xi - om_xj
            G[:, i, :] = (H * diff).sum(-1)  # (B, d); diagonal j==i -> 0
        return G


def antisymmetrise(
    raw_head: Callable[[Tensor, Tensor], Tensor], x: Tensor, t: Tensor
) -> Tensor:
    """Explicit antisymmetrisation of any raw pair score.

    G_swap(i, j | x) := 1/2 [ raw(i, j | x) - raw(i, j | Swap2(x, i, j)) ]

    is exactly antisymmetric under Swap2 for any raw_head, no hollowness
    required. raw_head is a callable (x, t) -> (B, d, d). One swapped pass
    per ordered pair (O(d^2)): the unit-test oracle, not the efficient path.
    """
    base = raw_head(x, t)
    batch, d, _ = base.shape
    G = base.new_zeros(batch, d, d)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            swapped = raw_head(swap2(x, i, j), t)
            G[:, i, j] = 0.5 * (base[:, i, j] - swapped[:, i, j])
    return G
