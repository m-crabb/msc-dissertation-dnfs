"""Paired-swap antisymmetric rate readout (swap-DNFS, instantiation A).

See docs/design/2026-06-29-swap-readout-antisymmetry-design.md.

A composition-preserving swap exchanges the spins at two sites (i, j). The swap
rate is read off a context HOLLOW IN BOTH sites against the token difference:

    G_swap(i, j | x) = < H_ij(x_{-{i,j}}),  omega_{x_i} - omega_{x_j} >

Doubly-hollow H_ij gives EXACT state-swap antisymmetry
    G_swap(i, j | x) = -G_swap(i, j | Swap2(x, i, j))
and free trivial-swap vanishing (x_i = x_j => G_swap = 0), at random init with
no training. Two heads share the readout and differ only in how H_ij is built:

  * DoublyHollowSwapHead - brute-force: mask BOTH i and j (O(d^2) passes; the
    architecture-agnostic correctness gate).
  * LeTFMaskOneSwapHead  - climax head: mask anchor i, reuse the single-site
    leTF hollowness at j. The d anchor passes are independent, so forward
    batches them into the model batch dimension (one stacked pass, optionally
    chunked for memory); forward_looped keeps the O(d)-sequential-pass
    reference.

Both read H_ij at the SECOND index j, so they agree numerically: the leTF
readout at j already ignores j's own input, so additionally masking j is a
no-op. The heads are NOT label-symmetric (H_ij != H_ji), so the downstream
swap residual must order each unordered pair by site index (i<j), never by
spin (design note section 4). LeTFRateMatrix is reused untouched.
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

    Replicates LeTFRateMatrix.compute_body plus the per-position output_norm +
    time line (letf.py:332), leaving LeTFRateMatrix untouched. Returns the
    hollow body H of shape (B, d, hidden_dim). H[:, k, :] never depends on x_k
    (leTF slice-and-mask hollowness) nor on any masked site's value (the
    zeroing is an unconditional override, independent of the true token).
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
    """Batched `_masked_body` over a stack of keep-masks: ONE stacked pass.

    `keep` is (n_masks, d) with 1.0 at sites whose token embedding survives
    and 0.0 at sites forced content-free. The masked passes are independent --
    they differ only in WHICH sites are zeroed -- so they ride the model batch
    dimension: build (n_masks*B, d, h), run the fwd/bwd stacks + readout once,
    and reshape back. Every kernel in the path reduces over non-batch dims, so
    each batch element's arithmetic (including reduction order) is identical
    to its looped counterpart -- observed bit-exact on CPU.

    Zeroing is an unconditional override (multiply by a value-independent
    mask), so H never depends on the true token at any masked site. That is
    the whole blindness argument, and it is why depth is free here: the
    masking is at the INPUT, so the two-hop leak that forces the one-pass
    heads' band content to be shallow (interval_swap_head.py) never arises.

    Returns (n_masks, B, d, h): [a, :, j, :] is the body under mask a read at
    site j -- blind to every site zeroed by keep[a] AND hollow in x_j (leTF
    single-site hollowness: the readout at j ignores j's own input).
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
    """Batched `_masked_body` over single-site anchors: ONE stacked pass.

    The single-anchor special case of `_keep_masked_bodies`: keep-mask a
    zeroes exactly site anchor_sites[a] (a diagonal zeroing mask). Delegating
    is arithmetically identical to building `emb` here, so the mask-one head's
    numerics are unchanged.

    Returns (n_anchors, B, d, h): [a, :, j, :] = H_ij for anchor i = anchor_sites[a].
    """
    keep = x.new_ones(anchor_sites.shape[0], model.d)
    keep[torch.arange(anchor_sites.shape[0], device=x.device), anchor_sites] = 0.0
    return _keep_masked_bodies(model, x, t, keep)


class DoublyHollowSwapHead(nn.Module):
    """Brute-force doubly-hollow swap head: mask BOTH sites. Gate-only, O(d^2).

    Architecture-agnostic correctness check. For each ordered pair (i, j) it
    masks i and j, reads the body at j, and reads out against the token
    difference. The diagonal (i == j) and same-spin pairs vanish automatically
    because omega_{x_i} - omega_{x_j} = 0 there.
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
            for j in range(d):
                if i == j:
                    continue
                H = _masked_body(m, x, t, (i, j))  # (B, d, h)
                H_ij = H[:, j, :]  # read at second index
                diff = om[:, i, :] - om[:, j, :]  # omega_{x_i} - omega_{x_j}
                G[:, i, j] = (H_ij * diff).sum(-1)
        return G


class LeTFMaskOneSwapHead(nn.Module):
    """Climax swap head: mask anchor i, reuse single-site leTF hollowness.

    For each anchor i, one masked body pass returns H_ij = H[:, j, :] for ALL
    j != i: blind to x_i (anchor masked) and hollow in x_j (leTF readout at j
    ignores j's own input). The readout against omega_{x_i} - omega_{x_j} then
    gives the full row G_swap(i, :). Diagonal and same-spin pairs vanish because
    the token difference is zero there. NOT label-symmetric (H_ij != H_ji).

    forward batches the d independent anchor passes into the model batch
    dimension (`_anchor_masked_bodies`) instead of looping them sequentially
    -- the loop cost ~d x the single-site head and made D >= 8 rungs
    infeasible (56 s/forward at d=256). `anchor_chunk_size` caps anchors per
    stacked pass: the readout attention buffer is (n_anchors*B, n_heads, d, 2d),
    which stops fitting memory at large d unchunked. None = all d anchors in
    one pass. forward_looped is the original sequential reference, kept as the
    test oracle.
    """

    def __init__(
        self, backbone: LeTFRateMatrix, anchor_chunk_size: int | None = None
    ):
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
        """Sequential reference: one masked body pass per anchor. ~d x slower
        than forward; kept as the readable form of the math and the oracle
        forward must match (tests/test_swap_head_vectorised.py)."""
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
    """Explicit antisymmetrisation of any raw pair score (design note 2.4).

    G_swap(i, j | x) := 1/2 [ raw(i, j | x) - raw(i, j | Swap2(x, i, j)) ]

    is exactly antisymmetric under Swap2 for ANY raw_head, with no hollowness
    required. raw_head is a callable (x, t) -> (B, d, d). Cost is one swapped
    forward pass per ordered pair (O(d^2)): the guaranteed-correct fallback and
    the unit-test oracle, NOT the efficient path.
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
