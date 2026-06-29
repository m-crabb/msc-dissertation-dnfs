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
    leTF hollowness at j (O(d) passes).

Both read H_ij at the SECOND index j, so they agree numerically: the leTF
readout at j already ignores j's own input, so additionally masking j is a
no-op. The heads are NOT label-symmetric (H_ij != H_ji), so the downstream
swap residual must order each unordered pair by site index (i<j), never by
spin (design note section 4). LeTFRateMatrix is reused untouched.
"""
from collections.abc import Iterable

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
    x_emb = model.token_embedder(x_idx).clone()        # (B, d, h)
    for s in mask_sites:
        x_emb[:, s, :] = 0.0                            # content-free override
    cond_t = model.time_embedder(t).unsqueeze(1)        # (B, 1, h)
    fwd_x = model.fwd_stack(torch.cat([cond_t, x_emb], dim=1))
    bwd_x = model.bwd_stack(torch.cat([cond_t, x_emb.flip(1)], dim=1)).flip(1)
    H = model.attention_readout(fwd_x, bwd_x, cond_t)    # (B, d, h)
    H = model.output_norm(H) + model.time_embedder(t).unsqueeze(1)
    return H


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
        om = m.omega(x_idx)                              # (B, d, h)
        batch, d = x.shape
        G = x.new_zeros(batch, d, d)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                H = _masked_body(m, x, t, (i, j))        # (B, d, h)
                H_ij = H[:, j, :]                         # read at second index
                diff = om[:, i, :] - om[:, j, :]         # omega_{x_i} - omega_{x_j}
                G[:, i, j] = (H_ij * diff).sum(-1)
        return G


class LeTFMaskOneSwapHead(nn.Module):
    """Climax swap head: mask anchor i, reuse single-site leTF hollowness. O(d).

    For each anchor i, one masked body pass returns H_ij = H[:, j, :] for ALL
    j != i: blind to x_i (anchor masked) and hollow in x_j (leTF readout at j
    ignores j's own input). The readout against omega_{x_i} - omega_{x_j} then
    gives the full row G_swap(i, :). Diagonal and same-spin pairs vanish because
    the token difference is zero there. NOT label-symmetric (H_ij != H_ji).
    """

    def __init__(self, backbone: LeTFRateMatrix):
        super().__init__()
        self.backbone = backbone
        self.d = backbone.d

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        m = self.backbone
        x_idx = ((x + 1) / 2).long()
        om = m.omega(x_idx)                              # (B, d, h)
        batch, d = x.shape
        G = x.new_zeros(batch, d, d)
        for i in range(d):
            H = _masked_body(m, x, t, (i,))             # (B, d, h); H[:, j, :] = H_ij
            diff = om[:, i : i + 1, :] - om             # (B, d, h): [:, j, :] = om_xi - om_xj
            G[:, i, :] = (H * diff).sum(-1)            # (B, d); diagonal j==i -> 0
        return G


def antisymmetrise(raw_head, x: Tensor, t: Tensor) -> Tensor:
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
