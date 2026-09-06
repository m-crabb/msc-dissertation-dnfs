"""Swap-neighbour helpers: pair enumeration + target evaluation at swaps.

A swap CTMC's neighbours are Swap2(x, i, j) over unordered site pairs. Every
pair is represented by its site-index-ordered member (i < j); the swap readout
head is label-asymmetric, so this ordering is load-bearing for the single-pass
reverse-rate identity (Eq. 8 / Eq. 10, swap form).
"""

import torch
from torch import Tensor

# Upper bound on the swap log-ratio before exp(), analogous to the single-site
# cap. A swap touches up to ~2x the bonds of a single-site flip, so the
# single-site value of 5.0 is effectively ~2x tighter for swaps and would bind
# at near-critical sigma; 30.0 is overflow-safe (exp(30) ~ 1e13) and non-binding
# across the gate sigma-range (max |log-ratio| = t*sigma*|Delta(xTAx)| <= 32*sigma).
SWAP_LOG_RATIO_CLAMP = 30.0


_PAIRS_CACHE: dict[tuple[int, torch.device], Tensor] = {}


def upper_tri_pairs(d: int, device) -> Tensor:
    """All (i, j) site pairs with i < j, shape (n_pairs, 2), n_pairs = d(d-1)/2.

    Cached per (d, device): every sampler step re-reads the same table and
    n_pairs grows as d^2, so rebuilding it per call is pure dispatch overhead.
    Callers treat the result as read-only.
    """
    key = (d, torch.device(device))
    if key not in _PAIRS_CACHE:
        _PAIRS_CACHE[key] = torch.combinations(torch.arange(d, device=device), r=2)
    return _PAIRS_CACHE[key]


def gather_pair_scores(G: Tensor, pairs: Tensor) -> Tensor:
    """Pick G[:, i, j] for each (i, j) in `pairs`, shape (B, n_pairs)."""
    return G[:, pairs[:, 0], pairs[:, 1]]


def _log_p_tilde_at_swap_neighbours(x: Tensor, t: Tensor, target) -> Tensor:
    """log p̃_t(Swap2(x, i, j)) for every i<j pair, shape (B, n_pairs).

    Same role for swap moves as `_neighbours._log_p_tilde_at_neighbours` for
    single-site flips. Same-spin pairs give Swap2(x,i,j) = x, so their column is
    log p̃_t(x) (downstream log-ratio 0), exactly as a trivial swap should.
    """
    batch_size, d = x.shape
    pairs = upper_tri_pairs(d, x.device)  # (P, 2)
    n_pairs = pairs.shape[0]
    i_col = pairs[:, 0].view(1, n_pairs, 1).expand(batch_size, n_pairs, 1)
    j_col = pairs[:, 1].view(1, n_pairs, 1).expand(batch_size, n_pairs, 1)
    y = x[:, None, :].expand(batch_size, n_pairs, d).clone()  # (B, P, d)
    spin_i = y.gather(2, i_col)
    spin_j = y.gather(2, j_col)
    y.scatter_(2, i_col, spin_j)
    y.scatter_(2, j_col, spin_i)
    flat = y.reshape(batch_size * n_pairs, d)
    t_per = t.repeat_interleave(n_pairs)
    return target.log_p_tilde_t(flat, t_per).reshape(batch_size, n_pairs)
