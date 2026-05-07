"""Shared neighbour-evaluation helper for Kolmogorov and CTMC modules."""
import torch
from torch import Tensor


def _log_p_tilde_at_neighbours(
    x: Tensor,
    t: Tensor,
    target,
    vocab_size: int,
) -> Tensor:
    """Evaluate log p̃_t at every single-site neighbour of `x`.

    For each (b, i, τ), construct y = x[b] with site i replaced by the
    spin value corresponding to vocab token τ, and evaluate
    `target.log_p_tilde_t(y, t)`. The τ = x_i_idx slot returns
    log p̃_t(x) itself — i.e. the corresponding log_ratio downstream is 0.

    Returns:
        log_p_neighbours: (B, D, vocab_size) tensor of log p̃_t values.
    """
    batch_size, n_sites = x.shape
    spin_of_idx = (
        2.0 * torch.arange(vocab_size, device=x.device, dtype=x.dtype) - 1.0
    )
    neighbours = (
        x[:, None, None, :]
        .expand(batch_size, n_sites, vocab_size, n_sites)
        .clone()
    )
    site_index = (
        torch.arange(n_sites, device=x.device)
        .view(1, n_sites, 1, 1)
        .expand(batch_size, n_sites, vocab_size, 1)
    )
    spin_at_site = spin_of_idx.view(1, 1, vocab_size, 1).expand(
        batch_size, n_sites, vocab_size, 1
    )
    neighbours.scatter_(-1, site_index, spin_at_site)

    flat = neighbours.reshape(batch_size * n_sites * vocab_size, n_sites)
    t_per = t.repeat_interleave(n_sites * vocab_size)
    return target.log_p_tilde_t(flat, t_per).reshape(
        batch_size, n_sites, vocab_size
    )
