"""Shared neighbour-evaluation helper for Kolmogorov and CTMC modules."""

import torch
from torch import Tensor

# Paper App. E.1.1 clips the neighbour ratio log p_t(y)/p_t(x) at 5. That
# value is calibrated for an *unmodified* Ising target, whose single-flip
# log-ratios are O(a few). A VCSGC-style composition penalty
# lambda*d*(c(x) - c_target)^2 adds -+2*lambda*Delta to the same quantity,
# because flipping one site moves c by exactly 1/d — so the ceiling starts
# binding once Delta > clamp/(2*lambda), a threshold that does NOT depend
# on d. Binding is a bias rather than noise: the inflow term is then
# evaluated at exp(clamp) instead of the true ratio, so no rate field
# zeroes the residual and the loss acquires an irreducible floor. Targets
# may override it; the default keeps the replication faithful.
DEFAULT_LOG_RATIO_CLAMP = 5.0


def log_ratio_clamp(target) -> float:
    """Ceiling in force for `log p_t(y)/p_t(x)`, read off the target.

    It lives on the target because it bounds a pure-target quantity, and
    because `target` is the one object in scope at all three places that
    must agree on it: the loss (`kolmogorov.residual_lenet`), the
    control-variate integrand (`ctmc._compute_xi_t_lenet`) and the
    `log_ratio_clamp_frac` diagnostic. If the first two ever disagreed, the
    control variate would be centred on a different quantity than the loss
    it corrects — a bias with no symptom that would reveal it.
    """
    return getattr(target, "log_ratio_clamp", DEFAULT_LOG_RATIO_CLAMP)


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
    spin_of_idx = 2.0 * torch.arange(vocab_size, device=x.device, dtype=x.dtype) - 1.0
    neighbours = (
        x[:, None, None, :].expand(batch_size, n_sites, vocab_size, n_sites).clone()
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
    return target.log_p_tilde_t(flat, t_per).reshape(batch_size, n_sites, vocab_size)
