"""Gibbs heat-bath sampler for Ising-like targets.

Reference oracle for DNFS sampler validation. Heat-bath update at site i:

    p(x_i = +1 | x_{≠i}) = σ( 4 h + 2 b ),    h = Σ_j J_ij x_j,  b = bias

derivation: the paper's J is symmetric (each edge contributes to BOTH J_ij
and J_ji), so the terms in x^T J x involving x_i come to 2 x_i Σ_j J_ij x_j.
Together with the bias term:

    log p(x_i=+1 | x_{≠i}) − log p(x_i=−1 | x_{≠i})
        = 2 · (+1) · Σ_j J_ij x_j + b · (+1)
        − 2 · (−1) · Σ_j J_ij x_j − b · (−1)
        = 4 h + 2 b.

So p(x_i=+1) = σ(4h + 2b). The textbook one-edge-per-pair convention gives
σ(2h_i); ours double-counts in J, so the factor doubles. Proposal IS the
conditional → acceptance is always 1 (no MH correction). One "sweep"
updates every site once; chains are vectorised, the inner loop is over
sites only.
"""

import torch
from torch import Tensor


def gibbs_sample(
    target,
    n_chains: int,
    n_sweeps: int,
    x_init: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Single-spin-flip heat-bath sampler.

    target: duck-types IsingTarget — needs `.J` (d×d), `.bias`, `.d`, `.device`.
    Returns: (n_chains, target.d) tensor in {-1, +1}.
    """
    if x_init is None:
        spins = torch.randint(
            0, 2, (n_chains, target.d), generator=generator, device=target.device
        ).float() * 2 - 1
    else:
        spins = x_init.to(device=target.device).float()

    for _ in range(n_sweeps):
        site_order = torch.randperm(target.d, generator=generator, device=target.device)
        for site in site_order.tolist():
            local_field = spins @ target.J[:, site]                       # Σ_j J_ij x_j, shape (n_chains,)
            log_odds_plus = 4 * local_field + 2 * target.bias              # log p(+1) − log p(−1)
            prob_plus = torch.sigmoid(log_odds_plus)
            uniform_draws = torch.rand(n_chains, generator=generator, device=target.device)
            spins[:, site] = (uniform_draws < prob_plus).to(spins.dtype) * 2 - 1
    return spins
