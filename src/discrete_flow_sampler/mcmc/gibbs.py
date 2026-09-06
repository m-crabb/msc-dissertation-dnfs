"""Gibbs heat-bath sampler for Ising-like targets.

Reference oracle for DNFS sampler validation. Unconstrained heat-bath update
at site i:

    p(x_i = +1 | x_{≠i}) = σ( 4 h + 2 b ),    h = Σ_j J_ij x_j,  b = bias

derivation: the paper's J is symmetric (each edge contributes to BOTH J_ij
and J_ji), so the terms in x^T J x involving x_i come to 2 x_i Σ_j J_ij x_j.
Together with the bias term:

    log p(x_i=+1 | x_{≠i}) - log p(x_i=-1 | x_{≠i})
        = 2 · (+1) · Σ_j J_ij x_j + b · (+1)
        - 2 · (-1) · Σ_j J_ij x_j - b · (-1)
        = 4 h + 2 b.

So p(x_i=+1) = σ(4h + 2b). If the target includes the soft composition
penalty λ d (c_+ - c_t)^2, the exact conditional subtracts

    λ · ( 2 · (S/d - c_t) + 1/d ),

where S is the number of +1 spins among sites other than i. The +1/d term is
the discrete single-site correction from comparing S+1 against S.

The textbook one-edge-per-pair convention gives σ(2h_i); ours double-counts
in J, so the factor doubles. Proposal IS the conditional, so acceptance is
always 1 (no MH correction). One "sweep" updates every site once; chains are
vectorised, the inner loop is over sites only.
"""

import torch
from torch import Tensor


def gibbs_sample(
    target,
    n_chains: int,
    n_sweeps: int,
    x_init: Tensor | None = None,
    generator: torch.Generator | None = None,
    record_energy_every: int | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Single-spin-flip heat-bath sampler.

    target: duck-types IsingTarget — needs `.J` (d×d), `.bias`, `.d`, `.device`;
      may expose `.target_composition` and `.composition_penalty_strength`;
      needs `.log_prob` if `record_energy_every` is set.
    Returns: (n_chains, target.d) tensor in {-1, +1}.

    If `record_energy_every=K`, also returns a (n_records, n_chains) tensor of
    `target.log_prob(spins)` at sweeps [0, K, 2K, ...]. Used as a mixing
    diagnostic for the long-chain reference: chain-mean trace should plateau,
    and runs from different `x_init` should converge to the same final
    distribution.
    """
    if x_init is None:
        spins = (
            torch.randint(
                0, 2, (n_chains, target.d), generator=generator, device=target.device
            ).float()
            * 2
            - 1
        )
    else:
        spins = x_init.to(device=target.device).float()

    energy_trace: list[Tensor] = []
    if record_energy_every is not None:
        energy_trace.append(target.log_prob(spins).detach())

    # Composition-penalty awareness. The soft constraint λ·d·(c₊ − c_t)² is a
    # GLOBAL term, so it must enter the single-site heat-bath conditional, not
    # only the diagnostic. Guard mirrors IsingTarget.composition_penalty
    # (ising.py): when inactive the log-odds and RNG stream are byte-for-byte
    # the original unconstrained sampler.
    target_composition = getattr(target, "target_composition", None)
    penalty_strength = getattr(target, "composition_penalty_strength", 0.0)
    penalty_active = target_composition is not None and penalty_strength != 0.0

    for sweep_idx in range(n_sweeps):
        site_order = torch.randperm(target.d, generator=generator, device=target.device)
        for site in site_order.tolist():
            local_field = spins @ target.J[:, site]  # Σ_j J_ij x_j, shape (n_chains,)
            log_odds_plus = 4 * local_field + 2 * target.bias  # log p(+1) - log p(-1)
            if penalty_active:
                # S = #{+1 among sites ≠ i}. With x ∈ {−1,+1} and d−1 other
                # sites: S = ((d−1) + Σ_{j≠i} x_j) / 2. Penalty contribution to
                # the log-odds is −λ·(2·(S/d − c_t) + 1/d), obtained by
                # completing the square on λd[((S+1)/d−c_t)² − (S/d−c_t)²].
                others_sum = spins.sum(dim=1) - spins[:, site]
                up_count_others = ((target.d - 1) + others_sum) / 2
                log_odds_plus = log_odds_plus - penalty_strength * (
                    2.0 * (up_count_others / target.d - target_composition)
                    + 1.0 / target.d
                )
            prob_plus = torch.sigmoid(log_odds_plus)
            uniform_draws = torch.rand(
                n_chains, generator=generator, device=target.device
            )
            spins[:, site] = (uniform_draws < prob_plus).to(spins.dtype) * 2 - 1
        if (
            record_energy_every is not None
            and (sweep_idx + 1) % record_energy_every == 0
        ):
            energy_trace.append(target.log_prob(spins).detach())

    if record_energy_every is None:
        return spins
    return spins, torch.stack(energy_trace)
