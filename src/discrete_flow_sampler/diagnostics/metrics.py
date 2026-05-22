"""Paper-faithful diagnostics for the DNFS Ising replication.

The DNFS paper (Ou/Zhang/Li 2025, Appendix D.1 + Table 2) evaluates the
Ising experiment with a single self-normalised IS pass, reporting:

- Effective Sample Size (Eq. 42),
- Free energy lower-bound estimate F/D = -log Z / (2σD) (Eq. 37),
- Internal energy E/D = E_p[E(x)] / D (Eq. 38),
- Entropy S/D = 2σ(E - F) / D (derived).

Sample budget: N = 2,048 with std taken across 10 independent seeds for
Table 2; N = 5,000 for the Figure 13 energy histogram. We standardise on
N = 5,000 (≥ 2,048 strictly) so a single eval pass feeds both.

All other distance metrics from earlier drafts (TVD, KL, 1-D Wasserstein
on log p̃) were off-paper and have been removed -- the paper does not
report them, and TVD in particular was sample-size-floored at our budget
(see auto-memory `project_tvd_floor_at_low_n`).

Note on the σ factor in the per-spin internal energy. Paper Eq. 53
defines E(x) := -σ x^T A x; paper Table 2's reported `E/D` is the
standard physics per-spin internal energy u = ⟨H⟩/N with H = -½ x^T A x
in J = 1 units, which equals -E_p[log p̃] / (2σD). The two differ by a
factor of β = 2σ (the matrix form `x^T A x` double-counts edges; β is
the inverse temperature). Cross-check against the reference repo
J-zin/DNFS:main.py::evaluate confirms the per-β normalisation in their
internal-energy expression. See `internal_energy_estimate` docstring
for the derivation.
"""

import itertools

import torch
from torch import Tensor


def ess_from_log_weights(log_w: Tensor) -> Tensor:
    """Effective sample size from importance log-weights (paper Eq. 42).

        ESS(w) = (Σ_n w_n)² / Σ_n w_n²
               = exp( 2·logsumexp(log_w) - logsumexp(2·log_w) )

    where log_w is a 1-D tensor of CTMC importance log-weights w_k =
    ∫₀¹ ∂_s log p̃_s(x_s^(k)) − Σ_y R_s(x_s^(k), y) p_s(y)/p_s(x_s^(k)) ds
    (Eq. 41). Computed in log-space because raw weights routinely span
    100+ orders of magnitude early in training; computing
    (Σw)² / Σw² directly under-/overflows.

    Returns a scalar tensor in [1, K]: K for uniform weights, 1 when one
    sample dominates. Caller divides by K for the [1/K, 1] normalised
    ESS the paper reports.
    """
    log_sum_w = torch.logsumexp(log_w, dim=0)
    log_sum_w_sq = torch.logsumexp(2 * log_w, dim=0)
    return torch.exp(2 * log_sum_w - log_sum_w_sq)


def composition_fraction_up(x: Tensor) -> Tensor:
    """Fraction of +1 spins/species in each sample, shape (B,)."""
    return ((x.float() + 1.0) * 0.5).mean(dim=-1)


def magnetisation(x: Tensor) -> Tensor:
    """Mean Ising spin in each sample, shape (B,)."""
    return x.float().mean(dim=-1)


def composition_observables(
    x: Tensor,
    *,
    target_composition: float | None = None,
    composition_penalty_strength: float | None = None,
) -> dict[str, float]:
    """Paper-adjacent composition diagnostics for eval samples.

    These are observables of the sample set, not training losses. When a target
    composition is supplied, the returned dict also includes violation metrics
    against that target.
    """
    c = composition_fraction_up(x)
    m = magnetisation(x)
    metrics = {
        "composition_mean": float(c.mean().item()),
        "composition_std": float(c.std(unbiased=False).item()),
        "magnetisation_mean": float(m.mean().item()),
        "magnetisation_std": float(m.std(unbiased=False).item()),
    }

    if target_composition is not None:
        error = c - float(target_composition)
        metrics.update(
            {
                "target_composition": float(target_composition),
                "composition_abs_error_mean": float(error.abs().mean().item()),
                "composition_sq_violation_mean": float(error.pow(2).mean().item()),
            }
        )
        if composition_penalty_strength is not None:
            metrics["composition_penalty_strength"] = float(
                composition_penalty_strength
            )

    return metrics


def enumerate_states(D: int) -> Tensor:
    """All 2^D binary spin states for a D-site lattice.

    Each spin takes values in {-1, +1} (Ising convention, not the
    {0, 1} software convention). Returns a (2^D, D) int64 tensor;
    ordering is lexicographic via itertools.product([-1, 1], repeat=D).

    Used for D ≤ 20 ground-truth computations (the analog of the
    "Optimal Value" row in paper Table 2 at sub-paper-scale lattices
    where direct enumeration is feasible). Beyond D = 20 this is memory-
    bound; D = 10×10 in the paper uses analytical Ferdinand & Fisher
    references instead, not enumeration.
    """
    bits = list(itertools.product([-1, 1], repeat=D))
    return torch.tensor(bits, dtype=torch.long)


def exact_log_probs(target, states: Tensor) -> Tensor:
    """Normalised log-probabilities under `target` over enumerated states.

    `target` must duck-type a `log_prob(x: Tensor) -> Tensor` method
    mapping (N, D) → (N,) un-normalised log-densities. The states tensor
    is cast to float at call time so the integer canonical form from
    `enumerate_states` can flow through float-typed target code without
    dtype mismatch.

    Returns (N,) log-probabilities satisfying logsumexp(...) = 0,
    i.e. exp(...).sum() == 1.
    """
    log_p_unnorm = target.log_prob(states.float())
    log_Z = torch.logsumexp(log_p_unnorm, dim=0)
    return log_p_unnorm - log_Z


def free_energy_lb_estimate(
    log_weights: Tensor, sigma: float, D: int
) -> Tensor:
    """Per-site free-energy lower-bound estimate F/D from CTMC IS weights.

    Implements paper Eq. 37 (Appendix D.1). The free energy of the Ising
    model with p(x) ∝ exp(σ x^T A x) is F = -(1 / 2σ) · log Z, equivalently
    F = -log Z / β with β = 2σ. The paper's lower bound is

        log Z ≥ E_{x ~ Q}[ ∫₀¹ ∂_s log p̃_s(x_s)
                          − Σ_y R_s(x_s, y) p_s(y)/p_s(x_s) ds ]
              = E_{x ~ Q}[ w(x) ]

    where w is the CTMC log-weight from Eq. 41. The Monte-Carlo estimate
    is the empirical mean

        log Ẑ_lb = (1/K) Σ_k w_k = log_weights.mean(),

    yielding F/D = -log Ẑ_lb / (2σD).

    Args:
        log_weights: (K,) tensor of CTMC IS log-weights from a single
            eval pass over t = 0 → 1 trajectories.
        sigma: Ising coupling parameter σ (matches `IsingTarget.sigma`).
        D: total number of spins (D = D_lin² for a D_lin × D_lin lattice).

    Returns:
        Scalar tensor: per-site free energy F/D in the paper's convention.
        For Ising D = 10×10, σ = 0.1, the analytic optimum per Ferdinand
        & Fisher (1969) is -3.6727 (Table 2 row 1).
    """
    return -log_weights.mean() / (2 * sigma * D)


def internal_energy_estimate(
    log_weights: Tensor,
    log_p_tilde: Tensor,
    sigma: float,
    D: int,
) -> Tensor:
    """Per-site internal-energy estimate E/D via self-normalised IS.

    Implements paper Eq. 38 (Appendix D.1). The numerical quantity Table 2
    reports as `E/D` is the standard physics per-spin internal energy
    u = ⟨H⟩/N (Onsager value u_c ≈ -√2 at criticality), *not* the
    paper-text-defined E_paper(x) := -σ x^T A x averaged over π. The two
    differ by β = 2σ:

        log p̃(x)   = σ x^T A x       (paper's exponent)
        E_paper(x) = -σ x^T A x = -log p̃(x)
        H(x)       = -½ x^T A x       (J = 1 physics convention; pair sum)
        E_paper    = β · H            (since β = 2σ, x^T A x = 2 Σ_<ij> s_i s_j)

    so u = ⟨H⟩/N = -E_p[log p̃]/(βN) = -E_p[log p̃]/(2σN). Eq. 38's
    self-normalised IS estimate of E_p[log p̃] is

        E_p[log p̃] ≈ Σ_k softmax(w)_k · log p̃(x_t^(k)),

    yielding

        E/D = u = -[ Σ_k softmax(w)_k · log p̃(x^(k)) ] / (2σD).

    Args:
        log_weights: (K,) tensor of CTMC IS log-weights w_k.
        log_p_tilde: (K,) tensor of un-normalised target log-densities
            log p̃(x_t^(k)) for the same K eval samples (`target.log_prob`
            applied to the t = 1 sample slice).
        sigma: Ising coupling parameter σ.
        D: total number of spins.

    Returns:
        Scalar tensor: per-site internal energy E/D. For Ising D = 10×10,
        σ = 0.1, the analytic optimum is -0.4282 (Table 2 row 1); at
        σ_c = 0.22305 it is -1.4763 ≈ -√2 (Onsager).
    """
    softmax_weights = torch.softmax(log_weights, dim=0)
    return -(softmax_weights * log_p_tilde).sum() / (2 * sigma * D)


def entropy_estimate(F_per_site: Tensor, E_per_site: Tensor, sigma: float) -> Tensor:
    """Per-site entropy estimate S/D = 2σ(E - F) / D from F/D and E/D.

    Direct algebraic combination of the free-energy and internal-energy
    estimates (paper Table 2 caption: S = 2σ(E − F) = β(E − F), the
    standard physics relation S = β(U − F) with β = 2σ). Holds in
    expectation; downstream std comes from the joint distribution of
    (F̂, Ê) over IS replicates, not a separate Monte-Carlo pass.

    Args:
        F_per_site: scalar tensor F/D from `free_energy_lb_estimate`.
        E_per_site: scalar tensor E/D from `internal_energy_estimate`.
        sigma: Ising coupling parameter σ.

    Returns:
        Scalar tensor: per-site entropy S/D. For Ising D = 10×10,
        σ = 0.1, the analytic optimum is 0.6489 (Table 2 row 1).
    """
    return 2 * sigma * (E_per_site - F_per_site)


def exact_free_energy(target, sigma: float, D: int) -> Tensor:
    """Exact F/D = -log Z / (2σD) by enumeration of all 2^D states.

    The "Optimal Value" analog at sub-paper-scale lattices (D ≤ 20).
    For D = 10×10 the paper uses the Ferdinand & Fisher (1969)
    analytical 2-D Ising solution instead; that helper is out of scope
    for this module.

    Args:
        target: object with `log_prob(x: Tensor) -> Tensor` and a
            `device` attribute.
        sigma: Ising σ.
        D: total number of spins. Must be ≤ 20 for tractability.

    Returns:
        Scalar tensor: exact F/D.
    """
    if D > 20:
        raise ValueError(
            f"exact_free_energy enumerates 2^D states; D={D} > 20 is "
            "intractable. Use Ferdinand & Fisher (1969) at D = 10×10."
        )
    states = enumerate_states(D).to(target.device).float()
    log_p_unnorm = target.log_prob(states)
    log_Z = torch.logsumexp(log_p_unnorm, dim=0)
    return -log_Z / (2 * sigma * D)


def conditional_pmf_at_composition(
    states: Tensor,
    log_pi: Tensor,
    n_plus_target: int,
) -> tuple[Tensor, Tensor]:
    """Exact conditional distribution restricted to a single composition slice.

    Given enumerated states with their normalised log-probabilities under π,
    selects states whose number of +1 spins equals `n_plus_target` and
    returns the corresponding state subset together with the renormalised
    conditional log-probabilities π(·|Σ(x+1)/2 = n_plus_target).

    At c_target = 0.5 the c-marginal is uninformative as a diagnostic (the
    Z_2 symmetry pins ⟨c⟩ = 0.5 by construction); the conditional energy
    distribution on the slice is what tells us whether DNFS is fitting the
    Ising free-energy *shape* correctly.

    Args:
        states: (N, D) tensor of enumerated ±1 spin configurations.
        log_pi: (N,) tensor of normalised log-probabilities (logsumexp = 0).
        n_plus_target: integer in [0, D] selecting the slice.

    Returns:
        (slice_states, log_pi_cond): the (M, D) selected states and the
        (M,) normalised conditional log-probabilities on the slice. M is
        the binomial coefficient C(D, n_plus_target).
    """
    n_plus = ((states + 1) // 2).sum(dim=-1)
    mask = n_plus == n_plus_target
    slice_states = states[mask]
    log_pi_slice = log_pi[mask]
    log_Z_slice = torch.logsumexp(log_pi_slice, dim=0)
    return slice_states, log_pi_slice - log_Z_slice


def z2_asymmetry_from_samples(
    samples: Tensor, log_weights: Tensor
) -> dict[str, float]:
    """Z_2 symmetry-breaking diagnostic for IS-weighted DNFS samples.

    For a Z_2-symmetric target (Ising with bias=0 at c_target=0.5),
    E_π[m(x)] = 0 exactly and the IS-weighted mass on m>0 should equal
    the mass on m<0. Any deviation measures how badly q_θ has failed to
    learn the Z_2 invariance.

    Returns:
        Dict with `e_m_is` (IS-weighted ⟨m⟩, should be 0), `mass_pos`,
        `mass_neg`, `mass_zero` (IS-weighted mass fractions in each
        magnetisation sector, summing to 1), and `asymmetry` (the absolute
        imbalance |mass_pos - mass_neg|).
    """
    w = torch.softmax(log_weights, dim=0)
    m = samples.float().mean(dim=-1)
    mass_pos = float(w[m > 0].sum().item())
    mass_neg = float(w[m < 0].sum().item())
    mass_zero = float(w[m == 0].sum().item())
    return {
        "e_m_is": float((w * m).sum().item()),
        "mass_pos": mass_pos,
        "mass_neg": mass_neg,
        "mass_zero": mass_zero,
        "asymmetry": abs(mass_pos - mass_neg),
    }


def exact_internal_energy(target, sigma: float, D: int) -> Tensor:
    """Exact E/D = u = -E_π[log p̃(x)] / (2σD) by enumeration.

    The standard physics per-spin internal energy under the exact
    Boltzmann distribution π(x) = exp(log p̃(x)) / Z. The same σ-factor
    derivation as `internal_energy_estimate` applies: u = ⟨H⟩/N with
    H = -½ x^T A x and β = 2σ gives u = -E_π[log p̃] / (βN). Same scope
    caveat: D ≤ 20.

    Args:
        target: object with `log_prob(x: Tensor) -> Tensor` and a
            `device` attribute.
        sigma: Ising σ.
        D: total number of spins. Must be ≤ 20.

    Returns:
        Scalar tensor: exact E/D.
    """
    if D > 20:
        raise ValueError(
            f"exact_internal_energy enumerates 2^D states; D={D} > 20 is "
            "intractable. Use Ferdinand & Fisher (1969) at D = 10×10."
        )
    states = enumerate_states(D).to(target.device).float()
    log_p_unnorm = target.log_prob(states)
    log_pi = log_p_unnorm - torch.logsumexp(log_p_unnorm, dim=0)
    return -(log_pi.exp() * log_p_unnorm).sum() / (2 * sigma * D)
