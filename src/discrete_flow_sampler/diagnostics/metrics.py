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
(see auto-memory `project_tvd_floor_at_low_n`). The removal landed when
the eval suite was re-aligned to the paper; see Stage 1 plan status at
`docs/plans/2026-05-05-dnfs-stage-1-vanilla.md`.

Authorship:
- ESS (already implemented) and the small enumeration helpers are
  plumbing.
- `free_energy_lb_estimate`, `internal_energy_estimate`, and
  `entropy_estimate` are research-bearing — bodies are written by the
  user (per `CLAUDE.md`); Claude provides the interface stub, the
  Eq.-anchored docstring, and the unit tests.
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
    model with p(x) ∝ exp(σ x^T A x) is F = -(1 / 2σ) · log Z; per the
    paper's lower bound,

        log Z ≥ E_{x ~ Q}[ ∫₀¹ ∂_s log p̃_s(x_s)
                          − Σ_y R_s(x_s, y) p_s(y)/p_s(x_s) ds ]
              = E_{x ~ Q}[ w(x) ]

    where w is the CTMC log-weight from Eq. 41. The Monte-Carlo estimate
    is the empirical mean

        log Ẑ_lb = (1/K) Σ_k w_k = log_weights.mean()

    yielding F/D = -log Ẑ_lb / (2σD).

    Args:
        log_weights: (K,) tensor of CTMC IS log-weights from a single
            eval pass over t = 0 → 1 trajectories.
        sigma: Ising coupling parameter σ (matches `IsingTarget.sigma`).
        D: total number of spins (D = D_lin² for a D_lin × D_lin lattice).

    Returns:
        Scalar tensor: per-site free energy F/D in the paper's
        convention. For Ising D = 10×10, σ = 0.1, the analytic optimum
        per Ferdinand & Fisher (1969) is -3.6727 (Table 2 row 1).

    Note (authorship): research-bearing per CLAUDE.md. Body intentionally
    deferred to the user; this stub raises NotImplementedError so the
    eval pipeline fails loudly until the body is filled in.
    """
    raise NotImplementedError(
        "free_energy_lb_estimate body deferred to user (paper Eq. 37). "
        "Stage 1 plan status, next-session todo step 2."
    )


def internal_energy_estimate(
    log_weights: Tensor,
    log_p_tilde: Tensor,
    sigma: float,
    D: int,
) -> Tensor:
    """Per-site internal-energy estimate E/D via self-normalised IS.

    Implements paper Eq. 38 (Appendix D.1). For Ising p(x) ∝ exp(σ x^T A x)
    the energy in the paper's convention is E(x) = -log p̃(x) / 1
    (since log p̃ = σ x^T A x = -E). The internal energy is

        E_p[E(x)] = -E_p[log p̃(x)]
                  ≈ -Σ_k softmax(w)_k · log p̃(x_t^(k))

    where the right-hand side is the self-normalised IS estimate of the
    test function φ = log p̃_t under the importance proposal Q (Eq. 38),
    using `softmax(w_k) = exp(w_k) / Σ_j exp(w_j)`.

    Per-site:

        E/D = E_p[E(x)] / D = -[ Σ_k softmax(w)_k · log p̃(x^(k)) ] / D

    Args:
        log_weights: (K,) tensor of CTMC IS log-weights w_k.
        log_p_tilde: (K,) tensor of un-normalised target log-densities
            log p̃(x_t^(k)) for the same K eval samples (`target.log_prob`
            applied to the t = 1 sample slice).
        sigma: Ising σ. (Carried for API symmetry with the F estimator
            and S formula; cancels here once the formula resolves.)
        D: total number of spins.

    Returns:
        Scalar tensor: per-site internal energy E/D. For Ising D = 10×10,
        σ = 0.1, the analytic optimum is -0.4282 (Table 2 row 1).

    Note (authorship): research-bearing per CLAUDE.md. Body deferred to
    the user; stub raises NotImplementedError.
    """
    raise NotImplementedError(
        "internal_energy_estimate body deferred to user (paper Eq. 38). "
        "Stage 1 plan status, next-session todo step 2."
    )


def entropy_estimate(F_per_site: Tensor, E_per_site: Tensor, sigma: float) -> Tensor:
    """Per-site entropy estimate S/D = 2σ(E - F) / D from F/D and E/D.

    Direct algebraic combination of the free-energy and internal-energy
    estimates (paper Table 2 caption). Holds in expectation; downstream
    std comes from the joint distribution of (F̂, Ê) over IS replicates,
    not from a separate Monte-Carlo pass.

    Args:
        F_per_site: scalar tensor F/D from `free_energy_lb_estimate`.
        E_per_site: scalar tensor E/D from `internal_energy_estimate`.
        sigma: Ising coupling parameter σ.

    Returns:
        Scalar tensor: per-site entropy S/D. For Ising D = 10×10,
        σ = 0.1, the analytic optimum is 0.6489 (Table 2 row 1).

    Note (authorship): trivial-derivation but research-adjacent. Body
    deferred to the user for consistency with the F and E estimators.
    """
    raise NotImplementedError(
        "entropy_estimate body deferred to user (S/D = 2σ(E - F)/D, "
        "Table 2 caption). Stage 1 plan status, next-session todo step 2."
    )


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
        D: total number of spins. Asserted ≤ 20 for tractability.

    Returns:
        Scalar tensor: exact F/D.

    Note (authorship): one-line wrapper over `exact_log_probs`. Body
    deferred to user for symmetry with the IS estimator.
    """
    raise NotImplementedError(
        "exact_free_energy body deferred to user "
        "(F/D = -logsumexp(log_p_tilde) / (2σD) by enumeration). "
        "Stage 1 plan status, next-session todo step 2."
    )


def exact_internal_energy(target, sigma: float, D: int) -> Tensor:
    """Exact E/D = -E_π[log p̃(x)] / D by enumeration of all 2^D states.

    Uses the normalised exact distribution π(x) = softmax(log p̃(x))
    over enumerated states and computes E_π[log p̃] in closed form.
    Same scope caveat as `exact_free_energy`: D ≤ 20.

    Args:
        target: object with `log_prob(x: Tensor) -> Tensor`.
        sigma: Ising σ.
        D: total number of spins. Asserted ≤ 20.

    Returns:
        Scalar tensor: exact E/D.

    Note (authorship): one-line wrapper over `exact_log_probs`. Body
    deferred to user for symmetry with the IS estimator.
    """
    raise NotImplementedError(
        "exact_internal_energy body deferred to user "
        "(E/D = -Σ_x π(x) log p̃(x) / D by enumeration). "
        "Stage 1 plan status, next-session todo step 2."
    )
