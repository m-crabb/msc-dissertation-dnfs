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

import numpy as np
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


def marginal_tvd(p: Tensor, q: Tensor) -> float:
    """½ Σ |p − q| between two pmfs on a shared 1-D support (both sum to 1)."""
    return 0.5 * (p - q).abs().sum().item()


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


def integrated_autocorr(x, c_window: float = 5.0) -> float:
    """Sokal automatic-windowing integrated autocorrelation time.

    τ_int = 1 + 2 Σ_{k=1}^{W} ρ(k), with W the smallest integer such that
    W >= c_window * τ_int(W). Uses an FFT for the autocovariance. Input is a
    1-D trace; the returned τ_int is in units of the trace's sampling interval
    (multiply by any thinning factor to recover raw-step units).

    Used to quantify MCMC mixing: ESS = n_samples / τ_int. For the §3.1
    Kawasaki failure demo, τ_int diverging with σ is the critical-slowing-down
    signature.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    var0 = np.dot(x, x) / n
    if var0 == 0:
        return 1.0
    n2 = 1
    while n2 < 2 * n:
        n2 *= 2
    f = np.fft.fft(x, n=n2)
    acov = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    acf = acov / (var0 * np.arange(n, 0, -1))
    tau = 1.0
    for W in range(1, n):
        tau += 2.0 * acf[W]
        if W >= c_window * tau:
            return float(tau)
    return float(tau)


def gelman_rubin(chains) -> float:
    """Between-chain potential scale reduction factor R̂.

    chains: (n_chains, n_samples) array of a scalar observable. R̂ ≈ 1 when the
    chains have mixed to a common distribution; R̂ ≫ 1 signals chains trapped in
    distinct modes — the §3.1 mode-coverage failure at low temperature.

        var_hat = (1 - 1/n) W + B/n,   R̂ = sqrt(var_hat / W),

    with W the mean within-chain variance and B the between-chain variance.
    """
    chains = np.asarray(chains, dtype=np.float64)
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    within_var = chains.var(axis=1, ddof=1).mean()
    between_var = n * chain_means.var(ddof=1)
    if within_var == 0:
        return float("inf")
    var_hat = (1 - 1 / n) * within_var + between_var / n
    return float(np.sqrt(var_hat / within_var))


def split_half_gelman_rubin(chains) -> float:
    """Split-half R̂ (Vehtari et al. 2021 convention): each chain is cut into
    two halves and the 2m halves enter the plain multi-chain `gelman_rubin`
    as if they were independent chains.

    Why the split matters: plain R̂ only compares BETWEEN-chain means, so m
    chains drifting in lockstep (all still relaxing from comparable inits,
    none stationary) can pass it while every chain is biased the same way.
    Halving makes each chain's own first-half/second-half disagreement count
    as a between-"chain" discrepancy — exactly the non-stationarity a
    reference-chain validity bar must catch. For odd chain lengths the middle
    sample is dropped so both halves are equal length (rank-normalisation and
    folding from the full Vehtari recipe are intentionally out of scope: the
    probe gates on location/scale mixing of near-Gaussian observables).
    """
    chains = np.asarray(chains, dtype=np.float64)
    n_samples = chains.shape[1]
    half = n_samples // 2
    if half < 2:
        raise ValueError(
            "split-half R-hat needs at least 4 samples per chain; got "
            f"{n_samples}"
        )
    halves = np.concatenate(
        [chains[:, :half], chains[:, n_samples - half:]], axis=0
    )
    return gelman_rubin(halves)


def nn_correlation(x: Tensor, adjacency: Tensor) -> Tensor:
    """Mean nearest-neighbour spin product <s_i s_j> over lattice edges, (B,).

    x^T A x sums s_i s_j over every ordered adjacent pair, i.e. twice per
    undirected edge; adjacency.sum() is the matching count of ordered pairs, so
    the ratio is the mean bond correlation. For a ferromagnet at c=0.5 this is
    ∝ the per-site energy; the diagonal correlation (out of scope) is the
    independent within-energy-level check.
    """
    x = x.float()
    quadratic = torch.einsum("bi,ij,bj->b", x, adjacency, x)
    return quadratic / adjacency.sum()


def diagonal_correlation(x: Tensor, D: int) -> Tensor:
    """Mean diagonal (next-nearest-neighbour) spin product over the DxD torus, (B,).

    Independent within-energy-level spatial check for the fixed-composition gate:
    nn_correlation is ∝ energy on the slice, so it cannot see within-level
    structure; the diagonal correlation can. Diagonal neighbours of site (r,c)
    are (r±1, c±1) with PBC. Same xᵀA_diag x / A_diag.sum() reduction as
    nn_correlation. For a checkerboard this is +1 (diagonal neighbours share
    colour), the opposite sign to nn_correlation (−1).
    """
    x = x.float()
    d = D * D
    A_diag = torch.zeros((d, d), device=x.device, dtype=x.dtype)
    for r in range(D):
        for c in range(D):
            i = r * D + c
            down_right = ((r + 1) % D) * D + (c + 1) % D
            down_left = ((r + 1) % D) * D + (c - 1) % D
            A_diag[i, down_right] = 1.0
            A_diag[i, down_left] = 1.0
    A_diag = A_diag + A_diag.T
    quadratic = torch.einsum("bi,ij,bj->b", x, A_diag, x)
    return quadratic / A_diag.sum()


def half_magnetisation_order_parameter(x: Tensor, D: int) -> Tensor:
    """Mode order parameter phi = (m_left - m_right) / 2 on the DxD torus, (B,).

    m_left / m_right are the mean spins of columns [0, D//2) and [D//2, D) of
    the row-major flattening; the half factor puts phi in [-1, 1]. NOTE the
    numba twin `mcmc.kawasaki.left_minus_right` omits the half factor, so its
    phi spans [-2, 2] — figures built from the two are on different scales.
    At fixed
    50/50 composition the two phase-separated modes give phi = +1 / -1 while
    total magnetisation is constant on the slice, so phi is the mode-coverage
    observable of the frozen mixing-probe metric (pre-registration
    2026-07-03); E_pi[phi] = 0 by the global spin-flip symmetry of the slice.
    """
    grid = x.float().reshape(*x.shape[:-1], D, D)
    m_left = grid[..., :, : D // 2].mean(dim=(-2, -1))
    m_right = grid[..., :, D // 2:].mean(dim=(-2, -1))
    return 0.5 * (m_left - m_right)
