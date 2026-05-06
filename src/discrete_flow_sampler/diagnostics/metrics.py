"""Diagnostics for DNFS replication: TVD, KL, ESS, exact enumeration,
and energy-distribution two-sample distances.

These are stage-agnostic measurement utilities used across the project:

- `tvd` / `kl` and `exact_log_probs` provide D=4 / D=10 ground-truth
  comparisons against the target's partition-function-normalised
  distribution over enumerated state space.
- `ess_from_log_weights` is the headline IS diagnostic of Ou et al.
  (Eq. 15); a *self-consistency* metric, NOT a coverage metric -- a
  proposal that lands confidently in spurious modes can have ESS = N.
  Always pair it with a coverage metric like TVD / KL / energy-W1.
- `wasserstein1_1d` and `log_prob_w1` give a coverage diagnostic that
  scales beyond enumeration: compare the empirical distribution of a
  scalar observable (default: log p̃) under model samples vs reference
  (oracle / exact) samples. Catches "wrong magnetisation" failure modes
  that ESS alone cannot.
"""

import itertools

import torch
from torch import Tensor


def tvd(p: Tensor, q: Tensor) -> Tensor:
    """Total variation distance between two probability vectors.

        TVD(p, q) = ½ · Σ_i |p_i - q_i|

    Both inputs must be 1-D tensors that sum to one and share the same
    ordering of states (i.e. p[i] and q[i] refer to the same state).
    No internal normalisation is performed; passing un-normalised vectors
    yields an un-normalised "distance" that won't have the usual [0, 1]
    range, so callers are responsible for normalising upstream.
    """
    return 0.5 * (p - q).abs().sum()


def kl(log_p: Tensor, log_q: Tensor) -> Tensor:
    """Kullback-Leibler divergence KL(p ‖ q) from log-probability vectors.

        KL(p ‖ q) = Σ_i p_i · (log p_i - log q_i)

    Asymmetric: order matters.

    - `kl(log_p_target, log_p_model)` — "forward / missing-mode KL".
      Penalises the model for assigning low mass where the target puts
      high mass. +∞ when the model puts zero mass on a target-supported
      state. Catches **missed modes**.
    - `kl(log_p_model, log_p_target)` — "reverse / spurious-mode KL".
      Penalises the model for putting mass where the target does not.
      Catches **spurious modes** (the failure that hides behind high ESS).

    Inputs are log-probabilities (1-D tensors over a shared, fixed
    ordering of states) so `+∞` from `log(0)` is naturally represented
    by `-inf`. The convention `0 · log 0 = 0` is enforced explicitly so
    a state with `p_i = 0` contributes nothing regardless of `log_q_i`.

    Returns a scalar Tensor in [0, +∞]. Returns `+inf` when q's support
    fails to cover p's.
    """
    p = log_p.exp()
    diff = log_p - log_q
    # 0 · log(0/q) := 0  (probability-theoretic convention).  Without
    # this guard, a zero-prob state with log_q = -inf produces 0 · inf
    # = nan and contaminates the sum.
    contribution = torch.where(p > 0, p * diff, torch.zeros_like(p))
    return contribution.sum()


def ess_from_log_weights(log_w: Tensor) -> Tensor:
    """Effective sample size from importance log-weights.

        ESS(w) = (Σ_n w_n)² / Σ_n w_n²
               = exp( 2·logsumexp(log_w) - logsumexp(2·log_w) )

    where log_w is a 1-D tensor of importance log-weights
    log(p̃_1(x_n) / q_θ(x_n)). Computed in log-space because raw weights
    routinely span 100+ orders of magnitude early in training; computing
    (Σw)² / Σw² directly under-/overflows.

    Returns a scalar tensor in [1, N]: N for uniform weights, 1 when one
    sample dominates.
    """
    log_sum_w = torch.logsumexp(log_w, dim=0)
    log_sum_w_sq = torch.logsumexp(2 * log_w, dim=0)
    return torch.exp(2 * log_sum_w - log_sum_w_sq)


def enumerate_states(D: int) -> Tensor:
    """All 2^D binary spin states for a D-site lattice.

    Each spin takes values in {-1, +1} (Ising convention, not the
    {0, 1} software convention). Returns a (2^D, D) int64 tensor;
    ordering is lexicographic via itertools.product([-1, 1], repeat=D).

    Used for D=4 (16 states) and D=10 (1024 states) ground-truth
    computations. Beyond D=20 this gets memory-bound and the function
    should not be called.
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


def wasserstein1_1d(samples_a: Tensor, samples_b: Tensor) -> Tensor:
    """1-Wasserstein distance between two 1-D empirical distributions.

        W₁(F_a, F_b) = ∫_ℝ |F_a(x) - F_b(x)| dx

    For equal sample sizes this collapses to the closed form

        W₁ = (1/N) · Σ_i |sorted(a)_i - sorted(b)_i|.

    For unequal sample sizes the integral is evaluated by stitching the
    two empirical CDFs onto a common axis and summing the staircase
    differences.

    Translation-invariant (W₁(a + c, b + c) = W₁(a, b)) and in the
    natural units of the input — for log-probability inputs that's
    "average displacement of one distribution to match the other in
    nats". Stronger than KS for diagnosing distribution shifts because
    KS is a sup-norm and ignores how far apart the CDFs are once the
    max gap is fixed.

    Args:
        samples_a, samples_b: 1-D float tensors. Need not be equal-sized.
    Returns:
        Scalar tensor.
    """
    if samples_a.dim() != 1 or samples_b.dim() != 1:
        raise ValueError(
            f"wasserstein1_1d requires 1-D inputs; got shapes "
            f"{tuple(samples_a.shape)} and {tuple(samples_b.shape)}."
        )

    sorted_a, _ = samples_a.sort()
    sorted_b, _ = samples_b.sort()

    if sorted_a.shape == sorted_b.shape:
        return (sorted_a - sorted_b).abs().mean()

    # Unequal sizes: stitch CDFs onto the merged-and-sorted x-axis and
    # integrate |F_a(x) - F_b(x)|.  Each interval [x_k, x_{k+1}) sees a
    # constant difference of CDFs, so the integral is a finite sum of
    # rectangles.
    n_a, n_b = sorted_a.numel(), sorted_b.numel()
    all_x = torch.cat([sorted_a, sorted_b]).sort().values  # (n_a + n_b,)
    deltas = all_x[1:] - all_x[:-1]                         # (n_a + n_b - 1,)
    cdf_a = torch.searchsorted(sorted_a, all_x[:-1], right=True).float() / n_a
    cdf_b = torch.searchsorted(sorted_b, all_x[:-1], right=True).float() / n_b
    return ((cdf_a - cdf_b).abs() * deltas).sum()


def log_prob_w1(
    samples_a: Tensor,
    samples_b: Tensor,
    target,
) -> Tensor:
    """1-Wasserstein distance between log-prob distributions of two
    sample sets under `target`.

    A coverage diagnostic that scales beyond enumeration: instead of
    asking "do the two empirical distributions match across all 2^d
    states?" (intractable for d ≥ 20), it asks "do the two distributions
    of the scalar `log p̃(x)` match?". Two distributions on {-1, +1}^d
    that agree on the law of every observable agree everywhere; in
    practice we only check this single scalar -- a 1-D summary that's
    nevertheless extremely sensitive to "wrong magnetisation" or
    "wrong energy mode" failures because log p̃ is a sufficient
    statistic for the Boltzmann family the target lives in.

    For Stage 1 D=4 we can use samples_b drawn from the exact target
    via multinomial; for Stage 1+ D=10 the same call works with
    samples_b drawn from a Gibbs / heat-bath oracle.

    Args:
        samples_a, samples_b: (N, d) tensors in {-1, +1}.  Sample sizes
            need not match.
        target: object exposing `log_prob(x) -> (N,)`.
    Returns:
        Scalar W₁ distance, in nats (units of log p̃).
    """
    log_prob_a = target.log_prob(samples_a.float())
    log_prob_b = target.log_prob(samples_b.float())
    return wasserstein1_1d(log_prob_a, log_prob_b)
