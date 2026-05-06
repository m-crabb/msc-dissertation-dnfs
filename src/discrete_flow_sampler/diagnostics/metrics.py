"""Diagnostics for DNFS replication: TVD, ESS, exact enumeration.

These are stage-agnostic measurement utilities used across the project:

- `tvd` and `exact_log_probs` provide the D=4 / D=10 ground-truth gates,
  comparing a sampler's empirical distribution to the target's exact
  partition-function-normalised distribution over enumerated state space.
- `ess_from_log_weights` is the headline diagnostic of Ou et al. (Eq. 15);
  computed in log-space for numerical stability under the weight regimes
  that arise early in DNFS training.
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
