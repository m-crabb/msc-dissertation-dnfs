"""Adaptive systematic resampling: the eval-time AIS -> SMC upgrade.

Plain annealed-IS eval accumulates log w = ∫₀¹ ξ_t(x_t) dt per particle
(Eq. 8 integrand / Eq. 41 weight) and reads ESS once at t = 1 (Eq. 42).
Since ESS/B ≈ exp(−var(log w)), the ESS ceiling is accumulated weight
variance. At checkpoints where interim ESS < τ·B the population is redrawn
(B ancestors ∝ normalised weights) and log-weights reset to zero; the
learned rates become the SMC proposal (AIS -> SMC, CRAFT lineage).

Resampling forgets the absolute weight scale, so each event first banks
the segment's contribution: with equal weights at segment start,
E[(1/B) Σᵢ e^{log wᵢ}] estimates the Z-ratio over that segment, and the
ratios telescope:

    log Ẑ = Σ_events log( (1/B) Σᵢ e^{log wᵢ} ) + logmeanexp(final log w).

`smc_log_z_estimate` implements this product form. The Eq. 37 Jensen
lower bound (`log_weights.mean()`) is not valid across resampled segments:
post-resample particles share ancestors, so segment weights are correlated.

Everything here touches only the (state, log_weights) ensemble, so
`sample_ctmc` and `sample_swap_ctmc` share `resample_if_needed`; in the
swap case whole on-manifold rows are duplicated and the composition stays
exact. Training uses the same hook in the buffer-rebuild rollout
(`TrainCfg.rollout_resample_ess_fraction`) for a different reason: the
surviving ensemble is the equally-weighted representation of p_t that the
c_t batch mean (Eq. 8) needs (LEAPS Algorithm 1, Algorithm 2 line 5).

A checkpoint that does not fire consumes no randomness, so a never-firing
config is bit-identical to the plain sampler (tests/test_resampling.py).
"""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights


@dataclass(frozen=True)
class ResamplingConfig:
    """Adaptive-resampling policy passed to the CTMC samplers.

    ess_threshold_fraction: fire when interim ESS < fraction·B. 0.0 never
        fires (parity with plain IS eval); 1.0 fires whenever the weights
        are not exactly uniform; 0.5 is the conventional default.
    check_every: evaluate the trigger every k-th Euler step.
    """

    ess_threshold_fraction: float = 0.5
    check_every: int = 1


@dataclass
class ResamplingStats:
    """SMC bookkeeping accumulated by the sampler loop across one eval draw.

    log_z_increment: scalar tensor, Σ over fired events of
        logmeanexp(segment log-weights); the banked part of the product
        form. Final estimate = `smc_log_z_estimate(stats, final_log_w)`.
    n_events / event_steps: how often and at which Euler steps resampling
        fired.
    """

    log_z_increment: Tensor
    n_events: int = 0
    event_steps: list[int] = field(default_factory=list)


def log_mean_exp(log_values: Tensor) -> Tensor:
    """log( (1/K) Σ_k exp(log_values_k) ), computed stably in log-space."""
    return torch.logsumexp(log_values, dim=0) - math.log(log_values.shape[0])


def systematic_resample_indices(
    log_weights: Tensor, uniform: Tensor | float | None = None
) -> Tensor:
    """Systematic (low-variance) ancestor indices ∝ softmax(log_weights).

    With w̄ = softmax(log_w) and CDF c_i = Σ_{j≤i} w̄_j, draw one jitter
    u ~ U[0,1) and query positions (u + k)/B for k = 0..B−1. Ancestor k is
    the first i with c_i > (u + k)/B: strict, so each particle owns the
    half-open interval [c_{i−1}, c_i) and a position exactly on a boundary
    maps to the right particle (ties matter for the u = 0, uniform-weight
    grid where positions coincide with CDF entries).

    An interval of length L_i = B·w̄_i on the scaled CDF axis contains
    ⌊L_i⌋ or ⌈L_i⌉ unit-spaced grid points, so

        counts_i ∈ {⌊B·w̄_i⌋, ⌈B·w̄_i⌉},   E[counts_i] = B·w̄_i:

    unbiased, with the smallest count variance of the standard schemes
    (multinomial adds ≈ B·w̄_i(1−w̄_i) per particle).

    `uniform` overrides the single RNG draw for deterministic tests.

    Returns: (B,) long tensor of ancestor indices into the particle batch.
    """
    batch_size = log_weights.shape[0]
    normalised_weights = torch.softmax(log_weights, dim=0)
    cdf = normalised_weights.cumsum(dim=0)
    if uniform is None:
        uniform = torch.rand((), device=log_weights.device, dtype=cdf.dtype)
    grid = torch.arange(batch_size, device=log_weights.device, dtype=cdf.dtype)
    positions = (uniform + grid) / batch_size
    # right=True implements the strict c_i > v convention above; the clamp
    # guards the last bin when float cumsum tops out slightly below 1.0.
    return torch.searchsorted(cdf, positions, right=True).clamp(max=batch_size - 1)


def resample_if_needed(
    state: Tensor, log_weights: Tensor, ess_threshold_fraction: float
) -> tuple[Tensor, Tensor, Tensor, bool]:
    """One adaptive-resampling checkpoint on the (state, log_weights) ensemble.

    Trigger: ESS(log_w) < fraction·B, with ESS per Eq. 42
    (`diagnostics.metrics.ess_from_log_weights`). On fire:

        increment  = logmeanexp(log_w)      # bank the segment's Ẑ factor
        state      = state[ancestors]        # systematic, ∝ softmax(log_w)
        log_w      = 0                       # equal weights restart

    Not fired: inputs are returned unchanged and no RNG is consumed, so a
    never-firing config replays the plain sampler bit-exactly.

    Returns: (state, log_weights, log_z_increment, fired) where
    log_z_increment is a scalar tensor (0 when not fired).
    """
    batch_size = log_weights.shape[0]
    ess = ess_from_log_weights(log_weights)
    if ess.item() >= ess_threshold_fraction * batch_size:
        zero_increment = torch.zeros(
            (), dtype=log_weights.dtype, device=log_weights.device
        )
        return state, log_weights, zero_increment, False
    log_z_increment = log_mean_exp(log_weights)
    ancestors = systematic_resample_indices(log_weights)
    return state[ancestors], torch.zeros_like(log_weights), log_z_increment, True


def smc_log_z_estimate(stats: ResamplingStats, final_log_weights: Tensor) -> Tensor:
    """Unbiased SMC product-form log Ẑ (banked increments + final segment).

    log Ẑ = stats.log_z_increment + logmeanexp(final_log_weights).
    With zero events this is the plain-IS estimator logmeanexp(log w). Do
    not substitute the Eq. 37 Jensen-LB form; see the module docstring.
    """
    return stats.log_z_increment + log_mean_exp(final_log_weights)
