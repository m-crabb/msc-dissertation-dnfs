"""Tests for the Wolff single-cluster sampler.

The sampler's role is ground-truth cross-check for the baseline chapter, so
its own correctness bar is exactness against enumeration on lattices small
enough to enumerate — the same gate every sampler in this project passes
before being trusted at size. Statistical assertions use fixed seeds and
tolerances a factor of ~3 above the binomial/SE scale of the sample counts,
so they are deterministic in CI and still tight enough to catch a wrong
p_add (the double-counted convention makes it 1 - exp(-4 sigma); using
1 - exp(-2 sigma) shifts the 3x3 energy mean by many standard errors).
"""

import itertools

import numpy as np
import pytest
import torch

from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget


def _exact_moments(D: int, sigma: float):
    """<pair_sum> and <|M|/d> by enumeration, plus the pair_sum pmf support.

    Energy/probability convention matches IsingTarget: p ∝ exp(2 sigma * S_p)
    with S_p = sum over unordered torus edges of s_i s_j.
    """
    n_sites = D * D
    bonds = []
    for r in range(D):
        for c in range(D):
            i = r * D + c
            bonds.append((i, r * D + (c + 1) % D))
            bonds.append((i, ((r + 1) % D) * D + c))
    states = np.array(
        list(itertools.product((-1, 1), repeat=n_sites)), dtype=np.float64
    )
    pair_sum = np.zeros(len(states))
    for i, j in bonds:
        pair_sum += states[:, i] * states[:, j]
    log_w = 2.0 * sigma * pair_sum
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    abs_mag = np.abs(states.sum(axis=1)) / n_sites
    return (
        float(weights @ pair_sum),
        float(weights @ abs_mag),
        pair_sum,
        weights,
    )


def _pair_sum(samples: torch.Tensor, D: int) -> torch.Tensor:
    grid = samples.reshape(-1, D, D)
    return (grid * grid.roll(1, dims=1)).sum(dim=(1, 2)) + (
        grid * grid.roll(1, dims=2)
    ).sum(dim=(1, 2))


def test_energy_and_magnetisation_match_enumeration_3x3():
    target = IsingTarget(D=3, sigma=0.2, bias=0.0, device="cpu")
    samples = wolff_sample(target, n_samples=20_000, seed=0)
    expected_pairs, expected_abs_mag, _, _ = _exact_moments(3, 0.2)
    pair_mean = _pair_sum(samples, 3).mean().item()
    abs_mag_mean = samples.sum(dim=1).abs().mean().item() / 9
    # SE of pair_sum over 20k near-independent draws is ~0.03; 3x margin.
    assert pair_mean == pytest.approx(expected_pairs, abs=0.1)
    assert abs_mag_mean == pytest.approx(expected_abs_mag, abs=0.02)


def test_energy_distribution_matches_enumeration_4x4_critical():
    """At sigma_c (exact, SIGMA_C): TV between the sampled and exact pmf of the
    pair-sum (the energy sufficient statistic) is small, on its full support."""
    sigma = SIGMA_C
    target = IsingTarget(D=4, sigma=sigma, bias=0.0, device="cpu")
    samples = wolff_sample(target, n_samples=20_000, seed=1)
    _, _, pair_sum, weights = _exact_moments(4, sigma)
    levels = np.unique(pair_sum)
    exact_pmf = np.array([weights[pair_sum == level].sum() for level in levels])
    sampled = _pair_sum(samples, 4).numpy()
    empirical_pmf = np.array([(sampled == level).mean() for level in levels])
    total_variation = 0.5 * np.abs(exact_pmf - empirical_pmf).sum()
    assert total_variation < 0.02
    assert empirical_pmf.sum() == pytest.approx(1.0)  # no off-support energies


def test_visits_both_z2_sectors_at_criticality():
    """Cluster moves tunnel between the all-up and all-down basins — the
    failure mode of the single-site reference this sampler exists to check."""
    target = IsingTarget(D=10, sigma=SIGMA_C, bias=0.0, device="cpu")
    samples = wolff_sample(target, n_samples=500, seed=2)
    magnetisation = samples.sum(dim=1)
    assert (magnetisation > 0).any() and (magnetisation < 0).any()


def test_rejects_nonzero_field():
    """The single-cluster rule is only detailed-balanced at h = 0."""
    target = IsingTarget(D=4, sigma=0.2, bias=0.1, device="cpu")
    with pytest.raises(ValueError, match="bias"):
        wolff_sample(target, n_samples=10, seed=0)


def test_output_shape_and_values():
    target = IsingTarget(D=5, sigma=0.1, bias=0.0, device="cpu")
    samples = wolff_sample(target, n_samples=64, seed=3)
    assert samples.shape == (64, 25)
    assert set(samples.unique().tolist()) <= {-1.0, 1.0}
