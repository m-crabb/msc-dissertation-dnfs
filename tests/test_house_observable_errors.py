"""House-table observable errors (MDNS Eq. 26/28, DASBS EW2) -- what correct looks like.

Written before the implementation. Three properties pin the definitions:
  * self-distance is zero (sampler == reference, uniform weights);
  * the metrics are weight-aware (a weighted sampler must reproduce a
    reference that the UNweighted sampler does not);
  * EW2 of a pure shift equals the shift (1-D W2 is the shift for translations).
"""

import torch

from discrete_flow_sampler.diagnostics.metrics import (
    correlation_profile_error,
    energy_wasserstein2,
    magnetisation_profile_error,
)

D = 4


def _spins(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, D * D), generator=g).float() * 2 - 1


def test_self_distance_is_zero():
    x = _spins(512, 0)
    uniform = torch.full((512,), 1 / 512)
    assert magnetisation_profile_error(x, uniform, x, D) == 0.0
    assert correlation_profile_error(x, uniform, x, D) == 0.0
    e = x.sum(-1)
    assert energy_wasserstein2(e, uniform, e) < 1e-6


def test_all_up_reference_against_random_sampler():
    # Reference all +1: M_row(k) = L for every row/col. Random sampler has
    # M_row ~ 0, so dMag = (1/2L) * 2L * L = L exactly in expectation.
    ref = torch.ones(8, D * D)
    x = _spins(4096, 1)
    uniform = torch.full((4096,), 1 / 4096)
    assert abs(magnetisation_profile_error(x, uniform, ref, D) - D) < 0.1


def test_weights_are_used():
    # Weighting a random sampler onto its all-up member reproduces an all-up reference.
    x = _spins(64, 2)
    x[0] = 1.0
    w = torch.zeros(64)
    w[0] = 1.0
    ref = torch.ones(4, D * D)
    assert magnetisation_profile_error(x, w, ref, D) == 0.0
    assert correlation_profile_error(x, w, ref, D) == 0.0


def test_ew2_of_shift_is_the_shift():
    g = torch.Generator().manual_seed(3)
    e = torch.randn(4000, generator=g)
    uniform = torch.full((4000,), 1 / 4000)
    assert abs(energy_wasserstein2(e + 0.3, uniform, e) - 0.3) < 0.01


def test_weighted_reference_equals_duplicated_reference():
    # The 4x4 hard house table's reference is the exactly enumerated slice
    # with Boltzmann probabilities, entering as unique states + weights
    # rather than as drawn samples. The extension is pinned by equivalence:
    # weighting a reference row 2/8 must equal physically duplicating it in
    # an unweighted reference of 8 rows.
    x = _spins(256, 4)
    w = torch.full((256,), 1 / 256)
    unique = _spins(6, 5)
    duplicated = torch.cat([unique, unique[:2]])
    ref_w = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0]) / 8
    for metric in (magnetisation_profile_error, correlation_profile_error):
        assert (
            abs(
                metric(x, w, duplicated, D)
                - metric(x, w, unique, D, reference_weights=ref_w)
            )
            < 1e-5
        )
    assert (
        abs(
            energy_wasserstein2(x.sum(-1), w, duplicated.sum(-1))
            - energy_wasserstein2(x.sum(-1), w, unique.sum(-1), reference_weights=ref_w)
        )
        < 1e-5
    )
