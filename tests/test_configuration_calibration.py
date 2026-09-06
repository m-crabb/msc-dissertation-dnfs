"""Configuration counts must detect errors hidden by energy marginals."""

import numpy as np
import torch
from scripts.configuration_calibration_4x4 import (
    calibration_metrics,
    count_configurations,
    exact_probabilities,
    reference_metrics,
)

from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def test_counts_include_unvisited_states_and_preserve_site_order():
    states = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    assert count_configurations(states).tolist() == [1, 2, 1, 0]


def test_equal_energy_redistribution_is_visible_without_weights():
    # All four states have equal target probability. An energy histogram
    # cannot distinguish this collapse from the target, but state counts can.
    p = np.full(4, 0.25)
    good = calibration_metrics(np.array([100, 100, 100, 100]), p)
    bad = calibration_metrics(np.array([200, 200, 0, 0]), p)
    assert good["tv"] == 0.0 and good["unvisited_target_mass"] == 0.0
    assert bad["tv"] == 0.5 and bad["unvisited_target_mass"] == 0.5


def test_off_slice_probability_is_zero_and_all_legal_states_are_present():
    p = exact_probabilities(FixedCompositionIsingTarget(2, 0.1, 0.5))
    assert len(p) == 16 and np.count_nonzero(p) == 6
    np.testing.assert_allclose(p.sum(), 1.0)
    assert p[0] == p[-1] == 0.0


def test_sampling_reference_uses_the_actual_draw_budget():
    p = np.full(100, 0.01)
    small = reference_metrics(p, 100, replicates=200, seed=123)
    large = reference_metrics(p, 10000, replicates=200, seed=123)
    # The observed TV has a nonzero finite-sample floor, shrinking with N.
    assert 0.3 < small["tv_median"] < 0.45
    assert 0.03 < large["tv_median"] < 0.05
    assert small["unvisited_target_mass_median"] > 0.3
    assert large["unvisited_target_mass_median"] == 0.0
