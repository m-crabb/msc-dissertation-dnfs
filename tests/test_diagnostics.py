import torch
import pytest

from discrete_flow_sampler.diagnostics.metrics import (
    tvd, ess_from_log_weights, enumerate_states, exact_log_probs,
)


def test_tvd_zero_for_identical():
    p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert tvd(p, p).item() == pytest.approx(0.0)


def test_tvd_one_for_disjoint():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    assert tvd(p, q).item() == pytest.approx(1.0)


def test_ess_equals_n_for_uniform_weights():
    log_w = torch.zeros(100)
    assert ess_from_log_weights(log_w).item() == pytest.approx(100.0)


def test_ess_one_for_dominant_weight():
    log_w = torch.full((100,), -100.0)
    log_w[0] = 0.0
    # one weight dominates → ESS → 1
    assert ess_from_log_weights(log_w).item() == pytest.approx(1.0, abs=1e-3)


def test_enumerate_states_shape_and_count():
    D = 3  # 2^3 = 8 states for binary spins
    states = enumerate_states(D)
    assert states.shape == (8, D)
    # all entries ∈ {-1, +1}
    assert torch.all((states == 1) | (states == -1))
    # all states unique
    assert len({tuple(s.tolist()) for s in states}) == 8


def test_exact_log_probs_normalises_to_one():
    # tiny target: log_prob(x) = sum(x). enumerate over D=2 → 4 states.
    class TinyTarget:
        def log_prob(self, x):
            return x.sum(dim=-1).float()
    states = enumerate_states(D=2)
    log_p = exact_log_probs(TinyTarget(), states)
    probs = log_p.exp()
    assert probs.sum().item() == pytest.approx(1.0)
