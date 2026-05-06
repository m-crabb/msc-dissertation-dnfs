import math

import torch
import pytest

from discrete_flow_sampler.diagnostics.metrics import (
    tvd, kl, ess_from_log_weights, enumerate_states, exact_log_probs,
    wasserstein1_1d, log_prob_w1,
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


def test_kl_zero_for_identical():
    log_p = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    assert kl(log_p, log_p).item() == pytest.approx(0.0, abs=1e-7)


def test_kl_matches_closed_form_on_two_state_bernoulli():
    # KL(Bern(0.7) || Bern(0.3)) = 0.7*log(0.7/0.3) + 0.3*log(0.3/0.7)
    log_p = torch.log(torch.tensor([0.7, 0.3]))
    log_q = torch.log(torch.tensor([0.3, 0.7]))
    expected = 0.7 * math.log(0.7 / 0.3) + 0.3 * math.log(0.3 / 0.7)
    assert kl(log_p, log_q).item() == pytest.approx(expected, abs=1e-6)


def test_kl_infinite_when_q_misses_p_support():
    # p has mass on state 1, q does not -> KL(p || q) = +inf.
    log_p = torch.log(torch.tensor([0.5, 0.5]))
    log_q = torch.tensor([0.0, -float("inf")])  # q = (1, 0) in prob space
    assert torch.isinf(kl(log_p, log_q)) and kl(log_p, log_q).item() > 0


def test_kl_finite_when_p_zeros_align_with_q_zeros():
    # p has zero mass on state where q has zero mass -> 0 * log(0/0) := 0,
    # the rest of the sum stays finite.  This pins the convention guard.
    log_p = torch.tensor([0.0, -float("inf")])  # p = (1, 0)
    log_q = torch.tensor([0.0, -float("inf")])  # q = (1, 0)
    assert kl(log_p, log_q).item() == pytest.approx(0.0)


def test_kl_asymmetric():
    # KL(p || q) != KL(q || p) on a non-uniform pair.
    log_p = torch.log(torch.tensor([0.8, 0.2]))
    log_q = torch.log(torch.tensor([0.5, 0.5]))
    forward = kl(log_p, log_q).item()
    reverse = kl(log_q, log_p).item()
    assert forward != pytest.approx(reverse)


def test_wasserstein1_zero_for_identical_samples():
    a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    assert wasserstein1_1d(a, a).item() == pytest.approx(0.0)


def test_wasserstein1_translation_invariant():
    a = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = a + 5.0
    # W1 between samples and a translated copy = the translation amount.
    assert wasserstein1_1d(a, b).item() == pytest.approx(5.0)


def test_wasserstein1_unequal_sample_sizes():
    # Sanity: W1 between two delta-at-different-points distributions of
    # different size should equal the gap between them.
    a = torch.tensor([0.0, 0.0, 0.0])  # all at 0
    b = torch.tensor([7.0, 7.0])        # all at 7
    assert wasserstein1_1d(a, b).item() == pytest.approx(7.0, abs=1e-6)


def test_log_prob_w1_zero_when_sample_sets_match():
    """If both sample sets are the same states, W1 of their log_probs is 0."""
    class TinyTarget:
        def log_prob(self, x):
            return x.sum(dim=-1).float()
    states = enumerate_states(D=3).float()
    assert log_prob_w1(states, states, TinyTarget()).item() == pytest.approx(0.0)


def test_log_prob_w1_positive_for_disjoint_log_prob_supports():
    """Two sample sets whose log_probs differ in mean should give W1 > 0."""
    class TinyTarget:
        def log_prob(self, x):
            return x.sum(dim=-1).float()
    # All-aligned samples: log_prob = +D ; All-anti-aligned samples: log_prob = -D
    aligned = torch.ones(10, 4)
    anti_aligned = -torch.ones(10, 4)
    w1 = log_prob_w1(aligned, anti_aligned, TinyTarget()).item()
    # log_prob differs by 8 (= 4 - (-4)) deterministically.
    assert w1 == pytest.approx(8.0, abs=1e-6)
