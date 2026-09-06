"""What correct looks like for the soft flip log-ratio decomposition.

The soft analogue of the hard chapter's local-field regression rests on one
closed form, derived here and pinned against brute force:

    Delta_i(x) = log pi~(flip_i x) - log pi~(x)
               = x_i * [ -4 sigma h_i + 2 lambda (c_null - c*) + lambda/d ]

with h_i = (A x)_i the local field (no self-bond, so hollow at i) and
c_null = c(x) - (x_i + 1)/(2d) the HOLE-EXCLUDED composition. The base term
is -4 sigma x_i h_i because base_log_prob = x^T J x counts every edge twice.
The naive penalty difference 2 lambda x_i (c - c*) - lambda/d looks x_i-EVEN
in part, but substituting c = c_null + (x_i+1)/(2d) cancels the lambda/d
pieces exactly, leaving a form that is odd in x_i with a hollow coefficient.

Why the oddness matters and is worth a test of its own: the leTF emits
G(i|x) = -x_i S_i(x) with S hollow at i, so the model can ONLY represent
x_i-odd functions of the state. If the true log-ratio had an x_i-even
component the architecture could never learn it and the regression's R^2
ceiling would be below 1 by construction; the closed form says the ceiling
is exactly 1, and test 2 asserts that.

Failure modes these tests guard: a sign slip in the field term (the double
edge count is easy to halve), the penalty's extensive scaling (lambda*d,
mirroring VC-SGC, not bare lambda), and a feature matrix built from the
hole-INCLUDED composition, which would silently leak x_i into a "hollow"
column and inflate every downstream R^2.
"""

import itertools

import pytest
import torch

from discrete_flow_sampler.targets.ising import IsingTarget


def enumerate_all(d):
    states = torch.tensor(
        list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float32
    )
    return states


def brute_force_flip_log_ratio(target, x):
    """(N, d) matrix of log pi~(flip_i x) - log pi~(x), by direct evaluation."""
    base = target.log_prob(x)
    columns = []
    for i in range(x.shape[1]):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        columns.append(target.log_prob(flipped) - base)
    return torch.stack(columns, dim=1)


@pytest.fixture(scope="module")
def soft_target():
    return IsingTarget(
        D=3,
        sigma=0.17,
        device="cpu",
        target_composition=0.3,
        composition_penalty_strength=50.0,
    )


def closed_form_flip_log_ratio(target, x):
    """The derived form, built ONLY from hollow features and x_i."""
    d = x.shape[1]
    h = x @ target.A  # local fields, (N, d)
    c = target.composition_fraction(x).unsqueeze(1)  # (N, 1)
    c_hollow = c - (x + 1.0) / (2.0 * d)  # hole-excluded, (N, d)
    lam = target.composition_penalty_strength
    return x * (
        -4.0 * target.sigma * h
        + 2.0 * lam * (c_hollow - target.target_composition)
        + lam / d
    )


def test_closed_form_matches_brute_force(soft_target):
    x = enumerate_all(soft_target.d)
    brute = brute_force_flip_log_ratio(soft_target, x)
    closed = closed_form_flip_log_ratio(soft_target, x)
    assert torch.allclose(brute, closed, atol=1e-4), (brute - closed).abs().max()


def test_closed_form_matches_brute_force_without_penalty():
    """The base half alone: a pure Ising target isolates the -4 sigma x h term."""
    target = IsingTarget(D=3, sigma=0.25, device="cpu")
    x = enumerate_all(target.d)
    brute = brute_force_flip_log_ratio(target, x)
    closed = x * (-4.0 * target.sigma * (x @ target.A))
    assert torch.allclose(brute, closed, atol=1e-4)


def test_log_ratio_is_exactly_odd_in_the_flipped_spin(soft_target):
    """Delta_i(flip_i x) = -Delta_i(x): the architecture's representable set
    contains the truth, so the regression's R^2 ceiling is 1."""
    x = enumerate_all(soft_target.d)
    brute = brute_force_flip_log_ratio(soft_target, x)
    for i in range(soft_target.d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        brute_flipped = brute_force_flip_log_ratio(soft_target, flipped)
        assert torch.allclose(brute_flipped[:, i], -brute[:, i], atol=1e-4)


def test_features_are_hollow(soft_target):
    """h_i and c_hollow_i must not move when x_i flips — the leak test."""
    x = enumerate_all(soft_target.d)
    d = soft_target.d
    h = x @ soft_target.A
    c_hollow = soft_target.composition_fraction(x).unsqueeze(1) - (x + 1.0) / (2.0 * d)
    for i in range(d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        h_f = flipped @ soft_target.A
        c_f = soft_target.composition_fraction(flipped).unsqueeze(1) - (
            flipped + 1.0
        ) / (2.0 * d)
        assert torch.allclose(h_f[:, i], h[:, i], atol=1e-6)
        assert torch.allclose(c_f[:, i], c_hollow[:, i], atol=1e-6)
