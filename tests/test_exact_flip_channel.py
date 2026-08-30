"""What correct looks like for the soft exact-field FLIP channel, before it.

The channel wraps a flip-rate model with the closed form the s90 regression
measured at ~95% of every trained specialist
(tests/test_soft_field_regression.py pins the derivation):

    G(i | x) <- G_head(i | x) + gain(t) * Delta_i(x),
    Delta_i  = x_i * [ -4 sigma h_i + 2 lambda (c_null_i - c*) + lambda/d ],
    gain(t)  = gain_constant + gain_slope * t,   both ZERO at init.

Contracts these tests freeze, each guarding a specific failure:

1. ZERO-INIT BIT-IDENTITY. A channel-on model must reproduce its parent
   exactly at initialisation — same construction RNG (the gains are
   zeros, no draws), same forward output. Without this the lambda-sweep
   twins would differ from Table 4.1's cells in their step-0 telemetry
   and the comparison would carry a second undeclared change.
2. GAIN=1 EQUALS BRUTE FORCE. With gain_constant=1 the added score at the
   flip slot must equal log pi~(flip_i x) - log pi~(x) computed directly
   from target.log_prob — tying the WIRING (not just the formula) to the
   target, so a sign slip or a hole-included composition cannot pass.
3. The current-token slot stays exactly zero (the DNFS convention the
   trainers rely on: summing over the vocab axis IS the flip score).
4. LIVE lambda. Curricula mutate the target in place
   (set_composition_penalty_strength); the channel must follow without
   rebuild, as the hard channel does for sigma.
5. REGISTRY PARITY. Every `_efc` twin differs from its parent in name and
   model.exact_field_channel ONLY — the one-declared-change rule that makes
   the rerun of tab:soft-lambda-sweep readable as "the channel did this".
"""

import itertools
from dataclasses import asdict

import pytest
import torch

from discrete_flow_sampler.constraints.exact_field_channel import (
    ExactFieldFlipModel)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.ising import IsingTarget


D_SIDE = 3  # 9 sites: big enough for a torus, small enough to enumerate


@pytest.fixture()
def target():
    return IsingTarget(
        D=D_SIDE, sigma=0.13, device="cpu",
        target_composition=0.4, composition_penalty_strength=50.0,
    )


def build_pair(target, seed=0):
    """(base, wrapped) leTF models with identical construction RNG."""
    def fresh():
        torch.manual_seed(seed)
        return LeTFRateMatrix(
            d=target.d, vocab_size=2, hidden_dim=32, n_layers=1, n_heads=2)
    return fresh(), ExactFieldFlipModel(fresh(), target)


def random_states(d, n=64, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, d), generator=generator).float() * 2 - 1


def test_zero_init_is_bit_identical(target):
    base, wrapped = build_pair(target)
    x = random_states(target.d)
    t = torch.rand(x.shape[0])
    assert torch.equal(base(x, t), wrapped(x, t))


def test_gain_one_matches_brute_force_log_ratio(target):
    _, wrapped = build_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
    x = random_states(target.d)
    t = torch.zeros(x.shape[0])  # gain(t)=1 exactly; isolates the feature
    added = wrapped(x, t) - wrapped.model(x, t)
    base_lp = target.log_prob(x)
    for i in range(target.d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        brute = target.log_prob(flipped) - base_lp
        flip_slot = (1 - ((x[:, i] + 1) / 2)).long()
        assert torch.allclose(
            added[torch.arange(x.shape[0]), i, flip_slot], brute, atol=1e-4)


def test_current_token_slot_stays_zero(target):
    _, wrapped = build_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(0.7)
        wrapped.gain_slope.fill_(-0.3)
    x = random_states(target.d)
    G = wrapped(x, torch.rand(x.shape[0]))
    current = ((x + 1) / 2).long().unsqueeze(-1)
    assert torch.equal(G.gather(-1, current), torch.zeros_like(current, dtype=G.dtype))


def test_channel_follows_live_lambda(target):
    _, wrapped = build_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
    x = random_states(target.d)
    t = torch.zeros(x.shape[0])
    before = wrapped(x, t) - wrapped.model(x, t)
    target.set_composition_penalty_strength(120.0)
    after = wrapped(x, t) - wrapped.model(x, t)
    assert not torch.allclose(before, after)
    base_lp = target.log_prob(x)  # brute force under the NEW lambda
    flipped = x.clone()
    flipped[:, 0] = -flipped[:, 0]
    brute = target.log_prob(flipped) - base_lp
    flip_slot = (1 - ((x[:, 0] + 1) / 2)).long()
    assert torch.allclose(
        after[torch.arange(x.shape[0]), 0, flip_slot], brute, atol=1e-4)


def test_wrapper_is_transparent_to_the_dispatchers(target):
    _, wrapped = build_pair(target)
    assert wrapped.is_locally_equivariant is True


def build_conditioned_pair(target, seed=0):
    """(base, wrapped) AMORTISED leTF models, identical construction RNG."""
    def fresh():
        torch.manual_seed(seed)
        return LeTFRateMatrix(
            d=target.d, vocab_size=2, hidden_dim=32, n_layers=1, n_heads=2,
            condition_on_composition=True)
    return fresh(), ExactFieldFlipModel(fresh(), target)


def test_zero_init_bit_identity_holds_amortised(target):
    """Contract 1 extended to the amortised route: passing c must not
    perturb the identity — the channel term is zero however c* is sourced."""
    base, wrapped = build_conditioned_pair(target)
    x = random_states(target.d)
    t = torch.rand(x.shape[0])
    c = torch.rand(x.shape[0])
    assert torch.equal(base(x, t, c), wrapped(x, t, c))


def test_per_row_composition_matches_brute_force(target):
    """The channel must aim at each ROW's composition, not the target scalar.

    Failure mode guarded: in an amortised batch the LOSS scores every row
    against its own c (bound via target.composition_batch), while a channel
    reading target.target_composition would inject a field aimed at one
    scalar c* for all rows — a silent disagreement between the transport the
    channel supplies and the target the loss trains toward, worst exactly at
    the off-centre windows amortisation exists to serve."""
    _, wrapped = build_conditioned_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
    x = random_states(target.d, n=8)
    t = torch.zeros(x.shape[0])  # gain(t)=1 exactly; isolates the feature
    c = torch.tensor([0.25, 0.375, 0.5, 0.625, 0.75, 0.3, 0.4, 0.6])
    added = wrapped(x, t, c) - wrapped.model(x, t, c)
    with target.composition_batch(c):
        base_lp = target.log_prob(x)
        for i in range(target.d):
            flipped = x.clone()
            flipped[:, i] = -flipped[:, i]
            brute = target.log_prob(flipped) - base_lp
            flip_slot = (1 - ((x[:, i] + 1) / 2)).long()
            assert torch.allclose(
                added[torch.arange(x.shape[0]), i, flip_slot], brute,
                atol=1e-4)


def test_amortised_adapter_stack_end_to_end(target):
    """The full amortised stack — CompositionConditioned(channel(leTF)) —
    must run under the samplers' plain (x, t) call convention, expand the
    per-block composition b-major, and mirror the dispatch attributes.
    Guards the integration gap the specialist twins never exercised."""
    from discrete_flow_sampler.composition import expand_b_major
    from discrete_flow_sampler.models.composition_conditioned import (
        CompositionConditioned)

    _, wrapped = build_conditioned_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
        wrapped.gain_slope.fill_(0.2)
    c_blocks = torch.tensor([0.25, 0.75])
    bound = CompositionConditioned(wrapped, c_blocks)
    assert bound.is_locally_equivariant is True
    x = random_states(target.d, n=4)  # two b-major blocks of two rows
    t = torch.rand(4)
    expected = wrapped(x, t, expand_b_major(c_blocks, 4))
    assert torch.equal(bound(x, t), expected)


def test_efc_twins_differ_from_parents_in_flag_and_name_only():
    from experiments.constrained_soft_02.configs import (
        CONFIGS, LAMBDA_SWEEP_PARENTS)

    assert len(LAMBDA_SWEEP_PARENTS) == 8  # 2 lattices x 4 lambdas
    for parent_name in LAMBDA_SWEEP_PARENTS:
        parent = asdict(CONFIGS[parent_name])
        twin = asdict(CONFIGS[f"{parent_name}_efc"])
        deviated = {k for k in parent if parent[k] != twin[k]}
        assert deviated == {"name", "model"}, (parent_name, deviated)
        model_dev = {
            k for k in parent["model"] if parent["model"][k] != twin["model"][k]
        }
        assert model_dev == {"exact_field_channel"}, (parent_name, model_dev)
        assert twin["model"]["exact_field_channel"] is True
