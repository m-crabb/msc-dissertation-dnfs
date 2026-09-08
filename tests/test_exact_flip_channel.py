"""The soft exact-field flip channel.

The channel wraps a flip-rate model with the closed form the field regression
measured at ~95% of every trained specialist
(tests/test_soft_field_regression.py pins the derivation):

    G(i | x) <- G_head(i | x) + gain(t) * Delta_i(x),
    Delta_i  = x_i * [ -4 sigma h_i + 2 lambda (c_null_i - c*) + lambda/d ],
    gain(t)  = gain_constant + gain_slope * t,   both zero at init.

Properties pinned:

1. Zero-init bit-identity: a channel-on model reproduces its parent exactly at
   initialisation — same construction RNG (the gains are zeros, no draws),
   same forward output — so the lambda-sweep twins match Table 4.1's cells in
   their step-0 telemetry.
2. Gain=1 equals brute force: with gain_constant=1 the added score at the flip
   slot equals log pi~(flip_i x) - log pi~(x) computed directly from
   target.log_prob, tying the wiring and not just the formula to the target.
3. The current-token slot stays exactly zero (the DNFS convention the
   trainers rely on: summing over the vocab axis is the flip score).
4. Live lambda: curricula mutate the target in place
   (set_composition_penalty_strength) and the channel follows without
   rebuild, as the hard channel does for sigma.
5. Config parity: every `_efc` twin differs from its parent in name and
   model.exact_field_channel alone.
"""

from dataclasses import asdict

import pytest
import torch

from discrete_flow_sampler.constraints.exact_field_channel import ExactFieldFlipModel
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec,
    ClusterExpansionTarget,
)
from discrete_flow_sampler.targets.ising import IsingTarget

D_SIDE = 3  # 9 sites: big enough for a torus, small enough to enumerate


@pytest.fixture()
def target():
    return IsingTarget(
        D=D_SIDE,
        sigma=0.13,
        device="cpu",
        target_composition=0.4,
        composition_penalty_strength=50.0,
    )


def build_pair(target, seed=0):
    """(base, wrapped) leTF models with identical construction RNG."""

    def fresh():
        torch.manual_seed(seed)
        return LeTFRateMatrix(
            d=target.d, vocab_size=2, hidden_dim=32, n_layers=1, n_heads=2
        )

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
            added[torch.arange(x.shape[0]), i, flip_slot], brute, atol=1e-4
        )


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
    base_lp = target.log_prob(x)  # brute force under the new lambda
    flipped = x.clone()
    flipped[:, 0] = -flipped[:, 0]
    brute = target.log_prob(flipped) - base_lp
    flip_slot = (1 - ((x[:, 0] + 1) / 2)).long()
    assert torch.allclose(
        after[torch.arange(x.shape[0]), 0, flip_slot], brute, atol=1e-4
    )


def test_wrapper_is_transparent_to_the_dispatchers(target):
    _, wrapped = build_pair(target)
    assert wrapped.is_locally_equivariant is True


def build_conditioned_pair(target, seed=0):
    """(base, wrapped) amortised leTF models, identical construction RNG."""

    def fresh():
        torch.manual_seed(seed)
        return LeTFRateMatrix(
            d=target.d,
            vocab_size=2,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            condition_on_composition=True,
        )

    return fresh(), ExactFieldFlipModel(fresh(), target)


def build_composition_gain_pair(target, seed=0):
    """Legacy and c-gain wrappers whose shared tensors are identical."""

    def fresh():
        torch.manual_seed(seed)
        return LeTFRateMatrix(
            d=target.d,
            vocab_size=2,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            condition_on_composition=True,
        )

    return (
        ExactFieldFlipModel(fresh(), target),
        ExactFieldFlipModel(fresh(), target, composition_conditioned_gain=True),
    )


def test_zero_init_bit_identity_holds_amortised(target):
    """Contract 1 extended to the amortised route: passing c must not
    perturb the identity — the channel term is zero however c* is sourced."""
    base, wrapped = build_conditioned_pair(target)
    x = random_states(target.d)
    t = torch.rand(x.shape[0])
    c = torch.rand(x.shape[0])
    assert torch.equal(base(x, t, c), wrapped(x, t, c))


def test_composition_gain_is_opt_in_zero_init_and_rng_neutral(target):
    """Adding the two c-gain scalars must not move the legacy trajectory at
    step zero or perturb any tensor shared with the parent arm."""
    legacy, conditioned = build_composition_gain_pair(target)
    legacy_state = legacy.state_dict()
    conditioned_state = conditioned.state_dict()

    extra = set(conditioned_state) - set(legacy_state)
    assert extra == {"composition_gain_constant", "composition_gain_slope"}
    for key, value in legacy_state.items():
        assert torch.equal(value, conditioned_state[key]), key

    x = random_states(target.d)
    t = torch.rand(x.shape[0])
    c = torch.rand(x.shape[0])
    assert torch.equal(legacy(x, t, c), conditioned(x, t, c))


def test_composition_gain_adds_centred_bilinear_correction(target):
    """The opt-in gain is g0 + g1*t + (c-c0)*(h0+h1*t)."""
    _, wrapped = build_composition_gain_pair(target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(0.2)
        wrapped.gain_slope.fill_(0.3)
        wrapped.composition_gain_constant.fill_(0.4)
        wrapped.composition_gain_slope.fill_(-0.1)

    x = random_states(target.d, n=4)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    c = torch.tensor([0.25, target.target_composition, 0.5, 0.75])
    added = wrapped(x, t, c) - wrapped.model(x, t, c)
    expected_gain = 0.2 + 0.3 * t + (c - target.target_composition) * (0.4 - 0.1 * t)
    expected = expected_gain.unsqueeze(1) * wrapped.exact_field(x, composition=c)
    flip_slot = (1 - ((x + 1) / 2)).long().unsqueeze(-1)
    assert torch.allclose(added.gather(-1, flip_slot).squeeze(-1), expected)


def test_composition_gain_requires_conditioned_model_and_runtime_c(target):
    torch.manual_seed(0)
    unconditioned = LeTFRateMatrix(
        d=target.d, vocab_size=2, hidden_dim=32, n_layers=1, n_heads=2
    )
    with pytest.raises(ValueError, match="composition-conditioned gain"):
        ExactFieldFlipModel(unconditioned, target, composition_conditioned_gain=True)

    _, wrapped = build_composition_gain_pair(target)
    x = random_states(target.d, n=4)
    t = torch.rand(x.shape[0])
    with pytest.raises(ValueError, match="composition c must be supplied"):
        wrapped(x, t)


def test_per_row_composition_matches_brute_force(target):
    """The channel must aim at each row's composition, not the target scalar.

    In an amortised batch the loss scores every row against its own c (bound
    via target.composition_batch), so a channel reading
    target.target_composition would inject a field aimed at one scalar c* for
    all rows: a silent disagreement between the transport the channel supplies
    and the target the loss trains toward, worst at the off-centre windows."""
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
                added[torch.arange(x.shape[0]), i, flip_slot], brute, atol=1e-4
            )


def test_amortised_adapter_stack_end_to_end(target):
    """The full amortised stack — CompositionConditioned(channel(leTF)) —
    must run under the samplers' plain (x, t) call convention, expand the
    per-block composition b-major, and mirror the dispatch attributes.
    Guards the integration gap the specialist twins never exercised."""
    from discrete_flow_sampler.composition import expand_b_major
    from discrete_flow_sampler.models.composition_conditioned import (
        CompositionConditioned,
    )

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
    from experiments.constrained_soft_02.configs import CONFIGS, LAMBDA_SWEEP_PARENTS

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


# --- the channel on a cluster expansion -----------------------------------
# The Cu-Au 16-site export carries pair and multi-body terms the Ising
# quadratic form cannot express; the channel must read the target's own
# flip log-ratio and match brute force on it exactly as it does on Ising.


@pytest.fixture()
def alloy_target():
    spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
    beta = 1.0 / (8.617333262e-5 * 500.0)
    return ClusterExpansionTarget(
        spec,
        beta,
        device="cpu",
        target_composition=0.5,
        composition_penalty_strength=10.0,
    )


@pytest.mark.parametrize("which", ["ising", "alloy"])
def test_base_flip_log_ratio_matches_brute_force(which, target, alloy_target):
    tgt = target if which == "ising" else alloy_target
    x = random_states(tgt.d)
    base = tgt.base_log_prob(x)
    for i in range(tgt.d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        torch.testing.assert_close(
            tgt.base_flip_log_ratio(x)[:, i],
            tgt.base_log_prob(flipped) - base,
            atol=1e-4,
            rtol=0,
        )


def test_alloy_gain_one_matches_brute_force_log_ratio(alloy_target):
    _, wrapped = build_pair(alloy_target)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
    x = random_states(alloy_target.d)
    t = torch.full((x.shape[0],), 0.3)
    added = wrapped(x, t) - wrapped.model(x, t)
    base_lp = alloy_target.log_prob(x)
    for i in range(alloy_target.d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        brute = alloy_target.log_prob(flipped) - base_lp
        flip_slot = (1 - ((x[:, i] + 1) / 2)).long()
        torch.testing.assert_close(
            added[torch.arange(x.shape[0]), i, flip_slot], brute, atol=1e-4, rtol=0
        )


def test_unconstrained_target_channel_is_the_bare_energy_log_ratio():
    """No penalty and no c* (the unconstrained rung): the channel must reduce
    to the base flip log-ratio instead of raising on the missing c*."""
    unconstrained = IsingTarget(D=D_SIDE, sigma=0.13, device="cpu")
    assert unconstrained.target_composition is None
    _, wrapped = build_pair(unconstrained)
    with torch.no_grad():
        wrapped.gain_constant.fill_(1.0)
    x = random_states(unconstrained.d)
    t = torch.full((x.shape[0],), 0.5)
    added = wrapped(x, t) - wrapped.model(x, t)
    base_lp = unconstrained.log_prob(x)
    for i in range(unconstrained.d):
        flipped = x.clone()
        flipped[:, i] = -flipped[:, i]
        flip_slot = (1 - ((x[:, i] + 1) / 2)).long()
        torch.testing.assert_close(
            added[torch.arange(x.shape[0]), i, flip_slot],
            unconstrained.log_prob(flipped) - base_lp,
            atol=1e-4,
            rtol=0,
        )
