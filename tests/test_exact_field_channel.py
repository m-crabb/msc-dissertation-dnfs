"""Falsification suite for the exact-field channel.

The channel adds gain(t) * sigma * Delta_ij to any swap head's score matrix,
where sigma * Delta_ij is the closed-form Kawasaki energy change of swapping
i and j (the equilibrium log-ratio at t=1). What "correct" means:

1. gain initialised at zero => the wrapped head is BIT-identical to its base
   (every archived cell is untouched when the flag is off or at init);
2. the channel matrix equals the brute-force log p(swap2(x,i,j)) - log p(x)
   for EVERY pair i < j (the loss reads i < j only; the energy change is
   label-symmetric and the mirror supplies the heads' index convention),
   adjacent ones included (the i-j bond is swap-invariant and
   must not be counted -- the failure mode the A_ij term guards);
3. the matrix is exactly antisymmetric with a zero diagonal, and zero on
   same-spin pairs;
4. target.set_sigma propagates (the curriculum moves sigma mid-run);
5. state_dict round-trips through the wrapper.
"""
import torch

from discrete_flow_sampler.constraints.exact_field_channel import ExactFieldSwapHead
from discrete_flow_sampler.constraints.interval_swap_head import IntervalSwapHead
from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget
from discrete_flow_sampler.targets.cluster_expansion import (
    BinaryExpansionSpec, FixedCompositionClusterExpansionTarget)


def _setup(D=4, sigma=0.223, seed=0):
    torch.manual_seed(seed)
    target = FixedCompositionIsingTarget(D=D, sigma=sigma, target_composition=0.5)
    backbone = LeTFRateMatrix(d=D * D, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    base = IntervalSwapHead(backbone, pair_offsets=(1, D))
    wrapped = ExactFieldSwapHead(base, target)
    x = target.sample_base(3, "cpu")
    t = torch.tensor([0.0, 0.4, 1.0])
    return target, base, wrapped, x, t


def test_zero_gain_is_bit_identical_to_base():
    _, base, wrapped, x, t = _setup()
    assert torch.equal(wrapped(x, t), base(x, t))


def test_channel_matches_brute_force_energy_change_on_every_pair():
    target, _, wrapped, x, _ = _setup()
    d = x.shape[1]
    channel = wrapped.exact_field(x)                       # (B, d, d), sigma*Delta
    log_p = target.log_prob(x)
    for i in range(d):
        for j in range(i + 1, d):
            expected = target.log_prob(swap2(x, i, j)) - log_p
            torch.testing.assert_close(channel[:, i, j], expected, atol=1e-5, rtol=0)


def test_channel_is_antisymmetric_zero_diagonal_and_zero_on_same_spin():
    _, _, wrapped, x, _ = _setup()
    channel = wrapped.exact_field(x)
    # atol=0: exact up to the sign of zero (torch.equal would reject -0.0).
    torch.testing.assert_close(channel, -channel.transpose(1, 2), atol=0, rtol=0)
    torch.testing.assert_close(
        channel.diagonal(dim1=1, dim2=2), torch.zeros_like(channel[:, :, 0]), atol=0, rtol=0
    )
    same_spin = x.unsqueeze(2) == x.unsqueeze(1)
    torch.testing.assert_close(
        channel[same_spin], torch.zeros_like(channel[same_spin]), atol=0, rtol=0
    )


def test_gain_is_applied_with_its_time_dependence_and_sigma_tracks_target():
    target, base, wrapped, x, t = _setup()
    with torch.no_grad():
        wrapped.gain_constant.fill_(0.5)
        wrapped.gain_slope.fill_(2.0)
    expected = base(x, t) + (0.5 + 2.0 * t)[:, None, None] * wrapped.exact_field(x)
    torch.testing.assert_close(wrapped(x, t), expected)
    before = wrapped.exact_field(x)
    target.set_sigma(2 * target.sigma)
    torch.testing.assert_close(wrapped.exact_field(x), 2 * before)


def test_state_dict_round_trip_and_backbone_passthrough():
    target, base, wrapped, x, t = _setup()
    with torch.no_grad():
        wrapped.gain_constant.fill_(0.3)
    fresh = ExactFieldSwapHead(
        IntervalSwapHead(LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2),
                         pair_offsets=(1, 4)),
        target,
    )
    fresh.load_state_dict(wrapped.state_dict())
    torch.testing.assert_close(fresh(x, t), wrapped(x, t))
    assert fresh.backbone is fresh.head.backbone


def test_channel_matches_brute_force_on_cluster_expansion():
    """The Cu-Au 16-site slice at 500 K: the channel must read the
    expansion's own swap energy change, not the Ising quadratic form."""
    torch.manual_seed(0)
    spec = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
    target = FixedCompositionClusterExpansionTarget(
        spec, beta=1.0 / (8.617333262e-5 * 500.0), target_composition=0.5)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    wrapped = ExactFieldSwapHead(IntervalSwapHead(backbone, pair_offsets=(1, 4)), target)
    x = target.sample_base(3, "cpu")
    channel = wrapped.exact_field(x)
    log_p = target.log_prob(x)
    for i in range(16):
        for j in range(i + 1, 16):
            expected = target.log_prob(swap2(x, i, j)) - log_p
            torch.testing.assert_close(channel[:, i, j], expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(channel, -channel.transpose(1, 2), atol=0, rtol=0)
