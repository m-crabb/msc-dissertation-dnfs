"""Checkpoint contract for the shared EMA instrument.

The failure this pins: a preemption resume that loses the shadow or the
update counter silently degrades the instrument — a re-seeded shadow sits
at the resume weights (the init-contamination failure in miniature) and a
reset counter restarts the warmup schedule, changing every subsequent
effective decay. Round-tripping through state_dict must reproduce the
continuation bit-exactly.
"""

import torch

from discrete_flow_sampler.ema import ExponentialMovingAverage


def _stepped_ema(n_updates: int) -> tuple[torch.nn.Parameter, object]:
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(5))
    ema = ExponentialMovingAverage([parameter], decay=0.9999, warmup=True)
    for step in range(n_updates):
        with torch.no_grad():
            parameter.add_(0.1 * (step + 1))
        ema.update()
    return parameter, ema


def test_state_dict_round_trip_continues_bit_exactly():
    parameter, ema = _stepped_ema(n_updates=7)

    restored = ExponentialMovingAverage([parameter], decay=0.9999, warmup=True)
    restored.load_state_dict(ema.state_dict())
    assert restored.updates == ema.updates

    # Continue both for a few more updates: identical trajectories require
    # both the shadow values and the warmup counter to have survived.
    for _ in range(5):
        with torch.no_grad():
            parameter.mul_(1.01)
        ema.update()
        restored.update()
    assert torch.equal(ema.shadow[0], restored.shadow[0])


def test_lost_counter_would_diverge():
    """The counterfactual the round trip guards against: same shadow but a
    reset counter yields a different effective decay, so the very next
    update differs — this is why `updates` travels in the checkpoint."""
    parameter, ema = _stepped_ema(n_updates=7)

    reset_counter = ExponentialMovingAverage([parameter], decay=0.9999, warmup=True)
    reset_counter.load_state_dict({"updates": 0, "shadow": ema.state_dict()["shadow"]})
    with torch.no_grad():
        parameter.mul_(1.5)
    ema.update()
    reset_counter.update()
    assert not torch.equal(ema.shadow[0], reset_counter.shadow[0])
