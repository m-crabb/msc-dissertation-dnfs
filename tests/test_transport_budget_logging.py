"""Transport-budget instrumentation for the swap-CTMC sampler.

A trajectory's ESS says nothing about whether the sampler MOVED: a head that
fires (almost) no swaps still returns finite IS weights, so an eval can look
healthy while its draw never left the base distribution's neighbourhood.
These tests pin the measured jump budget:

  - BOTH Euler step kinds (one-event categorical and vertex-disjoint matching)
    count proposed / accepted / state-changing swap events into the optional
    stats dict, in the same accumulate-into-dict style;
  - same-spin pairs matter: Swap2(x, i, j) = x when x_i == x_j, so `accepted`
    alone OVERSTATES productive transport — `accepted_state_changing` counts
    only swaps whose endpoint spins differ at fire time;
  - the frozen final eval integrates the counters along the trajectory and
    writes jumps_per_site_{proposed,accepted,state_changing} into the
    metrics.json every eval-variant dir already gets.
"""
import json
import math
from pathlib import Path

import torch
from experiments.constrained_hard_03.configs import HardStageCfg
from experiments.constrained_hard_03.run import build_target_and_head, final_eval
from experiments.dnfs_baseline_01.configs import (
    CTMCCfg,
    EvalCfg,
    IsingCfg,
    ModelCfg,
    TrainCfg,
)

from discrete_flow_sampler.samplers.swap_ctmc import (
    _euler_step_swap,
    _euler_step_swap_matching,
    sample_swap_ctmc,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

TRANSPORT_COUNTER_KEYS = {
    "proposed", "accepted", "accepted_state_changing", "state_steps",
}


class _ConstRateHead:
    """Fake swap head: every pair gets the same positive score.

    Exercises the counters at a controlled, nonzero firing rate without a
    trained network. Returns G of shape (B, d, d) like the real heads.
    """

    def __init__(self, d, value):
        self.d = d
        self.value = value

    def __call__(self, x, t):
        return torch.full(
            (x.shape[0], self.d, self.d), self.value, dtype=x.dtype
        )


class _SinglePairHead:
    """Fake swap head: positive score ONLY on pair (0, 1), negative elsewhere.

    Forces exactly that pair to fire (score * dt >= 1 saturates its firing
    probability while relu kills every other pair), so a test can hand-pick
    whether the firing pair is same-spin (state no-op) or mixed-spin.
    """

    def __init__(self, d, value=1000.0):
        self.d = d
        self.value = value

    def __call__(self, x, t):
        scores = torch.full(
            (x.shape[0], self.d, self.d), -1000.0, dtype=x.dtype
        )
        scores[:, 0, 1] = self.value
        return scores


def _int_stat(stats, key):
    return int(stats[key])


def test_multi_event_trajectory_counts_ordered_transport_budget():
    """(a) Matching-step trajectory: counters present, correctly ordered
    (state_changing <= accepted <= proposed), all positive for a nonzero-rate
    head, and state_steps = batch x n_euler_steps."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.1, target_composition=0.5
    )
    head = _ConstRateHead(target.d, value=2.0)
    batch_size, n_euler_steps = 8, 8
    ts = torch.linspace(0.0, 1.0, n_euler_steps + 1)
    x_initial = target.sample_base(batch_size, device="cpu")

    stats = {}
    x_final = sample_swap_ctmc(
        head, x_initial, ts, multi_event=True, matching_stats=stats
    )

    assert TRANSPORT_COUNTER_KEYS <= set(stats)
    proposed = _int_stat(stats, "proposed")
    accepted = _int_stat(stats, "accepted")
    state_changing = _int_stat(stats, "accepted_state_changing")
    assert 0 < state_changing <= accepted <= proposed
    assert _int_stat(stats, "state_steps") == batch_size * n_euler_steps
    target.assert_on_manifold(x_final)


def test_one_event_trajectory_produces_same_counter_keys():
    """(b) The one-event step now feeds the SAME counter keys through the same
    optional stats argument. Its single categorical draws the firing pair
    directly — there is no thinning/rejection stage — so proposed == accepted
    by construction."""
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.1, target_composition=0.5
    )
    head = _ConstRateHead(target.d, value=2.0)
    batch_size, n_euler_steps = 8, 8
    ts = torch.linspace(0.0, 1.0, n_euler_steps + 1)
    x_initial = target.sample_base(batch_size, device="cpu")

    stats = {}
    x_final = sample_swap_ctmc(
        head, x_initial, ts, multi_event=False, matching_stats=stats
    )

    assert TRANSPORT_COUNTER_KEYS <= set(stats)
    proposed = _int_stat(stats, "proposed")
    accepted = _int_stat(stats, "accepted")
    state_changing = _int_stat(stats, "accepted_state_changing")
    assert accepted == proposed
    assert 0 < state_changing <= accepted
    assert _int_stat(stats, "state_steps") == batch_size * n_euler_steps
    target.assert_on_manifold(x_final)


def _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4, use_matching_step=False):
    return HardStageCfg(
        name="tiny_hard_transport",
        ising=IsingCfg(D=4, sigma=0.1, bias=0.0, target_composition=0.5),
        train=TrainCfg(n_steps=2, batch_size=4, inner_steps_per_outer=2, seed=0),
        ctmc=CTMCCfg(n_euler_steps=8, use_matching_step=use_matching_step),
        eval=EvalCfg(
            eval_every=2,
            n_eval_samples=n_eval_samples,
            eval_sample_chunk=eval_sample_chunk,
        ),
        model=ModelCfg(kind="letf", hidden_dim=16, n_layers=2, n_heads=2,
                       vocab_size=2),
        estimator="control_variate",
        head_kind="mask_one",
        wandb_project="test",
    )


def test_final_eval_writes_integrated_jumps_per_site_fields(tmp_path):
    """(c) The chunked final eval accumulates transport stats across its draw
    slices and writes the three integrated jumps_per_site_* fields into the
    metrics.json it already writes. One-event canonical cell: proposed and
    accepted coincide (no thinning stage), and state_changing never exceeds
    accepted."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_eval_samples=10, eval_sample_chunk=4)
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path))

    saved = json.loads((tmp_path / "eval" / "metrics.json").read_text())
    for key in (
        "jumps_per_site_proposed",
        "jumps_per_site_accepted",
        "jumps_per_site_state_changing",
    ):
        assert key in saved and key in metrics
        assert math.isfinite(saved[key]) and saved[key] >= 0.0
    assert (
        saved["jumps_per_site_state_changing"]
        <= saved["jumps_per_site_accepted"]
        <= saved["jumps_per_site_proposed"]
    )
    # One-event step: a fired event is both the proposal and the acceptance.
    assert (
        saved["jumps_per_site_accepted"] == saved["jumps_per_site_proposed"]
    )


def test_final_eval_matching_step_jumps_fields_land_too(tmp_path):
    """(c cont.) The matching-canonical route through the same final_eval gets
    the fields as well (this is the path eval/ on a 16x16-rung-style cell and
    every eval-variant dir shares)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(
        n_eval_samples=10, eval_sample_chunk=4, use_matching_step=True
    )
    target, head = build_target_and_head(cfg, "cpu")

    metrics = final_eval(head, target, cfg, Path(tmp_path))

    assert metrics["multi_event"] is True
    assert (
        metrics["jumps_per_site_state_changing"]
        <= metrics["jumps_per_site_accepted"]
        <= metrics["jumps_per_site_proposed"]
    )


def test_same_spin_pair_fires_as_noop_in_matching_step():
    """(d) A same-spin pair's swap is a state no-op: on x = [+1, +1, -1, -1]
    with only pair (0, 1) able to fire, `accepted` increments but
    `accepted_state_changing` does not, and the state is unchanged. Flipping
    the state so the pair is mixed-spin turns the same fired event into a
    state-changing one — the counter tracks the spins, not the firing."""
    torch.manual_seed(0)
    d = 4
    same_spin_state = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    head = _SinglePairHead(d)
    step_dt = torch.tensor(0.01)  # 1000 * 0.01 clamps firing prob to 1
    t = torch.zeros(1)

    stats = {}
    new_state, _ = _euler_step_swap_matching(
        head, same_spin_state, t, step_dt, stats=stats
    )
    assert _int_stat(stats, "accepted") == 1
    assert _int_stat(stats, "accepted_state_changing") == 0
    assert torch.equal(new_state, same_spin_state)

    mixed_spin_state = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    stats = {}
    new_state, _ = _euler_step_swap_matching(
        head, mixed_spin_state, t, step_dt, stats=stats
    )
    assert _int_stat(stats, "accepted") == 1
    assert _int_stat(stats, "accepted_state_changing") == 1
    assert not torch.equal(new_state, mixed_spin_state)


def test_same_spin_pair_fires_as_noop_in_one_event_step():
    """(d cont.) Same no-op discipline in the one-event step: the saturated
    pair (0, 1) always wins the categorical, and on a same-spin state it
    counts as accepted but not state-changing."""
    torch.manual_seed(0)
    same_spin_state = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    head = _SinglePairHead(4)
    step_dt = torch.tensor(0.01)
    t = torch.zeros(1)

    stats = {}
    new_state, _ = _euler_step_swap(
        head, same_spin_state, t, step_dt, stats=stats
    )
    assert _int_stat(stats, "proposed") == 1
    assert _int_stat(stats, "accepted") == 1
    assert _int_stat(stats, "accepted_state_changing") == 0
    assert torch.equal(new_state, same_spin_state)


def test_stats_argument_stays_optional_and_inert():
    """(4) None = today's behaviour: both step kinds run without a stats dict
    and the stats accumulation consumes no RNG, so a stats-on and a stats-off
    trajectory from the same seed are byte-identical."""
    target = FixedCompositionIsingTarget(
        D=4, sigma=0.1, target_composition=0.5
    )
    head = _ConstRateHead(target.d, value=2.0)
    ts = torch.linspace(0.0, 1.0, 9)
    for multi_event in (False, True):
        torch.manual_seed(3)
        x_initial = target.sample_base(8, device="cpu")
        torch.manual_seed(4)
        x_without_stats = sample_swap_ctmc(
            head, x_initial, ts, multi_event=multi_event
        )
        torch.manual_seed(4)
        x_with_stats = sample_swap_ctmc(
            head, x_initial, ts, multi_event=multi_event, matching_stats={}
        )
        assert torch.equal(x_without_stats, x_with_stats)
