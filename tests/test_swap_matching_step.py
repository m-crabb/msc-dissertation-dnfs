"""Matching-based multi-event swap Euler step.

The one-event step fires ≤1 swap/step, so a trajectory needs O(d²) steps at the
critical coupling (Λ ∝ d²; scouting 2026-07-05). The matching step fires a
vertex-disjoint set of swaps per step — disjoint swaps commute, so tau-leaping
is exact within a matching — restoring O(d) trajectory length.

Pinned here:
  - the selected set is a valid matching (vertex-disjoint) and a subset of the
    proposals (so applied swaps commute and none is invented);
  - conflicts are resolved by priority (higher priority wins its vertices);
  - composition (n_plus) is preserved bit-exactly;
  - the step genuinely fires >1 swap (it is not the one-event step in disguise);
  - marginal-rate recovery: in the low-conflict limit the per-pair firing
    frequency tracks the proposal probability, the property the IS weight needs.
"""

import torch

from discrete_flow_sampler.samplers._swap_neighbours import upper_tri_pairs
from discrete_flow_sampler.samplers.swap_ctmc import (
    _euler_step_swap_matching,
    _vertex_disjoint_matching,
    sample_swap_ctmc,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


class _ConstRateHead:
    """Fake swap head: every pair gets constant score `value` (relu-positive).

    Lets the step's matching mechanics be exercised at a controlled firing rate
    without a trained network. Returns G of shape (B, d, d).
    """

    def __init__(self, d, value):
        self.d = d
        self.value = value

    def __call__(self, x, t):
        return torch.full((x.shape[0], self.d, self.d), self.value, dtype=x.dtype)


def _rows_are_matchings(accepted, pairs):
    """True iff every batch row's accepted pairs are vertex-disjoint."""
    for b in range(accepted.shape[0]):
        sites = pairs[accepted[b]].reshape(-1)
        if sites.numel() != torch.unique(sites).numel():
            return False
    return True


def test_matching_is_vertex_disjoint_subset_of_proposals():
    torch.manual_seed(0)
    d = 16
    pairs = upper_tri_pairs(d, "cpu")
    proposed = torch.rand(32, pairs.shape[0]) < 0.3  # dense -> many conflicts
    priority = torch.rand(32, pairs.shape[0])
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)

    assert torch.all(~accepted | proposed)  # accepted ⊆ proposed
    assert _rows_are_matchings(accepted, pairs)


def test_matching_keeps_already_disjoint_proposals():
    d = 16
    pairs = upper_tri_pairs(d, "cpu")
    proposed = torch.zeros(1, pairs.shape[0], dtype=torch.bool)
    # pairs (0,1), (2,3), (4,5): vertex-disjoint, so nothing should be dropped.
    disjoint_idx = [pairs.tolist().index([s, s + 1]) for s in (0, 2, 4)]
    proposed[0, disjoint_idx] = True
    accepted = _vertex_disjoint_matching(
        proposed, torch.rand(1, pairs.shape[0]), pairs, d
    )
    assert torch.equal(accepted, proposed)


def test_matching_resolves_conflict_by_priority():
    d = 4
    pairs = upper_tri_pairs(d, "cpu")  # [[0,1],[0,2],[0,3],[1,2],...]
    idx_01, idx_12 = pairs.tolist().index([0, 1]), pairs.tolist().index([1, 2])
    proposed = torch.zeros(1, pairs.shape[0], dtype=torch.bool)
    proposed[0, [idx_01, idx_12]] = True  # share vertex 1
    priority = torch.zeros(1, pairs.shape[0])
    priority[0, idx_01], priority[0, idx_12] = 0.9, 0.1
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    assert accepted[0, idx_01] and not accepted[0, idx_12]

    priority[0, idx_01], priority[0, idx_12] = 0.1, 0.9  # flip the winner
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    assert accepted[0, idx_12] and not accepted[0, idx_01]


def test_matching_step_preserves_composition():
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.223, target_composition=0.5)
    head = _ConstRateHead(target.d, value=2.0)
    state = target.sample_base(32, device="cpu")
    t = torch.full((32,), 0.5)
    for _ in range(20):
        state, _ = _euler_step_swap_matching(head, state, t, torch.tensor(0.1))
        target.assert_on_manifold(state)


def test_matching_step_fires_multiple_swaps():
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.223, target_composition=0.5)
    head = _ConstRateHead(target.d, value=5.0)  # high rate * dt -> many propose
    state = target.sample_base(64, device="cpu")
    t = torch.full((64,), 0.5)
    new_state, _ = _euler_step_swap_matching(head, state, t, torch.tensor(0.5))
    changed_sites = (new_state != state).sum(dim=-1)  # 2 per realised swap
    assert changed_sites.max() > 2  # some row fired >1 swap


def test_matching_low_conflict_recovers_proposal_rate():
    # As dt -> 0 the proposal density -> 0 and conflicts vanish, so the matching
    # drops almost nothing and per-pair firing frequency -> proposal probability.
    # This is the marginal-rate property the IS weight relies on (ξ_t assumes the
    # learned rate). At the 0.1-events/site operating cap retention is < 1 by
    # design (measured separately, monitored via the clip diagnostic).
    torch.manual_seed(0)
    d = 16
    pairs = upper_tri_pairs(d, "cpu")
    p = 0.0005  # ~0.06 proposals/row: the low-conflict (small-dt) limit
    trials = 20000
    proposed = torch.rand(trials, pairs.shape[0]) < p
    priority = torch.rand(trials, pairs.shape[0])
    accepted = _vertex_disjoint_matching(proposed, priority, pairs, d)
    retention = accepted.float().sum() / proposed.float().sum()
    assert retention > 0.97


def test_sample_swap_ctmc_multi_event_stays_on_manifold():
    torch.manual_seed(0)
    target = FixedCompositionIsingTarget(D=4, sigma=0.223, target_composition=0.5)
    head = _ConstRateHead(target.d, value=2.0)
    x0 = target.sample_base(16, device="cpu")
    ts = torch.linspace(0.0, 1.0, 20)
    x_final = sample_swap_ctmc(head, x0, ts, multi_event=True)
    assert x_final.shape == (16, target.d)
    target.assert_on_manifold(x_final)
