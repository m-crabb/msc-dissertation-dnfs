import torch

from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.samplers._swap_neighbours import (
    _log_p_tilde_at_swap_neighbours,
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def test_upper_tri_pairs_are_i_lt_j_and_complete():
    pairs = upper_tri_pairs(4, "cpu")
    assert pairs.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert torch.all(pairs[:, 0] < pairs[:, 1])


def test_gather_pair_scores_picks_upper_triangle():
    G = torch.arange(2 * 3 * 3).reshape(2, 3, 3).float()
    pairs = upper_tri_pairs(3, "cpu")  # [[0,1],[0,2],[1,2]]
    got = gather_pair_scores(G, pairs)  # (2, 3)
    assert torch.equal(got[0], torch.tensor([G[0, 0, 1], G[0, 0, 2], G[0, 1, 2]]))


def test_swap_neighbours_match_reference_loop():
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)  # d=16
    x = tgt.sample_base(5, device="cpu")
    t = torch.full((5,), 0.4)
    pairs = upper_tri_pairs(16, "cpu")
    got = _log_p_tilde_at_swap_neighbours(x, t, tgt)  # (5, n_pairs)
    # reference: build swap2 per pair and evaluate directly
    ref = torch.stack(
        [tgt.log_p_tilde_t(swap2(x, int(i), int(j)), t) for (i, j) in pairs.tolist()],
        dim=1,
    )
    assert torch.allclose(got, ref, atol=1e-6)


def test_same_spin_pair_neighbour_equals_x():
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.3, target_composition=0.5)
    x = tgt.sample_base(4, device="cpu")
    t = torch.full((4,), 0.5)
    got = _log_p_tilde_at_swap_neighbours(x, t, tgt)
    log_p_x = tgt.log_p_tilde_t(x, t)
    pairs = upper_tri_pairs(16, "cpu")
    for row in range(4):
        for k, (i, j) in enumerate(pairs.tolist()):
            if x[row, i] == x[row, j]:  # same-spin swap = identity
                assert torch.allclose(got[row, k], log_p_x[row], atol=1e-6)
