import torch
import torch.nn.functional as F

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead, swap2
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import compute_xi_t_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _head_and_target(D=2, seed=42):
    torch.manual_seed(seed)
    tgt = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=0.5)
    backbone = LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    return DoublyHollowSwapHead(backbone), tgt


def _exact_slice(target, D, t_scalar):
    states = enumerate_states(D * D).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]
    t = torch.full((slice_states.shape[0],), float(t_scalar))
    log_p = target.log_p_tilde_t(slice_states, t)
    p_cond = torch.softmax(log_p, dim=0)
    dt_log_Z = (p_cond * target.dt_log_p_tilde_t(slice_states, t)).sum()
    return slice_states, p_cond, dt_log_Z


def test_xi_t_swap_is_unbiased_for_dt_log_Z():
    # E_{p_t^C}[ξ_t] = ∂_t log Z_t^C for ANY valid rate (design §3.3), checked
    # exactly on the 2×2 slice at random init — no training.
    head, tgt = _head_and_target()
    for t_scalar in (0.1, 0.5, 0.9):
        slice_states, p_cond, dt_log_Z = _exact_slice(tgt, 2, t_scalar)
        t = torch.full((slice_states.shape[0],), t_scalar)
        xi = compute_xi_t_swap(slice_states, t, head, tgt)
        assert torch.isclose((p_cond * xi).sum(), dt_log_Z, atol=1e-5)


def test_single_pass_reverse_rate_identity():
    # [-G(i,j|x)]_+ == [G(i,j|Swap2 x)]_+ bit-exact (P1 antisymmetry) — the
    # identity the single-pass residual/ξ_t rely on.
    head, tgt = _head_and_target()
    x = tgt.sample_base(6, device="cpu")
    t = torch.full((6,), 0.5)
    pairs = upper_tri_pairs(4, "cpu")
    G_x = gather_pair_scores(head(x, t), pairs)
    reverse_from_x = F.relu(-G_x)
    reverse_true = torch.stack([
        F.relu(gather_pair_scores(head(swap2(x, int(i), int(j)), t), pairs))[:, k]
        for k, (i, j) in enumerate(pairs.tolist())
    ], dim=1)
    assert torch.allclose(reverse_from_x, reverse_true, atol=1e-6)


def test_orientation_negative_control_index_not_spin():
    # Reading the TRANSPOSED entry G[:, j, i] at the swapped state (what a
    # spin-based representative would do) breaks the reverse-rate identity: the
    # head is label-asymmetric, so the two disagree by >> 1e-4.
    head, tgt = _head_and_target()
    x = tgt.sample_base(6, device="cpu")
    t = torch.full((6,), 0.5)
    pairs = upper_tri_pairs(4, "cpu")
    G_x = gather_pair_scores(head(x, t), pairs)
    reverse_from_x = F.relu(-G_x)
    transposed = torch.stack([
        F.relu(head(swap2(x, int(i), int(j)), t)[:, int(j), int(i)])
        for k, (i, j) in enumerate(pairs.tolist())
    ], dim=1)
    # active (opposite-spin) pairs must disagree by a clear margin
    active = (x[:, pairs[:, 0]] != x[:, pairs[:, 1]])
    assert (reverse_from_x - transposed).abs()[active].max() > 1e-4
