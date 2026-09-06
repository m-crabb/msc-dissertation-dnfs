import pytest
import torch
import torch.nn.functional as F

from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
    swap2,
)
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_ctmc import (
    _euler_step_swap,
    compute_c_t_grid_swap,
    compute_xi_t_swap,
    sample_swap_ctmc,
)
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


@torch.no_grad()
def test_xi_t_swap_is_unbiased_for_dt_log_Z():
    # E_{p_t^C}[ξ_t] = ∂_t log Z_t^C for ANY valid rate, checked
    # exactly on the 2×2 slice at random init — no training.
    head, tgt = _head_and_target()
    for t_scalar in (0.1, 0.5, 0.9):
        slice_states, p_cond, dt_log_Z = _exact_slice(tgt, 2, t_scalar)
        t = torch.full((slice_states.shape[0],), t_scalar)
        xi = compute_xi_t_swap(slice_states, t, head, tgt)
        assert torch.isclose((p_cond * xi).sum(), dt_log_Z, atol=1e-5)


@torch.no_grad()
def test_single_pass_reverse_rate_identity():
    # [-G(i,j|x)]_+ == [G(i,j|Swap2 x)]_+ bit-exact (P1 antisymmetry) — the
    # identity the single-pass residual/ξ_t rely on.
    head, tgt = _head_and_target()
    x = tgt.sample_base(6, device="cpu")
    t = torch.full((6,), 0.5)
    pairs = upper_tri_pairs(4, "cpu")
    G_x = gather_pair_scores(head(x, t), pairs)
    reverse_from_x = F.relu(-G_x)
    reverse_true = torch.stack(
        [
            F.relu(gather_pair_scores(head(swap2(x, int(i), int(j)), t), pairs))[:, k]
            for k, (i, j) in enumerate(pairs.tolist())
        ],
        dim=1,
    )
    assert torch.allclose(reverse_from_x, reverse_true, atol=1e-6)


@torch.no_grad()
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
    transposed = torch.stack(
        [
            F.relu(head(swap2(x, int(i), int(j)), t)[:, int(j), int(i)])
            for k, (i, j) in enumerate(pairs.tolist())
        ],
        dim=1,
    )
    # active (opposite-spin) pairs must disagree by a clear margin
    active = x[:, pairs[:, 0]] != x[:, pairs[:, 1]]
    assert (reverse_from_x - transposed).abs()[active].max() > 1e-4


@torch.no_grad()
def test_euler_step_preserves_composition():
    head, tgt = _head_and_target(D=4)  # d=16, N_A=8
    state = tgt.sample_base(32, device="cpu")
    t = torch.full((32,), 0.5)
    for _ in range(50):
        state, _ = _euler_step_swap(head, state, t, torch.tensor(0.05))
        tgt.assert_on_manifold(state)  # bit-exact composition invariance


@torch.no_grad()
def test_sample_swap_ctmc_stays_on_manifold_and_shapes():
    head, tgt = _head_and_target(D=4)
    x0 = tgt.sample_base(16, device="cpu")
    ts = torch.linspace(0.0, 1.0, 20)
    x_final = sample_swap_ctmc(head, x0, ts, target=tgt)
    assert x_final.shape == (16, 16)
    tgt.assert_on_manifold(x_final)
    traj = sample_swap_ctmc(head, x0, ts, return_all_states=True)
    assert traj.shape == (20, 16, 16)


@torch.no_grad()
def test_sample_swap_ctmc_log_weights_finite_and_contract():
    head, tgt = _head_and_target(D=4)
    x0 = tgt.sample_base(16, device="cpu")
    ts = torch.linspace(0.0, 1.0, 20)
    x_final, log_w = sample_swap_ctmc(head, x0, ts, return_log_weights=True, target=tgt)
    assert log_w.shape == (16,) and torch.isfinite(log_w).all()
    with pytest.raises(ValueError):
        sample_swap_ctmc(head, x0, ts, return_log_weights=True)  # no target
    with pytest.raises(ValueError):
        sample_swap_ctmc(
            head,
            x0,
            ts,
            return_log_weights=True,
            return_all_states=True,
            target=tgt,
        )


@torch.no_grad()
def test_compute_c_t_grid_swap_modes():
    head, tgt = _head_and_target(D=4)
    x0 = tgt.sample_base(8, device="cpu")
    ts = torch.linspace(0.0, 1.0, 6)
    traj = sample_swap_ctmc(head, x0, ts, return_all_states=True)  # (6,8,16)
    for mode in ("naive_mc", "control_variate"):
        c_t, integrand = compute_c_t_grid_swap(ts, traj, tgt, head, mode=mode)
        assert c_t.shape == (6,) and integrand.shape == (6, 8)
        assert torch.isfinite(c_t).all()


@torch.no_grad()
def test_xi_t_swap_unbiased_d16_slice():
    # Unbiasedness at the GATE's dimension: d=16, N_A=8 (12,870 states), random
    # init, no training. mask_one head is numerically equal to doubly-hollow
    # (test_brute_force_matches_mask_one_d16_batch_all_pairs, d=16, batch, all
    # i<j pairs oracle in test_swap_readout.py: observed bit-exact, atol=1e-5
    # fallback), so this validates the same estimator at 120 pairs. sigma=0.1
    # -> clamp inert.
    torch.manual_seed(42)
    tgt = FixedCompositionIsingTarget(D=4, sigma=0.1, target_composition=0.5)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2)
    head = LeTFMaskOneSwapHead(backbone)
    for t_scalar in (0.1, 0.5, 0.9):
        slice_states, p_cond, dt_log_Z = _exact_slice(tgt, 4, t_scalar)
        t = torch.full((slice_states.shape[0],), t_scalar)
        xi = compute_xi_t_swap(slice_states, t, head, tgt)
        assert torch.isclose((p_cond * xi).sum(), dt_log_Z, atol=1e-4)
