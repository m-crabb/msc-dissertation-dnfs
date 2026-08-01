import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap, residual_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _head_and_target(D=2, seed=42):
    torch.manual_seed(seed)
    tgt = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=0.5)
    backbone = LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    return DoublyHollowSwapHead(backbone), tgt


def _exact_slice(target, D, t_scalar):
    """Exact p_t^C and ∂_t log Z_t^C on the fixed-N slice by enumeration."""
    states = enumerate_states(D * D).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]        # (M, d)
    t = torch.full((slice_states.shape[0],), float(t_scalar))
    log_p = target.log_p_tilde_t(slice_states, t)                # (M,)
    p_cond = torch.softmax(log_p, dim=0)                         # p_t^C
    dt_log_Z = (p_cond * target.dt_log_p_tilde_t(slice_states, t)).sum()
    return slice_states, p_cond, dt_log_Z


@torch.no_grad()
def test_residual_swap_shape_and_finite():
    head, tgt = _head_and_target()
    x = tgt.sample_base(8, device="cpu")
    t = torch.full((8,), 0.5)
    r = residual_swap(x, t, torch.zeros(()), head, tgt)
    assert r.shape == (8,)
    assert torch.isfinite(r).all()


@torch.no_grad()
def test_residual_swap_zero_mean_at_exact_dt_log_Z():
    # E_{p_t^C}[δ_t] = ∂_t log Z_t^C − dt_log_Zt = 0 when dt_log_Zt is exact,
    # for ANY head. Bit-close on the 6-state 2×2 slice.
    head, tgt = _head_and_target()
    for t_scalar in (0.1, 0.5, 0.9):
        slice_states, p_cond, dt_log_Z = _exact_slice(tgt, 2, t_scalar)
        t = torch.full((slice_states.shape[0],), t_scalar)
        r = residual_swap(slice_states, t, dt_log_Z, head, tgt)
        assert torch.isclose((p_cond * r).sum(), torch.zeros(()), atol=1e-5)


@torch.no_grad()
def test_loss_swap_is_nonneg_scalar():
    head, tgt = _head_and_target()
    x = tgt.sample_base(8, device="cpu")
    t = torch.full((8,), 0.5)
    val = loss_swap(x, t, torch.zeros(()), head, tgt)
    assert val.ndim == 0 and val.item() >= 0.0
