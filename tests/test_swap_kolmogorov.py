import torch

from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    c_t_offset_rms,
    loss_swap,
    residual_swap,
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
    """Exact p_t^C and ∂_t log Z_t^C on the fixed-N slice by enumeration."""
    states = enumerate_states(D * D).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]  # (M, d)
    t = torch.full((slice_states.shape[0],), float(t_scalar))
    log_p = target.log_p_tilde_t(slice_states, t)  # (M,)
    p_cond = torch.softmax(log_p, dim=0)  # p_t^C
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
    # for any head. Bit-close on the 6-state 2×2 slice.
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


# --- Δ, the c_t / loss-distribution offset -----------------------------------
#
# Objective identity these pin (see CTGridEMA's docstring for the derivation):
#
#     E_q[(ξ_t − c_t)^2] = Var_q[ξ_t] + Δ_t^2 ,   Δ_t ≜ E_q[ξ_t] − c_t
#
# Only Δ reaches the gradient from c_t's value, as 2·Δ·E_q[∇ξ]. The trainer
# estimates c_t on the fresh rollout but averages the loss over a replay
# buffer holding several past models, so Δ ≠ 0 by construction and had never
# been measured.


@torch.no_grad()
def test_residual_mean_recovers_c_t_offset():
    """−E_q[residual] is Δ. Offsetting c_t by a known k must move the
    residual mean by exactly −k, for any head — this is the contract the
    whole diagnostic rests on."""
    from discrete_flow_sampler.samplers.swap_ctmc import compute_xi_t_swap

    head, tgt = _head_and_target()
    x = tgt.sample_base(64, device="cpu")
    t = torch.full((64,), 0.5)
    xi_mean = compute_xi_t_swap(x, t, head, tgt).mean()
    for k in (0.0, 2.5, -1.25):
        residual = residual_swap(x, t, xi_mean + k, head, tgt)
        assert torch.isclose(residual.mean(), torch.as_tensor(-k), atol=1e-4)


@torch.no_grad()
def test_loss_swap_return_residual_matches_the_scalar_path():
    """The opt-in residual must be the sanitised one the loss squares, so the
    diagnostic and the gradient see the same numbers."""
    head, tgt = _head_and_target()
    x = tgt.sample_base(8, device="cpu")
    t = torch.full((8,), 0.5)
    c_t = torch.zeros(())

    scalar_only = loss_swap(x, t, c_t, head, tgt)
    loss, residual = loss_swap(x, t, c_t, head, tgt, return_residual=True)

    assert scalar_only.ndim == 0 and torch.equal(scalar_only, loss)
    assert residual.shape == (8,)
    assert torch.isfinite(residual).all()
    assert torch.isclose(residual.pow(2).mean(), loss)


def test_c_t_offset_rms_recovers_per_slot_offsets():
    deltas = torch.tensor([1.0, -3.0, 0.5])
    counts = torch.tensor([4.0, 4.0, 4.0])
    expected = deltas.pow(2).mean().sqrt().item()
    assert abs(c_t_offset_rms(deltas * counts, counts) - expected) < 1e-6


def test_c_t_offset_rms_does_not_cancel_across_slots():
    """The diagnostic is a per-slot RMS, not a signed batch mean: the
    objective pays Δ_t^2 in every slot, so equal-and-opposite offsets cost
    2.0 each while their signed average reads exactly 0.0 — a signed mean
    would report 'no mismatch' on a maximally mismatched run."""
    deltas = torch.tensor([2.0, -2.0])
    counts = torch.tensor([8.0, 8.0])
    assert (deltas * counts).sum().item() == 0.0
    assert abs(c_t_offset_rms(deltas * counts, counts) - 2.0) < 1e-6


def test_c_t_offset_rms_ignores_unvisited_slots():
    """A slot no inner batch happened to draw contributes nothing, rather
    than a spurious zero that would dilute the RMS toward 'healthy'."""
    deltas = torch.tensor([3.0, 0.0])
    counts = torch.tensor([5.0, 0.0])
    assert abs(c_t_offset_rms(deltas * counts, counts) - 3.0) < 1e-6


def test_c_t_offset_rms_is_nan_before_any_sample():
    import math

    assert math.isnan(c_t_offset_rms(torch.zeros(3), torch.zeros(3)))
