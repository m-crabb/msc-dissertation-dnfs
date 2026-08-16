"""Gradient-exactness pins for the micro-batched loss backward (2026-08-16).

The two 16x16 screen arms that OOM an A100-80GB (`H2_d256_scr5k_ma_h128_lr03`,
`H2_d256_scr5k_mo`) fit by slicing the ONE inner-step backward over batch rows.
That is admissible only because it is not a recipe change: the swap loss is a
per-row mean (residual_swap is row-wise, c_t is gathered per row from a grid
frozen for the cycle, nan_to_num is row-wise), so

    mean_N[r^2] = sum_slices (n_k / N) * mean_slice_k[r^2]

decomposes exactly, and autograd's linearity carries the identity to the
gradient: backwarding each slice's weighted loss accumulates the SAME total
gradient the single backward produces, up to float summation order. Clipping
and the optimiser step then see identical inputs. These tests pin that
identity — the proof the arm stays the twin its pin declares — and pin the
None path as literally the archived single-backward so the queued fleet is
untouched by deploying the lever switched off.
"""
import torch

from discrete_flow_sampler.constraints.masked_attention_swap_head import (
    MaskedAttentionSwapHead,
)
from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    loss_swap,
    loss_swap_backward_microbatched,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _backbone(D, seed):
    torch.manual_seed(seed)
    return LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )


def _mask_one_head_and_target(D=4, seed=42):
    tgt = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=0.5)
    # anchor_chunk_size exercised too: the d256 arm runs chunked forwards.
    head = LeTFMaskOneSwapHead(_backbone(D, seed), anchor_chunk_size=5)
    return head, tgt


def _masked_attention_head_and_target(D=4, seed=42):
    tgt = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=0.5)
    head = MaskedAttentionSwapHead(_backbone(D, seed), pair_offsets=(1, D))
    return head, tgt


def _grads(head):
    return [
        p.grad.clone() if p.grad is not None else None
        for p in head.parameters()
    ]


def _batch(tgt, batch_size, seed=7):
    torch.manual_seed(seed)
    x = tgt.sample_base(batch_size, device="cpu")
    t = torch.rand(batch_size)
    # Per-row c_t, the trainer's actual call shape (gathered from the grid).
    c_t = torch.randn(batch_size) * 0.1
    return x, t, c_t


def _full_batch_reference(head, tgt, x, t, c_t):
    head.zero_grad()
    loss, residual = loss_swap(x, t, c_t, head, tgt, return_residual=True)
    loss.backward()
    return loss.detach(), residual.detach(), _grads(head)


def _assert_parity(head_factory, microbatch_size, batch_size):
    head, tgt = head_factory()
    x, t, c_t = _batch(tgt, batch_size)
    ref_loss, ref_residual, ref_grads = _full_batch_reference(
        head, tgt, x, t, c_t
    )

    head.zero_grad()
    loss, residual = loss_swap_backward_microbatched(
        x, t, c_t, head, tgt, microbatch_size=microbatch_size
    )

    assert not loss.requires_grad and not residual.requires_grad
    assert torch.isclose(loss, ref_loss, rtol=1e-5, atol=1e-7)
    assert torch.allclose(residual, ref_residual, rtol=1e-5, atol=1e-7)
    for got, want in zip(_grads(head), ref_grads, strict=True):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.allclose(got, want, rtol=1e-4, atol=1e-6)


def test_microbatched_grads_match_full_batch_mask_one():
    # batch 10 with micro 4 -> slices of 4, 4, 2: the uneven tail exercises
    # the n_k/N weighting, which is exactly where a naive mean-of-means
    # implementation would silently deviate.
    _assert_parity(_mask_one_head_and_target, microbatch_size=4, batch_size=10)


def test_microbatched_grads_match_full_batch_masked_attention():
    _assert_parity(
        _masked_attention_head_and_target, microbatch_size=4, batch_size=10
    )


def test_microbatch_none_is_the_single_backward_path_bit_exactly():
    """None must reproduce the archived single-backward BIT-exactly (same
    ops in the same order), because the queued fleet imports this code with
    the field unset — statistical equivalence is not a strong enough
    contract there."""
    head, tgt = _mask_one_head_and_target()
    x, t, c_t = _batch(tgt, 8)
    ref_loss, ref_residual, ref_grads = _full_batch_reference(
        head, tgt, x, t, c_t
    )

    head.zero_grad()
    loss, residual = loss_swap_backward_microbatched(
        x, t, c_t, head, tgt, microbatch_size=None
    )
    assert torch.equal(loss, ref_loss)
    assert torch.equal(residual, ref_residual)
    for got, want in zip(_grads(head), ref_grads, strict=True):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.equal(got, want)


def test_microbatch_at_or_above_batch_is_the_single_backward_path():
    head, tgt = _mask_one_head_and_target()
    x, t, c_t = _batch(tgt, 8)
    ref_loss, _, ref_grads = _full_batch_reference(head, tgt, x, t, c_t)

    head.zero_grad()
    loss, _ = loss_swap_backward_microbatched(
        x, t, c_t, head, tgt, microbatch_size=64
    )
    assert torch.equal(loss, ref_loss)
    for got, want in zip(_grads(head), ref_grads, strict=True):
        if got is not None:
            assert torch.equal(got, want)


def test_microbatch_accepts_scalar_c_t():
    """Test call sites pass a 0-d dt_log_Zt; the slicer must broadcast it
    rather than index into it."""
    head, tgt = _mask_one_head_and_target()
    x, t, _ = _batch(tgt, 10)
    c_t = torch.zeros(())
    ref_loss, _, ref_grads = _full_batch_reference(head, tgt, x, t, c_t)

    head.zero_grad()
    loss, _ = loss_swap_backward_microbatched(
        x, t, c_t, head, tgt, microbatch_size=4
    )
    assert torch.isclose(loss, ref_loss, rtol=1e-5, atol=1e-7)
    for got, want in zip(_grads(head), ref_grads, strict=True):
        if got is not None:
            assert torch.allclose(got, want, rtol=1e-4, atol=1e-6)
