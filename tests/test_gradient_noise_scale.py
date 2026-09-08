"""Gradient-noise-scale instrumentation pins (2026-08-16).

The queued batch-scaling arm asks whether batch 128 sits below the critical
batch size at 16x16. McCandlish et al. (arXiv:1812.06162, App. A) make that
measurable from two gradient estimates at different batch sizes: for i.i.d.
per-row gradients with mean G and covariance Sigma,

    E |g_B|^2 = |G|^2 + tr(Sigma) / B,

so measuring the squared norm at two batch sizes (b, N) solves the 2x2
system for |G|^2 and tr(Sigma), and B_simple = tr(Sigma)/|G|^2 is the
critical-batch predictor. The micro-batched backward already walks slices
of size b inside the full batch of size N, so both measurements come from
the same already-paid backward — the accumulated-gradient increment after
slice k is (n_k/N) * g_slice_k, which rescales to the unweighted slice
gradient exactly.

These tests pin (1) the estimator algebra, (2) that collected slice
squared-norms equal independently computed slice gradients, and (3) that
collection is a pure observer: accumulated gradients stay bit-identical
and the single-backward paths never touch the out-list.
"""

import math

import torch

from discrete_flow_sampler.constraints.swap_readout import LeTFMaskOneSwapHead
from discrete_flow_sampler.diagnostics.metrics import (
    gradient_noise_scale_components,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers.swap_kolmogorov import (
    loss_swap,
    loss_swap_backward_microbatched,
)
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget


def _head_and_target(D=4, seed=42):
    torch.manual_seed(seed)
    backbone = LeTFRateMatrix(
        d=D * D, vocab_size=2, hidden_dim=16, n_layers=2, n_heads=2
    )
    tgt = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=0.5)
    return LeTFMaskOneSwapHead(backbone, anchor_chunk_size=5), tgt


def _batch(tgt, batch_size, seed=7):
    torch.manual_seed(seed)
    x = tgt.sample_base(batch_size, device="cpu")
    t = torch.rand(batch_size)
    c_t = torch.randn(batch_size) * 0.1
    return x, t, c_t


def _flat_grad(head):
    return torch.cat(
        [
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())
            for p in head.parameters()
        ]
    )


# ---------------------------------------------------------------- algebra


def test_components_invert_the_expectation_identity():
    """Feed the helper squared norms constructed from a known (|G|^2, trS)
    via E|g_B|^2 = |G|^2 + trS/B; it must return that pair exactly, which
    tests the 2x2 inversion."""
    grad_sqnorm_true, trace_sigma = 3.7, 250.0
    slice_size, batch_size = 16, 128
    slice_sqnorm = grad_sqnorm_true + trace_sigma / slice_size
    full_sqnorm = grad_sqnorm_true + trace_sigma / batch_size

    grad_est, trace_est = gradient_noise_scale_components(
        slice_sqnorm, full_sqnorm, slice_size, batch_size
    )
    assert math.isclose(grad_est, grad_sqnorm_true, rel_tol=1e-12)
    assert math.isclose(trace_est, trace_sigma, rel_tol=1e-12)


def test_components_reject_equal_sizes():
    """b == N makes the system singular; the helper must refuse rather
    than divide by zero."""
    try:
        gradient_noise_scale_components(1.0, 1.0, 64, 64)
    except ValueError:
        return
    raise AssertionError("equal slice and batch sizes must raise ValueError")


# ------------------------------------------------------------- collection


def test_slice_sqnorms_match_independent_slice_gradients():
    """Each collected (rows, sqnorm) must equal the squared norm of that
    slice's own unweighted gradient, computed by a separate backward. This
    pins the (N / n_k) rescaling of the accumulated increment — the step a
    naive implementation (logging the weighted increment) would get wrong
    by a factor of (n_k/N)^2."""
    batch_size, microbatch = 10, 4
    head, tgt = _head_and_target()
    x, t, c_t = _batch(tgt, batch_size)

    head.zero_grad()
    collected = []
    loss_swap_backward_microbatched(
        x,
        t,
        c_t,
        head,
        tgt,
        microbatch_size=microbatch,
        slice_grad_sqnorms_out=collected,
    )
    # batch 10 / micro 4 -> slices of 4, 4, 2 (ragged tail included).
    assert [rows for rows, _ in collected] == [4, 4, 2]

    for slice_index, (rows, sqnorm) in enumerate(collected):
        start = slice_index * microbatch
        head.zero_grad()
        slice_loss, _ = loss_swap(
            x[start : start + rows],
            t[start : start + rows],
            c_t[start : start + rows],
            head,
            tgt,
            return_residual=True,
        )
        slice_loss.backward()
        want = float(_flat_grad(head).pow(2).sum())
        assert math.isclose(sqnorm, want, rel_tol=1e-4), (
            f"slice {slice_index}: collected {sqnorm}, independent {want}"
        )


def test_collection_leaves_accumulated_gradients_bit_identical():
    """The observer must not perturb the update: accumulated grads with
    collection on must equal the collection-off run bit for bit."""
    head, tgt = _head_and_target()
    x, t, c_t = _batch(tgt, 10)

    head.zero_grad()
    loss_swap_backward_microbatched(
        x,
        t,
        c_t,
        head,
        tgt,
        microbatch_size=4,
    )
    reference = _flat_grad(head).clone()

    head.zero_grad()
    loss_swap_backward_microbatched(
        x,
        t,
        c_t,
        head,
        tgt,
        microbatch_size=4,
        slice_grad_sqnorms_out=[],
    )
    assert torch.equal(_flat_grad(head), reference)


def test_out_list_untouched_on_single_backward_paths():
    """None and microbatch >= batch are the archived single-backward paths:
    no slices exist, so the out-list must stay empty (a caller reading an
    empty list logs NaN, never a fabricated number)."""
    head, tgt = _head_and_target()
    x, t, c_t = _batch(tgt, 8)

    for microbatch_size in (None, 64):
        head.zero_grad()
        collected = []
        loss_swap_backward_microbatched(
            x,
            t,
            c_t,
            head,
            tgt,
            microbatch_size=microbatch_size,
            slice_grad_sqnorms_out=collected,
        )
        assert collected == []
