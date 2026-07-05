"""Tests for the anchor-batched (vectorised) LeTFMaskOneSwapHead forward.

Why this exists (2026-07-05): the original forward ran `for i in range(d)`
sequential masked leTF passes -- ~d x the single-site head (56 s/forward at
d=256), which made the D=8 sigma_c rung ~8 h/seed and D=16 infeasible. The d
anchor passes are independent (they differ only in which site's embedding is
zeroed), so they batch into the model batch dimension: build (n_anchors*B, d, h)
with a diagonal zeroing mask, run ONE fwd/bwd-stack + readout pass, reshape to
(n_anchors, B, d, h), and read out against omega_{x_i} - omega_{x_j}
vectorised. `anchor_chunk_size` caps how many anchors ride one pass, bounding
the readout attention buffer (n_anchors*B, n_heads, d, 2d) at large d.

Contracts pinned here:
  * forward == forward_looped (the preserved reference loop) == the
    DoublyHollowSwapHead oracle, at d=16 (gate dim) and d=64 (D=8 rung dim).
  * anchor chunking (incl. a ragged tail chunk) changes nothing numerically,
    and the backbone stacks genuinely see at most chunk_size*B rows per call.
  * gradients through the vectorised path match the loop (the head trains).

Equality convention: the re-batching applies identical kernels with identical
reduction orders per element, so torch.equal is expected on CPU; fall back to
ATOL if a backend makes the paths diverge in ULPs (same pattern as
test_swap_readout.test_brute_force_matches_mask_one_d16_batch_all_pairs).
Antisymmetry / trivial-swap / label-asymmetry properties of the vectorised
forward are covered by the existing tests in test_swap_readout.py, which call
head(x, t) directly.
"""

import torch

from discrete_flow_sampler.constraints.swap_readout import (
    DoublyHollowSwapHead,
    LeTFMaskOneSwapHead,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix

ATOL = 1e-5


def _backbone(d, seed=42, hidden_dim=8, n_heads=2, n_layers=2):
    torch.manual_seed(seed)
    m = LeTFRateMatrix(
        d=d, vocab_size=2, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads
    )
    m.eval()
    return m


def _state_batch(d, batch, seed=7):
    """A (batch, d) +-1 state batch; every row guaranteed to contain both spins."""
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    for b in range(batch):
        if bool((x[b] > 0).all()) or bool((x[b] < 0).all()):
            x[b, 0] *= -1
    return x


def _assert_matches(got, want, label):
    if not torch.equal(got, want):
        diff = (got - want).abs().max().item()
        assert diff < ATOL, f"{label}: max diff {diff:.2e}"


@torch.no_grad()
def test_vectorised_matches_loop_and_doubly_hollow_d16():
    """Triple agreement at the gate dimension, batch > 1, full (B, d, d)."""
    d = 16
    m = _backbone(d=d)
    head = LeTFMaskOneSwapHead(m)
    x = _state_batch(d=d, batch=3)
    t = torch.rand(3)

    G_vec = head(x, t)
    G_loop = head.forward_looped(x, t)
    G_dh = DoublyHollowSwapHead(m)(x, t)

    _assert_matches(G_vec, G_loop, "d=16 vectorised vs loop")
    _assert_matches(G_vec, G_dh, "d=16 vectorised vs doubly_hollow")


@torch.no_grad()
def test_vectorised_matches_loop_and_doubly_hollow_d64():
    """Triple agreement at the D=8 rung dimension (doubly_hollow is O(d^2)
    passes, so smallest viable backbone + batch 1)."""
    d = 64
    m = _backbone(d=d, n_layers=1)
    head = LeTFMaskOneSwapHead(m)
    x = _state_batch(d=d, batch=1)
    t = torch.rand(1)

    G_vec = head(x, t)
    G_loop = head.forward_looped(x, t)
    G_dh = DoublyHollowSwapHead(m)(x, t)

    _assert_matches(G_vec, G_loop, "d=64 vectorised vs loop")
    _assert_matches(G_vec, G_dh, "d=64 vectorised vs doubly_hollow")


@torch.no_grad()
def test_anchor_chunking_matches_unchunked_d256():
    """Chunked == unchunked == loop at d=256, with a ragged tail chunk, and the
    stacks genuinely never see more than chunk_size*B rows at once.

    d=256 is the size where the unchunked readout attention buffer
    (d*B, n_heads, d, 2d) stops fitting real-run memory; the chunk contract is
    what makes the D=16 rung feasible. chunk_size=96 does not divide 256, so
    the tail chunk (64 anchors) exercises the boundary arithmetic.
    """
    d = 256
    batch = 1
    chunk = 96
    m = _backbone(d=d, n_layers=1)
    x = _state_batch(d=d, batch=batch)
    t = torch.rand(batch)

    G_full = LeTFMaskOneSwapHead(m)(x, t)
    G_loop = LeTFMaskOneSwapHead(m).forward_looped(x, t)

    chunked_head = LeTFMaskOneSwapHead(m, anchor_chunk_size=chunk)
    stack_batches = []
    hook = m.fwd_stack.register_forward_pre_hook(
        lambda module, args: stack_batches.append(args[0].shape[0])
    )
    try:
        G_chunked = chunked_head(x, t)
    finally:
        hook.remove()

    _assert_matches(G_chunked, G_full, "d=256 chunked vs unchunked")
    _assert_matches(G_chunked, G_loop, "d=256 chunked vs loop")
    assert stack_batches == [chunk * batch, chunk * batch, (d - 2 * chunk) * batch], (
        f"anchor chunking not honoured by the stacks: saw batches {stack_batches}"
    )


def test_vectorised_gradients_match_loop():
    """The head trains: backbone gradients through the vectorised forward equal
    the loop's. Squared-sum loss so cancellations in G cannot mask a wrong
    gradient path (e.g. a broken diagonal zeroing mask)."""
    d = 16
    m = _backbone(d=d)
    m.train()
    head = LeTFMaskOneSwapHead(m)
    x = _state_batch(d=d, batch=2)
    t = torch.rand(2)

    head.zero_grad()
    head.forward_looped(x, t).square().sum().backward()
    loop_grads = {
        name: p.grad.clone()
        for name, p in head.named_parameters()
        if p.grad is not None
    }

    head.zero_grad()
    head(x, t).square().sum().backward()

    assert loop_grads, "loop backward produced no gradients"
    for name, p in head.named_parameters():
        if name not in loop_grads:
            continue
        _assert_matches(p.grad, loop_grads[name], f"grad mismatch: {name}")
