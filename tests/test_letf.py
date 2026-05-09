"""Tests for the Locally Equivariant Transformer.

Four correctness pillars:
1. CausalStack inclusive-causal masking - perturbing input position i changes
   outputs at positions >= i but NOT positions < i.
2. compute_body hollow (Def. 3) - full pipeline output at i independent of x_i.
   This is the load-bearing test; the readout's slice-and-mask trick + the
   inclusive-causal stacks are tested jointly here.
3. Local equivariance (Eq. 20) - G(tau, i | x) + G(x_i, i | Swap(x, i, tau)) ~= 0.
4. Spin/index input regression - model accepts both +-1 floats (training)
   and 0/1 longs (tests).

The original isolated AttentionReadout masks-self test is dropped: the
slice-and-mask design depends on inputs structured by the inclusive-causal
stacks, so an isolated readout test feeding arbitrary tensors is over-strict.
The full-pipeline hollow test in pillar 2 is the right level of granularity.
"""
import torch
import torch.nn as nn

from discrete_flow_sampler.models.letf import CausalStack, LeTFRateMatrix


def test_causal_stack_inclusive_causal():
    """Inclusive causal: perturbing input at position i leaves outputs at
    positions < i unchanged; outputs at positions >= i may change.
    """
    torch.manual_seed(0)
    stack = CausalStack(hidden_dim=8, n_layers=2, n_heads=2, seq_len=6)
    stack.eval()

    B, seq_len, h = 1, 6, 8
    x = torch.randn(B, seq_len, h)
    out = stack(x)

    i = 3
    x_pert = x.clone()
    x_pert[0, i] = torch.randn(h)
    out_pert = stack(x_pert)

    diff_before = (out[0, :i] - out_pert[0, :i]).abs().max().item()
    assert diff_before < 1e-6, (
        f"Causality violated: outputs at pos < i={i} changed by {diff_before:.2e}"
    )

    diff_after = (out[0, i:] - out_pert[0, i:]).abs().max().item()
    assert diff_after > 1e-3, (
        f"Trivial stack: outputs at pos >= i={i} unchanged (diff={diff_after:.2e}); "
        "layer is acting as identity on the perturbation, suggesting attention "
        "is broken"
    )


def _make_model(d: int = 9, vocab_size: int = 2, seed: int = 42) -> LeTFRateMatrix:
    torch.manual_seed(seed)
    model = LeTFRateMatrix(
        d=d,
        vocab_size=vocab_size,
        hidden_dim=8,
        n_layers=2,
        n_heads=2,
    )
    model.eval()
    return model


def test_compute_body_hollow():
    """Definition 3: H_HTF(x) at position i does not depend on x_i.

    Load-bearing test for the slice-and-mask architecture: verifies the
    joint behaviour of (inclusive-causal CausalStack) + (slice-and-mask
    AttentionReadout) actually produces hollow output at every position.
    """
    d, vocab_size = 9, 2
    model = _make_model(d=d, vocab_size=vocab_size)

    x1 = torch.randint(0, vocab_size, (1, d))
    t = torch.rand(1)
    H1 = model.compute_body(x1, t)

    for i in range(d):
        x2 = x1.clone()
        x2[0, i] = (x2[0, i].item() + 1) % vocab_size
        H2 = model.compute_body(x2, t)
        diff = (H1[0, i, :] - H2[0, i, :]).abs().max().item()
        assert diff < 1e-5, (
            f"Hollow violated at i={i}: ||H1[i] - H2[i]||_inf = {diff:.2e}"
        )


def test_local_equivariance_random_init():
    """Eq. 20 antisymmetry holds for every non-self transition."""
    d, vocab_size = 9, 2
    model = _make_model(d=d, vocab_size=vocab_size)

    x = torch.randint(0, vocab_size, (1, d))
    t = torch.rand(1)
    G = model(x, t)

    for i in range(d):
        x_i = x[0, i].item()
        for tau in range(vocab_size):
            if tau == x_i:
                continue
            x_swapped = x.clone()
            x_swapped[0, i] = tau
            G_swapped = model(x_swapped, t)
            sum_term = G[0, i, tau] + G_swapped[0, i, x_i]
            assert sum_term.abs().item() < 1e-5, (
                f"LE failed at i={i}, tau={tau}, x_i={x_i}: "
                f"G+Gswap = {sum_term.item():.2e}"
            )


def test_accepts_both_spin_and_index_input():
    """forward output identical for +-1 floats (training) and 0/1 longs (tests)."""
    d, vocab_size = 9, 2
    model = _make_model(d=d, vocab_size=vocab_size)

    x_idx = torch.randint(0, vocab_size, (2, d))
    x_spin = (2 * x_idx - 1).float()
    t = torch.rand(2)

    G_idx = model(x_idx, t)
    G_spin = model(x_spin, t)
    assert torch.allclose(G_idx, G_spin, atol=1e-6), (
        f"Output differs between spin/index input: "
        f"max diff = {(G_idx - G_spin).abs().max().item():.2e}"
    )


def test_reference_fidelity_components_present():
    """Pins reference-matching leTF details that materially affected D=10 runs."""
    d, vocab_size, hidden_dim, n_heads = 9, 2, 8, 2
    model = _make_model(d=d, vocab_size=vocab_size)

    for block in list(model.fwd_stack.blocks) + list(model.bwd_stack.blocks):
        assert block.pos_embed.shape == (1 + d, hidden_dim)

    assert model.attention_readout.pos_embed.shape == (d, hidden_dim // n_heads)
    assert isinstance(model.output_norm, nn.LayerNorm)
