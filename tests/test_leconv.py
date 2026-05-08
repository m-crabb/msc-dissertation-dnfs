"""Tests for LeConvRateMatrix.

Three correctness pillars:
1. Local equivariance (Eq. 20 in the DNFS paper / Definition 2):
   G(τ, i | x) + G(x_i, i | Swap(x, i, τ)) = 0.
2. Hollow body (Definition 3): H(x)_(r,c) does not depend on x_(r,c).
3. Translation equivariance: H(T_v x) = T_v H(x). Architectural property
   from the convolutional construction; load-bearing for the MARS framing.
"""
import torch

from discrete_flow_sampler.models.leconv import LeConvRateMatrix


def _make_model(D: int = 3, vocab_size: int = 2, seed: int = 42) -> LeConvRateMatrix:
    torch.manual_seed(seed)
    model = LeConvRateMatrix(
        D=D,
        vocab_size=vocab_size,
        hidden_dim=8,
        n_summands=2,
        kernel_size=3,
    )
    model.eval()
    return model


def test_local_equivariance_random_init():
    """Eq. 20: G(τ, i | x) + G(x_i, i | Swap(x, i, τ)) ≈ 0 for all i, τ ≠ x_i."""
    D, vocab_size = 3, 2
    d = D * D
    model = _make_model(D=D, vocab_size=vocab_size)

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


def test_compute_body_hollow():
    """Definition 3: H(x) at flat position i does not depend on x_i."""
    D, vocab_size = 3, 2
    d = D * D
    model = _make_model(D=D, vocab_size=vocab_size)

    x1 = torch.randint(0, vocab_size, (1, d))
    t = torch.rand(1)
    H1 = model.compute_body(x1, t)

    for i in range(d):
        x2 = x1.clone()
        x2[0, i] = (x2[0, i].item() + 1) % vocab_size
        H2 = model.compute_body(x2, t)
        diff = (H1[0, i, :] - H2[0, i, :]).abs().max().item()
        assert diff < 1e-6, (
            f"Hollow violated at i={i}: ||H1[i] - H2[i]||_inf = {diff:.2e}"
        )


def test_accepts_both_spin_and_index_input():
    """Forward output is identical for ±1 float spins (training convention)
    and 0/1 Long indices (test convention).

    Regression test for a bug where compute_body / forward only handled Long
    indices and crashed at nn.Embedding on the float spins that sample_ctmc
    actually produces in training. See leMLP.forward (lemlp.py:202) for the
    spin-to-index conversion convention.
    """
    D, vocab_size = 3, 2
    d = D * D
    model = _make_model(D=D, vocab_size=vocab_size)

    x_idx = torch.randint(0, vocab_size, (2, d))
    x_spin = (2 * x_idx - 1).float()
    t = torch.rand(2)

    G_idx = model(x_idx, t)
    G_spin = model(x_spin, t)
    assert torch.allclose(G_idx, G_spin, atol=1e-6), (
        f"Output differs between spin (±1) and index (0/1) input: "
        f"max diff = {(G_idx - G_spin).abs().max().item():.2e}"
    )


def test_translation_equivariant_body():
    """H(roll(x, v)) = roll(H(x), v) for any lattice shift v."""
    D, vocab_size = 4, 2
    model = _make_model(D=D, vocab_size=vocab_size)

    x_grid = torch.randint(0, vocab_size, (1, D, D))
    t = torch.rand(1)

    H_flat = model.compute_body(x_grid.flatten(start_dim=1), t)
    h_dim = H_flat.shape[-1]
    H_grid = H_flat.reshape(1, D, D, h_dim).permute(0, 3, 1, 2)

    for dr, dc in [(1, 0), (0, 1), (2, 3), (-1, -1)]:
        x_shifted = torch.roll(x_grid, shifts=(dr, dc), dims=(-2, -1))
        H_shifted_flat = model.compute_body(x_shifted.flatten(start_dim=1), t)
        H_shifted_grid = H_shifted_flat.reshape(1, D, D, h_dim).permute(0, 3, 1, 2)

        H_grid_rolled = torch.roll(H_grid, shifts=(dr, dc), dims=(-2, -1))
        diff = (H_grid_rolled - H_shifted_grid).abs().max().item()
        assert diff < 1e-5, (
            f"Translation equivariance violated at ({dr},{dc}): {diff:.2e}"
        )
