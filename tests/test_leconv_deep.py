"""Tests for LeConvDeepRateMatrix (LEAPS-style deep LEC).

Reference: Holderrieth, Albergo & Jaakkola (2025), papers/leaps.pdf,
Section 9 + Figure 3.

Three structural pillars (mirrors test_leconv.py) plus the architecture-
specific concern that hollow-ness must hold *through depth* of the
data-dependent-weight stacking trick.

1. Local equivariance (DNFS Eq. 20).
2. Hollow body (DNFS Definition 3) — verified at depth>1, since
   stacking is the new architectural mechanism.
3. Translation equivariance — verified at depth>1.
"""
import torch

from discrete_flow_sampler.models.leconv_deep import LeConvDeepRateMatrix


def _make_model(
    D: int = 3,
    vocab_size: int = 2,
    kernel_schedule: tuple[int, ...] = (3, 3),
    hidden_dim: int = 8,
    seed: int = 42,
    use_global_context: bool = False,
) -> LeConvDeepRateMatrix:
    torch.manual_seed(seed)
    model = LeConvDeepRateMatrix(
        D=D,
        vocab_size=vocab_size,
        kernel_schedule=kernel_schedule,
        hidden_dim=hidden_dim,
        use_global_context=use_global_context,
    )
    model.eval()
    return model


def test_local_equivariance_at_depth():
    """Eq. 20: G(τ, i | x) + G(x_i, i | Swap(x, i, τ)) ≈ 0 with depth>1."""
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
                f"LE failed at depth=2, i={i}, tau={tau}, x_i={x_i}: "
                f"G+Gswap = {sum_term.item():.2e}"
            )


def test_compute_body_hollow_at_depth():
    """Definition 3 holds at the FINAL layer output H = h_L for depth>1.

    The data-dependent-weight stacking (LEAPS Section 9) preserves
    hollow-ness through depth because: (i) k_t zeros the kernel center,
    so h_l[site] never receives a contribution from x[site], and (ii) the
    1×1 channel-mix A_l is per-site, so W_l[site] depends only on
    h_{l-1}[site] — which by induction over l excludes x[site].
    """
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
            f"Hollow violated at depth=2, i={i}: "
            f"||H1[i] - H2[i]||_inf = {diff:.2e}"
        )


def test_translation_equivariant_body_at_depth():
    """H(roll(x, v)) = roll(H(x), v) for any lattice shift v at depth>1."""
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
            f"Translation violated at depth=2, ({dr},{dc}): {diff:.2e}"
        )


def test_accepts_both_spin_and_index_input():
    """Forward output identical for ±1 float spins and 0/1 Long indices.

    Same regression pattern as test_leconv.py — ctmc.py:191 produces
    ±1 floats from torch.where(...,-state,state); leMLP/leConv handle
    via ((x+1)/2).long() conversion. leconv_deep must too.
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


def test_kernel_generation_state_depends_on_time():
    """The first generated kernel is conditioned on t, not only on a constant h_0."""
    D, vocab_size = 3, 2
    model = _make_model(D=D, vocab_size=vocab_size)

    captured: list[torch.Tensor] = []

    def capture_input(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.A[0].register_forward_pre_hook(capture_input)
    try:
        x = torch.randint(0, vocab_size, (1, D * D))
        model.compute_body(x, torch.tensor([0.1]))
        model.compute_body(x, torch.tensor([0.9]))
    finally:
        handle.remove()

    diff = (captured[0] - captured[1]).abs().max().item()
    assert diff > 1e-6, "first-layer kernel state is independent of time"


def test_hollow_global_context_preserves_structural_contracts():
    """Leave-one-out global context is hollow and translation equivariant."""
    D, vocab_size = 4, 2
    d = D * D
    model = _make_model(
        D=D,
        vocab_size=vocab_size,
        kernel_schedule=(3, 5),
        use_global_context=True,
    )

    x_grid = torch.randint(0, vocab_size, (1, D, D))
    x_flat = x_grid.flatten(start_dim=1)
    t = torch.rand(1)
    H = model.compute_body(x_flat, t)

    for i in range(d):
        x_flipped = x_flat.clone()
        x_flipped[0, i] = (x_flipped[0, i].item() + 1) % vocab_size
        H_flipped = model.compute_body(x_flipped, t)
        diff = (H[0, i] - H_flipped[0, i]).abs().max().item()
        assert diff < 1e-6, (
            f"hollow global context depends on x_i at site {i}: {diff:.2e}"
        )

    h_dim = H.shape[-1]
    H_grid = H.reshape(1, D, D, h_dim).permute(0, 3, 1, 2)
    x_shifted = torch.roll(x_grid, shifts=(1, 2), dims=(-2, -1))
    H_shifted = model.compute_body(x_shifted.flatten(start_dim=1), t)
    H_shifted_grid = H_shifted.reshape(1, D, D, h_dim).permute(0, 3, 1, 2)
    H_grid_rolled = torch.roll(H_grid, shifts=(1, 2), dims=(-2, -1))
    assert torch.allclose(H_grid_rolled, H_shifted_grid, atol=1e-5)


def test_varied_kernel_schedule_runs():
    """Depth-5 model with [3,5,7,9,15] kernels runs forward with correct shape.

    This is the LEAPS Figure 7 pattern (theirs: [5,7,15] for depth 3,
    [3,5,7,9,15] for depth 5 on a 15×15 lattice). We test the depth-5
    multi-scale schedule end-to-end on the D=10 torus used by our runs.
    """
    D, vocab_size = 10, 2
    model = _make_model(
        D=D, vocab_size=vocab_size,
        kernel_schedule=(3, 5, 7, 9, 15), hidden_dim=16,
    )
    x = torch.randint(0, vocab_size, (2, D * D))
    t = torch.rand(2)
    G = model(x, t)
    assert G.shape == (2, D * D, vocab_size), f"unexpected G shape {G.shape}"
