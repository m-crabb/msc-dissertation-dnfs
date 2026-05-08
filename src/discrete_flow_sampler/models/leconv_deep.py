"""LEAPS-style deep locally-equivariant convolutional rate matrix.

Reference: Holderrieth, Albergo & Jaakkola (2025), 'LEAPS: A discrete
neural sampler via locally equivariant networks', papers/leaps.pdf,
Section 9 + Figure 3 + Figure 7 (kernel-schedule ablation).

Architecture (LEAPS Section 9):
    h_0 = constant per-site, time-conditioned
    for layer l in 1..L:
        W_l = σ(A_l h_{l-1} + b_l) + c_l       # 1×1 conv (per-site channel-mix)
        h_l = k_t(W_l) * x                      # convolve ORIGINAL x with hollow kernel
    H = h_L

Hollow-preservation through depth (the LEAPS trick):
    1. k_t zeros the kernel center → x_i never appears at site i directly.
    2. A_l is a 1×1 conv → W_l[site] depends only on h_{l-1}[site].
    3. By induction over l, h_l[site] never depends on x[site].

Why this beats the static K-summand `LeConvRateMatrix` at criticality:
    Critical correlations are scale-free (power-law decay). A wavelet-like
    kernel schedule (e.g. [3, 5, 7, 9]) lets each layer pick up a distinct
    length-scale of fluctuations. K parallel hollow filters at one kernel
    size cannot represent multi-scale structure. LEAPS Figure 7 ablation
    confirms: depth-5 LEC > depth-3 LEC > LEA at the same parameter budget.
"""
import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.lemlp import TimestepEmbedder


class LeConvDeepRateMatrix(nn.Module):
    """LEAPS-style deep LEC rate matrix.

    Drop-in replacement for `LeConvRateMatrix` under the
    `is_locally_equivariant=True` interface — the readout (Prop. 2) and
    the (B, d, S) output shape are identical; only `compute_body` changes
    from K parallel hollow convs to L sequential layers with the
    data-dependent-weight stacking trick.

    Args:
        D: lattice side length. Flat dim d = D*D.
        vocab_size: number of token states S (2 for binary Ising).
        kernel_schedule: tuple of odd kernel sizes per layer; length L
            sets the depth. LEAPS Figure 7 caption uses (5, 7, 15) for
            depth 3 and (3, 5, 7, 9, 15) for depth 5 on 15×15 lattices.
            For our D=10 the lattice-spanning 15 is overkill; (3, 5, 7, 9)
            keeps the wavelet structure within reach.
        hidden_dim: channel dim for embeddings + the per-layer state.
            Implementation may use this directly as d_l for every layer
            OR set d_l = kernel_schedule[l]² per layer (paper-faithful;
            the latter is what makes A_l 1×1 conv reshape into a kernel).

    USER FILLS IN:
        - per-layer parameters in `__init__` (block marked below)
        - the `compute_body` recurrence

    Tests in `tests/test_leconv_deep.py` encode the structural contract.
    """

    is_locally_equivariant: bool = True

    def __init__(
        self,
        D: int,
        vocab_size: int,
        kernel_schedule: tuple[int, ...],
        hidden_dim: int,
    ):
        super().__init__()
        if not kernel_schedule:
            raise ValueError("kernel_schedule must be non-empty")
        for k in kernel_schedule:
            if k % 2 == 0:
                raise ValueError(f"kernel sizes must be odd, got {k}")
        self.D = D
        self.vocab_size = vocab_size
        self.kernel_schedule = tuple(kernel_schedule)
        self.depth = len(self.kernel_schedule)
        self.hidden_dim = hidden_dim

        # Readout (paper-faithful, mirrors LeConvRateMatrix; see DNFS Prop. 2).
        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.omega = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)

        nn.init.kaiming_uniform_(self.token_embedder.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.omega.weight, a=5 ** 0.5)

        # ----------------------------------------------------------------
        # USER: per-layer LEC parameters (LEAPS Section 9 / Figure 3).
        # Paper-default suggested structure (one block per kernel size k_l):
        #
        #   - A_l : (d_l, d_l)        # 1×1 channel-mix on h_{l-1}
        #   - b_l : (d_l,)            # bias added before σ
        #   - c_l : (d_l,)            # post-σ offset
        #   - hollow_mask: (k_l, k_l) buffer, ones with center 0
        #
        # Plus an initial constant `h_0` of shape (d_0,) — the 'Const' block
        # in Figure 3, broadcast over sites and combined with time conditioning
        # before the first layer.
        #
        # The paper's parameter count (~100k for depth 5 on 15×15) suggests
        # d_l = k_l² (the channel dim equals the kernel weight count, so W_l
        # reshapes directly into a (k_l, k_l) spatial kernel). That's the
        # paper-faithful choice; alternative is d_l = hidden_dim for all l.
        # ----------------------------------------------------------------

    def compute_body(self, x: Tensor, t: Tensor) -> Tensor:
        """Pre-readout body H(x), shape (B, d, hidden_dim).

        Must satisfy:
            - Hollow: H(x)[i] does not depend on x[i] (Definition 3).
            - Translation-equivariant under lattice shifts (verified by tests).

        LEAPS recurrence (Section 9):
            h_0 = constant + time conditioning, broadcast over sites
            for l in 1..L:
                W_l = σ(A_l · h_{l-1} + b_l) + c_l       # per-site 1×1
                W_l = W_l * hollow_mask                  # zero kernel center
                h_l = conv2d(x_emb, W_l, circular_pad)   # original x at every layer
            H = h_L

        Implementation notes:
            - Accept ±1 float spins (training; ctmc.py flip convention) OR
              0/1 Long indices (tests). The line `((x+1)/2).long()`
              converts in either case.
            - Use F.pad(..., mode='circular') for periodic Ising boundaries.
            - Time conditioning: simplest is to add `time_embedder(t)`
              to h_0 once before the layer loop.
            - The "data-dependent kernel" in the paper means W_l is computed
              from h_{l-1}, then used as the conv kernel that operates on x.
              The trick that keeps it efficient: W_l can be the same kernel
              everywhere (uniform-W variant) OR per-site (full data-dependent
              variant). Start with uniform-W (treat A_l h_{l-1}+b_l as a
              single per-site value averaged over sites, or use only h_0
              for W_1) — simpler to implement, still gets depth.
        """
        raise NotImplementedError(
            "Implement per LEAPS Section 9 + Figure 3. See class docstring "
            "and the suggested per-layer parameter scaffold in __init__. "
            "Tests in tests/test_leconv_deep.py encode the contract."
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Returns G(τ, i | x), shape (B, d, S). τ=x_i slot is exactly zero.

        Identical formula to `LeConvRateMatrix.forward`; only `compute_body`
        differs from the static K-summand version.

        Accepts ±1 float spins (training) or 0/1 Long indices (tests).
        """
        H = self.compute_body(x, t)
        x_idx = ((x + 1) / 2).long()
        omega_all = self.omega.weight
        omega_xi = self.omega(x_idx)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        G = torch.einsum("bdh,bdsh->bds", H, diff)
        G = G.scatter(-1, x_idx.unsqueeze(-1), 0.0)
        return G
