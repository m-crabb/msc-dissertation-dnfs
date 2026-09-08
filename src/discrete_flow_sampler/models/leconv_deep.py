"""LEAPS-style deep locally-equivariant convolutional rate matrix.

Reference: Holderrieth, Albergo & Jaakkola (2025), 'LEAPS: A discrete
neural sampler via locally equivariant networks', Section 9 + Figure 3 +
Figure 7 (kernel-schedule ablation).

Architecture (LEAPS Section 9, with explicit DNFS time conditioning and
optional hollow global context):
    h_0(t) = constant per-site + time embedding
    for layer l in 1..L:
        W_l = σ(A_l h_{l-1} + b_l) + c_l       # 1x1 conv (per-site channel-mix)
        h_l = k_t(W_l) * x                      # convolve ORIGINAL x with hollow kernel
    H = h_L

Hollow-preservation through depth (the LEAPS trick):
    1. k_t zeros the kernel center → x_i never appears at site i directly.
    2. A_l is a 1×1 conv → W_l[site] depends only on h_{l-1}[site].
    3. By induction over l, h_l[site] never depends on x[site].

A kernel schedule such as [3, 5, 7, 9] gives each layer a distinct length
scale, which K parallel hollow filters at one kernel size (`LeConvRateMatrix`)
cannot represent; LEAPS Figure 7: depth-5 LEC > depth-3 LEC > LEA at the
same parameter budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.models.lemlp import TimestepEmbedder


class LeConvDeepRateMatrix(nn.Module):
    """LEAPS-style deep LEC rate matrix.

    Drop-in replacement for `LeConvRateMatrix`: the readout (Prop. 2) and
    the (B, d, S) output shape are identical; only `compute_body` changes
    from K parallel hollow convs to L sequential data-dependent-kernel layers.

    Args:
        D: lattice side length. Flat dim d = D*D.
        vocab_size: number of token states S (2 for binary Ising).
        kernel_schedule: tuple of odd kernel sizes per layer; length L
            sets the depth. LEAPS Figure 7 uses (5, 7, 15) for depth 3 and
            (3, 5, 7, 9, 15) for depth 5 on 15×15 lattices; for D=10 the
            lattice-spanning 15 is dropped, (3, 5, 7, 9).
        hidden_dim: channel dim for embeddings + the per-layer state.
        use_global_context: if True, kernel generation at every site also
            receives a leave-one-out mean of the token embeddings (all but
            x_i), which preserves hollowness while exposing lattice-scale
            magnetisation.
    """

    is_locally_equivariant: bool = True

    def __init__(
        self,
        D: int,
        vocab_size: int,
        kernel_schedule: tuple[int, ...],
        hidden_dim: int,
        use_global_context: bool = False,
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
        self.use_global_context = use_global_context

        # Readout (paper-faithful, mirrors LeConvRateMatrix; see DNFS Prop. 2).
        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.omega = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)

        nn.init.kaiming_uniform_(self.token_embedder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.omega.weight, a=5**0.5)

        # h_0: spatially uniform learned constant, shifted by the time
        # embedding in compute_body, so layer 1 is a plain translation-
        # equivariant hollow conv; layers 2..L pick up structure from h_{l-1}.
        self.h_0 = nn.Parameter(torch.zeros(hidden_dim))

        # Per-layer 1×1 channel-mix A_l: hidden_dim -> k_l² channels per site,
        # reshaped to the (k_l, k_l) position-conditional kernel.
        self.A = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, k * k, kernel_size=1, bias=True)
                for k in self.kernel_schedule
            ]
        )
        if use_global_context:
            self.global_context_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        # Post-σ offset c_l (LEAPS Section 9: W_l = σ(A_l h + b_l) + c_l).
        self.c = nn.ParameterList(
            [nn.Parameter(torch.zeros(k * k)) for k in self.kernel_schedule]
        )

        # Hollow masks (zero center, ones elsewhere) per layer — preserves
        # Definition 3 by zeroing the kernel weight at the diagonal.
        for layer_idx, k in enumerate(self.kernel_schedule):
            mask = torch.ones(k, k)
            mask[k // 2, k // 2] = 0.0
            self.register_buffer(f"hollow_mask_{layer_idx}", mask)

    def _leave_one_out_global_context(self, x_emb: Tensor) -> Tensor:
        """Per-site global token summary excluding the site's own token.

        `x_emb` has shape (B, h, D, D). The output at lattice site i is
        mean({x_emb_j : j != i}), hence independent of x_i.
        """
        n_sites = self.D * self.D
        total = x_emb.sum(dim=(2, 3), keepdim=True)
        return (total - x_emb) / max(n_sites - 1, 1)

    def compute_body(self, x: Tensor, t: Tensor) -> Tensor:
        """Pre-readout body H(x), shape (B, d, hidden_dim); the LEAPS
        Section 9 recurrence in the module docstring, with W_l masked hollow
        before the position-conditional conv.

        Accepts ±1 float spins (training; ctmc.py flip convention) or 0/1
        Long indices (tests).
        """
        x_idx = ((x + 1) / 2).long()
        B = x_idx.shape[0]
        x_grid = x_idx.view(B, self.D, self.D)
        x_emb = self.token_embedder(x_grid).permute(0, 3, 1, 2)  # (B, h, D, D)
        global_context = None
        if self.use_global_context:
            global_context = self.global_context_proj(
                self._leave_one_out_global_context(x_emb)
            )

        cond_t = self.time_embedder(t)  # (B, h)
        x_in = x_emb + cond_t[:, :, None, None]  # (B, h, D, D)

        # h_0(t): spatially uniform and time-conditioned; cond_t does not
        # depend on x, so hollowness and translation equivariance hold.
        h = self.h_0[None, :, None, None] + cond_t[:, :, None, None]
        h = h.expand(-1, -1, self.D, self.D).contiguous()

        for layer_idx, k_size in enumerate(self.kernel_schedule):
            # 1) Per-site channel-mix produces kernel weights of shape k_l².
            kernel_state = h if global_context is None else h + global_context
            kernel_weights = F.gelu(self.A[layer_idx](kernel_state))  # (B, k², D, D)
            kernel_weights = kernel_weights + self.c[layer_idx].view(1, -1, 1, 1)

            # 2) Reshape to per-site (k, k) kernel and zero the center weight.
            W = kernel_weights.view(B, k_size, k_size, self.D, self.D)
            hollow_mask = getattr(self, f"hollow_mask_{layer_idx}")
            W = W * hollow_mask[None, :, :, None, None]

            # 3) Position-conditional convolution with circular padding: at
            #    site (r, s) apply W[..., r, s] to the k×k patch of x_in there.
            pad = k_size // 2
            x_padded = F.pad(x_in, (pad, pad, pad, pad), mode="circular")
            x_unfold = x_padded.unfold(2, k_size, 1).unfold(3, k_size, 1)
            # x_unfold: (B, h, D, D, k, k)

            # h_{l+1}[b,c,r,s] = Σ_{i,j} W[b,i,j,r,s] * x_patch[b,c,r,s,i,j]
            h = torch.einsum("bijrs,bcrsij->bcrs", W, x_unfold)

        # (B, hidden_dim, D, D) -> (B, d, hidden_dim) flat-position layout.
        return h.permute(0, 2, 3, 1).reshape(B, self.D * self.D, self.hidden_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Returns G(τ, i | x), shape (B, d, S); the τ=x_i slot is exactly
        zero. Same readout as `LeConvRateMatrix.forward`. Accepts ±1 float
        spins (training) or 0/1 Long indices (tests).
        """
        H = self.compute_body(x, t)
        x_idx = ((x + 1) / 2).long()
        omega_all = self.omega.weight
        omega_xi = self.omega(x_idx)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        G = torch.einsum("bdh,bdsh->bds", H, diff)
        G = G.scatter(-1, x_idx.unsqueeze(-1), 0.0)
        return G
