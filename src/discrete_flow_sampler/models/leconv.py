"""Locally-equivariant convolutional rate matrix (leConv, stage 3).

Derivation (DNFS paper Definition 3 + Proposition 2 + translation symmetry):

1. Ising on a 2D periodic lattice has translation symmetry:
   E(σ) = E(T_v σ) for any lattice shift v.
2. The DNFS rate matrix G(τ, i | x) should respect this:
       G(τ, T_v(i) | T_v x) = G(τ, i | x).
   Combined with local equivariance (Eq. 20):
       G(τ, i | x) = -G(x_i, i | Swap(x, i, τ)),
   we want G that is BOTH translation-equivariant AND locally equivariant.
3. By Prop. 2: any hollow H : X -> R^{d × h} (Def. 3) gives
       G(τ, i | x) = (ω_τ - ω_{x_i})^T H(x)_{i,:},
   which is locally equivariant. Translation equivariance of G follows
   if H is translation equivariant.
4. Hollow + translation-equivariant H: a 2D circular-padded convolution
   with kernel-centre weight = 0 is hollow at every spatial position
   AND translation equivariant. K parallel hollow convs sum to a hollow
   function; stacking would violate hollow-ness (paths through neighbours
   re-introduce dependence on x_i), so we widen via summands rather than
   depth — same trick as leMLP.
5. Result:
       H_Conv(x) = sum_{k=1}^K σ(W^k * x_emb + b^k),  W^k_(0,0) = 0,
       G^θ_t(τ, i | x) = (ω_τ - ω_{x_i})^T H_Conv(x)_{i,:}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_flow_sampler.models.lemlp import TimestepEmbedder


class LeConvRateMatrix(nn.Module):
    """Locally equivariant rate matrix via a hollow 2D convolution body.

    Args:
        D: lattice side length. Flat dimension is d = D * D.
        vocab_size: number of token states S (e.g. 2 for {-1, +1} Ising).
        hidden_dim: channel count for embeddings and conv body.
        n_summands: K parallel hollow convs (paper-aligned at 3).
        kernel_size: odd; 3 captures nearest-neighbour coupling on Ising.

    Forward expects integer tokens x of shape (B, d) in [0, S) and a
    time tensor t of shape (B,) in [0, 1]. Returns G of shape (B, d, S)
    with G[..., x_i] zero on every row i (one-way rate-matrix convention).
    """

    is_locally_equivariant: bool = True

    def __init__(
        self,
        D: int,
        vocab_size: int,
        hidden_dim: int,
        n_summands: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.D = D
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_summands = n_summands
        self.kernel_size = kernel_size

        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.omega = nn.Embedding(vocab_size, hidden_dim)

        scale = 1.0 / (hidden_dim * kernel_size)
        self.conv_weights = nn.Parameter(
            torch.randn(n_summands, hidden_dim, hidden_dim, kernel_size, kernel_size) * scale
        )
        self.conv_bias = nn.Parameter(torch.zeros(n_summands, hidden_dim))

        center = kernel_size // 2
        mask = torch.ones(kernel_size, kernel_size)
        mask[center, center] = 0.0
        self.register_buffer("hollow_mask", mask)

        nn.init.kaiming_uniform_(self.token_embedder.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.omega.weight, a=5 ** 0.5)

    def compute_body(self, x: Tensor, t: Tensor) -> Tensor:
        """Pre-readout body H(x), shape (B, d, h). Hollow + translation-equivariant."""
        B = x.shape[0]
        x_grid = x.view(B, self.D, self.D)
        x_emb = self.token_embedder(x_grid).permute(0, 3, 1, 2)
        cond_t = self.time_embedder(t)
        x_in = x_emb + cond_t[:, :, None, None]

        pad = self.kernel_size // 2
        x_padded = F.pad(x_in, (pad, pad, pad, pad), mode="circular")

        H = torch.zeros_like(x_in)
        for k in range(self.n_summands):
            W_k = self.conv_weights[k] * self.hollow_mask
            out_k = F.conv2d(x_padded, W_k, bias=self.conv_bias[k], padding=0)
            H = H + F.gelu(out_k)

        return H.permute(0, 2, 3, 1).reshape(B, self.D * self.D, self.hidden_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Returns G(τ, i | x), shape (B, d, S). τ=x_i slot is exactly zero."""
        H = self.compute_body(x, t)
        omega_all = self.omega.weight
        omega_xi = self.omega(x)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        G = torch.einsum("bdh,bdsh->bds", H, diff)
        G = G.scatter(-1, x.unsqueeze(-1), 0.0)
        return G
