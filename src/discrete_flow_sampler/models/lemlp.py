"""Locally Equivariant MLP rate-matrix parameterisation (DNFS Sec. 3.3 + App. B.2).

The generator G_theta(τ, i | x, t) must satisfy the antisymmetry of Eq. (20),

    G(τ, i | x) = -G(x_i, i | Swap(x, i, τ))          ...(20)

where Swap(x, i, τ) is x with site i replaced by τ. An architecture that
bakes this in is locally equivariant (LE); the Stage 1 MLP is not.

Hollow network (Definition 3): the output at site i does not depend on x_i,
only on the other sites. The hollow mask is a (D, D) ones matrix with a zero
diagonal; W_raw * mask stops site i reading its own embedding.

Prop. 2 readout (App. B.2): given hollow H(x_{-i}, t) ∈ R^h for site i,

    G(τ, i | x) = <H_i, ω_τ - ω_{x_i}>                ...(Prop. 2)

with learned token embeddings ω. Swapping x_i ↔ τ leaves H_i unchanged
(hollow) and negates (ω_τ - ω_{x_i}), which is Eq. (20).

Stacking hollow layers breaks LE (a second layer mixes the other sites'
outputs, which depend on x_i, back into site i), so K independent
single-layer hollow summands are summed before the readout instead.

States arrive as {-1, +1} tensors; idx = (x + 1) / 2 maps to {0, 1} for
embedding lookups.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding followed by a 2-layer SiLU MLP.

    Matches the reference architecture (paper Sec. 3.3 leTF figure).
    The sinusoidal basis uses cosine + sine at exponentially-spaced
    frequencies (identical to DDPM / DiT timestep embeddings).

    Args:
        hidden_size: output dimensionality (and MLP width).
        frequency_embedding_size: number of sinusoidal frequency channels
            used as the raw basis before the MLP head.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        """Build a (B, dim) sinusoidal basis for a batch of scalar times t.

        Frequencies are spaced exponentially from 1 to 1/max_period.
        Cosine channels cover indices 0..half-1, sine channels half..dim-1.

        Args:
            t: (B,) float tensor of times in [0, 1].
            dim: number of frequency channels (should be even).
            max_period: controls lowest frequency; larger = slower variation.

        Returns:
            (B, dim) embedding tensor.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]  # (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) time tensor.

        Returns:
            (B, hidden_size) time embedding.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LeMLPRateMatrix(nn.Module):
    """Locally equivariant MLP rate-matrix (DNFS paper Eq. 20, Prop. 2).

    G(τ, i | x, t) = <H_i, ω_τ - ω_{x_i}> with H_i the sum of K single-layer
    hollow projections (Definition 3) plus a shared sinusoidal time embedding.

    Args:
        d: number of sites (D² for a D×D Ising lattice).
        vocab_size: number of discrete token values (S=2 for binary spins).
        hidden_dim: width h of all embedding / hidden layers.
        n_summands: number K of hollow-MLP summands, >= 1.
        activation: nonlinearity inside each summand: "gelu", "relu", "silu".
    """

    is_locally_equivariant: bool = True

    def __init__(
        self,
        d: int,
        vocab_size: int,
        hidden_dim: int,
        n_summands: int,
        activation: str = "gelu",
    ):
        super().__init__()
        if n_summands < 1:
            raise ValueError(f"n_summands must be >= 1, got {n_summands}")

        self.d = d
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_summands = n_summands

        # Token embedder: maps {0, 1, ..., S-1} indices to R^h vectors.
        # Kaiming-uniform init (a=sqrt(5)) matches the reference EmbeddingLayer.
        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        nn.init.kaiming_uniform_(self.token_embedder.weight, a=math.sqrt(5))

        # Hollow MLP: K summands, each a (D, D) site-mixing weight and a bias.
        K = n_summands
        # Diagonal zero-init as well as masked at forward; the mask is
        # load-bearing, zeroed storage just keeps parameter histograms clean.
        W_init = torch.randn(K, d, d) / math.sqrt(d)
        W_init.diagonal(dim1=-2, dim2=-1).zero_()
        self.W_raw = nn.Parameter(W_init)
        self.b = nn.Parameter(torch.zeros(K, hidden_dim))
        self.register_buffer("hollow_mask", 1.0 - torch.eye(d))  # (D, D)

        # Activation applied element-wise inside each summand.
        _acts = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}
        if activation not in _acts:
            raise ValueError(
                f"activation must be one of {list(_acts)}, got {activation!r}"
            )
        self.activation = _acts[activation]

        # Sinusoidal time embedder — output shape (B, h) broadcast to (B, 1, h).
        self.time_embedder = TimestepEmbedder(hidden_dim)

        # Readout token embeddings ω (Prop. 2): one vector per vocab token.
        # Kaiming-uniform init for symmetry with token_embedder.
        self.omega = nn.Embedding(vocab_size, hidden_dim)
        nn.init.kaiming_uniform_(self.omega.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute the locally equivariant rate matrix G(τ, i | x, t).

        Args:
            x: (B, D) state tensor with entries in {-1, +1}.
            t: (B,) time tensor in [0, 1].

        Returns:
            G: (B, D, S) tensor where G[b, i, τ] is the rate of
               transitioning site i to token τ, given state x[b].
               G[b, i, x_i[b]] = 0 exactly (self-slot zeroed by scatter).
        """
        x_idx = ((x + 1) / 2).long()  # (B, D), values in {0, 1}

        x_emb = self.token_embedder(x_idx)  # (B, D, h)

        # Hollow MLP (Definition 3): the mask zeroes W's diagonal so site d
        # aggregates only the other sites j (kdj,bjh->kbdh), summed over K.
        W = self.W_raw * self.hollow_mask  # (K, D, D)
        pre = torch.einsum("kdj,bjh->kbdh", W, x_emb) + self.b[:, None, None, :]
        H = self.activation(pre).sum(dim=0)  # (B, D, h)

        # Time conditioning, broadcast over sites.
        H = H + self.time_embedder(t)[:, None, :]  # (B, D, h)

        # Prop. 2 readout: G(τ, i | x) = <H_i, ω_τ - ω_{x_i}>.
        omega_all = self.omega.weight  # (S, h)
        omega_xi = self.omega(x_idx)  # (B, D, h)
        # diff[b, d, s, h] = omega_all[s, h] - omega_xi[b, d, h]
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]  # (B, D, S, h)
        G = torch.einsum("bdh,bdsh->bds", H, diff)  # (B, D, S)

        # The τ=x_i slot is (ω_{x_i} - ω_{x_i})^T H_i = 0 algebraically; float32
        # leaves a residual, so scatter it to exact zero before [G]_+ / [-G]_+.
        G = G.scatter(-1, x_idx.unsqueeze(-1), 0.0)

        return G
