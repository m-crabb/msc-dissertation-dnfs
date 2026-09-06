"""Locally Equivariant MLP rate-matrix parameterisation (DNFS Sec. 3.3 + App. B.2).

Background — why local equivariance?
    The CTMC generator G_theta(τ, i | x, t) must satisfy the detailed-balance
    analogue of Eq. (10) in the paper, which requires the antisymmetry
    (Eq. 20):

        G(τ, i | x) = -G(x_i, i | Swap(x, i, τ))          ...(20)

    where Swap(x, i, τ) is x with site i replaced by τ.  Any architecture
    that bakes in this identity is called *locally equivariant* (LE).  The MLP
    of Stage 1 does not satisfy it — its outputs at different sites are coupled
    in an unconstrained way.

Hollow network (Definition 3):
    A network is hollow w.r.t. site i if its output at site i does not depend
    on the token x_i itself — only on the *other* sites.  The hollow mask is a
    (D, D) matrix of ones with a zero diagonal; multiplying W_raw by it zeroes
    out the diagonal weights so site i never reads its own embedding.

Proposition 2 readout (App. B.2):
    Given a hollow hidden representation H(x_{-i}, t) ∈ R^h for site i, a
    locally equivariant rate can be constructed as:

        G(τ, i | x) = <H_i, ω_τ - ω_{x_i}>                ...(Prop. 2)

    where ω_τ, ω_{x_i} are learned token embeddings.  The antisymmetry
    property follows algebraically: swapping x_i ↔ τ negates H_i (hollow,
    so H_i(x_{-i}) is unchanged) and negates (ω_τ - ω_{x_i}).

Single-layer constraint:
    *Stacking hollow layers violates LE.*  After the first hollow layer the
    hidden state at site i depends on {x_j : j ≠ i}; a second hollow layer
    would mix those outputs across sites, reintroducing x_i dependence.  We
    instead use multiple *summands* — K independent single-layer hollow
    projections summed before the readout — which preserves hollowness.

Spin convention:
    States arrive as {-1, +1} tensors (the codebase convention for Ising
    spins). Internally we map to {0, 1} indices via idx = (x + 1) / 2 for
    embedding lookups; we never change the external contract.
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

    Constructs G(τ, i | x, t) satisfying the antisymmetry condition
    (Eq. 20) by combining:

    1. Hollow MLP — K independent single-layer projections (summands),
       each masked so site i never reads x_i.  The mask enforces
       Definition 3 (hollow w.r.t. every site simultaneously).

    2. Prop. 2 readout — inner product of the hollow hidden state H_i
       with the difference of learned token embeddings: <H_i, ω_τ - ω_{x_i}>.
       This is exactly zero when τ = x_i, and antisymmetric under swap.

    3. Time conditioning — sinusoidal TimestepEmbedder added to H_i after
       the hollow projection; time is shared across all sites.

    Args:
        d: number of sites (D² for a D×D Ising lattice).
        vocab_size: number of discrete token values (S=2 for binary spins).
        hidden_dim: width h of all embedding / hidden layers.
        n_summands: number K of independent hollow-MLP summands.  Must be
            >= 1.  More summands increase expressivity without violating LE.
        activation: nonlinearity applied inside each hollow summand.
            One of "gelu" (default), "relu", "silu".
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

        # Hollow MLP weights: K summands, each a (D, D) weight matrix over
        # site indices and a bias.  The hollow_mask zeroes the diagonal so
        # site i never aggregates its own embedding.
        K = n_summands
        # Diagonal is zero-init AND zero-masked at every forward; the mask is
        # the load-bearing one but explicitly zeroing the storage avoids
        # confusing parameter-histogram readouts.
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
        # Step 1 — convert spins to 0/1 embedding indices.
        x_idx = ((x + 1) / 2).long()  # (B, D), values in {0, 1}

        # Step 2 — embed each site's current token.
        x_emb = self.token_embedder(x_idx)  # (B, D, h)

        # Step 3 — hollow MLP with K summands.
        #   W_raw is (K, D, D); multiplying by hollow_mask zeroes the diagonal
        #   so site i's row reads only from other sites (Definition 3).
        W = self.W_raw * self.hollow_mask  # (K, D, D)
        # einsum "kdj,bjh->kbdh": for each summand k and site d, aggregate the
        # weighted embeddings of *all other* sites j (diagonal is 0).
        # b: batch, d: target site, j: source site, h: embedding dim.
        pre = torch.einsum("kdj,bjh->kbdh", W, x_emb) + self.b[:, None, None, :]
        # (K, B, D, h) -> activation -> sum over K -> (B, D, h)
        H = self.activation(pre).sum(dim=0)  # (B, D, h)

        # Step 4 — add time conditioning (broadcast over sites).
        H = H + self.time_embedder(t)[:, None, :]  # (B, D, h)

        # Step 5 — Prop. 2 readout: G(τ, i | x) = <H_i, ω_τ - ω_{x_i}>.
        #   omega_all : (S, h)  — all token embeddings
        #   omega_xi  : (B, D, h) — embedding of the current token at each site
        omega_all = self.omega.weight  # (S, h)
        omega_xi = self.omega(x_idx)  # (B, D, h)
        # diff[b, d, s, h] = omega_all[s, h] - omega_xi[b, d, h]
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]  # (B, D, S, h)
        # Contract over hidden dim h to get (B, D, S) rates.
        G = torch.einsum("bdh,bdsh->bds", H, diff)  # (B, D, S)

        # Guard against fp rounding in the inner product: the τ=x_i slot is
        # (ω_{x_i} - ω_{x_i})^T H_i = 0 algebraically, but float32 arithmetic
        # can leave a small non-zero residual. Scatter to exact zero so
        # downstream [G]_+ / [-G]_+ aren't fed numerical garbage at the self-slot.
        G = G.scatter(-1, x_idx.unsqueeze(-1), 0.0)

        return G
