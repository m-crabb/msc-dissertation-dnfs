"""Locally Equivariant Transformer (leTF, stage 4).

Paper: DNFS Sec. 3.3 + App. B.3 + Eq. 22 + App. E.1.1.

Hollow Transformer:

    1. Embed x to (B, d, h). Compute cond_t = TimestepEmbedder(t) with
       shape (B, 1, h).
    2. fwd_in = cat([cond_t, x_emb], dim=1)   # (B, 1+d, h); cond_t at pos 0
    3. bwd_in = cat([cond_t, x_emb.flip(1)], dim=1)
    4. fwd_x = fwd_stack(fwd_in)               # per-block pos + causal (j <= i)
    5. bwd_x = bwd_stack(bwd_in).flip(1)       # inclusive causal then flip back
    6. H_HTF = attention_readout(fwd_x, bwd_x, cond_t)   # (B, d, h), hollow
    7. G(tau, i | x) = (omega_tau - omega_{x_i})^T H_HTF[:, i, :]   (Eq. 22)

Hollow argument (the "slice-and-mask" trick):

    Inside attention_readout:
      sliced_fwd = fwd_x[:, :-1, :]   # (B, d, h)
      sliced_bwd = bwd_x[:, 1:, :]    # (B, d, h)

    sliced_fwd[k] = fwd_x[k] for k in 0..d-1. Under inclusive causal,
    fwd_x[k] depends on fwd_in[0..k] = {cond_t, x_0, ..., x_{k-1}}, not
    x_k, so sliced_fwd[k] is hollow w.r.t. x_k.

    sliced_bwd[k] = bwd_x[k+1] (post-flip-back), which through the flip,
    inclusive-causal pass and final flip depends on
    {cond_t, x_{k+1}, ..., x_{d-1}}, not x_k.

    The readout fuses Q from combined, K and V from cat([sliced_fwd,
    sliced_bwd]), with masks M_L (target k attends L-keys j <= k) and M_R
    (target k attends R-keys j >= k). Q_k and every attended K_j, V_j are
    hollow w.r.t. x_k, so H[k] is.

Inclusive causal plus the slice, rather than strict causal with bracketing
tokens, avoids the all-masked row strict causal hits at the sequence boundary.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.lemlp import TimestepEmbedder


class _AttentionBlock(nn.Module):
    """Pre-norm Vaswani 2017 block: norm -> MHA + residual -> norm -> FFN + residual."""

    def __init__(self, hidden_dim: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class _CausalBlock(nn.Module):
    """One causal block: proj_in -> AttentionBlock -> raw-input skip.

    Per-block proj_in, learned position embedding and raw-input skip give the
    stack several residual paths for gradient flow at depth.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        seq_len: int,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden_dim) * 1e-2)
        self.attn_block = _AttentionBlock(hidden_dim, n_heads, ff_mult)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x_in = x
        x = self.proj_in(x) + self.pos_embed.unsqueeze(0)
        x = self.attn_block(x, attn_mask)
        return x + x_in


class CausalStack(nn.Module):
    """Stack of n_layers _CausalBlocks with inclusive causal masking.

    One direction only; LeTFRateMatrix holds two instances and does the bwd
    flip-process-flip itself. Inclusive causal (mask j > i): position k
    attends layer-input positions 0..k, which with the readout's slice gives
    hollow output at every k (module docstring).

    Args:
        hidden_dim: channel dimension for embeddings and Transformer hidden states.
        n_layers: number of _CausalBlocks in the stack.
        n_heads: number of attention heads. Must divide hidden_dim.
        seq_len: sequence length, including the prepended cond_t token.
        ff_mult: feed-forward expansion factor (Vaswani 2017 default = 4).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        ff_mult: int = 4,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )
        self.blocks = nn.ModuleList(
            [
                _CausalBlock(hidden_dim, n_heads, seq_len, ff_mult)
                for _ in range(n_layers)
            ]
        )

    def _cached_causal_mask(self, T: int, device) -> Tensor:
        """Inclusive-causal mask, rebuilt only on shape/device change.

        Plain attribute, not a registered buffer: it must stay out of
        state_dict so checkpoints keep their exact key set.
        """
        mask = getattr(self, "_causal_mask", None)
        if mask is None or mask.shape[0] != T or mask.device != device:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
            )
            self._causal_mask = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        # nn.MultiheadAttention attn_mask: True = excluded.
        # Inclusive causal: mask j > i (upper triangle excluding diag).
        mask = self._cached_causal_mask(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask)
        return x


class AttentionReadout(nn.Module):
    """Hollow attention readout via slice-and-mask (App B.3).

    Inputs: fwd_x, bwd_x both (B, 1+d, h); cond_t (B, 1, h).
        sliced_fwd = fwd_x[:, :-1, :]   # (B, d, h)  hollow at every position k
        sliced_bwd = bwd_x[:, 1:,  :]   # (B, d, h)  hollow at every position k
        combined   = (sliced_fwd + sliced_bwd) / sqrt(2) + cond_t
        Q  = q_proj(LN(combined))                                  # (B, d, h)
        K  = k_proj(LN(cat([sliced_fwd, sliced_bwd], dim=1) + cond_t))   # (B, 2d, h)
        V  = v_proj(LN(cat([sliced_fwd, sliced_bwd], dim=1) + cond_t))
        scores = Q K^T / sqrt(d_k)                                  # (B, h_n, d, 2d)
        joint mask: M_L allows L-keys j <= i (lower triangle inclusive);
                    M_R allows R-keys j >= i (upper triangle inclusive).
        H = (combined + softmax(scores) V_proj) + FFN(LN(...))      # (B, d, h)

    The cond_t triple-injection (combined + all-keys + sliced inputs) and the
    per-head readout position embeddings are independent of x_i, so
    hollowness holds.

    `use_sdpa` (default off) routes the attention through
    F.scaled_dot_product_attention, which never materialises the
    (B, n_heads, d, 2d) score buffer that forces small chunks at large d. Its
    reduction order differs from the manual branch (fp32-tolerance, not
    bit-exact); no new parameters, so checkpoints work under either flag.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        data_dim: int,
        ff_mult: int = 4,
        use_sdpa: bool = False,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.data_dim = data_dim
        self.d_k = hidden_dim // n_heads
        self.use_sdpa = use_sdpa
        self.pos_embed = nn.Parameter(torch.randn(data_dim, self.d_k) * 1e-2)

        self.norm_in = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def _cached_joint_mask(self, d: int, device) -> Tensor:
        """(d, 2d) joint score mask, rebuilt only on shape/device change.

        Score-mask convention: True = mask out (-inf).
        M_L allows L-keys j <= i, so mask j > i (upper triangle excl diag).
        M_R allows R-keys j >= i, so mask j < i (lower triangle excl diag).
        Plain attribute, not a registered buffer (kept out of state_dict).
        """
        mask = getattr(self, "_joint_mask", None)
        if mask is None or mask.shape[0] != d or mask.device != device:
            i_idx = torch.arange(d, device=device).unsqueeze(1)
            j_idx = torch.arange(d, device=device).unsqueeze(0)
            mask = torch.cat([j_idx > i_idx, j_idx < i_idx], dim=-1)
            self._joint_mask = mask
        return mask

    def forward(self, fwd_x: Tensor, bwd_x: Tensor, cond_t: Tensor) -> Tensor:
        sliced_fwd = fwd_x[:, :-1, :]  # (B, d, h)
        sliced_bwd = bwd_x[:, 1:, :]  # (B, d, h)

        combined = (sliced_fwd + sliced_bwd) / math.sqrt(2) + cond_t  # (B, d, h)
        all_keys = torch.cat([sliced_fwd, sliced_bwd], dim=1) + cond_t  # (B, 2d, h)

        Q = self.q_proj(self.norm_in(combined))  # (B, d, h)
        kv_norm = self.norm_in(all_keys)
        K = self.k_proj(kv_norm)  # (B, 2d, h)
        V = self.v_proj(kv_norm)  # (B, 2d, h)

        B, d, _ = Q.shape

        def split_heads(t: Tensor, T: int) -> Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q, d)  # (B, n_heads, d, d_k)
        K = split_heads(K, 2 * d)  # (B, n_heads, 2d, d_k)
        V = split_heads(V, 2 * d)  # (B, n_heads, 2d, d_k)

        pos = self.pos_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, d, d_k)
        Q = Q + pos
        K = torch.cat([K[:, :, :d, :] + pos, K[:, :, d:, :] + pos], dim=2)

        joint_mask = self._cached_joint_mask(d, Q.device)  # (d, 2d)

        if self.use_sdpa:
            # SDPA's default scale 1/sqrt(d_k) matches the manual branch; its
            # bool mask is True = attend, hence the negation.
            out = nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=~joint_mask
            )  # (B, n_heads, d, d_k)
        else:
            scale = math.sqrt(self.d_k)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / scale  # (B, n_heads, d, 2d)
            scores = scores.masked_fill(joint_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)  # (B, n_heads, d, 2d)
            out = torch.matmul(attn, V)  # (B, n_heads, d, d_k)
        out = out.transpose(1, 2).contiguous().view(B, d, self.hidden_dim)
        out = self.out_proj(out)  # (B, d, h)

        # `combined` and `out` are hollow at every position; the per-position
        # FF preserves that.
        h = combined + out
        h = h + self.ff(self.norm_ff(h))
        return h


class LeTFRateMatrix(nn.Module):
    """Locally Equivariant Transformer rate matrix (DNFS Eq. 22).

        G(tau, i | x) = (omega_tau - omega_{x_i})^T H_HTF(x)_{i,:}

    Args:
        d: number of sites (D*D for a D x D Ising lattice).
        vocab_size: number of token states S (2 for binary Ising).
        hidden_dim: width of all embeddings and Transformer hidden states.
        n_layers: depth of each directional CausalStack (App. E.1.1
            "3 bidirectional causal attention layers" -> n_layers=3).
        n_heads: attention heads per block. Must divide hidden_dim.
        use_sdpa_readout: fused-kernel readout attention (AttentionReadout
            docstring). Default off.
        condition_on_composition: take the target composition c as a second
            conditioning scalar, so one network serves the whole F(c) curve.
            Default off, which keeps parameter construction (RNG consumption,
            every archived checkpoint) bit-identical to the unconditioned
            model.

    Composition conditioning:

        cond = TimestepEmbedder(t) + CompositionEmbedder(c)      # (B, 1, h)

    c ∈ [0, 1] is a smooth scalar like t, so a second `TimestepEmbedder`
    serves; it is summed into the conditioning token rather than prepended,
    keeping the single token at position 0 that `fwd_x[:, :-1]` /
    `bwd_x[:, 1:]`, `AttentionReadout.pos_embed` (sized `data_dim`) and the
    (d, 2d) joint mask assume. c is independent of x, so hollowness holds.
    """

    is_locally_equivariant: bool = True

    def __init__(
        self,
        d: int,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int = 4,
        use_sdpa_readout: bool = False,
        condition_on_composition: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )
        self.d = d
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        nn.init.kaiming_uniform_(self.token_embedder.weight, a=math.sqrt(5))
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.condition_on_composition = condition_on_composition
        if condition_on_composition:
            # Registered here, in its historical position: an optimizer
            # state_dict maps moments by parameter order, so moving it would
            # mis-map an archived conditioned run on resume. Built under a
            # private RNG stream (offset from the run seed) so every shared
            # tensor after it stays same-seed paired to an unconditioned
            # specialist.
            composition_seed = (torch.initial_seed() + 0x5EED_C0DE_51A7) % (2**63)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(composition_seed)
                self.comp_embedder = TimestepEmbedder(hidden_dim)
            # Zero-init the output layer so the channel contributes exactly 0
            # at init and a conditioned model reproduces its same-seed
            # unconditioned twin bit-for-bit.
            nn.init.zeros_(self.comp_embedder.mlp[-1].weight)
            nn.init.zeros_(self.comp_embedder.mlp[-1].bias)
        seq_len = 1 + d
        self.fwd_stack = CausalStack(hidden_dim, n_layers, n_heads, seq_len)
        self.bwd_stack = CausalStack(hidden_dim, n_layers, n_heads, seq_len)
        self.attention_readout = AttentionReadout(
            hidden_dim, n_heads, d, use_sdpa=use_sdpa_readout
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.omega = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.omega.weight, std=0.002)

    def _conditioning(self, t: Tensor, c: Tensor | None) -> Tensor:
        """Conditioning token, shape (B, 1, h): prepended and readout-injected.

        TimestepEmbedder(t), plus the composition embedding when conditioned.

        Args:
            t: (B,) diffusion time in [0, 1].
            c: (B,) target composition in [0, 1], or None. Supplied iff the
                model was built with `condition_on_composition=True`; a
                mismatch raises, since a silently ignored c would train an
                "amortised" model that never saw its constraint.
        """
        if not self.condition_on_composition:
            if c is not None:
                raise ValueError(
                    "composition c was supplied but this model was built "
                    "with condition_on_composition=False; it would be "
                    "silently ignored."
                )
            return self.time_embedder(t).unsqueeze(1)

        if c is None:
            raise ValueError(
                "this model was built with condition_on_composition=True, "
                "so the target composition c must be supplied; running "
                "without it would train an 'amortised' model that never "
                "saw its constraint."
            )
        # `timestep_embedding` spaces frequencies from 1 down to 1/max_period,
        # so on [0, 1] the basis is near-linear in c (sin(cf) ≈ cf), biasing
        # the channel toward smooth behaviour in c. If it underfits, rescale
        # c before embedding first.
        return self.time_embedder(t).unsqueeze(1) + self.comp_embedder(c).unsqueeze(1)

    def compute_body(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        """Pre-readout body H_HTF(x), shape (B, d, hidden_dim). Hollow at every site.

        Accepts +-1 float spins (training) or 0/1 Long indices (tests).
        `c` is the target composition; see `_conditioning`.
        """
        x_idx = ((x + 1) / 2).long()
        x_emb = self.token_embedder(x_idx)  # (B, d, h)
        cond_t = self._conditioning(t, c)  # (B, 1, h)

        fwd_in = torch.cat([cond_t, x_emb], dim=1)  # (B, 1+d, h)
        fwd_x = self.fwd_stack(fwd_in)  # (B, 1+d, h)

        bwd_in = torch.cat([cond_t, x_emb.flip(1)], dim=1)
        bwd_x = self.bwd_stack(bwd_in).flip(1)  # (B, 1+d, h)

        return self.attention_readout(fwd_x, bwd_x, cond_t)

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        H = self.compute_body(x, t, c)
        H = self.output_norm(H) + self._conditioning(t, c)
        x_idx = ((x + 1) / 2).long()
        omega_all = self.omega.weight  # (S, h)
        omega_xi = self.omega(x_idx)  # (B, d, h)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]  # (B, d, S, h)
        G = torch.einsum("bdh,bdsh->bds", H, diff)  # (B, d, S)
        G = G.scatter(-1, x_idx.unsqueeze(-1), 0.0)
        return G
