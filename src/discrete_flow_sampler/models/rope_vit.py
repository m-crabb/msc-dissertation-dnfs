"""Periodic-RoPE, patch-key leTF backbone with torus position encoding.

`letf.py` carries free absolute position embeddings (one learned vector per
site), so the network must learn from d independent vectors that site 0 and
site d-L are bonded across the wrap. Here there are no position parameters:
positions enter through rotary embeddings whose phases are integer multiples
of 2 pi / L, so every attention logit is a function of the signed lattice
offset (delta_row, delta_col) mod L only --

    q_s . k_s' -> R(theta_s)^T q_s . R(theta_s') k_s' = q_s . R(theta_s' - theta_s) k_s',
    theta_s = 2 pi m (row_s, col_s) / L,   m integer,                          (1)

invariant under (row, col) -> (row + v) mod L for every integer v. Signed
offsets, not min-image distance: folding +delta onto -delta would impose a
reflection symmetry that fights the i < j antisymmetric readout.

(1) makes a bidirectional stack exactly torus-translation-equivariant,
H(roll x) = roll H(x), and a both-holes-masked pair oracle on it exactly
pair-equivariant (tests pin both). The heads in `constraints/` take their
blindness from the raster causal sweep, whose prefix set {x_<k} is not
shift-covariant, so the causal streams are not equivariant; they gain a
relative periodic position code and nothing more.

Patch keys: p x p ViT patches applied naively to the causal sweep break the
slice trick (a patch before patch(k) in patch-raster order can contain sites
after k in site-raster order). The compatible form keeps one query per site
and splits the keys into

    near  : individual sites in the query's own patch-row band (the p rows of
            sites that the query's patch-row spans), position <= query       [causal]
    far   : one pooled key per patch, allowed iff every member site precedes
            the query, i.e. the patch-row is strictly earlier                 [causal]
    cond  : the conditioning token, scored unrotated (a rotated query against
            an unrotated positionless key would reintroduce absolute position).

Pooled patch key = mean over members of the rotated site keys, so the patch
logit is the mean of the member logits and (1) holds term by term; pooled
value = mean of member values. Keys per query fall from d to 1 + pL + d/p^2
(p = 1 is dense causal attention in two pieces, identical up to reduction
order), attention cost per layer from O(d^2) to O(d (pL + d/p^2)). Pooling
aliases shifts that are not multiples of p, so at p > 1 equivariance holds on
p Z^2 only (pinned by test), and the far field is seen at patch resolution.
Blindness is untouched: every far patch lies entirely before the query.

Frequencies: each head has head_dim/2 rotation planes, half on rows and half
on columns; integer multipliers are spread geometrically from 1 to L/2
(Nyquist on the torus), so at the production width (hidden 32, 4 heads,
head_dim 8) each axis gets two planes, m in {1, L/2}. Every multiplier is an
integer, which is the periodicity argument.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from discrete_flow_sampler.models.letf import AttentionReadout, LeTFRateMatrix


def periodic_rope_angles(lattice_side: int, head_dim: int, device) -> Tensor:
    """Per-site rotation angles, (L*L, head_dim // 2): integer-multiple phases
    2 pi m coord / L, first half of the planes on rows, second half on cols."""
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim must be a multiple of 4 (row+col planes), got {head_dim}"
        )
    planes_per_axis = head_dim // 4
    nyquist = max(lattice_side // 2, 1)
    if planes_per_axis == 1:
        multipliers = torch.ones(1)
    else:
        exponents = torch.arange(planes_per_axis) / (planes_per_axis - 1)
        multipliers = (nyquist**exponents).round()
    sites = torch.arange(lattice_side * lattice_side)
    rows = (sites // lattice_side).float()
    cols = (sites % lattice_side).float()
    phase = 2 * math.pi / lattice_side
    row_angles = phase * rows[:, None] * multipliers[None, :]
    col_angles = phase * cols[:, None] * multipliers[None, :]
    return torch.cat([row_angles, col_angles], dim=1).to(device)


def apply_rope(t: Tensor, angles: Tensor) -> Tensor:
    """Rotate the last dim of t (..., head_dim) plane-wise by `angles`
    (..., head_dim // 2): plane k pairs channel k with channel k + head_dim/2."""
    planes = t.shape[-1] // 2
    first, second = t[..., :planes], t[..., planes:]
    cos, sin = angles.cos(), angles.sin()
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class RoPELatticeAttention(nn.Module):
    """Multi-head attention over (B, 1 + d, h) with the key split of the module
    docstring. Position 0 is the conditioning token (attends itself only);
    positions 1..d are sites in raster order, or reversed raster order when
    `reverse` (the bwd stack runs on the flipped sequence)."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        lattice_side: int,
        patch_size: int = 1,
        causal: bool = True,
        reverse: bool = False,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )
        if lattice_side % patch_size != 0:
            raise ValueError(
                f"patch_size {patch_size} must divide lattice_side {lattice_side}"
            )
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.lattice_side = lattice_side
        self.patch_size = patch_size
        self.causal = causal
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        d = lattice_side * lattice_side
        angles = periodic_rope_angles(lattice_side, self.head_dim, "cpu")
        if reverse:
            angles = angles.flip(0)
        # Index arithmetic on positions (the flipped raster is a raster too),
        # all non-persistent: reproducible from the constructor, out of state_dict.
        band_width = patch_size * lattice_side
        position = torch.arange(d)
        band = position // band_width
        patches_per_row = lattice_side // patch_size
        patch_of_position = (
            band * patches_per_row + (position % lattice_side) // patch_size
        )
        window = band[:, None] * band_width + torch.arange(band_width)[None, :]
        window_allowed = (
            window <= position[:, None]
            if causal
            else torch.ones_like(window, dtype=torch.bool)
        )
        patch_band = torch.arange(d // patch_size**2) // patches_per_row
        patch_allowed = (
            patch_band[None, :] < band[:, None]
            if causal
            else patch_band[None, :] != band[:, None]
        )
        patch_members = patch_of_position.argsort(stable=True).view(-1, patch_size**2)
        for name, tensor in (
            ("site_angles", angles),
            ("patch_of_position", patch_of_position),
            ("window_index", window),
            ("window_allowed", window_allowed),
            ("patch_allowed", patch_allowed),
            ("patch_members", patch_members),
        ):
            self.register_buffer(name, tensor, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, hidden = x.shape

        def heads_of(t: Tensor) -> Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = (
            heads_of(self.q_proj(x)),
            heads_of(self.k_proj(x)),
            heads_of(self.v_proj(x)),
        )
        angles = self.site_angles.view(1, 1, -1, self.head_dim // 2)
        q_sites = apply_rope(q[:, :, 1:], angles)  # (B, H, d, dk)
        k_sites = apply_rope(k[:, :, 1:], angles)
        v_sites = v[:, :, 1:]
        scale = 1.0 / math.sqrt(self.head_dim)

        cond_logit = (q[:, :, 1:] * k[:, :, :1]).sum(
            -1, keepdim=True
        ) * scale  # (B, H, d, 1)
        near_k = k_sites[:, :, self.window_index]  # (B, H, d, pL, dk)
        near_v = v_sites[:, :, self.window_index]
        near_logit = torch.einsum("bhik,bhiwk->bhiw", q_sites, near_k) * scale
        near_logit = near_logit.masked_fill(~self.window_allowed, float("-inf"))
        far_k = k_sites[:, :, self.patch_members].mean(dim=3)  # (B, H, N, dk)
        far_v = v_sites[:, :, self.patch_members].mean(dim=3)
        far_logit = torch.einsum("bhik,bhnk->bhin", q_sites, far_k) * scale
        far_logit = far_logit.masked_fill(~self.patch_allowed, float("-inf"))

        weights = torch.softmax(
            torch.cat([cond_logit, near_logit, far_logit], dim=-1), dim=-1
        )
        n_near = near_logit.shape[-1]
        out_sites = (
            weights[..., :1] * v[:, :, :1]
            + torch.einsum("bhiw,bhiwk->bhik", weights[..., 1 : 1 + n_near], near_v)
            + torch.einsum("bhin,bhnk->bhik", weights[..., 1 + n_near :], far_v)
        )
        out = torch.cat([v[:, :, :1], out_sites], dim=2)  # cond attends itself
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq_len, hidden))


class _RoPEBlock(nn.Module):
    """`letf._CausalBlock` minus its absolute position table: proj_in ->
    pre-norm attention + residual -> FFN + residual -> raw-input skip."""

    def __init__(
        self, hidden_dim, n_heads, lattice_side, patch_size, causal, reverse, ff_mult=4
    ):
        super().__init__()
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = RoPELatticeAttention(
            hidden_dim, n_heads, lattice_side, patch_size, causal, reverse
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        x = self.proj_in(x)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x + x_in


class RoPEStack(nn.Module):
    """Drop-in for `letf.CausalStack`: (B, 1 + d, h) -> (B, 1 + d, h).
    `causal=True, reverse=False/True` are the fwd/bwd stacks the heads slice;
    `causal=False` is the bidirectional body the equivariance tests exercise."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        lattice_side: int,
        patch_size: int = 1,
        causal: bool = True,
        reverse: bool = False,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _RoPEBlock(
                    hidden_dim,
                    n_heads,
                    lattice_side,
                    patch_size,
                    causal,
                    reverse,
                    ff_mult,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class RoPEAttentionReadout(AttentionReadout):
    """leTF's slice-and-mask readout with its per-head absolute position table
    replaced by the site rotation (1). Same joint (d, 2d) mask, same hollow
    argument: the rotation is a fixed function of the site index, not of x."""

    def __init__(self, hidden_dim, n_heads, lattice_side, use_sdpa=False):
        super().__init__(
            hidden_dim, n_heads, lattice_side * lattice_side, use_sdpa=use_sdpa
        )
        del self.pos_embed
        angles = periodic_rope_angles(lattice_side, self.d_k, "cpu")
        self.register_buffer("site_angles", angles, persistent=False)

    def forward(self, fwd_x: Tensor, bwd_x: Tensor, cond_t: Tensor) -> Tensor:
        sliced_fwd = fwd_x[:, :-1, :]
        sliced_bwd = bwd_x[:, 1:, :]
        combined = (sliced_fwd + sliced_bwd) / math.sqrt(2) + cond_t
        all_keys = torch.cat([sliced_fwd, sliced_bwd], dim=1) + cond_t
        Q = self.q_proj(self.norm_in(combined))
        kv_norm = self.norm_in(all_keys)
        K, V = self.k_proj(kv_norm), self.v_proj(kv_norm)
        B, d, _ = Q.shape

        def split_heads(t: Tensor, T: int) -> Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        angles = self.site_angles.view(1, 1, d, -1)
        Q = apply_rope(split_heads(Q, d), angles)
        K = split_heads(K, 2 * d)
        K = torch.cat(
            [apply_rope(K[:, :, :d], angles), apply_rope(K[:, :, d:], angles)], dim=2
        )
        V = split_heads(V, 2 * d)
        joint_mask = self._cached_joint_mask(d, Q.device)
        if self.use_sdpa:
            out = nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=~joint_mask
            )
        else:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
            out = torch.matmul(
                torch.softmax(scores.masked_fill(joint_mask, float("-inf")), dim=-1), V
            )
        out = self.out_proj(
            out.transpose(1, 2).contiguous().view(B, d, self.hidden_dim)
        )
        h = combined + out
        return h + self.ff(self.norm_ff(h))


class RoPEViTRateMatrix(LeTFRateMatrix):
    """`LeTFRateMatrix` with the two causal stacks and the readout swapped for
    their periodic-RoPE / patch-key versions. Identical public interface
    (constructor, `compute_body`, `forward`, the attributes the swap heads
    reach for: token_embedder, time_embedder, fwd_stack, bwd_stack,
    attention_readout, output_norm, omega, d, hidden_dim) plus `patch_size`.
    `d` must be a square and `patch_size` must divide its side."""

    def __init__(
        self,
        d: int,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int = 4,
        use_sdpa_readout: bool = False,
        condition_on_composition: bool = False,
        patch_size: int = 1,
    ):
        lattice_side = math.isqrt(d)
        if lattice_side * lattice_side != d:
            raise ValueError(f"d={d} is not a square lattice site count")
        super().__init__(
            d,
            vocab_size,
            hidden_dim,
            n_layers,
            n_heads,
            use_sdpa_readout=use_sdpa_readout,
            condition_on_composition=condition_on_composition,
        )
        self.lattice_side = lattice_side
        self.patch_size = patch_size
        self.fwd_stack = RoPEStack(
            hidden_dim, n_layers, n_heads, lattice_side, patch_size, reverse=False
        )
        self.bwd_stack = RoPEStack(
            hidden_dim, n_layers, n_heads, lattice_side, patch_size, reverse=True
        )
        self.attention_readout = RoPEAttentionReadout(
            hidden_dim, n_heads, lattice_side, use_sdpa=use_sdpa_readout
        )
