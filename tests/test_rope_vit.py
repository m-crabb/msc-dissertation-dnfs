"""Falsification tests for the periodic-RoPE / patch-key backbone
(`models/rope_vit.py`), written BEFORE the module body.

What the backbone claims, stated as tests:

  * RoPE primitive: a rotated dot product depends on the SIGNED lattice
    offset (delta_row, delta_col) mod L and on nothing else -- the torus wraps
    exactly because every phase frequency is an integer multiple of 2 pi / L.
    Signed, not min-image: +delta and -delta must differ (the swap readout is
    antisymmetric in the pair labels, a reflection-symmetric position code
    would fight it).
  * Patch keys: the attention layer's gathered-window + pooled-patch assembly
    equals the obvious dense formulation (same rotated keys, same mask rule),
    at patch sizes 1 and 2. p = 1 is dense causal attention in two pieces.
  * Head contract: the causal stacks keep the leTF slice-trick blindness, so
    `causal_stream_summaries` stays exact (prefix_i blind to x_{>= i},
    suffix_j blind to x_{<= j}) and every existing swap head can sit on top.
  * Equivariance: with the causal mask removed, H(roll x) = roll H(x) exactly
    for every torus shift at p = 1, and for shifts in p Z^2 at p = 2 (patch
    pooling aliases sub-patch shifts -- a ViT property, pinned rather than
    hidden). Input-masking commutes with the roll, so a both-holes-masked
    pair oracle on the bidirectional body is exactly pair-equivariant:
    G(T_v x)[i+v, j+v] = G(x)[i, j].
  * The CAUSAL streams are NOT equivariant (the raster prefix set is not
    shift-covariant); nothing here claims otherwise.
  * fimo2 (factorised + prefix band + row/col orderings) on this backbone is
    doubly blind and exactly swap-antisymmetric at 4x4 and 8x8, the swap
    Kolmogorov loss is finite and differentiable, and the backbone carries no
    absolute position parameter.
"""

import math

import pytest
import torch

from discrete_flow_sampler.constraints.factorised_swap_head import (
    FactorisedSwapHead,
)
from discrete_flow_sampler.constraints.interval_swap_head import (
    causal_stream_summaries,
)
from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.models.rope_vit import (
    RoPELatticeAttention,
    RoPEStack,
    RoPEViTRateMatrix,
    apply_rope,
    periodic_rope_angles,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import loss_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

ATOL = 1e-5


def _backbone(lattice_side, patch_size=1, hidden_dim=8, n_heads=2, n_layers=2, seed=0):
    torch.manual_seed(seed)
    return RoPEViTRateMatrix(
        d=lattice_side * lattice_side, vocab_size=2, hidden_dim=hidden_dim,
        n_layers=n_layers, n_heads=n_heads, patch_size=patch_size,
    ).eval()


def _fimo2(backbone, lattice_side, seed=1):
    torch.manual_seed(seed)
    return FactorisedSwapHead(
        backbone, bilinear_rank=3, factor_dim=4, global_feature_dim=6,
        position_dim=5, interior_band="prefix",
        site_orderings=("row", "col"), lattice_side=lattice_side,
    ).eval()


def _state(d, batch=1, seed=1):
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    x[:, 0] = 1.0
    x[:, 1] = -1.0  # both spins present in every row
    return x


def _roll_sites(x, lattice_side, shift):
    """Torus shift of a flattened (B, d, ...) site tensor."""
    grid = x.reshape(x.shape[0], lattice_side, lattice_side, *x.shape[2:])
    rolled = torch.roll(grid, shifts=shift, dims=(1, 2))
    return rolled.reshape(x.shape)


def _site(lattice_side, row, col):
    return (row % lattice_side) * lattice_side + (col % lattice_side)


# ---------------------------------------------------------------- primitive

def test_rope_dot_depends_only_on_signed_periodic_offset():
    lattice_side, head_dim = 8, 8
    angles = periodic_rope_angles(lattice_side, head_dim, torch.device("cpu"))
    assert angles.shape == (lattice_side * lattice_side, head_dim // 2)
    torch.manual_seed(0)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    def dot(query_site, key_site):
        rq = apply_rope(q.view(1, 1, head_dim), angles[query_site].view(1, 1, -1))
        rk = apply_rope(k.view(1, 1, head_dim), angles[key_site].view(1, 1, -1))
        return (rq * rk).sum().item()

    # same signed offset (+1, +2), two places -- one of them wrapping the torus
    a = dot(_site(lattice_side, 1, 1), _site(lattice_side, 2, 3))
    b = dot(_site(lattice_side, 7, 6), _site(lattice_side, 8, 8))
    assert abs(a - b) < 1e-5, "torus wrap is not exact"
    # opposite offset must differ: signed, never min-image
    c = dot(_site(lattice_side, 2, 3), _site(lattice_side, 1, 1))
    assert abs(a - c) > 1e-3, "offset sign lost (min-image folding)"
    # zero offset == unrotated dot
    assert abs(dot(5, 5) - (q * k).sum().item()) < 1e-5


def test_rope_angles_are_integer_multiples_of_2pi_over_L():
    lattice_side, head_dim = 4, 8
    angles = periodic_rope_angles(lattice_side, head_dim, torch.device("cpu"))
    multiples = angles * lattice_side / (2 * math.pi)
    assert torch.allclose(multiples, multiples.round(), atol=1e-5)


# -------------------------------------------------------- layer vs dense ref

def _dense_reference(layer: RoPELatticeAttention, x: torch.Tensor) -> torch.Tensor:
    """The obvious dense formulation the gathered implementation must equal:
    every site query scores the cond key unrotated, every site key rotated
    and allowed iff (same patch-row band and position <= query) [causal] or
    (same band) [bidirectional], and every patch key -- the MEAN of its
    members' rotated keys -- allowed iff its band is strictly earlier
    [causal] or different [bidirectional]. One softmax over all of them."""
    batch, seq_len, hidden = x.shape
    d = seq_len - 1
    heads, head_dim = layer.n_heads, hidden // layer.n_heads
    L, p = layer.lattice_side, layer.patch_size
    band_rows = p * L

    def heads_of(t):
        return t.view(batch, seq_len, heads, head_dim).transpose(1, 2)

    q, k, v = heads_of(layer.q_proj(x)), heads_of(layer.k_proj(x)), heads_of(layer.v_proj(x))
    angles = layer.site_angles.view(1, 1, d, -1)
    q_sites = apply_rope(q[:, :, 1:], angles)
    k_sites = apply_rope(k[:, :, 1:], angles)
    pos = torch.arange(d)
    band = pos // band_rows
    if layer.causal:
        site_allowed = (band[:, None] == band[None, :]) & (pos[None, :] <= pos[:, None])
    else:
        site_allowed = band[:, None] == band[None, :]
    patch_of = layer.patch_of_position
    n_patches = int(patch_of.max()) + 1
    pool = torch.zeros(n_patches, d)
    pool[patch_of, pos] = 1.0
    pool = pool / pool.sum(1, keepdim=True)
    k_patches = torch.einsum("nd,bhdk->bhnk", pool, k_sites)
    v_patches = torch.einsum("nd,bhdk->bhnk", pool, v[:, :, 1:])
    patch_band = torch.zeros(n_patches, dtype=torch.long)
    patch_band[patch_of] = band
    if layer.causal:
        patch_allowed = patch_band[None, :] < band[:, None]
    else:
        patch_allowed = patch_band[None, :] != band[:, None]

    scale = 1.0 / math.sqrt(head_dim)
    cond_logit = (q[:, :, 1:] * k[:, :, :1]).sum(-1, keepdim=True) * scale
    site_logits = torch.einsum("bhik,bhjk->bhij", q_sites, k_sites) * scale
    site_logits = site_logits.masked_fill(~site_allowed, float("-inf"))
    patch_logits = torch.einsum("bhik,bhnk->bhin", q_sites, k_patches) * scale
    patch_logits = patch_logits.masked_fill(~patch_allowed, float("-inf"))
    logits = torch.cat([cond_logit, site_logits, patch_logits], dim=-1)
    weights = torch.softmax(logits, dim=-1)
    values = torch.cat([v[:, :, :1], v[:, :, 1:], v_patches], dim=2)
    out_sites = torch.einsum("bhij,bhjk->bhik", weights, values)
    out = torch.cat([v[:, :, :1], out_sites], dim=2)  # cond attends itself only
    out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
    return layer.out_proj(out)


@pytest.mark.parametrize("lattice_side", [4, 8])
@pytest.mark.parametrize("patch_size", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_layer_matches_dense_reference(lattice_side, patch_size, causal):
    torch.manual_seed(3)
    layer = RoPELatticeAttention(
        hidden_dim=8, n_heads=2, lattice_side=lattice_side,
        patch_size=patch_size, causal=causal,
    )
    x = torch.randn(2, 1 + lattice_side ** 2, 8)
    assert (layer(x) - _dense_reference(layer, x)).abs().max().item() < 1e-5


# ------------------------------------------------------------- head contract

@pytest.mark.parametrize("lattice_side", [4, 8])
@pytest.mark.parametrize("patch_size", [1, 2])
@torch.no_grad()
def test_causal_streams_blind_exactly_as_letf(lattice_side, patch_size):
    backbone = _backbone(lattice_side, patch_size)
    d = lattice_side ** 2
    x = _state(d)
    t = torch.rand(1)
    prefix, suffix = causal_stream_summaries(backbone, x, t)
    assert prefix.shape == suffix.shape == (1, d, backbone.hidden_dim)
    for site in range(d):
        flipped = x.clone()
        flipped[0, site] *= -1
        prefix_f, suffix_f = causal_stream_summaries(backbone, flipped, t)
        assert (prefix_f[:, : site + 1] - prefix[:, : site + 1]).abs().max() < ATOL, (
            f"prefix leaks x_{site}"
        )
        assert (suffix_f[:, site:] - suffix[:, site:]).abs().max() < ATOL, (
            f"suffix leaks x_{site}"
        )
        if site < d - 1:
            assert (prefix_f[:, site + 1:] - prefix[:, site + 1:]).abs().max() > 1e-7, (
                "prefix stream is inert -- blindness test has no teeth"
            )


# ------------------------------------------------------------- equivariance

def _bidirectional_body(backbone, stack, x, t, mask_sites=()):
    """Non-causal body through `stack` (a RoPEStack(causal=False) sharing the
    backbone's width), with the embeddings at `mask_sites` zeroed."""
    x_emb = backbone.token_embedder(((x + 1) / 2).long()).clone()
    for site in mask_sites:
        x_emb[:, site] = 0.0
    cond = backbone.time_embedder(t).unsqueeze(1)
    return stack(torch.cat([cond, x_emb], dim=1))[:, 1:]


def _all_shifts(lattice_side, step=1):
    return [(r, c) for r in range(0, lattice_side, step) for c in range(0, lattice_side, step)]


@pytest.mark.parametrize("lattice_side", [4, 8])
@torch.no_grad()
def test_bidirectional_body_equivariant_under_every_torus_shift(lattice_side):
    backbone = _backbone(lattice_side, patch_size=1)
    torch.manual_seed(5)
    stack = RoPEStack(8, 2, 2, lattice_side, patch_size=1, causal=False).eval()
    x = _state(lattice_side ** 2)
    t = torch.rand(1)
    body = _bidirectional_body(backbone, stack, x, t)
    for shift in _all_shifts(lattice_side):
        shifted = _bidirectional_body(backbone, stack, _roll_sites(x, lattice_side, shift), t)
        drift = (shifted - _roll_sites(body, lattice_side, shift)).abs().max().item()
        assert drift < ATOL, f"shift {shift}: {drift:.2e}"


@pytest.mark.parametrize("lattice_side", [4, 8])
@torch.no_grad()
def test_patch_two_body_equivariant_under_patch_multiple_shifts_only(lattice_side):
    backbone = _backbone(lattice_side, patch_size=2)
    torch.manual_seed(5)
    stack = RoPEStack(8, 2, 2, lattice_side, patch_size=2, causal=False).eval()
    x = _state(lattice_side ** 2)
    t = torch.rand(1)
    body = _bidirectional_body(backbone, stack, x, t)
    for shift in _all_shifts(lattice_side, step=2):
        shifted = _bidirectional_body(backbone, stack, _roll_sites(x, lattice_side, shift), t)
        drift = (shifted - _roll_sites(body, lattice_side, shift)).abs().max().item()
        assert drift < ATOL, f"shift {shift}: {drift:.2e}"
    # Pinned aliasing: a one-site shift re-partitions the patches.
    shifted = _bidirectional_body(backbone, stack, _roll_sites(x, lattice_side, (0, 1)), t)
    assert (shifted - _roll_sites(body, lattice_side, (0, 1))).abs().max().item() > 1e-4


@torch.no_grad()
def test_masked_pair_oracle_is_exactly_pair_equivariant():
    """Both holes zeroed at the INPUT, bidirectional body read at j, readout
    against omega_i - omega_j: the O(d^2)-pass oracle. Masking commutes with
    the roll, so the pair score field shifts with the lattice."""
    lattice_side = 4
    d = lattice_side ** 2
    backbone = _backbone(lattice_side, patch_size=1)
    torch.manual_seed(5)
    stack = RoPEStack(8, 2, 2, lattice_side, patch_size=1, causal=False).eval()
    t = torch.rand(1)

    def pair_scores(x):
        omega = backbone.omega(((x + 1) / 2).long())
        G = torch.zeros(d, d)
        for i in range(d):
            for j in range(d):
                if i != j:
                    H = _bidirectional_body(backbone, stack, x, t, (i, j))[0, j]
                    G[i, j] = (H * (omega[0, i] - omega[0, j])).sum()
        return G

    x = _state(d)
    G = pair_scores(x)
    for shift in [(1, 0), (0, 1), (2, 3), (3, 3)]:
        G_shifted = pair_scores(_roll_sites(x, lattice_side, shift))
        for i in range(d):
            for j in range(d):
                r_i, c_i = divmod(i, lattice_side)
                r_j, c_j = divmod(j, lattice_side)
                i_v = _site(lattice_side, r_i + shift[0], c_i + shift[1])
                j_v = _site(lattice_side, r_j + shift[0], c_j + shift[1])
                assert abs(G_shifted[i_v, j_v] - G[i, j]) < ATOL, (shift, i, j)


# ------------------------------------------------------- fimo2 on the backbone

@pytest.mark.parametrize("lattice_side", [4, 8])
@pytest.mark.parametrize("patch_size", [1, 2])
@torch.no_grad()
def test_fimo2_on_rope_backbone_blind_and_antisymmetric(lattice_side, patch_size):
    d = lattice_side ** 2
    backbone = _backbone(lattice_side, patch_size)
    head = _fimo2(backbone, lattice_side)
    x = _state(d)
    t = torch.rand(1)
    H = head.compute_pair_context(x, t)
    G = head(x, t)
    assert G.shape == (1, d, d)
    assert torch.isfinite(G).all()
    assert (G + G.transpose(1, 2)).abs().max().item() == 0.0
    torch.manual_seed(11)
    pairs = [(0, d - 1), (0, 1), (d - 2, d - 1), (lattice_side, 2 * lattice_side + 1)]
    pairs += [tuple(sorted(torch.randperm(d)[:2].tolist())) for _ in range(12)]
    for i, j in pairs:
        for sites in ((i,), (j,), (i, j)):
            flipped = x.clone()
            for s in sites:
                flipped[0, s] *= -1
            drift = (head.compute_pair_context(flipped, t)[:, i, j] - H[:, i, j]).abs().max().item()
            assert drift < ATOL, f"H[{i},{j}] leaks {sites}: {drift:.2e}"
        if x[0, i] != x[0, j]:
            G_swapped = head(swap2(x, i, j), t)
            assert abs(G[0, i, j] + G_swapped[0, i, j]) < ATOL, (i, j)


def test_swap_kolmogorov_loss_finite_and_trains_backbone():
    lattice_side = 4
    backbone = _backbone(lattice_side, patch_size=2).train()
    head = _fimo2(backbone, lattice_side).train()
    target = FixedCompositionIsingTarget(D=lattice_side, sigma=0.223, target_composition=0.5)
    x = _state(16, batch=4)
    t = torch.rand(x.shape[0])
    loss = loss_swap(x, t, 0.0, head, target)
    assert torch.isfinite(loss)
    loss.backward()
    stack_grads = [p.grad for p in backbone.fwd_stack.parameters() if p.grad is not None]
    assert stack_grads and any(g.abs().sum() > 0 for g in stack_grads)


# ------------------------------------------------------------ shapes / API

@pytest.mark.parametrize("lattice_side", [4, 8])
@pytest.mark.parametrize("patch_size", [1, 2])
@torch.no_grad()
def test_flip_forward_shapes_and_hollow_diagonal(lattice_side, patch_size):
    d = lattice_side ** 2
    backbone = _backbone(lattice_side, patch_size)
    x = _state(d, batch=3)
    t = torch.rand(3)
    G = backbone(x, t)
    assert G.shape == (3, d, 2)
    assert torch.isfinite(G).all()
    x_idx = ((x + 1) / 2).long()
    assert (G.gather(-1, x_idx.unsqueeze(-1)) == 0).all()
    assert backbone.compute_body(x, t).shape == (3, d, backbone.hidden_dim)


def test_backbone_has_no_absolute_position_parameters_and_same_interface():
    backbone = _backbone(4, patch_size=2)
    assert not [n for n, _ in backbone.named_parameters() if "pos_embed" in n]
    for attribute in ("d", "hidden_dim", "vocab_size", "token_embedder", "time_embedder",
                      "fwd_stack", "bwd_stack", "attention_readout", "output_norm", "omega",
                      "is_locally_equivariant"):
        assert hasattr(backbone, attribute), attribute
    with pytest.raises(ValueError):
        RoPEViTRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2, patch_size=3)
    with pytest.raises(ValueError):
        RoPEViTRateMatrix(d=12, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)


# ------------------------------------------------------------- config wire

@pytest.mark.parametrize("patch_size", [1, 2])
def test_rope_cells_mirror_fimo2_rung_except_backbone(patch_size):
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    name = f"H2_d64_c50_s223_rope{patch_size}_fimo2_50k_curr"
    cell = CONFIGS[name]
    twin = CONFIGS["H2_d64_c50_s223_letf_fimo2_50k_curr"]
    assert cell.model.kind == "rope_vit" and cell.model.patch_size == patch_size
    assert replace(cell, name=twin.name, model=twin.model) == twin
    _, head = build_target_and_head(cell, torch.device("cpu"))
    assert isinstance(head.backbone, RoPEViTRateMatrix)
    assert head.backbone.patch_size == patch_size
