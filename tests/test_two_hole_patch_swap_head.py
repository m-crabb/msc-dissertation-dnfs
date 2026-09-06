"""Falsification tests for the two-hole patch swap head (locality-based
blindness: hollow local patch with the partner zeroed + multi-scale pooled
context with both holes subtracted + torus-relative pair position).

Written BEFORE the head body. The claims under test, each as a property the
implementation cannot fake:

  * two-hole blindness: the pair context H_ij is invariant to x_i, x_j and
    both together, for EVERY pair of a 4x4 lattice (adjacent, diagonal,
    distance 2, wrap-around) and a probe set at 8x8;
  * the context is not trivially blind: it moves when a neighbour of a hole
    moves AND when a far site moves (the pooled levels reach it);
  * the vectorised assembly equals a slow per-pair reference that zeroes the
    partner in the patch and subtracts the holes from the pooled means by
    hand -- the index arithmetic of the torus scatter is what this pins;
  * exact state-swap antisymmetry and exact index antisymmetry at init;
  * torus translation equivariance of the PAIR output G;
  * the swap Kolmogorov residual averages to zero under the exact p_t on the
    enumerable 2x2 and 4x4 slices when dt_log_Z is exact -- the identity
    the loss relies on, which holds only if the reverse rate read off
    -G is the true reverse rate (antisymmetry + the readout convention).
"""

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
    torus_neighbour_offsets,
)
from discrete_flow_sampler.diagnostics.metrics import enumerate_states
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.samplers._swap_neighbours import (
    gather_pair_scores,
    upper_tri_pairs,
)
from discrete_flow_sampler.samplers.swap_kolmogorov import residual_swap
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

ATOL = 1e-5  # suite bar: pooled-level hole removal is subtractive (ulp residue)


def _head(lattice_side=4, seed=42, hidden_dim=8, patch_radius=1, feature_dim=6):
    torch.manual_seed(seed)
    d = lattice_side * lattice_side
    backbone = LeTFRateMatrix(
        d=d,
        vocab_size=2,
        hidden_dim=hidden_dim,
        n_layers=1,
        n_heads=2,
    )
    head = TwoHolePatchSwapHead(
        backbone,
        lattice_side=lattice_side,
        patch_radius=patch_radius,
        feature_dim=feature_dim,
    )
    head.eval()
    return head


def _state(d=16, seed=1, batch=1):
    torch.manual_seed(seed)
    x = (torch.randint(0, 2, (batch, d)) * 2 - 1).float()
    for b in range(batch):
        if bool((x[b] > 0).all()) or bool((x[b] < 0).all()):
            x[b, 0] *= -1
    return x


def _flip(x, *sites):
    y = x.clone()
    for s in sites:
        y[0, s] *= -1
    return y


def _drift(a, b):
    return (a - b).abs().max().item()


def _all_pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


@torch.no_grad()
def test_pair_context_blind_to_both_holes_all_pairs_4x4():
    """Every pair of the 4x4 torus, every flip pattern of the two holes."""
    patch_radius = 1
    head = _head(lattice_side=4, patch_radius=patch_radius)
    x = _state(16)
    t = torch.rand(1)
    H = head.compute_pair_context(x, t)
    worst = 0.0
    for i, j in _all_pairs(16):
        base = H[:, i, j, :]
        for flipped in (_flip(x, i), _flip(x, j), _flip(x, i, j)):
            drift = _drift(head.compute_pair_context(flipped, t)[:, i, j, :], base)
            worst = max(worst, drift)
            assert drift < ATOL, f"H_[{i},{j}] leaks a hole: {drift:.2e}"
    print(f"4x4 R={patch_radius}: worst hole leak {worst:.2e}")


@torch.no_grad()
def test_pair_context_blind_to_both_holes_8x8_probe():
    """8x8 probe set: adjacent, diagonal, straddling a pooled-level edge,
    wrap-around, and far pairs."""
    head = _head(lattice_side=8, patch_radius=1)
    x = _state(64)
    t = torch.rand(1)
    H = head.compute_pair_context(x, t)
    probe_pairs = [(0, 1), (0, 9), (0, 2), (0, 7), (0, 56), (0, 63), (18, 45), (27, 36)]
    worst = 0.0
    for i, j in probe_pairs:
        base = H[:, i, j, :]
        for flipped in (_flip(x, i), _flip(x, j), _flip(x, i, j)):
            drift = _drift(head.compute_pair_context(flipped, t)[:, i, j, :], base)
            worst = max(worst, drift)
            assert drift < ATOL, f"H_[{i},{j}] leaks a hole: {drift:.2e}"
    print(f"8x8 R=1: worst hole leak {worst:.2e}")


@torch.no_grad()
def test_pair_context_sensitive_near_and_far():
    """Anti-triviality: H_ij must move when a neighbour of a hole moves (patch
    term) and when a site far from both holes moves (pooled levels)."""
    head = _head(lattice_side=8, patch_radius=1)
    x = _state(64)
    t = torch.rand(1)
    i, j = 0, 9  # (0,0) and (1,1): diagonal neighbours
    base = head.compute_pair_context(x, t)[:, i, j, :]
    for site, region in ((1, "neighbour of i"), (36, "far site (4,4)")):
        drift = _drift(head.compute_pair_context(_flip(x, site), t)[:, i, j, :], base)
        assert drift > 1e-7, f"H_[{i},{j}] ignores its {region} (site {site})"


@torch.no_grad()
def test_blindness_probe_has_teeth():
    """Negative control for the probe: the per-site patch feature with NO
    partner zeroing must register a neighbour flip loudly."""
    head = _head(lattice_side=4, patch_radius=1)
    x = _state(16)
    t = torch.rand(1)
    leaky = head.site_patch_features(x, t)[:, 0, :]
    leaky_flipped = head.site_patch_features(_flip(x, 1), t)[:, 0, :]
    assert _drift(leaky_flipped, leaky) > 1e-4


@torch.no_grad()
@pytest.mark.parametrize("lattice_side,patch_radius", [(4, 1), (8, 1), (8, 2), (8, 3)])
def test_vectorised_context_matches_per_pair_reference(lattice_side, patch_radius):
    """The scatter/gather assembly must equal a slow reference that builds
    each pair's context by hand: zero the partner inside the other hole's
    patch, re-run the patch MLP, and subtract both holes from every pooled
    level whose box contains them."""
    head = _head(lattice_side=lattice_side, patch_radius=patch_radius)
    d = lattice_side * lattice_side
    x = _state(d, batch=2)
    t = torch.rand(2)
    H = head.compute_pair_context(x, t)
    for i, j in _all_pairs(d)[:: max(1, d // 8)] + [(0, 1), (0, d - 1)]:
        reference = head.pair_context_reference(x, t, i, j)
        assert _drift(H[:, i, j, :], reference) < ATOL, (i, j)


@torch.no_grad()
@pytest.mark.parametrize("lattice_side,patch_radius", [(4, 1), (8, 1), (8, 2), (8, 3)])
def test_antisymmetric_at_init(lattice_side, patch_radius):
    head = _head(lattice_side=lattice_side, patch_radius=patch_radius)
    d = lattice_side * lattice_side
    x = _state(d)
    t = torch.rand(1)
    G = head(x, t)
    pairs = [(i, j) for i, j in _all_pairs(d) if x[0, i] != x[0, j]]
    worst = 0.0
    for i, j in pairs[:: max(1, len(pairs) // 60)]:
        G_swapped = head(swap2(x, i, j), t)
        worst = max(worst, (G[0, i, j] + G_swapped[0, i, j]).abs().item())
    print(f"{lattice_side}x{lattice_side}: antisymmetry residual {worst:.2e}")
    assert worst < ATOL


@torch.no_grad()
def test_trivial_swap_vanishes_and_index_antisymmetry_exact():
    head = _head(lattice_side=4)
    x = _state(16)
    G = head(x, torch.rand(1))
    same = [(i, j) for i, j in _all_pairs(16) if x[0, i] == x[0, j]]
    assert max(G[0, i, j].abs().item() for i, j in same) < ATOL
    assert (G + G.transpose(1, 2)).abs().max().item() == 0.0


@torch.no_grad()
@pytest.mark.parametrize("lattice_side,patch_radius", [(4, 1), (8, 1), (8, 2), (8, 3)])
def test_pair_output_translation_equivariant_on_torus(lattice_side, patch_radius):
    """The PHYSICAL rate of the unordered pair, G[min, max], must satisfy
    G(roll x)(roll i, roll j) == G(x)(i, j) for every lattice shift: nothing
    in the head may know an absolute position, and a shift that moves the
    lower index to the other hole must not change the rate. The stored
    matrix is index-antisymmetric by convention, so compare G * sign(j - i),
    the label-symmetric physical matrix."""
    head = _head(lattice_side=lattice_side, patch_radius=patch_radius)
    D = lattice_side
    d = D * D
    x = _state(d)
    t = torch.rand(1)
    index_sign = torch.sign(torch.arange(d)[None, :] - torch.arange(d)[:, None]).float()
    G = (head(x, t)[0] * index_sign).view(D, D, D, D)
    for shift in ((1, 0), (0, 1), (2, 3), (-1, -1)):
        x_grid = x.view(1, D, D)
        x_shifted = torch.roll(x_grid, shifts=shift, dims=(1, 2)).reshape(1, -1)
        G_shifted = (head(x_shifted, t)[0] * index_sign).view(D, D, D, D)
        G_rolled = torch.roll(G, shifts=shift * 2, dims=(0, 1, 2, 3))
        drift = _drift(G_shifted, G_rolled)
        assert drift < ATOL, f"shift {shift}: {drift:.2e}"


@torch.no_grad()
@pytest.mark.parametrize("lattice_side,composition", [(3, 1 / 3), (4, 0.5)])
def test_kolmogorov_residual_zero_mean_on_exact_slice(lattice_side, composition):
    """E_{p_t^C}[delta_t] = 0 with exact dt_log_Z: holds only if the reverse
    rate the residual reads off -G is the true reverse rate. The bar is
    relative to the residual RMS: the fp32 enumeration floor is ~3e-5 of
    it (measured identically for the factorised head), not an absolute."""
    D = lattice_side
    head = _head(lattice_side=D)
    target = FixedCompositionIsingTarget(D=D, sigma=0.3, target_composition=composition)
    states = enumerate_states(D * D).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]
    for t_scalar in (0.1, 0.5, 0.9):
        t = torch.full((slice_states.shape[0],), t_scalar)
        log_p = target.log_p_tilde_t(slice_states, t)
        p_cond = torch.softmax(log_p, dim=0)
        dt_log_Z = (p_cond * target.dt_log_p_tilde_t(slice_states, t)).sum()
        residual = residual_swap(slice_states, t, dt_log_Z, head, target)
        mean = (p_cond * residual).sum().item()
        rms = residual.pow(2).mean().sqrt().item()
        print(f"{D}x{D} t={t_scalar}: E_p[residual] = {mean:.2e} (rms {rms:.2f})")
        assert abs(mean) < 1e-4 * rms


@torch.no_grad()
def test_forward_matches_context_readout():
    head = _head(lattice_side=4)
    x = _state(16)
    t = torch.rand(1)
    omega_f = head.omega_projection(head.backbone.omega(((x + 1) / 2).long()))
    token_difference = omega_f.unsqueeze(2) - omega_f.unsqueeze(1)
    scores = (head.compute_pair_context(x, t) * token_difference).sum(-1)
    # H is label-odd and the omega difference too, so the direct readout is
    # the label-SYMMETRIC physical matrix; forward stores it index-antisymmetric.
    assert _drift(scores, scores.transpose(1, 2)) < ATOL
    upper = torch.triu(scores, diagonal=1)
    assert _drift(head(x, t), upper - upper.transpose(1, 2)) < ATOL


@torch.no_grad()
def test_shapes_finite_and_pair_gather():
    head = _head(lattice_side=4)
    x = _state(16, batch=3)
    t = torch.rand(3)
    H = head.compute_pair_context(x, t)
    assert H.shape == (3, 16, 16, head.feature_dim)
    G = head(x, t)
    assert G.shape == (3, 16, 16) and torch.isfinite(G).all()
    scores = gather_pair_scores(G, upper_tri_pairs(16, x.device))
    assert scores.shape == (3, 120)


def test_head_parameters_receive_grad_and_causal_stacks_dead():
    head = _head(lattice_side=4)
    x = _state(16)
    head(x, torch.rand(1)).sum().backward()
    for name, param in head.named_parameters():
        if "backbone" in name:
            continue
        assert param.grad is not None, f"head module dead in graph: {name}"
    stacks = list(head.backbone.fwd_stack.parameters()) + list(
        head.backbone.bwd_stack.parameters()
    )
    assert all(p.grad is None for p in stacks), (
        "leTF stacks live: this head is ordering-free"
    )


@torch.no_grad()
def test_G_stays_fp32_under_bf16_autocast():
    head = _head(lattice_side=4)
    x = _state(16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        G = head(x, torch.rand(1))
    assert G.dtype == torch.float32


def test_patch_radius_must_fit_the_torus():
    """Offsets of two distinct patch entries must never alias to the same
    site through the wrap (2R+1 <= D); otherwise partner-zeroing is ambiguous."""
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    with pytest.raises(ValueError):
        TwoHolePatchSwapHead(backbone, lattice_side=4, patch_radius=2)


def test_torus_neighbour_offsets_are_hollow_and_symmetric():
    offsets = torus_neighbour_offsets(radius=1)
    assert (0, 0) not in offsets and len(offsets) == 8
    for dr, dc in offsets:
        assert (-dr, -dc) in offsets


def test_build_swap_head_wires_lattice_side_and_exact_field_wrapper():
    """head_kind='two_hole_patch' must reach the head with cfg.ising.D and
    compose with the exact-field channel like every other head."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS, build_swap_head

    cfg = replace(
        CONFIGS["H2_d64_c50_s223_letf_fimo2_50k_curr"],
        head_kind="two_hole_patch",
        patch_radius=2,
        exact_field_channel=True,
    )
    target = FixedCompositionIsingTarget(D=8, sigma=0.223, target_composition=0.5)
    backbone = LeTFRateMatrix(d=64, vocab_size=2, hidden_dim=32, n_layers=2, n_heads=4)
    head = build_swap_head(cfg, backbone, target)
    assert isinstance(head.head, TwoHolePatchSwapHead)
    assert head.head.patch_radius == 2 and head.head.lattice_side == 8
    assert head.head.pooling_radii == (1, 2)
    x = _state(64, batch=2)
    G = head(x, torch.rand(2))
    assert G.shape == (2, 64, 64) and torch.isfinite(G).all()


def test_d64_thp_cell_mirrors_fimo2_rung_except_head_kind():
    """The d64 twin differs from the fimo2 rung ONLY in head_kind (and the
    fimo2-specific band/ordering knobs that head_kind makes inert), so any
    outcome difference is attributable to the head."""
    from dataclasses import replace

    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    cell = CONFIGS["H2_d64_c50_s223_letf_thp_50k_curr"]
    twin = CONFIGS["H2_d64_c50_s223_letf_fimo2_50k_curr"]
    assert cell.head_kind == "two_hole_patch"
    assert (
        replace(
            cell,
            name=twin.name,
            head_kind=twin.head_kind,
            interior_band=twin.interior_band,
            site_orderings=twin.site_orderings,
        )
        == twin
    )
    _, head = build_target_and_head(cell, torch.device("cpu"))
    assert isinstance(head, TwoHolePatchSwapHead)
    assert head.patch_radius == 1 and head.lattice_side == 8
