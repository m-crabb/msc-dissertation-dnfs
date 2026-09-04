"""Falsification tests for the two-hole patch head on a Bravais supercell
(the 64-site fcc Cu-Au cell), written BEFORE the geometry body.

The head's every geometric fact is a statement about the lattice's
TRANSLATION GROUP: the hollow window is a list of offsets, the pooled
levels are balls of offsets, the pair position code is the offset class of
j from i, and "opposite offset" is negation. On the D x D torus that group
is Z_D^2; on a one-atom-per-primitive-cell supercell repeated (n1, n2, n3)
times it is Z_n1 x Z_n2 x Z_n3, with offsets read as minimum-image
fractional displacements. `bravais_patch_geometry` builds the same tensors
the torus constructor builds, so the head's math (blindness by locality,
partner-zeroed recompute, hole-subtracted linear pools, label oddness,
translation equivariance) is unchanged; these tests check that each
property survives the change of group, and that the torus path is
untouched (its tensors and its pooled means are reproduced exactly).

Claims under test:
  * geometry: the one-shell window of the 4x4x4 cell is the twelve fcc
    nearest neighbours, agrees with the expansion's own neighbour list,
    is closed under negation, and never aliases; the displacement table is
    a translation-group table (each row a permutation, negation consistent);
  * the 2x2x4 cell is REFUSED: two repeats make an offset and its negation
    the same site, so partner-zeroing would be ambiguous;
  * the torus geometry reproduces the head's original buffers, and the
    mask-matmul pooling equals the circular conv2d pooling;
  * on the fcc cell: two-hole blindness for every pair, sensitivity near and
    far, vectorised assembly == per-pair oracle, state-swap and index
    antisymmetry, translation equivariance under the supercell group;
  * build_swap_head wires the cluster-expansion cell to the fcc geometry,
    and the registered 64-site patch cells mirror their mask-one parents.
"""

from dataclasses import replace

import pytest
import torch

from discrete_flow_sampler.constraints.swap_readout import swap2
from discrete_flow_sampler.constraints.two_hole_patch_swap_head import (
    TwoHolePatchSwapHead,
    bravais_patch_geometry,
    torus_patch_geometry,
)
from discrete_flow_sampler.models.letf import LeTFRateMatrix
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

ATOL = 1e-5
FCC_64 = "data/ce/cuau_fcc_4x4x4.json"
FCC_16 = "data/ce/cuau_fcc_2x2x4.json"


def _spec(path=FCC_64):
    return BinaryExpansionSpec.from_json(path)


def _fcc_head(seed=42, hidden_dim=8, feature_dim=6, patch_shells=1, pooling_shells=None):
    torch.manual_seed(seed)
    spec = _spec()
    geometry = bravais_patch_geometry(
        spec.positions, spec.cell, patch_shells=patch_shells, pooling_shells=pooling_shells,
    )
    backbone = LeTFRateMatrix(
        d=spec.n_sites, vocab_size=2, hidden_dim=hidden_dim, n_layers=1, n_heads=2,
    )
    head = TwoHolePatchSwapHead(backbone, geometry=geometry, feature_dim=feature_dim)
    head.eval()
    return head, geometry


def _state(d, seed=1, batch=1):
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


# ---- geometry -----------------------------------------------------------


def test_fcc_one_shell_window_is_the_twelve_nearest_neighbours():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    assert geometry.neighbour_site.shape == (64, 12)
    adjacency = spec.nn_adjacency()
    for site in range(64):
        window = geometry.neighbour_site[site].tolist()
        assert len(set(window)) == 12, f"site {site}: window aliases {window}"
        assert site not in window, "window must be hollow"
        assert set(window) == set(torch.nonzero(adjacency[site]).flatten().tolist())


def test_fcc_window_closed_under_negation():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    opposite = geometry.opposite_offset
    assert (opposite[opposite] == torch.arange(12)).all(), "negation is an involution"
    assert (opposite != torch.arange(12)).all(), "no offset is its own negation"
    # Site i sits at the OPPOSITE offset inside its neighbour's window.
    for site in range(64):
        for k, partner in enumerate(geometry.neighbour_site[site].tolist()):
            assert geometry.neighbour_site[partner, opposite[k]] == site


def test_fcc_two_shells_adds_the_six_second_neighbours():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=2)
    assert geometry.neighbour_site.shape == (64, 18)


def test_fcc_displacement_table_is_a_translation_group_table():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    table = geometry.pair_displacement
    assert table.shape == (64, 64) and geometry.n_displacements == 64
    assert (table.diagonal() == 0).all(), "class 0 is the identity translation"
    for row in table:
        assert sorted(row.tolist()) == list(range(64)), "each row enumerates the group"
    negation = geometry.negate_displacement
    assert (negation[table] == table.transpose(0, 1)).all(), "class(j->i) = -class(i->j)"
    # The class of j from i is the site the identity-site's translate lands on.
    for k in range(64):
        assert table[0, k] == k


def test_fcc_translation_permutations_are_group_actions():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    for k in (1, 5, 17, 63):
        perm = geometry.translation(k)
        assert sorted(perm.tolist()) == list(range(64))
        # Translating preserves every offset class.
        table = geometry.pair_displacement
        assert (table[perm][:, perm] == table).all()


def test_fcc_pooling_levels_are_balls_with_the_centre_and_global_last():
    spec = _spec()
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    assert len(geometry.level_masks) == len(geometry.level_sizes)
    for mask, size in zip(geometry.level_masks, geometry.level_sizes):
        assert (mask == mask.transpose(0, 1)).all()
        assert (mask.diagonal() == 1).all(), "a ball contains its centre"
        assert (mask.sum(dim=1) == size).all()
    # Default levels: first- and second-shell balls (13 and 19 sites).
    assert geometry.level_sizes == (13, 19)


def test_two_by_two_by_four_cell_is_refused():
    spec = _spec(FCC_16)
    with pytest.raises(ValueError, match="alias"):
        bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)


# ---- torus path is untouched ----------------------------------------------


@pytest.mark.parametrize("lattice_side,patch_radius", [(4, 1), (8, 1), (8, 2)])
def test_torus_geometry_reproduces_the_original_buffers(lattice_side, patch_radius):
    """The formulas the head used before the geometry object existed."""
    D = lattice_side
    d = D * D
    geometry = torus_patch_geometry(D, patch_radius)
    offsets = [
        (dr, dc)
        for dr in range(-patch_radius, patch_radius + 1)
        for dc in range(-patch_radius, patch_radius + 1)
        if (dr, dc) != (0, 0)
    ]
    sites = torch.arange(d)
    rows, cols = sites // D, sites % D
    neighbour_site = torch.stack(
        [((rows + dr) % D) * D + (cols + dc) % D for dr, dc in offsets], dim=1
    )
    assert (geometry.neighbour_site == neighbour_site).all()
    opposite = torch.tensor([offsets.index((-dr, -dc)) for dr, dc in offsets])
    assert (geometry.opposite_offset == opposite).all()
    displacement = ((rows[None, :] - rows[:, None]) % D) * D + (cols[None, :] - cols[:, None]) % D
    assert (geometry.pair_displacement == displacement).all()
    dr = (rows[:, None] - rows[None, :]).abs()
    dc = (cols[:, None] - cols[None, :]).abs()
    dr, dc = torch.minimum(dr, D - dr), torch.minimum(dc, D - dc)
    radii = tuple(2**k for k in range(5) if 2 * 2**k + 1 <= D)
    assert geometry.level_sizes == tuple((2 * r + 1) ** 2 for r in radii)
    for mask, r in zip(geometry.level_masks, radii):
        assert (mask == (torch.maximum(dr, dc) <= r).float()).all()
    assert geometry.lattice_side == D


def test_torus_head_constructor_signature_unchanged():
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    head = TwoHolePatchSwapHead(backbone, lattice_side=4, patch_radius=1)
    assert head.lattice_side == 4 and head.patch_radius == 1 and head.n_patch == 8
    assert head.pooling_radii == (1,)
    with pytest.raises(ValueError):
        TwoHolePatchSwapHead(backbone, lattice_side=4, patch_radius=2)


@torch.no_grad()
def test_mask_pooling_equals_circular_conv_pooling_on_the_torus():
    """The general (graph) pooled mean and the torus conv2d pooled mean are
    the same linear map; this pins the general path to the archived one."""
    torch.manual_seed(0)
    backbone = LeTFRateMatrix(d=64, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    head = TwoHolePatchSwapHead(backbone, lattice_side=8, patch_radius=1)
    values = torch.randn(3, 64, head.feature_dim)
    for level, radius in enumerate(head.pooling_radii):
        conv = head._level_box_mean(values, level)
        mask = getattr(head, f"level_mask_{level}")
        matmul = torch.einsum("ij,bjf->bif", mask, values) / head.geometry.level_sizes[level]
        assert _drift(conv, matmul) < 1e-5


# ---- head properties on the fcc cell ------------------------------------


@torch.no_grad()
def test_fcc_pair_context_blind_to_both_holes_all_pairs():
    head, geometry = _fcc_head()
    x = _state(64)
    t = torch.rand(1)
    H = head.compute_pair_context(x, t)
    worst = 0.0
    for i in (0, 21, 63):
        for j in range(64):
            if j == i:
                continue
            base = H[:, i, j, :]
            for flipped in (_flip(x, i), _flip(x, j), _flip(x, i, j)):
                drift = _drift(head.compute_pair_context(flipped, t)[:, i, j, :], base)
                worst = max(worst, drift)
                assert drift < ATOL, f"H_[{i},{j}] leaks a hole: {drift:.2e}"
    print(f"fcc 64 one shell: worst hole leak {worst:.2e}")


@torch.no_grad()
def test_fcc_pair_context_sensitive_near_and_far():
    head, geometry = _fcc_head()
    x = _state(64)
    t = torch.rand(1)
    i = 0
    j = int(geometry.neighbour_site[i, 0])              # a nearest neighbour of i
    other_neighbour = int(geometry.neighbour_site[i, 1])
    far = [s for s in range(64) if geometry.level_masks[-1][i, s] == 0 and geometry.level_masks[-1][j, s] == 0][0]
    base = head.compute_pair_context(x, t)[:, i, j, :]
    for site, region in ((other_neighbour, "neighbour of i"), (far, "far site")):
        drift = _drift(head.compute_pair_context(_flip(x, site), t)[:, i, j, :], base)
        assert drift > 1e-7, f"H_[{i},{j}] ignores its {region} (site {site})"


@torch.no_grad()
@pytest.mark.parametrize("patch_shells", [1, 2])
def test_fcc_vectorised_context_matches_per_pair_reference(patch_shells):
    head, geometry = _fcc_head(patch_shells=patch_shells)
    x = _state(64, batch=2)
    t = torch.rand(2)
    H = head.compute_pair_context(x, t)
    pairs = [(0, int(geometry.neighbour_site[0, 0])), (0, 63), (5, 40), (17, 18), (30, 31)]
    for i, j in pairs:
        reference = head.pair_context_reference(x, t, i, j)
        assert _drift(H[:, i, j, :], reference) < ATOL, (i, j)


@torch.no_grad()
def test_fcc_antisymmetric_at_init_and_trivial_swaps_vanish():
    head, _ = _fcc_head()
    x = _state(64)
    t = torch.rand(1)
    G = head(x, t)
    assert (G + G.transpose(1, 2)).abs().max().item() == 0.0
    worst = 0.0
    for i in range(64):
        for j in range(i + 1, 64, 7):
            if x[0, i] == x[0, j]:
                assert G[0, i, j].abs().item() < ATOL
                continue
            G_swapped = head(swap2(x, i, j), t)
            worst = max(worst, (G[0, i, j] + G_swapped[0, i, j]).abs().item())
    assert worst < ATOL


@torch.no_grad()
def test_fcc_pair_output_translation_equivariant():
    """G_phys(translate x)[perm i, perm j] == G_phys(x)[i, j] for supercell
    translations, on the label-symmetric physical matrix G * sign(j - i)."""
    head, geometry = _fcc_head()
    x = _state(64)
    t = torch.rand(1)
    index_sign = torch.sign(torch.arange(64)[None, :] - torch.arange(64)[:, None]).float()
    G = head(x, t)[0] * index_sign
    for k in (1, 5, 17, 63):
        perm = geometry.translation(k)                    # site i -> perm[i]
        x_shifted = torch.empty_like(x)
        x_shifted[:, perm] = x
        G_shifted = head(x_shifted, t)[0] * index_sign
        drift = _drift(G_shifted[perm][:, perm], G)
        assert drift < ATOL, f"translation {k}: {drift:.2e}"


def test_fcc_head_parameters_receive_grad():
    head, _ = _fcc_head()
    x = _state(64)
    head(x, torch.rand(1)).sum().backward()
    for name, param in head.named_parameters():
        if "backbone" not in name:
            assert param.grad is not None, name


# ---- config wiring ------------------------------------------------------


def test_build_swap_head_gives_the_cluster_expansion_cell_the_fcc_geometry():
    from experiments.constrained_hard_03.configs import CONFIGS
    from experiments.constrained_hard_03.run import build_target_and_head

    cell = CONFIGS["H2_cuau64_c25_T500_thp_50k_curr"]
    assert cell.head_kind == "two_hole_patch"
    _, head = build_target_and_head(cell, torch.device("cpu"))
    assert isinstance(head, TwoHolePatchSwapHead)
    assert head.n_patch == 18 and head.geometry.lattice_side is None
    x = _state(64, batch=2)
    G = head(x, torch.rand(2))
    assert G.shape == (2, 64, 64) and torch.isfinite(G).all()


def test_cuau64_patch_cells_mirror_their_mask_one_parents():
    """Only the head (and the diagnostic-only in-training eval draw) moves."""
    from experiments.constrained_hard_03.configs import CONFIGS

    for c_tag in ("c25", "c50"):
        cell = CONFIGS[f"H2_cuau64_{c_tag}_T500_thp_50k_curr"]
        parent = CONFIGS[f"H2_cuau64_{c_tag}_T500_mask_one_50k_curr"]
        assert replace(
            cell, name=parent.name, head_kind=parent.head_kind, patch_shells=None,
            eval=replace(cell.eval, n_eval_samples_training=None),
        ) == parent
        assert cell.eval.n_eval_samples_training == 256 and cell.patch_shells == 2


# ---- a Bravais cell that IS enumerable: the square-lattice expansion -------
#
# `data/ce/square_cuau_4x4.json` is a 4x4 square lattice (2.5 A spacing, one
# layer in a 10 x 10 x 20 A box). Its translation group is the 4x4 torus's,
# so the Bravais geometry can be checked against the torus one, and the
# swap Kolmogorov identity can run on its enumerable c=0.5 slice through the
# cluster-expansion target -- the identity the loss relies on, here through
# the same target class and geometry route the 64-site cell uses.

SQUARE_16 = "data/ce/square_cuau_4x4.json"


def _square_site_to_raster(spec):
    """JSON site index -> row-major torus index, from the positions."""
    positions = torch.tensor(spec.positions)[:, :2]
    spacing = positions[positions[:, 0] > 0, 0].min()
    rows = torch.round(positions[:, 1] / spacing).long()
    cols = torch.round(positions[:, 0] / spacing).long()
    return rows * 4 + cols


def test_square_expansion_two_shells_is_the_torus_radius_one_window():
    spec = _spec(SQUARE_16)
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=2)
    torus = torus_patch_geometry(4, 1)
    to_raster = _square_site_to_raster(spec)
    assert geometry.neighbour_site.shape == (16, 8)
    for site in range(16):
        window = set(to_raster[geometry.neighbour_site[site]].tolist())
        assert window == set(torus.neighbour_site[to_raster[site]].tolist())
    one_shell = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=1)
    assert one_shell.neighbour_site.shape == (16, 4)


@torch.no_grad()
def test_kolmogorov_residual_zero_mean_on_square_expansion_slice(patch_shells=1):
    """E_{p_t^C}[delta_t] = 0 with exact dt_log_Z on the enumerable c=0.5
    slice of the square-lattice expansion, with the head on the Bravais
    geometry and the cluster-expansion target: the reverse rate read off -G
    is the true reverse rate on this route too.

    The bar is the doubly-hollow reference head's own residual mean on the
    SAME target, not an absolute: eV-scale energies at beta 20 give a 26-nat
    log-p spread, and the head-independent term of the residual carries an
    fp32 floor of ~2.4e-4 of the rms at t=0.1 that is identical across heads
    and seeds (1e-5 on the Ising target). A head defect would move the
    patch head OFF that floor; matching it to 20% is the pass. Geometry
    does not enter the identity, so one shell count and the worst-floor
    time point suffice (each residual pass over the 12870-state slice is
    ~20 s)."""
    from discrete_flow_sampler.constraints.swap_readout import DoublyHollowSwapHead
    from discrete_flow_sampler.diagnostics.metrics import enumerate_states
    from discrete_flow_sampler.samplers.swap_kolmogorov import residual_swap
    from discrete_flow_sampler.targets.cluster_expansion import (
        FixedCompositionClusterExpansionTarget,
    )

    spec = _spec(SQUARE_16)
    geometry = bravais_patch_geometry(spec.positions, spec.cell, patch_shells=patch_shells)
    target = FixedCompositionClusterExpansionTarget(spec, beta=20.0, target_composition=0.5)
    states = enumerate_states(16).float()
    n_plus = ((states + 1) * 0.5).sum(dim=-1)
    slice_states = states[n_plus == target.n_plus_target]

    def residual_mean_over_rms(head):
        out = []
        for t_scalar in (0.1,):
            t = torch.full((slice_states.shape[0],), t_scalar)
            log_p = target.log_p_tilde_t(slice_states, t)
            p_cond = torch.softmax(log_p, dim=0)
            dt_log_Z = (p_cond * target.dt_log_p_tilde_t(slice_states, t)).sum()
            residual = residual_swap(slice_states, t, dt_log_Z, head, target)
            out.append(abs((p_cond * residual).sum().item()) / residual.pow(2).mean().sqrt().item())
        return out

    torch.manual_seed(3)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    patch = residual_mean_over_rms(
        TwoHolePatchSwapHead(backbone, geometry=geometry, feature_dim=6).eval()
    )
    torch.manual_seed(3)
    backbone = LeTFRateMatrix(d=16, vocab_size=2, hidden_dim=8, n_layers=1, n_heads=2)
    reference = residual_mean_over_rms(DoublyHollowSwapHead(backbone).eval())
    for t_scalar, ours, floor in zip((0.1,), patch, reference):
        assert ours < 1e-3, (t_scalar, ours)
        assert ours < 1.2 * floor + 5e-5, (t_scalar, ours, floor)


def test_cuau64_temperature_grid_cells_stop_their_ladder_at_the_row_temperature():
    """MetaDNS grid rows: same cell as the 500 K thp / flip parents, ladder
    truncated at 1200 K (one stage) or 680 K (four stages linear in beta)."""
    from experiments.constrained_hard_03.configs import CONFIGS as HARD, cuau_sigma
    from experiments.constrained_soft_02.configs import CONFIGS as SOFT

    def final_temperature(cfg):
        return 1.0 / (2.0 * 8.617333262e-5 * cfg.curriculum.stages[-1].sigma)

    grid = [(HARD[f"H2_cuau64_{c}_{v}"], HARD[f"H2_cuau64_{c}_T500_thp_50k_curr"])
            for c in ("c25", "c50") for v in ("T1200_thp_10k", "T680_thp_30k_l4")]
    grid += [(SOFT[v], SOFT["A1_cuau64_T500_letf_50k_curr"])
             for v in ("A1_cuau64_T1200_letf_10k", "A1_cuau64_T680_letf_30k_l4")]
    for cell, parent in grid:
        one_stage = "T1200" in cell.name
        assert len(cell.curriculum.stages) == (1 if one_stage else 4)
        assert final_temperature(cell) == pytest.approx(1200.0 if one_stage else 680.0)
        assert cell.curriculum.stages[0].sigma == pytest.approx(cuau_sigma(1200.0))
        assert cell.train.n_steps == (10_000 if one_stage else 30_000)
        assert replace(cell, name=parent.name, curriculum=parent.curriculum,
                       train=parent.train) == parent


def test_cuau16_composition_sweep_cells_differ_from_the_house_cell_only_by_composition():
    from experiments.constrained_hard_03.configs import CONFIGS

    house = CONFIGS["H2_cuau16_c50_T500_mask_one_50k_house"]
    for c, tag in ((0.3125, "c31"), (0.375, "c38"), (0.4375, "c44")):
        cell = CONFIGS[f"H2_cuau16_{tag}_T500_mask_one_50k_house"]
        assert cell.ising.target_composition == c and round(c * 16) == int(c * 16)
        assert replace(cell, name=house.name, ising=house.ising) == house


def test_cuau16_amortised_cell_is_the_house_cell_plus_the_mixture_knob():
    from experiments.constrained_hard_03.configs import CONFIGS

    house = CONFIGS["H2_cuau16_c50_T500_mask_one_50k_house"]
    cell = CONFIGS["H2_cuau16_camort_T500_mask_one_50k_house"]
    assert cell.composition_mixture == (0.5, 0.4375, 0.375, 0.3125, 0.25)
    assert all(c * 16 == round(c * 16) for c in cell.composition_mixture)
    assert replace(cell, name=house.name, composition_mixture=None) == house


def test_cuau16_free_grid_cells_stop_the_ladder_at_the_row_temperature():
    from experiments.constrained_soft_02.configs import CONFIGS

    house = CONFIGS["A1_cuau16_T500_letf_50k_house"]
    for name, n_stages, n_steps, T in (("A1_cuau16_T1200_letf_10k", 1, 10_000, 1200.0),
                                       ("A1_cuau16_T680_letf_30k_l4", 4, 30_000, 680.0)):
        cell = CONFIGS[name]
        assert len(cell.curriculum.stages) == n_stages and cell.train.n_steps == n_steps
        assert 1.0 / (2.0 * 8.617333262e-5 * cell.curriculum.stages[-1].sigma) == pytest.approx(T)
        assert replace(cell, name=house.name, curriculum=house.curriculum, train=house.train) == house
