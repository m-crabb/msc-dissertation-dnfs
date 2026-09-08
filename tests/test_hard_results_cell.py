"""Hard chapter's house results cell: energy marginal plus a Z2-odd panel.

Two facts distinguish the hard instantiation from the soft and unconstrained
ones, so a figure built on the wrong observable or the wrong bin grid fails
here rather than in the thesis.

The magnetisation marginal the other two chapters use is degenerate here: the
swap process cannot leave c = 0.5, so sum_i s_i = 0 on every draw. What
survives is the half-magnetisation order parameter phi = (m_left - m_right)/2,
at +-1 on the two phase-separated configurations. E_pi[phi] = 0 by the global
spin-flip symmetry of the slice, so a symmetric reference and a mode-collapsed
sampler differ in the width and shape of the marginal, not its mean.

phi has a discrete support and must be binned on it. On the c = 0.5 slice the
two halves' magnetisations are equal and opposite, so phi collapses to m_left,
which ranges over the d/2 + 1 values (2k - d/2)/(d/2) for k up-spins in the
left half -- spacing 2/(d/2), i.e. 0.0625 at 8x8 and 0.015625 at 16x16. A
uniform grid that does not divide that spacing aliases as the energy panel
would (fig 3.2 left, 40 uniform bins); a naive 17-bin histogram of the
certified D8 pool reads ... 16283 28522 16371 28226 16191 ...

Reference-side conventions (composition exactness, the chain as the unit of
independence, the floor drawn at the neural cells' own N) are tested with the
8x8 house table; only what the figure adds is tested here.
"""

import numpy as np
import pytest
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    half_magnetisation_order_parameter,
    magnetisation,
)


def _balanced_spins(n, seed, d):
    """n draws from the c=0.5 slice: every row exactly d/2 up, d/2 down."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.cat([torch.ones(d // 2), -torch.ones(d // 2)])
    return torch.stack([base[torch.randperm(d, generator=generator)] for _ in range(n)])


def _phase_separated(lattice_edge, side):
    """The left (side=0) or right (side=1) half all up, the other all down.

    These are the two configurations phi is built to separate; a sampler that
    reaches only one of them has collapsed.
    """
    grid = -torch.ones(lattice_edge, lattice_edge)
    half = (
        slice(None, lattice_edge // 2) if side == 0 else slice(lattice_edge // 2, None)
    )
    grid[:, half] = 1.0
    return grid.reshape(1, -1)


# --- fact one: the constraint kills magnetisation, phi survives -----------


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_magnetisation_is_degenerate_on_the_slice(lattice_edge):
    """Magnetisation is a delta at zero for every head, seed and coupling, so
    the panel the other two chapters use carries no information here."""
    d = lattice_edge * lattice_edge
    states = _balanced_spins(256, seed=0, d=d)
    assert torch.allclose(magnetisation(states), torch.zeros(len(states)), atol=1e-6)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_separates_the_two_phase_separated_modes(lattice_edge):
    """phi = +-1 on the two modes it exists to tell apart, with the sign
    tracking which half is up; 0 would mean the coverage panel has lost its
    axis."""
    left_up = half_magnetisation_order_parameter(
        _phase_separated(lattice_edge, side=0), lattice_edge
    )
    right_up = half_magnetisation_order_parameter(
        _phase_separated(lattice_edge, side=1), lattice_edge
    )
    assert left_up.item() == pytest.approx(1.0)
    assert right_up.item() == pytest.approx(-1.0)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_is_z2_odd(lattice_edge):
    """phi(-x) = -phi(x). An even observable (energy, |m|, nn-correlation)
    cannot see a sampler collapsed onto one of a symmetric pair of modes."""
    d = lattice_edge * lattice_edge
    states = _balanced_spins(64, seed=1, d=d)
    assert torch.allclose(
        half_magnetisation_order_parameter(-states, lattice_edge),
        -half_magnetisation_order_parameter(states, lattice_edge),
        atol=1e-6,
    )


# --- fact two: the support, and binning on it ----------------------------


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_support_matches_the_analytic_grid(lattice_edge):
    """Every observed phi lands on (2k - d/2)/(d/2), and the builder's grid is
    that set: bin edges come from the support, never from a bin count."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    support = hrc.phi_support(lattice_edge)
    assert len(support) == d // 2 + 1
    assert support[0] == pytest.approx(-1.0)
    assert support[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(support), 2.0 / (d / 2))

    observed = half_magnetisation_order_parameter(
        _balanced_spins(512, seed=2, d=d), lattice_edge
    ).numpy()
    nearest = np.abs(observed[:, None] - support[None, :]).min(axis=1)
    assert nearest.max() < 1e-6


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_pmf_is_normalised_and_lands_on_the_right_atoms(lattice_edge):
    """A pmf over the support, not a density over bins: the two phase-
    separated states must put all their mass on the two end atoms."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    both_modes = torch.cat(
        [_phase_separated(lattice_edge, 0), _phase_separated(lattice_edge, 1)]
    )
    pmf = hrc.phi_pmf(both_modes, lattice_edge)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[-1] == pytest.approx(0.5)  # phi = +1
    assert pmf[0] == pytest.approx(0.5)  # phi = -1
    assert pmf[1:-1].sum() == pytest.approx(0.0)


def test_phi_pmf_is_weight_aware():
    """The neural cells are importance-weighted, so the pmf takes the same
    weights the house table's error columns do: uniform weights reproduce the
    unweighted count, a degenerate weight the single surviving draw."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    states = torch.cat([_phase_separated(8, 0), _phase_separated(8, 1)])
    uniform = hrc.phi_pmf(states, 8, weights=torch.full((2,), 0.5))
    assert np.allclose(uniform, hrc.phi_pmf(states, 8))

    collapsed = hrc.phi_pmf(states, 8, weights=torch.tensor([1.0, 0.0]))
    assert collapsed[-1] == pytest.approx(1.0)
    assert collapsed[0] == pytest.approx(0.0)


# --- the energy panel: same anti-aliasing contract ------------------------


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_energy_support_is_the_exact_level_set(lattice_edge):
    """On the periodic DxD lattice every bond flip moves E by 4, so the support
    is {-2d, -2d+4, ..., 2d} and the panel bins on it. The E/d axis matches the
    house table's EW2 convention."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    levels = hrc.energy_support(lattice_edge)
    assert levels[0] == pytest.approx(-2.0 * d)
    assert levels[-1] == pytest.approx(2.0 * d)
    assert np.allclose(np.diff(levels), 4.0)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_bare_energy_lands_on_its_level_set(lattice_edge):
    """Real draws from the slice hit the analytic levels exactly, catching a
    sign or normalisation slip in the figure's own energy evaluation."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    energies = hrc.bare_energy(_balanced_spins(128, seed=3, d=d), lattice_edge)
    levels = hrc.energy_support(lattice_edge)
    nearest = np.abs(energies.numpy()[:, None] - levels[None, :]).min(axis=1)
    assert nearest.max() < 1e-4


# --- the reference pool, as the figure consumes it ------------------------


def test_pooled_reference_splits_into_equal_chain_blocks():
    """The d256 reference ships as one pooled, chain-block-contiguous tensor
    (generate_kawasaki_reference_d256.py concatenates the thinned chains in
    order). The floor needs the chain as the unit of independence, so the
    figure must recover the blocks; an off-by-one in the block width mixes two
    chains into every replicate."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    pooled = torch.arange(8 * 5).reshape(8 * 5, 1).float()
    blocks = hrc.split_pooled_into_chains(pooled, n_chains=8)
    assert len(blocks) == 8
    assert all(len(b) == 5 for b in blocks)
    assert torch.equal(torch.cat(blocks), pooled)


def test_reference_floor_is_positive_and_shrinks_with_draws():
    """The floor is how far from the reference a draw of N lands when it is the
    reference: non-zero at finite N, and falling as N grows."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    pool = _balanced_spins(4000, seed=4, d=64)
    chains = hrc.split_pooled_into_chains(pool, n_chains=8)
    small = hrc.phi_floor(chains, lattice_edge=8, n_draws=100, n_replicates=24, seed=0)
    large = hrc.phi_floor(chains, lattice_edge=8, n_draws=1000, n_replicates=24, seed=0)
    assert small > 0.0
    assert large < small


# --- the tripwire guard --------------------------------------------------


def test_tripwire_halted_cells_are_excluded(tmp_path):
    """A cell the cold-CV tripwire halted must never reach a panel: its frozen
    eval reads ESS fraction ~0.0009 against a healthy twin's ~0.898 only
    because training stopped at step 5000 of 50000. Four d256 w3 cells carry
    the marker.
    """
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    def make_cell(name, halted):
        run = tmp_path / name
        (run / "eval_ema").mkdir(parents=True)
        (run / "eval_ema" / "metrics.json").write_text('{"ess_fraction": 0.5}')
        torch.save(_balanced_spins(8, seed=0, d=256), run / "eval_ema" / "samples.pt")
        torch.save(torch.zeros(8), run / "eval_ema" / "log_weights.pt")
        if halted:
            (run / "cv_inversion_halt.json").write_text('{"step": 5000}')
        return run

    healthy = make_cell("H2_d256_c50_s010_letf_ma_50k_w3_seed42_tag-r2", halted=False)
    halted = make_cell("H2_d256_c50_s010_letf_ma_50k_w3_seed42_tag", halted=True)
    assert not hrc.is_tripwire_truncated(healthy)
    assert hrc.is_tripwire_truncated(halted)

    loaded = hrc.load_cells(tmp_path, 16, "s010", "ma", "eval_ema")
    assert [c["name"] for c in loaded] == [healthy.name]


def test_head_token_does_not_match_a_longer_head(tmp_path):
    """`thp` must not sweep up `thp2`: separate rows of every house table, so
    averaging them merges a head with its own R=2 variant."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    for name in (
        "H2_d256_c50_s220_letf_thp_100k_w3_seed42_tag",
        "H2_d256_c50_s220_letf_thp2_100k_w3_seed42_tag",
    ):
        run = tmp_path / name
        (run / "eval_ema").mkdir(parents=True)
        (run / "eval_ema" / "metrics.json").write_text("{}")
        torch.save(_balanced_spins(8, seed=0, d=256), run / "eval_ema" / "samples.pt")
        torch.save(torch.zeros(8), run / "eval_ema" / "log_weights.pt")

    assert len(hrc.load_cells(tmp_path, 16, "s220", "thp", "eval_ema")) == 1
    assert len(hrc.load_cells(tmp_path, 16, "s220", "thp2", "eval_ema")) == 1
