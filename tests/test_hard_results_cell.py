"""What correct looks like for the hard chapter's house results cell, before it.

Every results chapter carries the same two-panel cell -- energy marginal
on exact levels, plus a Z2-ODD order-parameter marginal for mode coverage
-- and the hard body carries that cell at BOTH 8x8 and 16x16. This module
encodes the two facts that make the hard instantiation different from the
soft and unconstrained ones, so that a figure built on the wrong observable
or the wrong bin grid fails here rather than in the thesis.

FACT ONE: THE OBVIOUS Z2-ODD PANEL IS DEAD ON THIS CHAPTER'S SLICE.
The other two chapters use the magnetisation marginal. Here the swap
process cannot leave c = 0.5, so sum_i s_i = 0 on EVERY draw of every run:
the magnetisation marginal is a spike at zero carrying no information, and
a figure that plotted it would look flawless while saying nothing. The
observable that survives the constraint is the
half-magnetisation order parameter phi = (m_left - m_right)/2, whose two
phase-separated configurations sit at +-1 while the total stays pinned.
E_pi[phi] = 0 by the global spin-flip symmetry of the slice, so a symmetric
reference and a mode-collapsed sampler are distinguishable by the WIDTH and
SHAPE of the marginal, not merely its mean.

FACT TWO: PHI HAS A DISCRETE SUPPORT AND MUST BE BINNED ON IT.
On the c = 0.5 slice the two halves' magnetisations are equal and opposite,
so phi collapses to m_left exactly, and m_left ranges over the d/2 + 1
values (2k - d/2)/(d/2) for k up-spins in the left half -- spacing 2/(d/2),
i.e. 0.0625 at 8x8 and 0.015625 at 16x16. Binning phi on a uniform grid
that does not divide that spacing produces the same alternating high/low
aliasing already excluded from the energy panel (fig 3.2 left, 40
uniform bins). A naive 17-bin histogram of the certified D8 pool reads
... 16283 28522 16371 28226 16191 ... -- pure aliasing, not structure.

The reference-side conventions (composition exactness, the chain as the
unit of independence, the floor drawn at the neural cells' own N) are the
8x8 house table's and are tested there; what is tested here is only what
the FIGURE adds.
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
    return torch.stack([base[torch.randperm(d, generator=generator)]
                        for _ in range(n)])


def _phase_separated(lattice_edge, side):
    """The left (side=0) or right (side=1) half all up, the other all down.

    These are the two configurations phi is built to separate, and they are
    the fixed points of the coverage question: a sampler that reaches only
    one of them has collapsed.
    """
    grid = -torch.ones(lattice_edge, lattice_edge)
    half = slice(None, lattice_edge // 2) if side == 0 else slice(lattice_edge // 2, None)
    grid[:, half] = 1.0
    return grid.reshape(1, -1)


# --- fact one: the constraint kills magnetisation, phi survives -----------

@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_magnetisation_is_degenerate_on_the_slice(lattice_edge):
    """The panel the other two chapters use carries no information here.

    Guards against porting the unconstrained cell verbatim: its panel (b)
    would be a delta at zero for every head, every seed and both couplings.
    """
    d = lattice_edge * lattice_edge
    states = _balanced_spins(256, seed=0, d=d)
    assert torch.allclose(magnetisation(states),
                          torch.zeros(len(states)), atol=1e-6)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_separates_the_two_phase_separated_modes(lattice_edge):
    """phi = +-1 on the two modes it exists to tell apart, and the sign
    tracks which half is up. If this ever reads 0, the observable has lost
    the axis the coverage panel is drawn on."""
    left_up = half_magnetisation_order_parameter(
        _phase_separated(lattice_edge, side=0), lattice_edge)
    right_up = half_magnetisation_order_parameter(
        _phase_separated(lattice_edge, side=1), lattice_edge)
    assert left_up.item() == pytest.approx(1.0)
    assert right_up.item() == pytest.approx(-1.0)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_is_z2_odd(lattice_edge):
    """phi(-x) = -phi(x). The coverage panel must be
    Z2-ODD; an even observable (energy, |m|, nn-correlation) cannot see a
    sampler that has collapsed onto one of a symmetric pair of modes."""
    d = lattice_edge * lattice_edge
    states = _balanced_spins(64, seed=1, d=d)
    assert torch.allclose(
        half_magnetisation_order_parameter(-states, lattice_edge),
        -half_magnetisation_order_parameter(states, lattice_edge), atol=1e-6)


# --- fact two: the support, and binning on it ----------------------------

@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_support_matches_the_analytic_grid(lattice_edge):
    """Every observed phi lands on (2k - d/2)/(d/2), and the builder's grid
    is exactly that set. This is the anti-aliasing contract: bin edges are
    derived from the support, never from a bin count."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    support = hrc.phi_support(lattice_edge)
    assert len(support) == d // 2 + 1
    assert support[0] == pytest.approx(-1.0)
    assert support[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(support), 2.0 / (d / 2))

    observed = half_magnetisation_order_parameter(
        _balanced_spins(512, seed=2, d=d), lattice_edge).numpy()
    nearest = np.abs(observed[:, None] - support[None, :]).min(axis=1)
    assert nearest.max() < 1e-6


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_phi_pmf_is_normalised_and_lands_on_the_right_atoms(lattice_edge):
    """A pmf over the support, not a density over bins: the two phase-
    separated states must put all their mass on the two end atoms."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    both_modes = torch.cat([_phase_separated(lattice_edge, 0),
                            _phase_separated(lattice_edge, 1)])
    pmf = hrc.phi_pmf(both_modes, lattice_edge)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[-1] == pytest.approx(0.5)   # phi = +1
    assert pmf[0] == pytest.approx(0.5)    # phi = -1
    assert pmf[1:-1].sum() == pytest.approx(0.0)


def test_phi_pmf_is_weight_aware():
    """The neural cells are importance-weighted, so the figure's pmf must
    take the same weights the house table's error columns do. Uniform
    weights must reproduce the unweighted count, and a degenerate weight
    must reproduce the single surviving draw."""
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
    """On the periodic DxD lattice every bond flip moves E by 4, so the
    support is {-2d, -2d+4, ..., 2d} and the panel bins ON it. The E/d axis
    matches the house table's EW2 convention."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    levels = hrc.energy_support(lattice_edge)
    assert levels[0] == pytest.approx(-2.0 * d)
    assert levels[-1] == pytest.approx(2.0 * d)
    assert np.allclose(np.diff(levels), 4.0)


@pytest.mark.parametrize("lattice_edge", [8, 16])
def test_bare_energy_lands_on_its_level_set(lattice_edge):
    """Real draws from the slice hit the analytic levels exactly. Catches a
    sign or normalisation slip in the figure's own energy evaluation, which
    is what puts the panel on the wrong axis relative to the table."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    d = lattice_edge * lattice_edge
    energies = hrc.bare_energy(_balanced_spins(128, seed=3, d=d), lattice_edge)
    levels = hrc.energy_support(lattice_edge)
    nearest = np.abs(energies.numpy()[:, None] - levels[None, :]).min(axis=1)
    assert nearest.max() < 1e-4


# --- the reference pool, as the figure consumes it ------------------------

def test_pooled_reference_splits_into_equal_chain_blocks():
    """The d256 reference ships as ONE pooled tensor, chain-block
    contiguous (generate_kawasaki_reference_d256.py concatenates the
    thinned chains in order). The floor needs the chain as the unit of
    independence, so the figure must recover the blocks; an off-by-one in
    the block width would silently mix two chains into every replicate."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    pooled = torch.arange(8 * 5).reshape(8 * 5, 1).float()
    blocks = hrc.split_pooled_into_chains(pooled, n_chains=8)
    assert len(blocks) == 8
    assert all(len(b) == 5 for b in blocks)
    assert torch.equal(torch.cat(blocks), pooled)


def test_reference_floor_is_positive_and_shrinks_with_draws():
    """The floor answers "how far from the reference does a draw of N land
    when it IS the reference". It must be non-zero (finite N) and must fall
    as N grows -- a floor that ignored N would make every cell look at the
    floor at large N and above it at small N."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    pool = _balanced_spins(4000, seed=4, d=64)
    chains = hrc.split_pooled_into_chains(pool, n_chains=8)
    small = hrc.phi_floor(chains, lattice_edge=8, n_draws=100, n_replicates=24,
                          seed=0)
    large = hrc.phi_floor(chains, lattice_edge=8, n_draws=1000, n_replicates=24,
                          seed=0)
    assert small > 0.0
    assert large < small


# --- the tripwire guard --------------------------------------------------

def test_tripwire_halted_cells_are_excluded(tmp_path):
    """A cell the cold-CV tripwire halted must never reach a panel.

    Its frozen eval reads ESS fraction ~0.0009 against a healthy twin's
    ~0.898 purely because training stopped at step 5000 of 50000, so a
    loader that kept it would put an infrastructure artefact on the page as
    a catastrophic head. Four d256 w3 cells carry the marker.
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
    """`thp` must not sweep up `thp2`. The two are separate rows of every
    house table and averaging them would silently merge a head with its
    own R=2 variant."""
    from experiments.constrained_hard_03.analysis import hard_results_cell as hrc

    for name in ("H2_d256_c50_s220_letf_thp_100k_w3_seed42_tag",
                 "H2_d256_c50_s220_letf_thp2_100k_w3_seed42_tag"):
        run = tmp_path / name
        (run / "eval_ema").mkdir(parents=True)
        (run / "eval_ema" / "metrics.json").write_text("{}")
        torch.save(_balanced_spins(8, seed=0, d=256), run / "eval_ema" / "samples.pt")
        torch.save(torch.zeros(8), run / "eval_ema" / "log_weights.pt")

    assert len(hrc.load_cells(tmp_path, 16, "s220", "thp", "eval_ema")) == 1
    assert len(hrc.load_cells(tmp_path, 16, "s220", "thp2", "eval_ema")) == 1
