"""The 16x16 house-table fill.

`house_table_16x16` imports the 8x8 fill's reference machinery (half-split
standard error, estimated sampling floor, FLOP provenance read from each cell's
saved config) rather than restating it: that code is lattice-generic and already
verified at 8x8. Tested here are the four things the top rung does differently,
each a way to print a wrong number the 8x8 tests cannot catch.

(1) The reference ships in a different format and the FLOP bill is in different
    units. The 8x8 pool is one npz per chain carrying `n_trial_steps`, a count
    of swap proposals. The d256 pool is a single pooled tensor whose provenance
    records sweeps, and one sweep is `N_SITES = 256` proposals
    (generate_kawasaki_reference_d256.py:137, `burn_proposals = BURN_IN_SWEEPS *
    N_SITES`). Passing sweeps where the bill wants trials would under-price the
    reference chain by 256x, the one direction that flatters the neural sampler
    in the FLOP/es column.

(2) Chain identity must be recovered by slicing. The half-split standard error
    needs the chain as its unit of independence, but the d256 reference is
    `np.concatenate(thinned_per_chain)` -- chain-block contiguous, equal width,
    with the boundaries recorded nowhere in the tensor. An off-by-one in the
    block width mixes two chains into every split and understates the
    reference's own error.

(3) The coupling must be asserted, not assumed. `kawasaki_ref_d256_sc` is
    mislabelled: it sits at sigma = 0.22305, not the frozen sigma_c of
    0.220343. It certifies cleanly against the legacy 0.588 anchor, so only
    reading its provenance reveals the problem. Mixing it into the sigma_c
    column would compare heads against a reference at a different temperature.

(4) The arm set is four heads and carries no rejection row. The skeleton
    declared a plain-`fimo2` row that was never run at either coupling (since
    deleted), and the rejection rows stay at 4x4 and 8x8 because neither the
    unconstrained nor the soft chapter has a d256 case to reject off. A fill
    that emitted either would print a row with no run behind it.
"""

import json

import pytest
import torch

L = 16
D_SITES = L * L


def _balanced_spins(n, seed, d=D_SITES):
    """n draws from the c=0.5 slice: every row exactly d/2 up, d/2 down."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.cat([torch.ones(d // 2), -torch.ones(d // 2)])
    return torch.stack([base[torch.randperm(d, generator=generator)] for _ in range(n)])


# --- (1) the FLOP bill's units -------------------------------------------


def test_trial_count_converts_sweeps_to_proposals():
    """One sweep is N_SITES proposals, and burn-in counts.

    The generator runs `BURN_IN_SWEEPS * N_SITES` burn-in proposals and
    `SAMPLING_SWEEPS * N_SITES` sampling proposals per chain; the bill
    prices proposals, and burn-in is paid before the first usable record so
    it belongs in the total (kawasaki_run_flops' own docstring).
    """
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    provenance = {
        "n_chains": 8,
        "burn_in_sweeps": 100_000,
        "sampling_sweeps_per_chain": 102_400,
    }
    counts = h16.chain_trial_counts(provenance, lattice_edge=L)
    assert len(counts) == 8
    assert all(c == (100_000 + 102_400) * D_SITES for c in counts)


def test_trial_count_scales_with_the_lattice():
    """The sweep -> proposal factor is the site count, not a constant. A
    hard-coded 256 would silently mis-bill any other rung built this way."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    provenance = {"n_chains": 2, "burn_in_sweeps": 10, "sampling_sweeps_per_chain": 10}
    assert h16.chain_trial_counts(provenance, lattice_edge=8)[0] == 20 * 64
    assert h16.chain_trial_counts(provenance, lattice_edge=16)[0] == 20 * 256


# --- (2) chain identity from the pooled tensor ---------------------------


def test_pooled_reference_splits_into_equal_chain_blocks():
    """Eight equal blocks whose concatenation is the pool, in order."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    pooled = torch.arange(8 * 7).reshape(8 * 7, 1).float()
    blocks = h16.split_pooled_into_chains(pooled, n_chains=8)
    assert len(blocks) == 8
    assert all(len(b) == 7 for b in blocks)
    assert torch.equal(torch.cat(blocks), pooled)


def test_chain_blocks_are_not_interleaved():
    """Blocks must be contiguous slices, not a stride.

    The generator concatenates whole chains; a strided split would look
    identical in shape while putting one snapshot of every chain into each
    block, which would make the half-split read the within-chain correlation as
    if it were between-chain and collapse the reference's stated error toward
    zero.
    """
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    pooled = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    blocks = h16.split_pooled_into_chains(pooled, n_chains=2)
    assert blocks[0].unique().tolist() == [0.0]
    assert blocks[1].unique().tolist() == [1.0]


# --- (3) the coupling guard ----------------------------------------------


def test_reference_rejects_a_mislabelled_coupling(tmp_path):
    """A pool whose provenance sigma disagrees with the column's must raise.

    This is the `kawasaki_ref_d256_sc` case (0.22305 against the frozen
    0.220343): it certifies against the legacy anchor and looks healthy, so
    only the assertion catches it.
    """
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    directory = tmp_path / "kawasaki_ref_d256_s220"
    directory.mkdir()
    (directory / "provenance.json").write_text(
        json.dumps({"sigma": 0.22305, "n_chains": 8})
    )
    torch.save(_balanced_spins(16, seed=0), directory / "samples.pt")

    with pytest.raises(AssertionError, match="sigma"):
        h16.load_reference(directory, sigma_key="s220")


def test_reference_accepts_the_exact_frozen_sigma_c(tmp_path):
    """The frozen value passes; the guard must not be so tight that the
    real reference trips it on floating-point representation."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    directory = tmp_path / "kawasaki_ref_d256_s220"
    directory.mkdir()
    (directory / "provenance.json").write_text(
        json.dumps({"sigma": 0.22034339675488573, "n_chains": 2})
    )
    torch.save(_balanced_spins(16, seed=0), directory / "samples.pt")

    chains, provenance = h16.load_reference(directory, sigma_key="s220")
    assert len(chains) == 2
    assert provenance["sigma"] == pytest.approx(0.22034339675488573)


def test_reference_must_be_composition_exact(tmp_path):
    """Every reference state sits on the c=0.5 slice; otherwise the error
    columns measure the composition gap instead of the structure."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    directory = tmp_path / "kawasaki_ref_d256_s010"
    directory.mkdir()
    (directory / "provenance.json").write_text(
        json.dumps({"sigma": 0.1, "n_chains": 2})
    )
    off_slice = _balanced_spins(16, seed=0)
    off_slice[3, 0] = -off_slice[3, 0]
    torch.save(off_slice, directory / "samples.pt")

    with pytest.raises(AssertionError, match="slice"):
        h16.load_reference(directory, sigma_key="s010")


# --- (4) the arm set -----------------------------------------------------


def test_arm_set_is_the_heads_that_ran():
    """The three w3 heads (ma, thp, thp2) plus the raster-ordering ladder that
    ran at this rung (tag 20260830-rasterord-d256): the two band families at one
    and two sweeps, with and without the exact field, and the separable `ma`
    twin at the floor. The plain-`fimo2` row the skeleton declared has no run at
    either coupling and was deleted; `fimo2ef` left with the factorised head,
    which is demoted to an exterior-combiner note and prints no results row at
    any rung. Its cells and config still exist -- an editorial removal, not a
    deletion."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    assert set(h16.ARMS) == {
        "ma",
        "thp",
        "thp2",
        "masep",
        "mamo2",
        "mamo2ef",
        "iv",
        "ivmo2",
        "ivmo2ef",
    }
    assert not {"fimo2", "fimo2ef", "fmo2ef"} & set(h16.ARMS)
    # Every arm prints a row, and the ladder sits in its own block.
    printed = [key for row in h16.LATEX_ROWS if row for key in [row[0]]]
    # masep is scored (it anchors the floor caveat) but prints no row: one
    # masked-attention row per rung, billed separable.
    assert set(h16.ARMS) - {"masep"} <= set(printed)


def test_gfn_rows_stay_outside_the_bold_comparison():
    """The 8x8 rule carried up: the GFN rows are a different sampling paradigm,
    so even when a GFN cell holds the best number in a column the bold lands on
    the best swap cell. `best` is computed over ARMS, which the GFN arms are not
    in; a refactor computing it over every table key would silently move the
    bold."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    assert not set(h16.GFN_ARMS) & set(h16.ARMS)
    printed = [row[0] for row in h16.LATEX_ROWS if row]
    assert set(h16.GFN_ARMS) <= set(printed)

    def entry(ess, flops):
        return {
            "ESS": (ess, 0.001),
            "dMag": (0.05, 0.01),
            "dCorr": (0.05, 0.01),
            "EW2": (0.05, 0.01),
            "FLOP/es": (flops, 0.0),
        }

    table = {
        "thp_s010": entry(0.90, 1.0e9),
        "gfn_tb_s010": entry(0.99, 1.0e6),
    }  # best ESS and FLOP/es
    body = h16.latex_table(table)
    lines = body.splitlines()
    gfn_line = next(line for line in lines if "trajectory balance" in line)
    thp_line = next(line for line in lines if "two-hole patch head" in line)
    assert "mathbf" not in gfn_line
    assert "mathbf" in thp_line


def test_no_rejection_row_is_emitted():
    """Rejection rows stay at 4x4 and 8x8: neither the unconstrained
    nor the soft chapter has a d256 case to reject off, so a row here would
    have no counterpart."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    emitted = [row[0] for row in h16.LATEX_ROWS if row is not None]
    assert not any("reject" in key for key in emitted)


def test_row_order_puts_reference_and_floor_above_the_heads():
    """The reference and its floor are the rows every head is read
    against, so they precede the heads as in both smaller tables."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    emitted = [row[0] for row in h16.LATEX_ROWS if row is not None]
    assert emitted[:2] == ["reference", "floor"]
    assert set(emitted[2:]) == (set(h16.ARMS) - {"masep"}) | set(h16.GFN_ARMS)


# --- cell selection ------------------------------------------------------


def _make_cell(tmp_path, name, halted=False):
    run = tmp_path / name
    (run / "eval_ema").mkdir(parents=True)
    (run / "eval_ema" / "metrics.json").write_text('{"ess_fraction": 0.5}')
    torch.save(_balanced_spins(4, seed=0), run / "eval_ema" / "samples.pt")
    torch.save(torch.zeros(4), run / "eval_ema" / "log_weights.pt")
    if halted:
        (run / "cv_inversion_halt.json").write_text('{"step": 5000}')
    return run


def test_tripwire_halted_cells_are_excluded(tmp_path):
    """Four d256 w3 cells were halted at step 5000 of 50000 by the cold-CV
    inversion tripwire. Their evals read ESS fraction ~0.0009 purely from the
    truncation, so keeping one would print a training-infrastructure artefact as
    a catastrophic head."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    good = _make_cell(tmp_path, "H2_d256_c50_s010_letf_ma_50k_w3_seed42_t-r2")
    _make_cell(tmp_path, "H2_d256_c50_s010_letf_ma_50k_w3_seed42_t", halted=True)
    found = h16.find_cells(tmp_path, "s010", "ma")
    assert [p.name for p in found] == [good.name]


def test_thp_does_not_match_thp2(tmp_path):
    """Both heads exist at this rung and are separate table rows; matching
    on a bare substring would merge a head with its own R=2 variant."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    _make_cell(tmp_path, "H2_d256_c50_s220_letf_thp_100k_w3_seed42_t")
    _make_cell(tmp_path, "H2_d256_c50_s220_letf_thp2_100k_w3_seed42_t")
    assert len(h16.find_cells(tmp_path, "s220", "thp")) == 1
    assert len(h16.find_cells(tmp_path, "s220", "thp2")) == 1


def test_missing_arm_yields_no_cells_rather_than_raising(tmp_path):
    """A head whose runs have not landed must leave its row blank, not
    abort the fill -- the ma sigma_c trio lands after the rest."""
    from experiments.constrained_hard_03.analysis import house_table_16x16 as h16

    assert h16.find_cells(tmp_path, "s220", "ma") == []
