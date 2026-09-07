"""fig:hard-clean-ladder (and the older per-rung cells): the hard results cells.

Every results chapter carries the same two-panel results cell (energy
marginal on exact levels + a Z2-ODD order-parameter marginal). The hard
body printed that cell at 8x8 and 16x16 as two separate figures until
2026-09-07; it now prints ONE critical-coupling grid, `--layout grid`:
columns are the four rungs (8, 16, 20, 24), rows are the two marginals,
and each column carries the patch head at the radius its house table
prints (GRID_CELLS). The per-rung `--lattice-edge` cells remain for the
appendix and for a single-rung read. Only sigma_c is gridded: the 24x24
rung was run at the critical coupling alone, and the sigma = 0.1 rows sit
at the floor in every table.

WHAT CHANGES FROM THE OTHER TWO CHAPTERS' CELLS, and why it is not a port.

(1) THE COVERAGE PANEL CANNOT BE THE MAGNETISATION MARGINAL. The other two
    chapters read mode coverage off total magnetisation. Here the swap
    process cannot leave c = 0.5, so sum_i s_i = 0 on every draw of every
    run and that marginal is a spike at zero: it would look perfect for
    every head, at both couplings, while measuring only that the constraint
    holds -- which the chapter already establishes bitwise. The observable
    that survives the constraint is the half-magnetisation order parameter

        phi = (m_left - m_right) / 2,

    the mean spin of the left half minus that of the right, halved into
    [-1, 1]. The two phase-separated configurations sit at phi = +-1 while
    the total stays pinned at zero, and E_pi[phi] = 0 by the global
    spin-flip symmetry of the slice, so phi is Z2-odd as the cell requires.
    On the c = 0.5 slice the halves' magnetisations are equal and opposite,
    so phi reduces to m_left exactly.

    WHAT THE PANEL ACTUALLY READS AT THESE SIZES. The certified references
    are unimodal and symmetric about zero at both couplings and both rungs
    -- SD 0.098 (16x16, sigma = 0.1) rising to 0.260 at sigma_c, and 0.181
    to 0.296 at 8x8 -- and every chain's own mean sits at zero whatever side
    it started from. So at these sizes the equilibrium is NOT two locked
    modes to be covered; it is one broad distribution whose WIDTH is the
    critical-fluctuation signal, roughly tripling from sigma = 0.1 to
    sigma_c. The panel therefore discriminates on width and shape: a
    sampler that under-orders or collapses gives a phi marginal too narrow,
    and one that has lost the symmetry gives a skewed one. Stated here
    rather than in the caption because it bounds what the figure may be
    claimed to show -- it is not evidence of two-mode coverage.

(2) BOTH COUPLINGS ARE ON THE FIGURE. The soft and unconstrained cells are
    1x2 at a single headline condition; the hard chapter carries sigma =
    0.1 and sigma_c as a first-class axis in every one of its house tables,
    and the phi width is the observable that most visibly separates them, so
    the cell is 2x2: rows are the coupling, columns are the two panels.

(3) BINNING. Both panels bin on their EXACT support, never on a bin count.
    Energy: the periodic lattice moves E by multiples of 4, so the support
    is {-2d, -2d+4, ..., 2d} and the axis is E/d, matching the house
    table's EW2 convention. Phi: m_left takes the d/2 + 1 values
    (2k - d/2)/(d/2), spacing 0.0625 at 8x8 and 0.015625 at 16x16. A
    uniform grid that does not divide that spacing aliases -- a 17-bin
    histogram of the certified D8 pool reads ... 16283 28522 16371 28226
    16191 ..., alternating high/low, which is the bin grid and not the
    physics. This is the same failure a 40-bin uniform grid produces on the
    energy panel.

REFERENCES, and their two shipping formats. At 8x8 the reference is the
per-chain mchammer Kawasaki pool (results/03_hard/kawasaki_w2,
kawasaki_D8_{s100,s220}_seed*.npz), the same pool tab:eval-hard-8x8 scores
against. At 16x16 it is the certified single-tensor pool
(results/kawasaki_ref_d256_{s010,s220}/samples.pt), 8 chains concatenated
chain-block contiguous after thinning: chains 0-1 start phase-separated on
one side, 2-3 on the other, 4-7 random, so the pool's phi symmetry is
earned rather than assumed. Chain identity is recovered by equal-width
slicing because the floor needs the chain as its unit of independence.
`kawasaki_ref_d256_sc` is MISLABELLED (actually sigma = 0.22305, not
0.220343) and is never read here.

THE FLOOR is the mean TVD of N independently resampled reference frames
against the full pool. It represents ideal draw noise conditional on that
pool. The earlier hierarchical bootstrap (chains, then frames) also included
reference uncertainty and was not comparable to the table's iid floor.
Reference uncertainty is now reported separately in the tables. Neither
proximity to this mean nor a smaller error establishes equivalence.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE,
    FIGSIZE_FULL_2X2,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    REFERENCE_INK,
    SAMPLER_HUE,
    SAVEFIG_DPI,
    parameter_ramp,
    seed_band,
    style_axes,
    use_house_style,
)
from discrete_flow_sampler.diagnostics.metrics import (
    half_magnetisation_order_parameter,
    marginal_tvd,
)
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]

# The two couplings the hard house tables carry. sigma_c is the EXACT
# critical value used project-wide; 0.223 is the legacy coupling of the
# older archive and must never share a panel with it.
SIGMA = {"s010": 0.1, "s220": 0.22034339675488573}
SIGMA_LABEL = {"s010": r"$\sigma = 0.1$", "s220": r"$\sigma = \sigma_c$"}

# The grid's column per rung: (head token, a run-dir token the cell must ALSO
# carry). The 20x20 fp32 and bf16-trained twins share the thp3 token and only
# the fp32 row is printed bold-eligible in tab:eval-hard-20x20, so the fp32
# `_w4_` twin is pinned; 24x24 exists only as the bf16-trained cell.
GRID_RUNGS = (8, 16, 20, 24)
GRID_CELLS = {
    8: ("thp", None, "$R=1$"),
    16: ("thp2", None, "$R=2$"),
    20: ("thp3", "_w4_", "$R=3$"),
    24: ("thp4", None, "$R=4$"),
}

# Head label -> the run-dir token that identifies it, per rung.
HEAD_LABEL = {
    "mo": "mask-one head",
    "ma": "masked-attention head",
    "thp": "two-hole patch head",
    "thp2": "two-hole patch head, $R=2$",
    "fimo2ef": "factorised head, prefix band $+$ exact field",
}


# --- supports, and the pmfs built on them --------------------------------


def phi_support(lattice_edge):
    """The d/2 + 1 values phi can take on the c = 0.5 slice.

    phi = m_left there, and m_left is the mean of d/2 spins of which k are
    up, so phi = (2k - d/2)/(d/2) for k = 0 .. d/2. Returned as the bin
    CENTRES; the figure draws atoms, never a density.
    """
    half = lattice_edge * lattice_edge // 2
    return (2.0 * np.arange(half + 1) - half) / half


def phi_pmf(states, lattice_edge, weights=None):
    """Weighted pmf of phi on its exact support.

    `weights` are the self-normalised importance weights the house table's
    error columns use; None means uniform (chain snapshots carry no
    weights). Mass is placed by index on the support rather than by
    histogram edges, so the result cannot alias.
    """
    half = lattice_edge * lattice_edge // 2
    phi = half_magnetisation_order_parameter(states.float(), lattice_edge)
    index = ((phi * half + half) / 2.0).round().long().clamp(0, half)
    if weights is None:
        weights = torch.full((len(states),), 1.0 / len(states))
    return torch.zeros(half + 1).index_add_(0, index, weights.float()).numpy()


def energy_support(lattice_edge):
    """The exact bare-Ising energy levels {-2d, -2d+4, ..., 2d}."""
    d = lattice_edge * lattice_edge
    return np.arange(-2.0 * d, 2.0 * d + 1.0, 4.0)


def _bare_target(lattice_edge):
    """Bare Ising at sigma = 1, i.e. the coupling-free energy.

    E = -log p~ / (2 sigma) and log p~ = x^T J x is linear in sigma, so the
    energy the panel plots does not depend on which coupling the draws came
    from -- which is exactly what lets both rows share one energy axis. No
    composition penalty: the constraint is enforced by the process, so the
    chapter's energy is the BARE one, matching the soft cell's convention
    and the house table's EW2 column.
    """
    return IsingTarget(D=lattice_edge, sigma=1.0)


def bare_energy(states, lattice_edge):
    """Bare Ising energy of each state, on the exact level set."""
    target = _bare_target(lattice_edge)
    return -target.base_log_prob(states.float()) / 2.0


def energy_pmf(states, lattice_edge, weights=None):
    """Weighted pmf on the exact energy levels, with an aliasing guard."""
    d = lattice_edge * lattice_edge
    level = (bare_energy(states, lattice_edge) + 2.0 * d) / 4.0
    assert (level - level.round()).abs().max() < 1e-2, (
        "state energies off the exact-level support"
    )
    index = level.round().long().clamp(0, d)
    if weights is None:
        weights = torch.full((len(states),), 1.0 / len(states))
    return torch.zeros(d + 1).index_add_(0, index, weights.float()).numpy()


# --- references ----------------------------------------------------------


def split_pooled_into_chains(pooled, n_chains):
    """Recover chain blocks from the d256 reference's single pooled tensor.

    generate_kawasaki_reference_d256.py concatenates the thinned chains in
    index order and all chains record the same number of snapshots, so the
    blocks are equal-width. Any remainder is dropped rather than smeared
    across blocks: a mis-set block width would mix two chains into every
    bootstrap replicate and silently understate the floor.
    """
    block = len(pooled) // n_chains
    return [pooled[i * block : (i + 1) * block] for i in range(n_chains)]


def load_reference(lattice_edge, sigma_key, burn_in_fraction=0.2):
    """Reference chains for this rung and coupling, one tensor per chain."""
    if lattice_edge == 8:
        tag = {"s010": "s100", "s220": "s220"}[sigma_key]
        paths = sorted(
            (REPO_ROOT / "results" / "03_hard" / "kawasaki_w2").glob(
                f"kawasaki_D8_{tag}_seed*.npz"
            )
        )
        if not paths:
            raise FileNotFoundError(f"no D8 {tag} reference chains")
        chains = [torch.from_numpy(np.load(p)["spins"]).float() for p in paths]
        return [c[int(len(c) * burn_in_fraction) :] for c in chains]

    # Certified single-tensor pools: d256 is split by coupling key, the two
    # larger sigma_c pools carry the `_sc` suffix (kawasaki_ref_d256_sc is
    # the mislabelled 0.22305 pool and is never read here).
    d = lattice_edge * lattice_edge
    suffix = (
        sigma_key if lattice_edge == 16 else {"s010": "s010", "s220": "sc"}[sigma_key]
    )
    directory = REPO_ROOT / "results" / f"kawasaki_ref_d{d}_{suffix}"
    provenance = json.loads((directory / "provenance.json").read_text())
    stated = provenance["sigma"]
    assert abs(stated - SIGMA[sigma_key]) < 1e-9, (
        f"{directory.name} is at sigma={stated}, not {SIGMA[sigma_key]} -- "
        "couplings must never be mixed in one panel"
    )
    pooled = torch.load(directory / "samples.pt", weights_only=True).float()
    # already burnt in and thinned by the generator; no further burn-in
    return split_pooled_into_chains(pooled, provenance["n_chains"])


def _floor(chains, n_draws, n_replicates, seed, pmf_of):
    """Mean TVD of n_draws iid frames against the empirical reference pool.

    Estimate E[TV(p_hat_N, p_pool)], excluding chain-pool uncertainty. The
    conditional benchmark assumes the empirical pool represents the target;
    whole-chain splitting diagnoses its uncertainty separately.
    """
    pool = torch.cat(chains)
    reference = pmf_of(pool)
    generator = torch.Generator().manual_seed(seed)
    distances = []
    for _ in range(n_replicates):
        rows = torch.randint(len(pool), (n_draws,), generator=generator)
        distances.append(
            marginal_tvd(
                torch.from_numpy(pmf_of(pool[rows])), torch.from_numpy(reference)
            )
        )
    return float(np.mean(distances))


def phi_floor(chains, lattice_edge, n_draws, n_replicates=64, seed=0):
    return _floor(
        chains, n_draws, n_replicates, seed, lambda x: phi_pmf(x, lattice_edge)
    )


def energy_floor(chains, lattice_edge, n_draws, n_replicates=64, seed=0):
    return _floor(
        chains, n_draws, n_replicates, seed, lambda x: energy_pmf(x, lattice_edge)
    )


# --- the neural cells ----------------------------------------------------


def is_tripwire_truncated(run_dir):
    """True if the cold-CV inversion tripwire halted this run early.

    `halt_on_cv_inversion_after` was a SCREENING rule that rode into the
    d256 w3 production cells through the shared house-cell builder and
    stopped four of them at step 5000 of 50000 on controlled-to-naive
    ratios of just 1.08-1.76. The tell is this file in the run dir; the
    cross-check is the row count (5001 against a healthy twin's 50001).

    Excluding them is not tidiness. Their frozen evals read ESS fraction
    0.0009 against the relaunches' 0.898, so a loader that swept both into
    one seed band would put a training-infrastructure artefact on the page
    as a catastrophic head. The relaunches carry the `-r2` tag and are the
    cells that count.
    """
    return (run_dir / "cv_inversion_halt.json").exists()


def load_cells(
    results_dir, lattice_edge, sigma_key, head, eval_subdir, require_token=None
):
    """Every healthy seed of one head at one coupling, as samples + IS weights.

    Matched on the run-dir naming the w2/w3 waves use, so a cell trained at
    another coupling or another size cannot enter the panel. Tripwire-halted
    cells are dropped and named on stderr rather than silently, so a rung
    that loses a seed says so. `require_token` pins one twin where the head
    token alone is ambiguous (the 20x20 fp32/bf16 pair).
    """
    d = lattice_edge * lattice_edge
    pattern = f"H2_d{d}_c50_{sigma_key}_letf_{head}_*"
    runs, dropped = [], []
    for run_dir in sorted(Path(results_dir).glob(pattern)):
        # `thp` must not match `thp2`: the token is delimited by underscores.
        if f"_{head}_" not in run_dir.name:
            continue
        if require_token is not None and require_token not in run_dir.name:
            continue
        metrics_path = run_dir / eval_subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        if is_tripwire_truncated(run_dir):
            dropped.append(run_dir.name)
            continue
        samples = torch.load(
            run_dir / eval_subdir / "samples.pt", weights_only=True
        ).float()
        log_w = torch.load(run_dir / eval_subdir / "log_weights.pt", weights_only=True)
        runs.append(
            {
                "name": run_dir.name,
                "samples": samples,
                "weights": torch.softmax(log_w, dim=0),
                "metrics": json.loads(metrics_path.read_text()),
            }
        )
    for name in dropped:
        print(f"dropped (cv-inversion tripwire halt): {name}", file=sys.stderr)
    return runs


def _occupied_limits(support, *pmfs, pad_fraction=0.04):
    """x-limits covering every atom any curve puts mass on.

    Both supports are far wider than the occupied window -- the energy level
    set spans [-2, 2] in E/d while the mass sits in a band of width ~1 --
    so drawing the full support flattens the panel into a spike. Taken over
    reference AND sampler so a head with a fatter tail is not cropped into
    looking like the reference.
    """
    occupied = np.nonzero(np.sum([np.asarray(p) for p in pmfs], axis=0) > 1e-9)[0]
    low, high = support[occupied.min()], support[occupied.max()]
    pad = pad_fraction * (high - low)
    return low - pad, high + pad


# Dash patterns paired with the lightness ramp so the heads separate in
# greyscale print as well as in colour; the ramp alone is too subtle once
# four curves sit on top of each other.
HEAD_DASHES = ((), (4.5, 1.6), (1.4, 1.4), (6.0, 1.5, 1.4, 1.5))


def _multi_head_panel(ax, support, reference_pmf, per_head, floors, xlabel):
    """One panel carrying every head, mean-over-seeds lines only.

    NO seed bands here. Four min-max bands at 0.18 alpha overlap into a
    single wash that hides the very separation the figure is drawn to show;
    the per-head seed spread is the single-head figure's job. Colours come
    from parameter_ramp on SAMPLER_HUE rather than from new palette
    entries: every curve is OUR sampler and the contrast is the head, which
    is a parameter within one role, so borrowing CLASSICAL_HUE for a third
    head would tell a reader who has learned the palette that the
    factorised head is a classical chain.
    """
    ax.plot(
        support,
        reference_pmf,
        color=REFERENCE_INK,
        linewidth=1.3,
        zorder=5,
        label="Kawasaki reference (certified)",
    )
    # darkest = best head, so the ramp orders the way the legend does
    hues = parameter_ramp(SAMPLER_HUE, len(per_head))[::-1]
    for index, (label, pmfs) in enumerate(per_head):
        if not pmfs:
            continue
        mean_pmf = np.mean(pmfs, axis=0)
        tvd = np.mean(
            [
                marginal_tvd(torch.from_numpy(p), torch.from_numpy(reference_pmf))
                for p in pmfs
            ]
        )
        ax.plot(
            support,
            mean_pmf,
            color=hues[index],
            linewidth=1.5,
            dashes=HEAD_DASHES[index % len(HEAD_DASHES)],
            zorder=4 - index,
            label=f"{label} ({tvd:.3f}, {len(pmfs)} seeds)",
        )
    ax.annotate(
        f"floor {np.mean(floors):.3f}",
        xy=(0.03, 0.93),
        xycoords="axes fraction",
        fontsize=FONT_SIZE_ANNOTATION,
        color=REFERENCE_INK,
    )
    every = [p for _, pmfs in per_head for p in pmfs] + [reference_pmf]
    ax.set_xlim(*_occupied_limits(support, *every))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    style_axes(ax)


def _panel(
    ax, support, reference_pmf, seed_pmfs, floor, xlabel, hue, title=None, n_ticks=4
):
    """`title`: when given (the row and grid layouts), the TVD/floor read goes
    into the panel title after it instead of an in-axes annotation, which
    collides at row height; an empty string prints the read alone. `n_ticks`
    caps the x locator: the grid's narrow panels take three."""
    ax.plot(
        support,
        reference_pmf,
        color=REFERENCE_INK,
        linewidth=1.3,
        zorder=4,
        label="Kawasaki reference (certified)",
    )
    if len(seed_pmfs):
        # DNFS spelled out: matplotlib has no glossary, so a \gls{} would
        # print verbatim into the figure.
        seed_band(ax, support, seed_pmfs, hue, "DNFS")
        tvds = [
            marginal_tvd(torch.from_numpy(p), torch.from_numpy(reference_pmf))
            for p in seed_pmfs
        ]
        read = f"TVD {np.mean(tvds):.3f}, floor {floor:.3f}"
        if title is not None:
            heading = f"{title}\n{read}" if title else read
            ax.set_title(heading, fontsize=FONT_SIZE_ANNOTATION, loc="left")
        else:
            ax.annotate(
                f"TVD {np.mean(tvds):.3f}   floor {floor:.3f}",
                xy=(0.03, 0.93),
                xycoords="axes fraction",
                fontsize=FONT_SIZE_ANNOTATION,
                color=REFERENCE_INK,
            )
    else:
        ax.annotate(
            "no cell at this coupling",
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            fontsize=FONT_SIZE_ANNOTATION,
            color=CLASSICAL_HUE,
        )
    ax.set_xlim(*_occupied_limits(support, reference_pmf, *seed_pmfs))
    # Cap tick density: the default locator puts six labels on the narrow
    # sigma_c energy window and they run together at print width.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    style_axes(ax)


def build(
    results_dir,
    lattice_edge,
    heads,
    eval_subdir,
    out_path,
    n_replicates,
    layout="2x2",
    couplings=("s010", "s220"),
):
    """One results cell. `heads` is a list; >1 switches to the all-head form.
    `layout="row"` lays the four panels out in one full-width row (half the page
    height of the 2x2, decided 2026-09-03 for the float budget); panel order is
    energy, phi at sigma=0.1, then energy, phi at sigma_c. `couplings` picks the
    rows; a single coupling with the row layout gives a two-panel cell (the
    critical-only cells decided 2026-09-03)."""
    use_house_style()
    if layout == "row":
        n_couplings = len(couplings)
        width = FIGSIZE_FULL_2X2[0] * (1.0 if n_couplings == 2 else 0.62)
        figure, flat = plt.subplots(
            1, 2 * n_couplings, figsize=(width, 2.5), gridspec_kw=dict(wspace=0.45)
        )
        axes = {(r, c): flat[2 * r + c] for r in range(n_couplings) for c in range(2)}
    else:
        assert len(couplings) == 2, "the 2x2 layout needs both couplings"
        figure, grid = plt.subplots(2, 2, figsize=FIGSIZE_FULL_2X2)
        axes = {(r, c): grid[r, c] for r in range(2) for c in range(2)}
    d = lattice_edge * lattice_edge
    summary = {}

    for row, sigma_key in enumerate(couplings):
        chains = load_reference(lattice_edge, sigma_key)
        pool = torch.cat(chains)
        per_head = [
            (
                HEAD_LABEL.get(h, h),
                load_cells(results_dir, lattice_edge, sigma_key, h, eval_subdir),
            )
            for h in heads
        ]
        n_draws = next((len(c[0]["samples"]) for _, c in per_head if c), 5000)
        energy_ref = energy_pmf(pool, lattice_edge)
        phi_ref = phi_pmf(pool, lattice_edge)
        e_floor = energy_floor(chains, lattice_edge, n_draws, n_replicates)
        p_floor = phi_floor(chains, lattice_edge, n_draws, n_replicates)
        energy_axis = energy_support(lattice_edge) / d
        phi_axis = phi_support(lattice_edge)

        if len(heads) == 1:
            cells = per_head[0][1]
            panel_title = SIGMA_LABEL[sigma_key] if layout == "row" else None
            _panel(
                axes[row, 0],
                energy_axis,
                energy_ref,
                [energy_pmf(c["samples"], lattice_edge, c["weights"]) for c in cells],
                e_floor,
                "$E/d$",
                SAMPLER_HUE,
                panel_title,
            )
            _panel(
                axes[row, 1],
                phi_axis,
                phi_ref,
                [phi_pmf(c["samples"], lattice_edge, c["weights"]) for c in cells],
                p_floor,
                r"$\phi = (m_\mathrm{left} - m_\mathrm{right})/2$",
                SAMPLER_HUE,
                panel_title,
            )
        else:
            _multi_head_panel(
                axes[row, 0],
                energy_axis,
                energy_ref,
                [
                    (
                        label,
                        [
                            energy_pmf(c["samples"], lattice_edge, c["weights"])
                            for c in cells
                        ],
                    )
                    for label, cells in per_head
                ],
                [e_floor],
                "$E/d$",
            )
            _multi_head_panel(
                axes[row, 1],
                phi_axis,
                phi_ref,
                [
                    (
                        label,
                        [
                            phi_pmf(c["samples"], lattice_edge, c["weights"])
                            for c in cells
                        ],
                    )
                    for label, cells in per_head
                ],
                [p_floor],
                r"$\phi = (m_\mathrm{left} - m_\mathrm{right})/2$",
            )

        if layout == "row":
            if row == 0:
                axes[row, 0].set_ylabel("probability mass", fontsize=FONT_SIZE_LABEL)
        else:
            axes[row, 0].set_ylabel(
                f"{SIGMA_LABEL[sigma_key]}\nprobability mass", fontsize=FONT_SIZE_LABEL
            )
        summary[sigma_key] = {
            "n_draws": n_draws,
            "energy_floor": round(e_floor, 4),
            "phi_floor": round(p_floor, 4),
            "heads": {label: [c["name"] for c in cells] for label, cells in per_head},
        }

    # One figure-level legend under the panels: per-axes legends collide with
    # the TVD annotations, and all four panels carry the same curves.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=FONT_SIZE_ANNOTATION,
        frameon=False,
        loc="lower center",
        ncol=4 if layout == "row" else 2,
        bbox_to_anchor=(
            0.5,
            -0.16 if layout == "row" else (-0.10 if len(heads) > 1 else -0.02),
        ),
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(
        json.dumps(
            {
                "figure": str(out_path),
                "lattice_edge": lattice_edge,
                "heads": heads,
                "eval": eval_subdir,
                **summary,
            },
            indent=2,
        )
    )


def build_grid(results_dir, eval_subdir, out_path, n_replicates, sigma_key="s220"):
    """The critical-coupling ladder: 2 rows (energy, phi) x 4 columns (rungs).

    Each column is the single-head cell of its rung -- seed band, TVD and
    floor in the column title -- so the read is the same as the per-rung
    figures, side by side. What the grid adds is the trend the tables carry
    only as numbers: the energy marginal drifting off the reference as the
    lattice grows while phi stays near its floor. Panels do NOT share an
    x-axis: E/d and phi both narrow with size (the occupied window is set
    per panel, as in the single cells), so a shared axis would flatten the
    large rungs into spikes. Widths follow the print: four columns across
    the text width, two rows at the row layout's panel height."""
    use_house_style()
    n_cols = len(GRID_RUNGS)
    figure, grid = plt.subplots(
        2,
        n_cols,
        figsize=(FIGSIZE_FULL_2X2[0], 3.6),
        gridspec_kw=dict(wspace=0.5, hspace=0.7),
    )
    summary = {}
    for col, lattice_edge in enumerate(GRID_RUNGS):
        head, require_token, radius = GRID_CELLS[lattice_edge]
        d = lattice_edge * lattice_edge
        chains = load_reference(lattice_edge, sigma_key)
        pool = torch.cat(chains)
        cells = load_cells(
            results_dir, lattice_edge, sigma_key, head, eval_subdir, require_token
        )
        n_draws = cells[0]["samples"].shape[0] if cells else 5000
        energy_ref = energy_pmf(pool, lattice_edge)
        phi_ref = phi_pmf(pool, lattice_edge)
        e_floor = energy_floor(chains, lattice_edge, n_draws, n_replicates)
        p_floor = phi_floor(chains, lattice_edge, n_draws, n_replicates)
        energy_seeds = [
            energy_pmf(c["samples"], lattice_edge, c["weights"]) for c in cells
        ]
        phi_seeds = [phi_pmf(c["samples"], lattice_edge, c["weights"]) for c in cells]
        title = f"${lattice_edge}\\times{lattice_edge}$, {radius}"
        _panel(
            grid[0, col],
            energy_support(lattice_edge) / d,
            energy_ref,
            energy_seeds,
            e_floor,
            "$E/d$",
            SAMPLER_HUE,
            title,
            n_ticks=3,
        )
        _panel(
            grid[1, col],
            phi_support(lattice_edge),
            phi_ref,
            phi_seeds,
            p_floor,
            r"$\phi$",
            SAMPLER_HUE,
            "",
            n_ticks=3,
        )
        tvd = lambda seeds, ref: [  # noqa: E731
            marginal_tvd(torch.from_numpy(p), torch.from_numpy(ref)) for p in seeds
        ]
        summary[lattice_edge] = {
            "cells": [c["name"] for c in cells],
            "n_draws": n_draws,
            "energy_tvd_mean_sd": [
                round(float(np.mean(tvd(energy_seeds, energy_ref))), 4),
                round(float(np.std(tvd(energy_seeds, energy_ref))), 4),
            ],
            "energy_floor": round(e_floor, 4),
            "phi_tvd_mean_sd": [
                round(float(np.mean(tvd(phi_seeds, phi_ref))), 4),
                round(float(np.std(tvd(phi_seeds, phi_ref))), 4),
            ],
            "phi_floor": round(p_floor, 4),
        }
    grid[0, 0].set_ylabel("probability mass", fontsize=FONT_SIZE_LABEL)
    grid[1, 0].set_ylabel("probability mass", fontsize=FONT_SIZE_LABEL)
    handles, labels = grid[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=FONT_SIZE_ANNOTATION,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.savefig(out_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(
        json.dumps(
            {
                "figure": str(out_path),
                "eval": eval_subdir,
                "sigma": sigma_key,
                "n_replicates": n_replicates,
                **{str(k): v for k, v in summary.items()},
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice-edge", type=int, choices=(8, 16, 20, 24))
    parser.add_argument(
        "--heads",
        default="thp",
        help="comma-separated run-dir head tokens, best "
        "first; more than one switches to the all-head "
        "form (mean lines, no seed bands)",
    )
    parser.add_argument(
        "--eval-subdir", default="eval_ema", choices=("eval", "eval_ema")
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "03_hard"))
    parser.add_argument("--n-replicates", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout", choices=("2x2", "row", "grid"), default="2x2")
    parser.add_argument(
        "--couplings",
        default="s010,s220",
        help="comma-separated coupling keys; 's220' alone gives the critical-only cell",
    )
    args = parser.parse_args()
    if args.layout == "grid":
        build_grid(
            args.results_dir, args.eval_subdir, Path(args.out), args.n_replicates
        )
        return
    assert args.lattice_edge is not None, (
        "--lattice-edge is required outside --layout grid"
    )
    build(
        args.results_dir,
        args.lattice_edge,
        [h.strip() for h in args.heads.split(",") if h.strip()],
        args.eval_subdir,
        Path(args.out),
        args.n_replicates,
        args.layout,
        tuple(k.strip() for k in args.couplings.split(",") if k.strip()),
    )


if __name__ == "__main__":
    main()
