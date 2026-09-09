"""The ladder figure for the hard chapter: ESS and cost against lattice size, at
sigma_c.

Two panels over the five rungs the chapter prints. (a) Frozen ESS fraction at the exact
critical coupling for selected heads that have a cell at that rung, plus the GFlowNet
(TB) comparator; (b) FLOP per effective sample at sigma_c, with the certified Kawasaki
chain as the classical line. The chapter's central trend -- the patch head's ESS falling
slowly with size while the causal-stream bands need a second sweep to survive -- is
otherwise spread across four tables.

VALUES ARE THE PRINTED HOUSE-TABLE CELLS (tab:eval-hard-{4x4,8x8,16x16,20x20,24x24}),
each of which is itself the output of its emitter and was re-verified cell by cell on
2026-09-03 (24x24 added 2026-09-06 from house_table_24x24.py; the GFlowNet TB 20x20
point added 2026-09-07 from house_table_20x20.py, tag 20260904-gfn-d400-tb; its 24x24
point 2026-09-08 from house_table_24x24.py, tag 20260906-gfn-d576-sc). The read
convention follows the tables: raw at 4x4 and 8x8, averaged (EMA) at 16x16 and above;
the marker fill encodes which. The 24x24 rung is sigma_c only, two patch radii. Update
this dict when a table changes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE,
    MUTED,
    NEURAL_COMPARATOR_HUE,
    SAMPLER_HUE,
    parameter_ramp,
    style_axes,
    use_house_style,
)

SIDES = (4, 8, 16, 20, 24)
EMA_RUNGS = {16, 20, 24}

# (label, hue, linestyle, {side: (ess, flop_per_es)})
LIGHT, MID, DARK, DARKEST = parameter_ramp(SAMPLER_HUE, 4)
GREY_LIGHT, GREY_DARK = parameter_ramp(MUTED, 2)
SERIES = [
    ("Mask-one", "#1a1a19", ":", {4: (0.975, 3.8e9), 8: (0.903, 8.0e10)}),
    (
        "Attention, 1 sweep",
        GREY_LIGHT,
        "-",
        {4: (0.975, 8.6e8), 8: (0.781, 2.1e10), 16: (0.015, 3.6e14)},
    ),
    (
        "Attention, 2 sweeps + field",
        GREY_DARK,
        "-",
        {4: (0.983, 1.3e9), 8: (0.918, 2.4e10), 16: (0.812, 3.0e11)},
    ),
    (
        "Prefix-sum, 2 sweeps + field",
        GREY_DARK,
        "--",
        {4: (0.981, 1.2e9), 8: (0.931, 1.9e10), 16: (0.823, 1.6e11)},
    ),
    (
        "Patch, $R=1$",
        LIGHT,
        "-",
        {4: (0.992, 2.6e8), 8: (0.921, 3.2e9), 16: (0.638, 6.0e10)},
    ),
    ("Patch, $R=2$", MID, "-", {16: (0.826, 5.3e10), 20: (0.702, 1.4e11)}),
    ("Patch, $R=3$", DARK, "-", {20: (0.798, 1.4e11), 24: (0.605, 3.6e11)}),
    ("Patch, $R=4$", DARKEST, "-", {24: (0.720, 3.6e11)}),
    (
        "GFlowNet TB",
        NEURAL_COMPARATOR_HUE,
        "-",
        {
            4: (0.991, 3.2e6),
            8: (0.955, 1.3e7),
            16: (0.883, 6.4e7),
            20: (0.735, 1.4e8),
            24: (0.144, 2.0e9),
        },
    ),
]
# Between-seed SD from the same printed cells, in SERIES order. These are
# point uncertainties, not confidence bands between different trained sizes.
ESS_SD = [
    {4: 0.006, 8: 0.022},
    {4: 0.003, 8: 0.016, 16: 0.019},
    {4: 0.002, 8: 0.005, 16: 0.010},
    {4: 0.002, 8: 0.003, 16: 0.005},
    {4: 0.006, 8: 0.003, 16: 0.029},
    {16: 0.015, 20: 0.025},
    {20: 0.004, 24: 0.054},
    {24: 0.007},
    {4: 0.002, 8: 0.005, 16: 0.018, 20: 0.068, 24: 0.095},
]
KAWASAKI_FLOP_PER_ES = {4: 1.6e3, 8: 8.4e3, 16: 3.4e5, 20: 8.2e5, 24: 1.6e6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ess_ladder.png"))
    args = parser.parse_args()
    use_house_style()
    fig, (ax_ess, ax_cost) = plt.subplots(1, 2, figsize=(6.3, 3.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.34, top=0.93, wspace=0.38)
    handles = []
    for (label, hue, style, cells), deviations in zip(SERIES, ESS_SD):
        sizes = sorted(cells)
        ax_ess.plot(
            sizes,
            [cells[size][0] for size in sizes],
            style,
            color=hue,
            lw=1.15,
            zorder=3,
        )
        # Keep a collapsed cell's ESS point, but omit its meaningless cost.
        active = [size for size in sizes if cells[size][0] > 0.05]
        ax_cost.plot(
            active,
            [cells[size][1] for size in active],
            style,
            color=hue,
            lw=1.15,
            zorder=3,
        )
        for size in sizes:
            ess, cost = cells[size]
            marker = dict(
                marker="o",
                ms=3.7,
                mec=hue,
                mfc=hue if size in EMA_RUNGS else "white",
                zorder=4,
            )
            ax_ess.errorbar(
                size,
                ess,
                yerr=deviations[size],
                color=hue,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                **marker,
            )
            if size in active:
                ax_cost.plot(size, cost, color=hue, **marker)
        handles.append(Line2D([], [], color=hue, ls=style, lw=1.3, label=label))

    ax_cost.plot(
        SIDES,
        [KAWASAKI_FLOP_PER_ES[size] for size in SIDES],
        color=CLASSICAL_HUE,
        lw=1.5,
    )
    handles.append(
        Line2D([], [], color=CLASSICAL_HUE, lw=1.5, label="Kawasaki reference")
    )
    ax_ess.set_ylabel("ESS fraction")
    ax_ess.set_ylim(0, 1.045)
    ax_ess.set_yticks([step / 5 for step in range(6)])
    ax_cost.set_ylabel("FLOP per effective sample")
    ax_cost.set_yscale("log")
    for ax, identifier in ((ax_ess, "(a)"), (ax_cost, "(b)")):
        ax.set_xlabel(r"Lattice edge $D$")
        ax.set_xticks(SIDES)
        ax.set_xlim(3, 25.5)
        style_axes(ax)
        ax.text(0, 1.025, identifier, transform=ax.transAxes, fontsize=9)
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.52, 0.10),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.35,
        handletextpad=0.55,
        labelspacing=0.5,
    )
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
