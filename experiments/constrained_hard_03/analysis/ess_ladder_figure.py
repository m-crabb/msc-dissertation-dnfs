"""The ladder figure for the hard chapter: ESS and cost against lattice size, at sigma_c.

Two panels over the four rungs the chapter prints. (a) Frozen ESS fraction at the exact
critical coupling for every head that has a cell at that rung, plus the GFlowNet (TB)
comparator; (b) FLOP per effective sample at sigma_c, with the certified Kawasaki chain as
the classical line. The chapter's central trend -- the patch head's ESS falling slowly
with size while the causal-stream bands need a second sweep to survive -- is otherwise
spread across four tables.

VALUES ARE THE PRINTED HOUSE-TABLE CELLS (tab:eval-hard-{4x4,8x8,16x16,20x20}), each of
which is itself the output of its emitter and was re-verified cell by cell on 2026-09-03.
The read convention follows the tables: raw at 4x4 and 8x8, averaged (EMA) at 16x16 and
20x20; the marker fill encodes which. Update this dict when a table changes.
"""
import argparse
from pathlib import Path

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE, FONT_SIZE_ANNOTATION, FULL_WIDTH_IN, MUTED,
    NEURAL_COMPARATOR_HUE, SAMPLER_HUE, SAVEFIG_DPI, parameter_ramp,
    style_axes, use_house_style)
import matplotlib.pyplot as plt

SIDES = (4, 8, 16, 20)
EMA_RUNGS = {16, 20}

# (label, hue, linestyle, {side: (ess, flop_per_es)})
LIGHT, MID, DARK = parameter_ramp(SAMPLER_HUE, 3)
GREY_LIGHT, GREY_DARK = parameter_ramp(MUTED, 2)
SERIES = [
    ("mask-one reference", "#1a1a19", ":",
     {4: (0.975, 3.8e9), 8: (0.903, 8.0e10)}),
    ("masked-attention band, 1 sweep", GREY_LIGHT, "-",
     {4: (0.975, 8.6e8), 8: (0.781, 2.1e10), 16: (0.015, 3.6e14)}),
    ("masked-attention band, 2 sweeps + exact field", GREY_DARK, "-",
     {4: (0.983, 1.3e9), 8: (0.918, 2.4e10), 16: (0.812, 3.0e11)}),
    ("prefix-sum band, 2 sweeps + exact field", GREY_DARK, "--",
     {4: (0.981, 1.2e9), 8: (0.931, 1.9e10), 16: (0.823, 1.6e11)}),
    ("two-hole patch head, $R=1$", LIGHT, "-",
     {4: (0.992, 2.6e8), 8: (0.921, 3.2e9), 16: (0.638, 6.0e10)}),
    ("two-hole patch head, $R=2$", MID, "-",
     {16: (0.826, 5.3e10), 20: (0.702, 1.4e11)}),
    ("two-hole patch head, $R=3$", DARK, "-",
     {20: (0.798, 1.4e11)}),
    ("GFlowNet (TB)", NEURAL_COMPARATOR_HUE, "-",
     {4: (0.991, 3.2e6), 8: (0.955, 1.3e7)}),
]
KAWASAKI_FLOP_PER_ES = {4: 1.6e3, 8: 8.4e3, 16: 3.4e5, 20: 8.2e5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ess_ladder.png"))
    args = parser.parse_args()
    use_house_style()
    fig, (ax_ess, ax_cost) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.7))

    for label, hue, style, cells in SERIES:
        xs = sorted(cells)
        ys = [cells[x][0] for x in xs]
        costs = [cells[x][1] for x in xs]
        # a collapsed cell (ESS at the floor) has no meaningful cost; plot its ESS only
        alive = [x for x in xs if cells[x][0] > 0.05]
        ax_ess.plot(xs, ys, style, color=hue, lw=1.3, zorder=3, label=label)
        ax_cost.plot(alive, [cells[x][1] for x in alive], style, color=hue, lw=1.3, zorder=3)
        for x, y, c in zip(xs, ys, costs):
            filled = x in EMA_RUNGS
            kw = dict(marker="o", ms=4.5, mec=hue, mfc=hue if filled else "white", zorder=4)
            ax_ess.plot([x], [y], **kw)
            if x in alive:
                ax_cost.plot([x], [c], **kw)

    ks = sorted(KAWASAKI_FLOP_PER_ES)
    ax_cost.plot(ks, [KAWASAKI_FLOP_PER_ES[k] for k in ks], "-", color=CLASSICAL_HUE,
                 lw=1.6, zorder=3)
    ax_ess.plot([], [], "-", color=CLASSICAL_HUE, lw=1.6, label="Kawasaki chain (certified reference)")

    ax_ess.set_ylabel("ESS fraction at $\\sigma_c$")
    ax_ess.set_ylim(0, 1.04)
    ax_cost.set_ylabel("FLOP per effective sample at $\\sigma_c$")
    ax_cost.set_yscale("log")
    for ax in (ax_ess, ax_cost):
        ax.set_xlabel("lattice edge $D$")
        ax.set_xticks(SIDES)
        ax.set_xlim(3, 24)
        style_axes(ax)
    ax_ess.text(0.03, 0.05, "open: raw read (4$\\times$4, 8$\\times$8)\nfilled: averaged read (16$\\times$16, 20$\\times$20)",
                transform=ax_ess.transAxes, fontsize=FONT_SIZE_ANNOTATION - 1.5, color=MUTED, va="bottom")
    handles, labels = ax_ess.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FONT_SIZE_ANNOTATION - 1, frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    ax_ess.set_title("(a) effective sample size", fontsize=FONT_SIZE_ANNOTATION + 1, loc="left")
    ax_cost.set_title("(b) cost", fontsize=FONT_SIZE_ANNOTATION + 1, loc="left")
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
