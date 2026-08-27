"""The whole of app:logp-scatters as ONE 3x2 figure, one row per paradigm.

Replaces the three separate floats (hard / unconstrained / soft) that the
appendix carried until s76. Two reasons, in order of importance:

  * IT IS THE BETTER EXHIBIT. The appendix exists to compare per-configuration
    recovery ACROSS the three constraint paradigms at the one size where an
    exact reference exists. Three floats on three pages make that comparison a
    page-turning exercise; three rows on one page make it immediate, and the
    shared vertical axis means a wider cloud is visibly a wider cloud rather
    than something the reader has to infer from two quoted standard deviations.
  * IT FITS ON ONE PAGE. Three floats each carry a section head, a paragraph
    and a caption -- about 4.8 in of the 9.72 in text height in furniture
    alone, which forces the figures below the size at which a scatter's
    diagonal stays readable. One float pays that overhead once.

The data comes from each chapter's own script via its `panel_series()`, never
re-derived here, so this figure and the per-chapter ones can never disagree:

    experiments/constrained_hard_03/analysis/plot_logp_scatter_4x4.py
    experiments/dnfs_baseline_01/analysis/10_logp_scatter_4x4.py
    experiments/constrained_soft_02/analysis/23_logp_scatter_4x4.py

WHAT THE ROWS DO NOT SHARE, and why that is correct rather than sloppy.
The hard row's horizontal axis is the enumerated CONDITIONAL over the
C(16,8) = 12,870 feasible states, because that chapter conditions on a fixed
composition; the other two enumerate all 2^16 states, the unconstrained one
under the plain Ising log-density and the soft one under the penalised one.
And the soft row's two panels are COMPOSITIONS (c = 0.50, 0.80) where the
upper two rows' are COUPLINGS (sigma = 0.1, sigma_c): the soft 4x4 family is
run at a single coupling, so two sigma panels there would print the same
distribution twice. Every panel is therefore titled in its own terms and the
caption states the axis difference.
"""
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Module names begin with a digit, so they cannot be imported by name.
ROWS = (
    ("Hard", "experiments.constrained_hard_03.analysis.plot_logp_scatter_4x4"),
    ("Unconstrained", "experiments.dnfs_baseline_01.analysis.10_logp_scatter_4x4"),
    ("Soft", "experiments.constrained_soft_02.analysis.23_logp_scatter_4x4"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="output PNG path (the Overleaf assets file)")
    args = parser.parse_args(argv)

    # Style annex: in-figure labels 9pt, annotations 8pt. Figure is authored at
    # the printed width (6.3 in = \textwidth) so the point sizes are true.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7})
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.15))

    for row, (row_name, module_path) in enumerate(ROWS):
        panels = importlib.import_module(module_path).panel_series()
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            for label, colour, x, y in panel["series"]:
                ax.scatter(x, y, s=3, alpha=0.25, lw=0, color=colour,
                           label=label, rasterized=True)
            ax.plot(panel["lims"], panel["lims"], color="black", lw=0.8,
                    zorder=0)
            ax.set_xlim(panel["lims"])
            ax.set_ylim(panel["lims"])
            ax.set_title(panel["title"], fontsize=9, pad=3)
            ax.set_xlabel(panel["xlabel"], labelpad=1)
        # Row identity on the left panel; the shared quantity goes once, in
        # the middle row, so the three names read as a column of labels.
        axes[row, 0].set_ylabel(row_name, fontweight="bold", labelpad=2)
        if len(panels[0]["series"]) > 1:
            axes[row, 0].legend(loc="upper left", frameon=False,
                                handletextpad=0.1, borderpad=0.1)

    fig.supylabel(r"estimated $\log \hat q(x)$ (offset removed)", fontsize=9,
                  x=0.005)
    fig.tight_layout(h_pad=0.8, w_pad=1.0, rect=(0.02, 0, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[scatter] wrote {args.out}")


if __name__ == "__main__":
    main()
