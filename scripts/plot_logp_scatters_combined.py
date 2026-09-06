"""Archived 3x2 scatter for app:logp-scatters, one row per paradigm.

ARCHIVED FIGURE (6 September 2026). The rationale below is historical:
DNFS path weights cannot generally be inverted into an endpoint log-density.
For configuration-probability validation use scripts/configuration_calibration_4x4.py
and scripts/plot_configuration_calibration_4x4.py instead. This script remains
only to reproduce the retired image, whose density interpretation was incorrect.

The shared-axis layout fits all three paradigms on one appendix page.
Each chapter's `panel_series()` supplies the data:

    experiments/constrained_hard_03/analysis/plot_logp_scatter_4x4.py
    experiments/dnfs_baseline_01/analysis/10_logp_scatter_4x4.py
    experiments/constrained_soft_02/analysis/23_logp_scatter_4x4.py

The hard row enumerates the conditional over C(16,8) = 12,870 feasible states.
The other rows enumerate all 2^16 states under the plain or penalised Ising
log-density. Soft panels vary composition (c = 0.50, 0.80) at a single coupling;
hard and unconstrained panels vary coupling (sigma = 0.1, sigma_c). Panel titles
and the caption distinguish these axes.
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

    # Render at the printed width (6.3 in) to preserve label sizes.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7})
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.15))

    rows = [(name, importlib.import_module(path).panel_series())
            for name, path in ROWS]

    # Shared limits keep the same scale across all six panels.
    #
    # Taken from the PLOTTED points, not from each panel's own `lims`: the
    # hard panels set theirs from the full enumerated support, whose
    # low-probability tail is never drawn, and that alone stretched the
    # sigma_c panel to -25 against a cloud ending near -15.
    finite = [t for _, panels in rows for panel in panels
              for _l, _c, x, y in panel["series"] for t in (x, y)]
    lo = min(float(t.min()) for t in finite) - 0.3
    hi = max(float(t.max()) for t in finite) + 0.3
    shared_lims = (lo, hi)

    for row, (row_name, panels) in enumerate(rows):
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            for label, colour, x, y in panel["series"]:
                ax.scatter(x, y, s=3, alpha=0.25, lw=0, color=colour,
                           label=label, rasterized=True)
            ax.plot(shared_lims, shared_lims, color="black", lw=0.8, zorder=0)
            ax.set_xlim(shared_lims)
            ax.set_ylim(shared_lims)
            ax.set_title(panel["title"], fontsize=9, pad=3)
            ax.set_xlabel(panel["xlabel"], labelpad=1)
        # Label each row on the left; the shared quantity uses fig.supylabel.
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
