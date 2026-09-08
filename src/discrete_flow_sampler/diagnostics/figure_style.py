"""House figure style for every thesis figure.

One module owns the palette and the uncertainty grammar. The five-hue
palette was validated for colour-vision deficiency on the light surface
(worst adjacent CVD dE 9.1, normal dE 21.6, dataviz six-check validator
2026-08-13). Two hues sit below the 3:1 surface-contrast bar, so every
figure carries direct labels or a legend; colour is never the only
identity channel.

Colour follows the role, never the figure: reference/ground truth is ink,
our sampler is the same blue everywhere, classical baselines stay in one
family. The one exception is a figure whose contrast is a parameter level
(two lambdas, two lattice sizes) where every series shares one role;
``parameter_ramp`` gives that role a lightness ramp, light = low level,
dark = high level, which survives greyscale print.

Uncertainty grammar (one convention per data shape):
- curves with seed spread   -> ``seed_band``: mean line + min-max band,
  n in the legend entry.
- any other shaded interval -> ``uncertainty_band``: same ribbon, for
  series with their own centre marks. A band implies interpolation along
  a continuous x, which is why it is wrong for the two cases below.
- point estimates           -> ``point_errorbars``: capped bars, for
  categorical x or one experiment per abscissa.
- seed-by-seed structure    -> ``per_seed_traces``: thin per-seed lines,
  only when the seed split itself is the figure's point.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, to_hex, to_rgb

# --- roles (never reassign per figure) -----------------------------------
REFERENCE_INK = "#1a1a19"  # exact enumeration / certified chain / TI truth
REFERENCE_FILL = "#4a4943"  # the reference role as a filled area (bars,
# patches): a bar-sized block of ink dominates the panel. Ink for strokes.
SAMPLER_HUE = "#2a78d6"  # our sampler (DNFS / masked attention), every chapter
NEURAL_COMPARATOR_HUE = "#1baf7a"  # second neural head or matched neural baseline
CLASSICAL_HUE = "#eda100"  # classical MCMC baseline (Kawasaki nonlocal, Gibbs, VC-SGC)
CLASSICAL_ALT_HUE = "#8e63c5"  # second classical variant (Kawasaki local)
HARD_DELTA_HUE = "#c8503c"  # the hard-constraint delta / limit marker
ANALYTIC_GUIDE = "#6f6e66"  # analytic envelopes and guides (dashed, muted)
MUTED = "#6f6e66"
GRID = "#e6e5df"

# Spin-lattice rendering, established at background.tex fig:ising-phases:
# indigo = spin -1 (down), gold = spin +1 (up); separates in greyscale.
# Lattices plot as (x + 1) / 2 so index 0 maps to down. Every raw-spin
# figure uses this map.
SPIN_DOWN_COLOUR = "#3B3A6B"
SPIN_UP_COLOUR = "#F2C14E"
SPIN_CMAP = ListedColormap([SPIN_DOWN_COLOUR, SPIN_UP_COLOUR])

# House geometry: figures are designed at print size, 1:1, so a point of
# script font is a point on the page. Two tex widths only: \textwidth for
# multi-panel figures, 0.72\textwidth for single panels (A4, 2.5 cm margins,
# 11 pt body -> text block 6.3 in). Label size 9 pt matches the
# \footnotesize house tables; dpi 300 is print quality at these sizes.
FULL_WIDTH_IN = 6.3
SINGLE_PANEL_WIDTH_IN = 4.54
FIGSIZE_FULL_1X2 = (FULL_WIDTH_IN, 2.9)
FIGSIZE_FULL_1X2_SHORT = (FULL_WIDTH_IN, 2.4)  # 1x2 with a few marks per
# panel, not a dense curve; at 2.9 in it printed 7.4 cm tall.
FIGSIZE_FULL_1X4 = (FULL_WIDTH_IN, 2.5)  # four panels in a row: 6.4 cm tall
# vs 14.1 cm for the old 2x2. Width stays 6.3 in and the panels get narrow;
# an 18 in figure shrunk by LaTeX printed ~3 pt type.
FIGSIZE_FULL_2X2 = (FULL_WIDTH_IN, 5.6)
FIGSIZE_FULL_WIDE_SINGLE = (FULL_WIDTH_IN, 2.6)  # one panel at full width,
# legend below the axes: a six-entry legend inside covers the peak.
FIGSIZE_SINGLE = (SINGLE_PANEL_WIDTH_IN, 3.2)
FIGSIZE_SINGLE_2X2 = (SINGLE_PANEL_WIDTH_IN, 3.8)  # 2x2 at 0.72\textwidth:
# the 1x4 at full width left ~0.63 in of data axis per panel (60% went to
# labels); this trades +3.3 cm of print height for ~2.6x the data area.

FONT_SIZE_TITLE = 9
FONT_SIZE_LABEL = 9
FONT_SIZE_ANNOTATION = 8
SAVEFIG_DPI = 300

RC_PARAMS = {
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_LABEL,
    "ytick.labelsize": FONT_SIZE_LABEL,
    "legend.fontsize": FONT_SIZE_ANNOTATION,
    "figure.dpi": 110,
    "savefig.dpi": SAVEFIG_DPI,
    # Body math is Computer Modern; matching mathtext keeps axis math
    # indistinguishable from the surrounding text.
    "mathtext.fontset": "cm",
    "axes.edgecolor": MUTED,
    "text.color": REFERENCE_INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}


def use_house_style() -> None:
    """Install the shared rcParams; call once at the top of a plot script."""
    mpl.rcParams.update(RC_PARAMS)


def style_axes(ax, grid_axis: str = "y") -> None:
    """Shared axis treatment: recessive grid below the data, no top or right
    spine, muted remaining spines and ticks."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=FONT_SIZE_LABEL)


def parameter_ramp(hue, n_levels, lightest=0.55):
    """Lightness ramp within one role, ordered low level -> high level.

    The darkest entry is the role's own hue, so a single-level figure is
    unchanged. ``lightest`` is the blend toward white at the low end; above
    about 0.65 a 1.6pt line disappears on the light surface.
    """
    base = np.array(to_rgb(hue))
    if n_levels == 1:
        return [hue]
    blend_fractions = np.linspace(lightest, 0.0, n_levels)
    return [
        to_hex(base + (np.ones(3) - base) * fraction) for fraction in blend_fractions
    ]


BAND_ALPHA = 0.18  # one alpha for every shaded interval: two overlapping
# bands (0.33 effective) still read as a third shade, not an opaque block.


def uncertainty_band(ax, x, lower, upper, hue, zorder=2, label=None, step=False):
    """The house shaded interval: series hue, BAND_ALPHA, no edge line.

    No edge because a stroked band boundary reads as a data curve. zorder 2
    sits above the grid (0) and below the centre line or markers (3+).
    For point estimates use ``point_errorbars``; see the module docstring.
    ``step`` fills mid-centred steps, matching ``seed_band(step=True)``.
    """
    return ax.fill_between(
        x,
        lower,
        upper,
        color=hue,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=zorder,
        label=label,
        step="mid" if step else None,
    )


def seed_band(ax, x, per_seed_values, hue, label, step=False):
    """Mean line + min-max shaded band for a family of seed curves; the
    legend entry carries n. min-max rather than +/-sd because seed counts
    are 3-6.

    ``step`` draws mean and band as mid-centred steps. Use it for every
    probability mass function on a discrete support (the marginal panels):
    a line through the levels invents mass between them, and drawn next to
    a stepped reference it made the sampler look smoother than the chain.
    Training curves keep the default line.
    """
    per_seed_values = np.asarray(per_seed_values)
    n_seeds = per_seed_values.shape[0]
    mean = per_seed_values.mean(axis=0)
    uncertainty_band(
        ax,
        x,
        per_seed_values.min(axis=0),
        per_seed_values.max(axis=0),
        hue,
        step=step,
    )
    draw = ax.step if step else ax.plot
    draw(
        x,
        mean,
        color=hue,
        linewidth=1.6,
        zorder=3,
        label=f"{label} (mean, band = min-max over {n_seeds} seeds)",
        **({"where": "mid"} if step else {}),
    )


def point_errorbars(ax, x, y, yerr, hue, label, marker="o"):
    """Discrete capped error bars for point estimates (replicate spread or
    a stated interval; state which in the legend label)."""
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=hue,
        label=label,
        marker=marker,
        markersize=4,
        linestyle="none",
        capsize=2.5,
        linewidth=1.2,
        zorder=3,
    )


def per_seed_traces(ax, x, per_seed_values, hue, label, highlight_index=None):
    """Thin per-seed lines for figures whose point is the seed-by-seed
    split. ``highlight_index`` draws one seed at full weight."""
    per_seed_values = np.asarray(per_seed_values)
    for index, trace in enumerate(per_seed_values):
        is_highlighted = index == highlight_index
        ax.plot(
            x,
            trace,
            color=hue,
            linewidth=1.6 if is_highlighted else 0.9,
            alpha=1.0 if is_highlighted else 0.55,
            zorder=3 if is_highlighted else 2,
            label=label if index == 0 else None,
        )
