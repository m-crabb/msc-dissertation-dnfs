"""House figure style for every thesis figure.

One module owns the palette and the uncertainty grammar so that a reader
moving between chapters sees one visual language. The palette anchors on
the two newest deliberately-designed figure sets (the 4x4 demo pack and
the 8x8 probe figures) and was validated for colour-vision deficiency as
a five-hue set on the light surface (worst adjacent CVD dE 9.1, normal
dE 21.6, dataviz six-check validator 2026-08-13). Two hues sit below the
3:1 surface-contrast bar, so every figure must carry direct labels or a
legend naming its series -- colour is never the only identity channel.

Colour follows the ROLE, never the figure: the reference/ground truth is
always ink, our sampler is always the same blue, classical baselines stay
in one family. A new figure picks roles, not colours.

Uncertainty grammar (one convention per data shape):
- curves with seed spread   -> ``seed_band``: mean line + shaded min-max
  band, band labelled with n in the legend entry.
- point estimates           -> ``point_errorbars``: discrete capped bars.
- seed-by-seed structure    -> ``per_seed_traces``: thin per-seed lines,
  ONLY when the seed split itself is the figure's point (e.g. the
  penalty-variance traces, where one escaping seed is the story).
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# --- roles (never reassign per figure) -----------------------------------
REFERENCE_INK = "#1a1a19"          # exact enumeration / certified chain / TI truth
SAMPLER_HUE = "#2a78d6"            # our sampler (DNFS / masked attention), every chapter
NEURAL_COMPARATOR_HUE = "#1baf7a"  # second neural head or matched neural baseline
CLASSICAL_HUE = "#eda100"          # classical MCMC baseline (Kawasaki nonlocal, Gibbs, VC-SGC)
CLASSICAL_ALT_HUE = "#8e63c5"      # second classical variant (Kawasaki local)
HARD_DELTA_HUE = "#c8503c"         # the hard-constraint delta / limit marker
ANALYTIC_GUIDE = "#6f6e66"         # analytic envelopes and guides (dashed, muted)
MUTED = "#6f6e66"
GRID = "#e6e5df"

FONT_SIZE_TITLE = 9
FONT_SIZE_LABEL = 8
FONT_SIZE_ANNOTATION = 7
SAVEFIG_DPI = 180

RC_PARAMS = {
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_LABEL,
    "ytick.labelsize": FONT_SIZE_LABEL,
    "legend.fontsize": FONT_SIZE_ANNOTATION,
    "figure.dpi": 110,
    "savefig.dpi": SAVEFIG_DPI,
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
    """The shared axis treatment: recessive grid below the data, no top or
    right spine, muted remaining spines/ticks (lifted verbatim from the two
    anchor scripts so restyled figures match them exactly)."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=FONT_SIZE_LABEL)


def seed_band(ax, x, per_seed_values, hue, label):
    """Mean line + min-max shaded band for a family of seed curves; the
    legend entry carries n so the band's meaning is on the figure, not in
    the caption. min-max (not +/-sd) because thesis seed counts are 3-6:
    a standard deviation over so few seeds implies a precision it lacks."""
    per_seed_values = np.asarray(per_seed_values)
    n_seeds = per_seed_values.shape[0]
    mean = per_seed_values.mean(axis=0)
    ax.fill_between(x, per_seed_values.min(axis=0), per_seed_values.max(axis=0),
                    color=hue, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mean, color=hue, linewidth=1.6, zorder=3,
            label=f"{label} (mean, band = min-max over {n_seeds} seeds)")


def point_errorbars(ax, x, y, yerr, hue, label, marker="o"):
    """Discrete capped error bars for point estimates (replicate spread or
    a stated interval; state which in the legend label)."""
    ax.errorbar(x, y, yerr=yerr, color=hue, label=label, marker=marker,
                markersize=4, linestyle="none", capsize=2.5, linewidth=1.2,
                zorder=3)


def per_seed_traces(ax, x, per_seed_values, hue, label, highlight_index=None):
    """Thin per-seed lines; reserved for figures whose point IS the
    seed-by-seed split. ``highlight_index`` draws one seed at full weight
    (the escaping seed of the penalty-variance figure)."""
    per_seed_values = np.asarray(per_seed_values)
    for index, trace in enumerate(per_seed_values):
        is_highlighted = index == highlight_index
        ax.plot(x, trace, color=hue,
                linewidth=1.6 if is_highlighted else 0.9,
                alpha=1.0 if is_highlighted else 0.55,
                zorder=3 if is_highlighted else 2,
                label=label if index == 0 else None)
