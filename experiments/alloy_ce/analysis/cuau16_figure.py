"""The 16-site Cu-Au exhibit: ordered structures, composition histograms and the canonical F(c),
every reading against exact truth.

Layout (MetaDNS Fig. 5 as the model, exact enumeration where they have a chain):
  (a,b)   Cu3Au (L1_2) and CuAu (L1_0) conventional cells, lit-sphere renders;
  (c,d,e) Au-concentration marginal of the free ensemble at 1200 / 680 / 500 K: exact bars
          (all 2^16 states weighted by exp(-beta E)) under the free cell's raw draws, mean
          step over seeds with a min-max band (house seed grammar, n in the legend);
  (f)     canonical free energy per site F(c) = -log Z_c / (beta d) at 500 K over every slice
          n_Au = 0..16 (ink, a discrete curve: one value per slice), with the specialist cells
          (one per composition) and the composition-amortised cell (one checkpoint, its draws
          split by slice) as importance-sampling estimates, mean over seeds;
  (g)     the residual F_IS - F_exact of the same points in meV/site, min-max over seeds,
          where the result actually lives: on (f) every point sits on the curve.

The histograms show what the sampler itself produces (the MetaDNS panels are KDEs of raw x_Au);
the free energy is by definition the IS normaliser read off the weights (paper Eq. 37 gives the
bound; -logmeanexp is the estimate). Units are absolute meV/site to match the table's dF column,
where -41 meV/site at CuAu is a reading an alloy reader can use.

Usage: pixi run -e dev python -m experiments.alloy_ce.analysis.cuau16_figure \\
           --free "results/02_constrained_soft/A1_cuau16_T1200*fc" \\
                  "results/02_constrained_soft/A1_cuau16_T680*fc" \\
                  "results/02_constrained_soft/A1_cuau16_T500_letf_50k_house_seed*free" \\
           --temperatures 1200 680 500 \\
           --specialists "results/03_hard/H2_cuau16_c[234]*_T500_mask_one_50k_house_seed*fc" \\
                         "results/03_hard/H2_cuau16_c50_T500_mask_one_50k_house_seed*house" \\
           --amortised "results/03_hard/H2_cuau16_camort*fc" --out assets/cuau16_exhibit.pdf
"""

import argparse
import glob
import itertools
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.alloy_ce.analysis.fcc_render import (
    JMOL,
    conventional_cell,
    draw_structure,
    rectangular_tiling,
)
from experiments.alloy_ce.probes.patch_reach_probe import ordered_states
from matplotlib.gridspec import GridSpec

from discrete_flow_sampler.diagnostics.figure_style import (
    BAND_ALPHA,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FULL_WIDTH_IN,
    MUTED,
    NEURAL_COMPARATOR_HUE,
    REFERENCE_FILL,
    REFERENCE_INK,
    SAMPLER_HUE,
    SAVEFIG_DPI,
    point_errorbars,
    style_axes,
    use_house_style,
)
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B = 8.617333262e-5
D = 16
SPEC = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
STATES = torch.tensor(
    list(itertools.product([-1.0, 1.0], repeat=D)), dtype=torch.float64
)
ENERGY = SPEC.energy(STATES)
N_AU = ((STATES + 1) / 2).sum(1).long()
COMPOSITIONS = np.arange(D + 1) / D
DODGE = 0.009  # in c, so the specialist and amortised markers at one slice both stay visible
MEV = 1e3


def exact_marginal(T):
    log_p = torch.log_softmax(-ENERGY / (K_B * T), 0)
    return (
        torch.zeros(D + 1, dtype=torch.float64).index_add_(0, N_AU, log_p.exp()).numpy()
    )


def exact_slice_free_energy(T):
    """-log Z_n / (beta d) for n = 0..16, eV/site."""
    beta = 1.0 / (K_B * T)
    return np.array(
        [
            -torch.logsumexp(-beta * ENERGY[N_AU == n], 0).item() / beta / D
            for n in range(D + 1)
        ]
    )


def au_count(samples):
    return ((samples.double() + 1) / 2).sum(1).long()


def is_free_energy(log_w, beta):
    return (
        -(torch.logsumexp(log_w.double(), 0) - math.log(len(log_w))).item() / beta / D
    )


def landed(patterns):
    return sorted(
        r
        for p in patterns
        for r in glob.glob(p)
        if os.path.exists(f"{r}/eval/log_weights.pt")
    )


def panel_letter(ax, letter):
    ax.text(
        -0.02,
        1.06,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_LABEL,
        color=REFERENCE_INK,
        ha="right",
        va="bottom",
    )


def draw_structures(ax_l12, ax_l10):
    spec = BinaryExpansionSpec.from_json(
        "data/ce/cuau_fcc_4x4x4.json"
    )  # any fcc supercell gives the same cube
    tiled = rectangular_tiling(spec)
    for ax, phase, title in (
        (ax_l12, "l12", "Cu$_3$Au (L1$_2$)\n$c_\\mathrm{Au} = 0.25$"),
        (ax_l10, "l10", "CuAu (L1$_0$)\n$c_\\mathrm{Au} = 0.5$"),
    ):
        # Show the L1_0 variant layered along z, so the Cu/Au planes stack
        # vertically in the shared camera view. These are ideal reference
        # cells, not selected sampler draws; symmetry-equivalent to index 0.
        variant = 4 if phase == "l10" else 0
        draw_structure(
            ax, *conventional_cell(tiled, ordered_states(spec, phase)[variant].numpy())
        )
        ax.set_title(title, fontsize=FONT_SIZE_ANNOTATION, pad=2)
    for name in ("Au", "Cu"):
        ax_l10.scatter(
            [], [], s=40, color=JMOL[name], edgecolor="black", lw=0.3, label=name
        )
    ax_l10.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_ANNOTATION,
        handletextpad=0.3,
        columnspacing=1.0,
    )


def draw_histogram(ax, runs, T):
    ax.bar(
        COMPOSITIONS,
        exact_marginal(T),
        width=1 / D,
        color=REFERENCE_FILL,
        alpha=0.45,
        zorder=1,
        label="exact",
        linewidth=0,
    )
    masses = (
        np.array(
            [
                np.bincount(
                    au_count(torch.load(f"{r}/eval/samples.pt")).numpy(),
                    minlength=D + 1,
                )
                / 5000
                for r in runs
            ]
        )
        if runs
        else np.zeros((0, D + 1))
    )
    if len(runs):
        ax.fill_between(
            COMPOSITIONS,
            masses.min(0),
            masses.max(0),
            step="mid",
            color=SAMPLER_HUE,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax.step(
            COMPOSITIONS,
            masses.mean(0),
            where="mid",
            color=SAMPLER_HUE,
            lw=1.4,
            zorder=3,
            label=f"free cell, raw draws (mean, band = min-max over {len(runs)} seeds)",
        )
    ax.set_title(f"$T = {T}$ K", fontsize=FONT_SIZE_LABEL, pad=3)
    ax.set_xlim(0.03, 0.85)
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_xlabel("$c_\\mathrm{Au}$", labelpad=1)
    ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
    style_axes(ax)


def slice_estimates(runs, beta, split_by_slice):
    """{composition: [F_IS per seed]} in meV/site. A specialist is one composition per run; the
    amortised run mixes five slices and each slice's rows alone are that slice's importance sample
    (each row's weight is exact against its own slice's base density, so no selection correction)."""
    by_c = {}
    for run in runs:
        counts, log_w = (
            au_count(torch.load(f"{run}/eval/samples.pt")),
            torch.load(f"{run}/eval/log_weights.pt"),
        )
        slices = (
            sorted(set(counts.tolist()))
            if split_by_slice
            else [round(counts.double().mean().item())]
        )
        for n in slices:
            rows = log_w[counts == n] if split_by_slice else log_w
            by_c.setdefault(n / D, []).append(MEV * is_free_energy(rows, beta))
    return by_c


def mixture_of(runs):
    """The composition slices the amortised checkpoints trained on (config.json)."""
    mixtures = {
        tuple(json.load(open(f"{run}/config.json"))["composition_mixture"])
        for run in runs
    }
    assert len(mixtures) == 1, mixtures
    return sorted(mixtures.pop())


def draw_free_energy(ax_curve, ax_resid, specialists, amortised, T):
    beta = 1.0 / (K_B * T)
    exact = MEV * exact_slice_free_energy(T)
    ax_curve.plot(
        COMPOSITIONS,
        exact,
        "-",
        color=REFERENCE_INK,
        lw=1.2,
        zorder=2,
        label="exact (all $2^{16}$ states)",
    )
    ax_curve.plot(COMPOSITIONS, exact, "o", color=REFERENCE_INK, ms=2.2, zorder=2)
    for n, name, dx, ha in ((4, "Cu$_3$Au", -5, "right"), (8, "CuAu", 6, "left")):
        ax_curve.annotate(
            name,
            (n / D, exact[n]),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=FONT_SIZE_ANNOTATION,
            color=MUTED,
        )
    series = (
        (
            slice_estimates(specialists, beta, False),
            SAMPLER_HUE,
            "o",
            -DODGE,
            "specialist cells (one per $c$)",
        ),
        (
            slice_estimates(amortised, beta, True),
            NEURAL_COMPARATOR_HUE,
            "s",
            +DODGE,
            "amortised cell (one checkpoint)",
        ),
    )
    mixture = mixture_of(amortised)
    for by_c, hue, marker, dodge, label in series:
        cs = np.array(sorted(by_c))
        values = np.array([by_c[c] for c in cs])  # (n_c, n_seeds)
        residual = values - exact[(cs * D).round().astype(int)][:, None]
        spread = np.stack(
            [residual.mean(1) - residual.min(1), residual.max(1) - residual.mean(1)]
        )
        # An amortised read outside the slices it trained on is zero-shot
        # composition transfer of the same checkpoint: hollow, not filled.
        outside = ~np.isin(cs, mixture) if marker == "s" else np.zeros(len(cs), bool)
        for keep, hollow, series_label in (
            (~outside, False, label),
            (outside, True, "same checkpoint, slices outside its mixture"),
        ):
            if not keep.any():
                continue
            ax_curve.plot(
                cs[keep] + dodge,
                values[keep].mean(1),
                marker,
                color=hue,
                ms=4.5,
                zorder=4,
                label=series_label,
                markerfacecolor="white" if hollow else hue,
                markeredgecolor=hue if hollow else "white",
                markeredgewidth=0.8 if hollow else 0.6,
            )
            point_errorbars(
                ax_resid,
                cs[keep] + dodge,
                residual[keep].mean(1),
                spread[:, keep],
                hue,
                None,
                marker=marker,
                hollow=hollow,
            )
    ax_curve.set_xlim(-0.03, 1.03)
    ax_curve.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_curve.set_xlabel("$c_\\mathrm{Au}$", labelpad=1)
    ax_curve.set_ylabel("$F(c)$ (meV/site)", labelpad=2)
    ax_curve.set_title(
        f"canonical free energy, $T = {T:.0f}$ K", fontsize=FONT_SIZE_LABEL, pad=3
    )
    ax_curve.legend(
        frameon=False,
        fontsize=FONT_SIZE_ANNOTATION - 1,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.0),
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.4,
    )
    ax_resid.axhline(0, color=REFERENCE_INK, lw=0.8, zorder=1)
    if len(mixture) < len(cs):
        ax_resid.set_xlim(-0.03, 1.03)
        ax_resid.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax_resid.set_xlim(0.2, 0.55)
        ax_resid.set_xticks([0.25, 0.375, 0.5])
        ax_resid.set_xticklabels(["0.25", "0.375", "0.5"])
        ax_resid.set_ylim(-0.2, 0.45)
        ax_resid.set_yticks([0, 0.2, 0.4])
    ax_resid.set_xlabel("$c_\\mathrm{Au}$", labelpad=1)
    ax_resid.set_ylabel("$F_\\mathrm{IS} - F_\\mathrm{exact}$ (meV/site)", labelpad=2)
    ax_resid.set_title("residual", fontsize=FONT_SIZE_LABEL, pad=3)
    for ax in (ax_curve, ax_resid):
        ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
        style_axes(ax)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--free", nargs="+", required=True, help="one glob per temperature"
    )
    parser.add_argument("--temperatures", nargs="+", type=int, required=True)
    parser.add_argument("--specialists", nargs="+", required=True)
    parser.add_argument("--amortised", nargs="+", required=True)
    parser.add_argument("--fc-temperature", type=float, default=500.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    use_house_style()

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.1))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[0.95, 1, 1, 1],
        height_ratios=[1, 1.15],
        wspace=0.55,
        hspace=0.65,
        left=0.01,
        right=0.99,
        top=0.90,
        bottom=0.10,
    )
    ax_l12, ax_l10 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    histogram_axes = [fig.add_subplot(grid[0, k]) for k in (1, 2, 3)]
    ax_curve, ax_resid = fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3])

    draw_structures(ax_l12, ax_l10)
    for ax, pattern, T in zip(histogram_axes, args.free, args.temperatures):
        draw_histogram(ax, landed([pattern]), T)
    handles, labels = histogram_axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.62, 1.0),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_ANNOTATION,
        handlelength=1.6,
        columnspacing=1.5,
    )
    histogram_axes[0].set_ylabel("probability mass", labelpad=2)
    draw_free_energy(
        ax_curve,
        ax_resid,
        landed(args.specialists),
        landed(args.amortised),
        args.fc_temperature,
    )
    for ax, letter in zip(
        (ax_l12, ax_l10, *histogram_axes, ax_curve, ax_resid), "abcdefg"
    ):
        panel_letter(ax, letter)
    fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
