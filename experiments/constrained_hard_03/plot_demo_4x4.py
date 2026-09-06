"""Figures for the 4x4 supervisor demo pack (2026-07-08).

Reads the demo analysis outputs (neural_estimates.json + neff_table.json)
and renders two PNGs:

  energy_marginals.png -- exact conditional (black ink reference) vs the
      IS-weighted sampled energy marginal, one panel per cell, seeds
      overlaid in the cell's head hue.
  neff_per_compute.png -- N_eff(O) per 1e6 compute-currency units, sigma
      rows x currency columns. The two currencies (backbone rows vs energy
      evaluations) are NEVER drawn on one axis: each column carries exactly
      one currency, per the frozen two-currency rule (FREEZE-2).

Hues are fixed per sampler entity (never re-assigned by panel):
masked_attention #2a78d6, mask_one #1baf7a, Kawasaki #eda100; palette
validated colourblind-safe (worst adjacent CVD dE 47.2, light surface).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discrete_flow_sampler.diagnostics.figure_style import CLASSICAL_HUE as KAWASAKI_HUE
from discrete_flow_sampler.diagnostics.figure_style import (
    MUTED,
    NEURAL_COMPARATOR_HUE,
    SAMPLER_HUE,
)
from discrete_flow_sampler.diagnostics.figure_style import REFERENCE_INK as INK

HEAD_HUES = {"masked_attention": SAMPLER_HUE, "mask_one": NEURAL_COMPARATOR_HUE}
OBSERVABLE_LABELS = {
    "energy": "energy",
    "nn_correlation": "nn corr.",
    "diagonal_correlation": "diag corr.",
    "phi": "phi",
}


def _style_axis(ax):
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_energy_marginals(neural, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, cell in zip(axes.flat, neural):
        hue = HEAD_HUES[cell["head_kind"]]
        first = cell["fidelity"][0]
        ax.step(
            first["energy_centres"],
            first["hist_exact"],
            where="mid",
            color=INK,
            linewidth=1.8,
            zorder=3,
            label=r"exact $\pi(\cdot\,|\,C)$",
        )
        for fidelity in cell["fidelity"]:
            ax.step(
                fidelity["energy_centres"],
                fidelity["hist_dnfs"],
                where="mid",
                color=hue,
                linewidth=1.1,
                alpha=0.6,
                zorder=2,
                label=f"seed {fidelity['seed']}",
            )
        sigma_label = f"$\\sigma$ = {cell['sigma']}"
        ax.set_title(f"{cell['head_kind']}  ({sigma_label})", fontsize=9, color=INK)
        ax.set_xlabel(r"slice energy $x^\top A x$", fontsize=8, color=MUTED)
        _style_axis(ax)
    axes[0, 0].set_ylabel("probability mass", fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel("probability mass", fontsize=8, color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].legend(handles, labels, fontsize=7, frameon=False)
    # No suptitle: this figure prints as hard.tex fig:hard-4x4-marginals,
    # whose LaTeX caption carries the description (layout review 2026-08-08).
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_neff_per_compute(table, out_path):
    observables = list(OBSERVABLE_LABELS)
    # sharey per column: each column is one currency, so its two sigma rows
    # must be read on a common scale (the across-sigma drop IS the story).
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharey="col")
    currency_titles = {
        "backbone_rows": "neural swap-CTMC\nper $10^6$ network passes (backbone rows)",
        "energy_evals": "Kawasaki (mchammer, non-local)\nper $10^6$ energy evaluations",
    }
    for row, sigma in enumerate(sorted({r["sigma"] for r in table})):
        for col, currency in enumerate(("backbone_rows", "energy_evals")):
            ax = axes[row, col]
            samplers = sorted(
                {
                    (r["sampler"], r["head_kind"])
                    for r in table
                    if r["sigma"] == sigma and r["currency"] == currency
                }
            )
            width = 0.8 / len(samplers)
            for k, (sampler, head_kind) in enumerate(samplers):
                hue = HEAD_HUES.get(head_kind, KAWASAKI_HUE)
                rows = {
                    r["observable"]: r
                    for r in table
                    if r["sampler"] == sampler and r["sigma"] == sigma
                }
                values = [rows[o]["n_eff_per_1e6"] for o in observables]
                errors = [
                    rows[o]["n_eff_se"] / (rows[o]["mean_cost"] / 1e6)
                    for o in observables
                ]
                positions = [
                    i + (k - (len(samplers) - 1) / 2) * width
                    for i in range(len(observables))
                ]
                label = head_kind if currency == "backbone_rows" else "kawasaki"
                ax.bar(
                    positions, values, width * 0.92, color=hue, zorder=2, label=label
                )
                # Bars on bars: x here is the categorical observable list,
                # not a continuous axis, so there is nothing for a shaded
                # band to be shaded along.
                ax.errorbar(
                    positions,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=INK,
                    elinewidth=0.9,
                    capsize=2,
                    zorder=3,
                )
            ax.set_xticks(range(len(observables)))
            ax.set_xticklabels([OBSERVABLE_LABELS[o] for o in observables], fontsize=8)
            _style_axis(ax)
            if row == 0:
                ax.set_title(currency_titles[currency], fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(
                    f"$\\sigma$ = {sigma}\n$N_{{\\rm eff}}(O)$ per $10^6$ units",
                    fontsize=9,
                    color=INK,
                )
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "N_eff(O) per compute, 4x4 demo -- currencies are NOT comparable "
        "across columns (frozen two-currency rule)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_phi_coverage(phi_hists, kawasaki_dir, out_path):
    """Mode-coverage exhibit: exact enumerated pmf of the order parameter phi
    vs the IS-weighted neural histograms and the pooled Kawasaki histogram,
    one panel per sigma. Recovery of the full phi distribution (both Z2
    sectors, equal mass) is the distributional coverage check the scalar
    E[phi]=0 rows only imply statistically."""
    from experiments.constrained_hard_03.demo_4x4 import (
        exact_phi_pmf,
        kawasaki_phi_mass,
        phi_support,
    )

    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    sigmas = sorted({cell["sigma"] for cell in phi_hists})
    support = phi_support(4)
    fig, axes = plt.subplots(1, len(sigmas), figsize=(9.5, 3.8), sharey=True)
    for ax, sigma in zip(axes, sigmas):
        target = FixedCompositionIsingTarget(
            D=4, sigma=sigma, target_composition=0.5, bias=0.0, device="cpu"
        )
        ax.plot(
            support,
            exact_phi_pmf(target),
            color=INK,
            linewidth=1.8,
            marker="o",
            markersize=4,
            zorder=4,
            label=r"exact $\pi(\phi)$",
        )
        for cell in phi_hists:
            if abs(cell["sigma"] - sigma) > 1e-9:
                continue
            ax.plot(
                cell["phi_support"],
                cell["phi_mass"],
                color=HEAD_HUES[cell["head_kind"]],
                linewidth=1.4,
                marker="o",
                markersize=3.5,
                alpha=0.85,
                zorder=3,
                label=cell["head_kind"],
            )
        ax.plot(
            support,
            kawasaki_phi_mass(kawasaki_dir, sigma, target, 20_000),
            color=KAWASAKI_HUE,
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            alpha=0.85,
            zorder=2,
            label="kawasaki",
        )
        ax.set_title(f"$\\sigma$ = {sigma}", fontsize=9, color=INK)
        ax.set_xlabel(r"mode order parameter $\phi$", fontsize=8, color=MUTED)
        _style_axis(ax)
    axes[0].set_ylabel("probability mass", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Mode coverage at the enumerable size: full distribution of $\\phi$ "
        "vs exact enumeration",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", default="results/03_hard/demo_4x4")
    args = parser.parse_args(argv)
    demo_dir = Path(args.demo_dir)
    figures_dir = demo_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    neural = json.loads((demo_dir / "neural_estimates.json").read_text())
    table = json.loads((demo_dir / "neff_table.json").read_text())
    plot_energy_marginals(neural, figures_dir / "energy_marginals.png")
    plot_neff_per_compute(table, figures_dir / "neff_per_compute.png")
    n_figures = 2
    phi_json = demo_dir / "phi_hists.json"
    if phi_json.exists():
        plot_phi_coverage(
            json.loads(phi_json.read_text()),
            demo_dir / "kawasaki",
            figures_dir / "phi_coverage.png",
        )
        n_figures = 3
    print(f"[plot] wrote {n_figures} figures to {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
