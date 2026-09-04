"""Au-concentration distributions of the free-ensemble 64-site Cu-Au cell against the reference chain.

The like-for-like exhibit with MetaDNS Fig. 5(c-e): at each temperature the free sampler's
draws (one histogram per seed, raw draws) beside the single-flip Metropolis chain
(reference_chain.py, filled). Composition is discrete, n_Au / 64, so the bins are the sites.
Missing cells or chains leave their panel annotated rather than failing the figure.

Usage: pixi run -e dev python -m experiments.alloy_ce.tools.composition_histograms \\
           --cells "results/02_constrained_soft/A1_cuau64_T1200*grid" \\
                   "results/02_constrained_soft/A1_cuau64_T680*grid" \\
                   "results/02_constrained_soft/A1_cuau64_T500*house" \\
           --temperatures 1200 680 500 --out assets/cuau64_composition_histograms.pdf
"""
import argparse, glob, os
import matplotlib.pyplot as plt, torch
from discrete_flow_sampler.diagnostics.figure_style import (
    FULL_WIDTH_IN, FONT_SIZE_ANNOTATION, FONT_SIZE_LABEL, REFERENCE_FILL, SAMPLER_HUE, SAVEFIG_DPI,
    parameter_ramp, style_axes, use_house_style)

N_SITES = 64


def au_fraction(states):
    return ((states.double() + 1) / 2).mean(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True, help="one run-dir glob per temperature")
    parser.add_argument("--temperatures", nargs="+", type=int, required=True)
    parser.add_argument("--reference-dir", default="results/alloy_ref")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    use_house_style()
    bins = [(k - 0.5) / N_SITES for k in range(N_SITES + 2)]
    fig, axes = plt.subplots(1, len(args.temperatures), figsize=(FULL_WIDTH_IN, 2.1), sharey=False)
    for ax, pattern, T in zip(axes, args.cells, args.temperatures):
        chain = f"{args.reference_dir}/cuau64_chain_free_T{T}.pt"
        if os.path.exists(chain):
            ax.hist(au_fraction(torch.load(chain)).numpy(), bins=bins, density=True, color=REFERENCE_FILL,
                    alpha=0.45, label="Metropolis chain")
        else:
            ax.text(0.5, 0.5, "no chain", transform=ax.transAxes, ha="center", fontsize=FONT_SIZE_ANNOTATION)
        runs = sorted(r for r in glob.glob(pattern) if os.path.exists(f"{r}/eval/samples.pt"))
        for run, hue in zip(runs, parameter_ramp(SAMPLER_HUE, max(len(runs), 2))):
            ax.hist(au_fraction(torch.load(f"{run}/eval/samples.pt")).numpy(), bins=bins, density=True,
                    histtype="step", color=hue, lw=1.1, label="sampler, one seed" if run == runs[0] else None)
        if not runs:
            ax.text(0.5, 0.35, "no cell", transform=ax.transAxes, ha="center", fontsize=FONT_SIZE_ANNOTATION)
        ax.set_title(f"$T = {T}$ K", fontsize=FONT_SIZE_LABEL)
        ax.set_xlabel("Au concentration", fontsize=FONT_SIZE_LABEL); ax.set_xlim(0.1, 0.8)
        style_axes(ax)
    axes[0].set_ylabel("density", fontsize=FONT_SIZE_LABEL)
    axes[0].legend(frameon=False, fontsize=FONT_SIZE_ANNOTATION, loc="upper left")
    fig.tight_layout(); fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    main()
