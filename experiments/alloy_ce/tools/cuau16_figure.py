"""The 16-site Cu-Au exhibit: composition histograms and the canonical F(c), all against exact truth.

Every panel's reference is the 2^16 enumeration of the 2x2x4 cell (no chain):
  (c,d,e) Au-concentration marginal of the free ensemble at each temperature, exact
          bars vs the free cell's draws (one step histogram per seed, raw draws);
  (f)     canonical free energy per site F(c) = -log Z_c / (beta d) over every slice
          n_Au = 0..16 at 500 K, in kT per cell above the minimum, with the specialist
          cells (one per composition, mean +- sd over seeds of the IS estimate) and the
          composition-amortised cell (one checkpoint, per-slice IS estimate) on top.

Usage: pixi run -e dev python -m experiments.alloy_ce.tools.cuau16_figure \\
           --free "results/02_constrained_soft/A1_cuau16_T1200*fc" \\
                  "results/02_constrained_soft/A1_cuau16_T680*fc" \\
                  "results/02_constrained_soft/A1_cuau16_T500_letf_50k_house*" \\
           --temperatures 1200 680 500 \\
           --specialists "results/03_hard/H2_cuau16_c*_T500_mask_one_50k_house_seed*" \\
           --amortised "results/03_hard/H2_cuau16_camort*fc" --out assets/cuau16_exhibit.pdf
"""
import argparse, glob, itertools, math, os
import matplotlib.pyplot as plt, numpy as np, torch
from discrete_flow_sampler.diagnostics.figure_style import (
    FULL_WIDTH_IN, FONT_SIZE_ANNOTATION, FONT_SIZE_LABEL, NEURAL_COMPARATOR_HUE, REFERENCE_FILL, REFERENCE_INK,
    SAMPLER_HUE, SAVEFIG_DPI, parameter_ramp, style_axes, use_house_style)
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

K_B = 8.617333262e-5
D = 16
SPEC = BinaryExpansionSpec.from_json("data/ce/cuau_fcc_2x2x4.json")
STATES = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=D)), dtype=torch.float64)
ENERGY = SPEC.energy(STATES)
N_AU = ((STATES + 1) / 2).sum(1).long()


def exact_marginal(T):
    log_p = torch.log_softmax(-ENERGY / (K_B * T), 0)
    return torch.zeros(D + 1, dtype=torch.float64).index_add_(0, N_AU, log_p.exp())


def exact_slice_free_energy(T):
    """-log Z_n / (beta d) for n = 0..16, eV/site."""
    beta = 1.0 / (K_B * T)
    return torch.tensor([-torch.logsumexp(-beta * ENERGY[N_AU == n], 0).item() / beta / D for n in range(D + 1)])


def au_count(samples):
    return ((samples.double() + 1) / 2).sum(1).long()


def sampler_slice_free_energy(log_w, beta):
    return -(torch.logsumexp(log_w.double(), 0) - math.log(len(log_w))).item() / beta / D


def landed(pattern):
    return sorted(r for r in glob.glob(pattern) if os.path.exists(f"{r}/eval/log_weights.pt"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", nargs="+", required=True)
    parser.add_argument("--temperatures", nargs="+", type=int, required=True)
    parser.add_argument("--specialists", required=True)
    parser.add_argument("--amortised", required=True)
    parser.add_argument("--fc-temperature", type=float, default=500.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    use_house_style()
    n_hist = len(args.temperatures)
    fig, axes = plt.subplots(1, n_hist + 1, figsize=(FULL_WIDTH_IN, 2.0),
                             gridspec_kw={"width_ratios": [1] * n_hist + [1.35]})

    for ax, pattern, T in zip(axes, args.free, args.temperatures):
        ax.bar(np.arange(D + 1) / D, exact_marginal(T).numpy(), width=1 / D, color=REFERENCE_FILL, alpha=0.45,
               label="exact")
        runs = landed(pattern)
        for run, hue in zip(runs, parameter_ramp(SAMPLER_HUE, max(len(runs), 2))):
            counts = au_count(torch.load(f"{run}/eval/samples.pt"))
            mass = torch.bincount(counts, minlength=D + 1).double() / len(counts)
            ax.step(np.arange(D + 1) / D, mass.numpy(), where="mid", color=hue, lw=1.1,
                    label="free cell, one seed" if run == runs[0] else None)
        if not runs:
            ax.text(0.5, 0.5, "no cell", transform=ax.transAxes, ha="center", fontsize=FONT_SIZE_ANNOTATION)
        ax.set_title(f"$T = {T}$ K", fontsize=FONT_SIZE_LABEL); ax.set_xlim(0.05, 0.75)
        ax.set_xticks([0.25, 0.5, 0.75]); ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
        ax.set_xlabel("$c_\\mathrm{Au}$", fontsize=FONT_SIZE_LABEL); style_axes(ax)
    axes[0].set_ylabel("mass", fontsize=FONT_SIZE_LABEL)
    axes[0].legend(frameon=False, fontsize=FONT_SIZE_ANNOTATION, loc="upper left")

    ax = axes[-1]; T = args.fc_temperature; beta = 1.0 / (K_B * T); kT_cell = K_B * T / D
    exact = exact_slice_free_energy(T); floor = exact.min()
    ax.plot(np.arange(D + 1) / D, ((exact - floor) / kT_cell).numpy(), "-", color=REFERENCE_INK, lw=1.2, label="exact")
    by_composition = {}
    for run in landed(args.specialists):
        log_w = torch.load(f"{run}/eval/log_weights.pt")
        c = round(au_count(torch.load(f"{run}/eval/samples.pt")).double().mean().item() / D * D) / D
        by_composition.setdefault(c, []).append(sampler_slice_free_energy(log_w, beta))
    if by_composition:
        cs = sorted(by_composition)
        means = np.array([np.mean(by_composition[c]) for c in cs]); sds = np.array([np.std(by_composition[c]) for c in cs])
        ax.errorbar(cs, (means - floor.item()) / kT_cell, yerr=sds / kT_cell, fmt="o", ms=4, color=SAMPLER_HUE,
                    capsize=2, label="specialist cells")
    amortised = {}
    for run in landed(args.amortised):
        samples, log_w = torch.load(f"{run}/eval/samples.pt"), torch.load(f"{run}/eval/log_weights.pt")
        counts = au_count(samples)
        for n in sorted(set(counts.tolist())):
            amortised.setdefault(n / D, []).append(sampler_slice_free_energy(log_w[counts == n], beta))
    if amortised:
        cs = sorted(amortised)
        means = np.array([np.mean(amortised[c]) for c in cs]); sds = np.array([np.std(amortised[c]) for c in cs])
        ax.errorbar(np.array(cs) + 0.006, (means - floor.item()) / kT_cell, yerr=sds / kT_cell, fmt="s", ms=4,
                    color=NEURAL_COMPARATOR_HUE, capsize=2, label="amortised cell")
    ax.set_xlim(0.05, 0.75); ax.set_ylim(-0.5, None)
    ax.set_title(f"$F(c)$ at $T = {T:.0f}$ K", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("$c_\\mathrm{Au}$", fontsize=FONT_SIZE_LABEL); ax.set_ylabel("$F / k_BT$ per cell", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([0.25, 0.5, 0.75]); ax.tick_params(labelsize=FONT_SIZE_ANNOTATION)
    ax.legend(frameon=False, fontsize=FONT_SIZE_ANNOTATION, loc="lower right"); style_axes(ax)
    fig.tight_layout(); fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    main()
