"""Render sampled 64-site Cu-Au states on the fcc lattice beside the ordered references.

The 4x4x4 primitive supercell (vectors 7.6 (0,1,1) A etc., cube edge 3.8 A)
is a rhombohedral chunk in which no (001) plane is complete, so each state is
re-tiled into the equivalent 7.6 x 7.6 x 15.2 A rectangular cell (its edges
a2+a3-a1, a1+a3-a2, a1+a2-a3 are supercell lattice vectors of the same
volume) and drawn as its eight (001) layers of eight sites, every supercell
site appearing exactly once. Au = the house spin-up colour, Cu = spin-down.
Rows: the two ordered references (L1_0 for c=0.5, L1_2 for c=0.25), a c=0.25
draw from the patch-head cell at 500 K (ordered: on an L1_2 variant), and two
c=0.5 draws from the same recipe (a typical multi-domain state and the
top-weight draw).

Usage: pixi run -e dev python -m experiments.alloy_ce.tools.fcc_render \\
           --c25 <run dir> --c50 <run dir> --out assets/cuau64_fcc_renders.pdf
"""
import argparse, itertools
import matplotlib.pyplot as plt, torch
from discrete_flow_sampler.diagnostics.figure_style import (
    FULL_WIDTH_IN, FONT_SIZE_ANNOTATION, GRID, SAVEFIG_DPI, SPIN_DOWN_COLOUR,
    SPIN_UP_COLOUR, use_house_style)
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec
from experiments.alloy_ce.tools.patch_reach_probe import ordered_states

MEV = 1000.0


BOX = torch.tensor([7.6, 7.6, 15.2], dtype=torch.float64)


def rectangular_tiling(spec):
    """(site index, position) for the 64 crystal sites inside the rectangular cell."""
    positions = torch.tensor(spec.positions, dtype=torch.float64)
    cell = torch.tensor(spec.cell, dtype=torch.float64)
    tiled = {}
    for site, shift in itertools.product(range(len(positions)), itertools.product(range(-2, 3), repeat=3)):
        q = positions[site] + torch.tensor(shift, dtype=torch.float64) @ cell
        if ((q > -1e-6) & (q < BOX - 1e-6)).all():
            tiled[tuple((q / 1.9).round().long().tolist())] = site
    assert len(tiled) == len(positions) and len(set(tiled.values())) == len(positions)
    return tiled


def draw_state(axes, tiled, state, label):
    """One row of eight (001) layer tiles; sites at 1.9 A grid coordinates."""
    for layer, ax in enumerate(axes):
        for (x, y, z), site in tiled.items():
            if z == layer:
                ax.scatter(x, y, s=34, color=SPIN_UP_COLOUR if state[site] > 0 else SPIN_DOWN_COLOUR,
                           edgecolor="black", lw=0.3)
        ax.set_xlim(-0.7, 3.7); ax.set_ylim(-0.7, 3.7); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel(label, fontsize=FONT_SIZE_ANNOTATION, rotation=0, ha="right", va="center", labelpad=6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="data/ce/cuau_fcc_4x4x4.json")
    parser.add_argument("--c25", required=True, help="run dir of a c=0.25 patch-head cell")
    parser.add_argument("--c50", required=True, help="run dir of a c=0.5 patch-head cell")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    use_house_style()
    spec = BinaryExpansionSpec.from_json(args.spec)
    tiled = rectangular_tiling(spec)
    n_sites = spec.n_sites

    def energy_label(state):
        return f"{spec.energy(state[None]).item() / n_sites * MEV:.1f} meV/site"

    panels = [(ordered_states(spec, "l10")[0], "L1$_0$ reference, $c=0.5$"),
              (ordered_states(spec, "l12")[0], "L1$_2$ reference, $c=0.25$")]
    for run, label in ((args.c25, "$c=0.25$ draw, 500 K"), (args.c50, "$c=0.5$ typical draw, 500 K"),
                       (args.c50, "$c=0.5$ top-weight draw, 500 K")):
        samples = torch.load(f"{run}/eval/samples.pt").double()
        log_w = torch.load(f"{run}/eval/log_weights.pt").double()
        energies = spec.energy(samples)
        index = log_w.argmax() if "top-weight" in label else (energies - energies.median()).abs().argmin()
        panels.append((samples[index], label))

    fig, axes = plt.subplots(len(panels), 8, figsize=(FULL_WIDTH_IN, 0.62 * len(panels) + 0.3))
    for row, (state, label) in zip(axes, panels):
        draw_state(row, tiled, state.numpy(), f"{label}\n{energy_label(state)}")
    for layer, ax in enumerate(axes[0]):
        ax.set_title(f"layer {layer}", fontsize=FONT_SIZE_ANNOTATION, pad=2)
    for colour, name in ((SPIN_UP_COLOUR, "Au"), (SPIN_DOWN_COLOUR, "Cu")):
        axes[0][0].scatter([], [], s=34, color=colour, edgecolor="black", lw=0.3, label=name)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(1.0, 1.02), fontsize=FONT_SIZE_ANNOTATION)
    fig.subplots_adjust(wspace=0.08, hspace=0.25)
    fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
