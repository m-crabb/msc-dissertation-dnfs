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

Usage: pixi run -e dev python -m experiments.alloy_ce.analysis.fcc_render \\
           --c25 <run dir> --c50 <run dir> --out assets/cuau64_fcc_renders.pdf
       ... --structures --out assets/cuau_ordered_structures.pdf
           (sphere renders of Cu3Au (L1_2) and CuAu (L1_0) in a 2x2x2 conventional cube)
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import torch
from experiments.alloy_ce.probes.patch_reach_probe import ordered_states

from discrete_flow_sampler.diagnostics.figure_style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TITLE,
    FULL_WIDTH_IN,
    GRID,
    SAVEFIG_DPI,
    SPIN_DOWN_COLOUR,
    SPIN_UP_COLOUR,
    use_house_style,
)
from discrete_flow_sampler.targets.cluster_expansion import BinaryExpansionSpec

MEV = 1000.0


BOX = torch.tensor([7.6, 7.6, 15.2], dtype=torch.float64)


def rectangular_tiling(spec):
    """(site index, position) for the 64 crystal sites inside the rectangular cell."""
    positions = torch.tensor(spec.positions, dtype=torch.float64)
    cell = torch.tensor(spec.cell, dtype=torch.float64)
    tiled = {}
    for site, shift in itertools.product(
        range(len(positions)), itertools.product(range(-2, 3), repeat=3)
    ):
        q = positions[site] + torch.tensor(shift, dtype=torch.float64) @ cell
        if ((q > -1e-6) & (q < BOX - 1e-6)).all():
            tiled[tuple((q / 1.9).round().long().tolist())] = site
    assert len(tiled) == len(positions) and len(set(tiled.values())) == len(positions)
    return tiled


JMOL = {
    "Au": (1.0, 0.82, 0.14),
    "Cu": (0.78, 0.50, 0.20),
}  # jmol element colours (ASE's default)


def rotation(degrees_x, degrees_y, degrees_z):
    """Rotation matrix for successive rotations about x, y, z (degrees), as ASE's 'ax,by,cz' strings."""
    import math

    def about(axis, angle):
        c, sn = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        m = torch.eye(3, dtype=torch.float64)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -sn, sn, c
        return m

    return about(2, degrees_z) @ about(1, degrees_y) @ about(0, degrees_x)


def sphere_sprite(colour, n=192):
    """RGBA image of a lit sphere in `colour` (Lambert + a highlight), transparent outside."""
    import numpy as np

    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    r2 = x * x + y * y
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    light = np.array([-0.4, 0.5, 0.75])
    light /= np.linalg.norm(light)
    lambert = np.clip(x * light[0] + y * light[1] + z * light[2], 0, 1)
    shade = 0.52 + 0.48 * lambert
    highlight = np.exp(-((x + 0.35) ** 2 + (y - 0.4) ** 2) / 0.07) * 0.18
    rgb = np.clip(
        np.array(colour)[None, None, :] * shade[..., None] + highlight[..., None], 0, 1
    )
    alpha = np.clip((1 - np.sqrt(r2)) * n / 2, 0, 1)
    return np.dstack([rgb, alpha])


def draw_structure(
    ax, positions, symbols, cell, rotate=(-65.0, -25.0, 0.0), radius=0.64
):
    """Orthographic lit-sphere render of an Atoms-like (positions A, symbols, cell) with the
    cell outline; spheres drawn back to front so nearer atoms occlude farther ones.

    Radius is a display choice shared by both species, not a fitted atomic
    radius. Cell edges are clipped against the visible sphere surfaces in
    camera coordinates, so a frontmost outline cannot cut through an atom.
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    R = rotation(*rotate).numpy()
    view = positions @ R.T  # x, y on the page, z toward the viewer
    corners = (
        np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        @ cell
        @ R.T
    )
    visible_edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:  # cube edges of the cell
                points = np.linspace(corners[a], corners[b], 161)
                midpoint = (points[:-1] + points[1:]) / 2
                distance_sq = ((midpoint[:, None, :2] - view[None, :, :2]) ** 2).sum(-1)
                surface_z = view[None, :, 2] + np.sqrt(
                    np.maximum(radius**2 - distance_sq, 0)
                )
                hidden = (
                    (distance_sq < radius**2) & (surface_z > midpoint[:, None, 2])
                ).any(1)
                segments = np.stack((points[:-1, :2], points[1:, :2]), axis=1)
                visible_edges.extend(segments[~hidden])
    ax.add_collection(
        LineCollection(
            visible_edges, colors="#95958f", linewidths=0.65, zorder=len(symbols) + 3
        )
    )
    sprites = {name: sphere_sprite(JMOL[name]) for name in set(symbols)}
    for depth_rank, i in enumerate(np.argsort(view[:, 2])):
        x, y = view[i, :2]
        ax.imshow(
            sprites[symbols[i]],
            extent=(x - radius, x + radius, y - radius, y + radius),
            zorder=2 + depth_rank,
            interpolation="bilinear",
            origin="lower",
        )
    lo, hi = view[:, :2].min(0) - radius, view[:, :2].max(0) + radius
    ax.set_xlim(min(lo[0], corners[:, 0].min()), max(hi[0], corners[:, 0].max()))
    ax.set_ylim(min(lo[1], corners[:, 1].min()), max(hi[1], corners[:, 1].max()))
    ax.set_aspect("equal")
    ax.set_axis_off()


def conventional_cell(tiled, state):
    """The 14-atom conventional fcc cell (corners + face centres) coloured from a periodic state."""
    import numpy as np

    points = [(x, y, z) for x in (0, 2) for y in (0, 2) for z in (0, 2)] + [
        (1, 1, 0),
        (1, 1, 2),
        (1, 0, 1),
        (1, 2, 1),
        (0, 1, 1),
        (2, 1, 1),
    ]
    symbols = ["Au" if state[tiled[q]] > 0 else "Cu" for q in points]
    return 1.9 * np.array(points, dtype=float), symbols, 3.8 * np.eye(3)


def draw_state(axes, tiled, state, label):
    """One row of eight (001) layer tiles; sites at 1.9 A grid coordinates."""
    for layer, ax in enumerate(axes):
        for (x, y, z), site in tiled.items():
            if z == layer:
                ax.scatter(
                    x,
                    y,
                    s=34,
                    color=SPIN_UP_COLOUR if state[site] > 0 else SPIN_DOWN_COLOUR,
                    edgecolor="black",
                    lw=0.3,
                )
        ax.set_xlim(-0.7, 3.7)
        ax.set_ylim(-0.7, 3.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel(
        label,
        fontsize=FONT_SIZE_ANNOTATION,
        rotation=0,
        ha="right",
        va="center",
        labelpad=6,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="data/ce/cuau_fcc_4x4x4.json")
    parser.add_argument("--c25", help="run dir of a c=0.25 patch-head cell")
    parser.add_argument("--c50", help="run dir of a c=0.5 patch-head cell")
    parser.add_argument(
        "--structures",
        action="store_true",
        help="sphere renders of the two ordered phases only",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    use_house_style()
    spec = BinaryExpansionSpec.from_json(args.spec)
    tiled = rectangular_tiling(spec)
    n_sites = spec.n_sites

    if args.structures:
        fig = plt.figure(figsize=(FULL_WIDTH_IN * 0.7, 2.4))
        for k, (phase, title) in enumerate(
            (
                ("l12", "Cu$_3$Au (L1$_2$), $c_\\mathrm{Au}=0.25$"),
                ("l10", "CuAu (L1$_0$), $c_\\mathrm{Au}=0.5$"),
            )
        ):
            ax = fig.add_subplot(1, 2, k + 1)
            draw_structure(
                ax, *conventional_cell(tiled, ordered_states(spec, phase)[0].numpy())
            )
            ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        for name in ("Au", "Cu"):
            ax.scatter(
                [], [], s=60, color=JMOL[name], edgecolor="black", lw=0.3, label=name
            )
        fig.legend(
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=FONT_SIZE_ANNOTATION,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=0.1)
        fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")
        print("wrote", args.out)
        return

    def energy_label(state):
        return f"{spec.energy(state[None]).item() / n_sites * MEV:.1f} meV/site"

    panels = [
        (ordered_states(spec, "l10")[0], "L1$_0$ reference, $c=0.5$"),
        (ordered_states(spec, "l12")[0], "L1$_2$ reference, $c=0.25$"),
    ]
    for run, label in (
        (args.c25, "$c=0.25$ draw, 500 K"),
        (args.c50, "$c=0.5$ typical draw, 500 K"),
        (args.c50, "$c=0.5$ top-weight draw, 500 K"),
    ):
        samples = torch.load(f"{run}/eval/samples.pt").double()
        log_w = torch.load(f"{run}/eval/log_weights.pt").double()
        energies = spec.energy(samples)
        index = (
            log_w.argmax()
            if "top-weight" in label
            else (energies - energies.median()).abs().argmin()
        )
        panels.append((samples[index], label))

    fig, axes = plt.subplots(
        len(panels), 8, figsize=(FULL_WIDTH_IN, 0.62 * len(panels) + 0.3)
    )
    for row, (state, label) in zip(axes, panels):
        draw_state(row, tiled, state.numpy(), f"{label}\n{energy_label(state)}")
    for layer, ax in enumerate(axes[0]):
        ax.set_title(f"layer {layer}", fontsize=FONT_SIZE_ANNOTATION, pad=2)
    for colour, name in ((SPIN_UP_COLOUR, "Au"), (SPIN_DOWN_COLOUR, "Cu")):
        axes[0][0].scatter(
            [], [], s=34, color=colour, edgecolor="black", lw=0.3, label=name
        )
    fig.legend(
        *axes[0][0].get_legend_handles_labels(),
        loc="upper right",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1.0, 1.02),
        fontsize=FONT_SIZE_ANNOTATION,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.25)
    fig.savefig(args.out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
