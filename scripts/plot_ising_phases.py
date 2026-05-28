"""Three Ising snapshots (disordered / critical / ordered) for the report's
running-example section.

Temperature is implicit in this codebase: sigma is the coupling in J = sigma*A,
so it plays the role of inverse temperature beta. Large sigma -> strong
coupling -> low temperature -> ordered; small sigma -> high temperature ->
disordered. The repo pins the 2D-Ising critical point at sigma = 0.22305
(experiments/dnfs_baseline_01/configs.py), consistent with the exact
K_c = 0.5*ln(1+sqrt(2)) once the symmetrised-adjacency double-counting (K = 2*sigma)
is accounted for.

Each snapshot is a single equilibrated Gibbs state reshaped from length d = D*D
to a D*D grid. The ordered panel is initialised from a fully aligned lattice:
with periodic boundaries a random start can freeze into a metastable two-domain
stripe that single-spin Gibbs will not heal on short timescales, which would
misrepresent the phase. Off-critical panels mix quickly; the critical panel gets
the largest sweep budget because correlation time diverges there (critical
slowing-down).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget

D = 64
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "assets" / "ising_phases.png"
# Sampling is the slow part (~3 min); colour is just rendering. Cache the
# equilibrated lattices so palette tweaks re-render in ~1 s. Delete this file
# (or change D/SEED/PANELS) to force a resample.
CACHE = OUT.with_name("ising_phases_states.pt")

# Two-colour map. Values come from (x+1)/2, so the first colour is spin -1 (down)
# and the second is spin +1 (up). Indigo/gold: a muted, colourblind-safe pair that
# also separates in greyscale print.
DOWN_COLOR, UP_COLOR = "#3B3A6B", "#F2C14E"
CMAP = ListedColormap([DOWN_COLOR, UP_COLOR])

# (label, sigma, n_sweeps, aligned_init)
PANELS = [
    ("High $T$ (disordered)", 0.05, 200, False),
    ("Critical $T_c$", 0.22305, 1500, False),
    ("Low $T$ (ordered)", 0.45, 300, True),
]


def snapshot(sigma: float, n_sweeps: int, aligned_init: bool, seed: int) -> torch.Tensor:
    target = IsingTarget(D=D, sigma=sigma)
    generator = torch.Generator().manual_seed(seed)
    x_init = torch.ones(1, target.d) if aligned_init else None
    spins = gibbs_sample(target, n_chains=1, n_sweeps=n_sweeps, x_init=x_init, generator=generator)
    return spins.reshape(D, D)


def load_grids() -> dict[str, torch.Tensor]:
    if CACHE.exists():
        return torch.load(CACHE)
    grids = {
        label: snapshot(sigma, n_sweeps, aligned_init, SEED)
        for label, sigma, n_sweeps, aligned_init in PANELS
    }
    torch.save(grids, CACHE)
    return grids


def main() -> None:
    grids = load_grids()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3 * len(PANELS), 3.2))
    for ax, (label, sigma, _n_sweeps, _aligned_init) in zip(axes, PANELS):
        ax.imshow((grids[label] + 1) * 0.5, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{label}\n$\\sigma = {sigma:g}$", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"2D Ising lattice ({D}$\\times${D}, periodic)", fontsize=12)
    legend_handles = [
        Patch(facecolor=UP_COLOR, edgecolor="none", label="spin $+1$ (up)"),
        Patch(facecolor=DOWN_COLOR, edgecolor="none", label="spin $-1$ (down)"),
    ]
    # Reserve a band at the bottom so the legend clears the panels.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.savefig(OUT, dpi=200)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
