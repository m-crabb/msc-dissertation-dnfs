"""Three Ising snapshots (disordered / critical / ordered) for the report's
running-example section.

Temperature is implicit in this codebase: sigma is the coupling in J = sigma*A,
so it plays the role of inverse temperature beta. Large sigma -> strong
coupling -> low temperature -> ordered; small sigma -> high temperature ->
disordered. The critical panel sits at the exact 2D-Ising critical point
sigma_c = ln(1+sqrt(2))/4 = 0.220343 (targets/ising.py SIGMA_C; K_c =
0.5*ln(1+sqrt(2)) with the symmetrised-adjacency double-counting K = 2*sigma).
Regenerated at the exact value in the s58 sigma_c migration (2026-08-24);
the pre-migration panel used the legacy 0.22305.

Each snapshot is a single equilibrated state reshaped from length d = D*D
to a D*D grid. The ordered panel is initialised from a fully aligned lattice:
with periodic boundaries a random start can freeze into a metastable two-domain
stripe that single-spin Gibbs will not heal on short timescales, which would
misrepresent the phase. Off-critical panels use Gibbs and mix quickly; the
critical panel is drawn by Wolff, since Gibbs equilibration there needs on the
order of D^z sweeps (z ~ 2.2, far beyond any reasonable budget at D = 64) while
cluster moves sidestep the critical slowing-down entirely.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

from discrete_flow_sampler.diagnostics.figure_style import (
    SPIN_CMAP,
    SPIN_DOWN_COLOUR,
    SPIN_UP_COLOUR,
)
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.mcmc.wolff import wolff_sample
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

D = 64
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "assets" / "ising_phases.png"
# Sampling is the slow part (~3 min); colour is just rendering. Cache the
# equilibrated lattices so palette tweaks re-render in ~1 s. Delete this file
# (or change D/SEED/PANELS) to force a resample.
CACHE = OUT.with_name("ising_phases_states.pt")

# Two-colour spin map: shared house constant (this script established it;
# it now lives in figure_style so every spin figure agrees).
DOWN_COLOR, UP_COLOR = SPIN_DOWN_COLOUR, SPIN_UP_COLOUR
CMAP = SPIN_CMAP

# (label, sigma, n_sweeps, aligned_init)
PANELS = [
    ("High $T$ (disordered)", 0.05, 200, False),
    ("Critical $T_c$", SIGMA_C, 1500, False),
    ("Low $T$ (ordered)", 0.45, 300, True),
]


def snapshot(sigma: float, n_sweeps: int, aligned_init: bool, seed: int) -> torch.Tensor:
    target = IsingTarget(D=D, sigma=sigma)
    if sigma == SIGMA_C:
        # Wolff for the critical panel (see module docstring); 2000 cluster
        # flips is generous burn-in even at D = 64.
        return wolff_sample(target, n_samples=1, burn_in_clusters=2000, seed=seed).reshape(D, D)
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
