"""fig:soft-clean -- the soft chapter's K2 results cell, house standard.

The approved house figure set (s62 board, K2; soft instantiation decided
s64: c=0.50 window, lambda-trade figure #6 stays separate) gives the soft
chapter the same two-panel results-cell as the other results chapters, at
the headline 10x10 size, lambda=50, c_target=0.50, sigma=0.1:

  (a) energy marginal on the EXACT levels of the BARE Ising energy, E/d
      axis. The penalty is excluded, matching the house table's EW2
      convention (the quality question is whether the sampler gets the
      *physics* right inside the constrained ensemble; the penalty term is
      shared bookkeeping). On the periodic lattice E = -2d + 4k, so binning
      on that support carries no aliasing.
  (b) composition marginal on its exact 101-point support k/d -- the
      chapter's order-parameter/coverage read: the soft target holds c in a
      band of width 1/sqrt(2*lambda*d) around the target, and the panel
      shows the sampler landing on that band, not merely near it.

Reference = the pooled mchammer VC-SGC chains at matched kappa=lambda (the
chapter's like-for-like ensemble, sec:fc), order-checked against the
validated embedding exactly as in 19_house_table_soft. The caption quotes
each panel's total-variation distance beside the reference's own sampling
floor. The floor is 19's construction verbatim -- N_EVAL-frame BLOCK
bootstrap replicates of the pool scored against the pool -- NOT the
chain-resampling bootstrap the unconstrained chapter uses: with only 4
VC-SGC chains a chain-level resample is too coarse to price sampling
noise, and reusing 19's blocks keeps "at the floor" meaning the same
thing in this chapter's figure and table.
"""
import argparse
import json
from pathlib import Path

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_FULL_1X2, FONT_SIZE_ANNOTATION, FONT_SIZE_LABEL, REFERENCE_INK,
    SAMPLER_HUE, SAVEFIG_DPI, seed_band, style_axes, use_house_style)
import matplotlib.pyplot as plt
import numpy as np
import torch

from discrete_flow_sampler.diagnostics.metrics import marginal_tvd
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"
L, SIGMA = 10, 0.1
D_SITES = L * L
N_EVAL = 5000            # replicate size = the neural draw count (19's floor)
FLOOR_BLOCK = 10         # frames per bootstrap block (19's BLOCK)
N_FLOOR_BOOTSTRAP = 200


TARGET = IsingTarget(D=L, sigma=SIGMA, bias=0.0)


def composition(x: torch.Tensor) -> torch.Tensor:
    return (x.mean(dim=-1) + 1.0) * 0.5


def energy_level_index(x: torch.Tensor) -> torch.Tensor:
    """Exact-level index of the bare Ising energy (penalty excluded).

    E = -base_log_prob / (2 sigma); on the periodic lattice E = -2d + 4k.
    The rounding assert guards the support assumption, as in the
    unconstrained cell: off-level energies mean a changed convention and
    the figure must not silently rebin.
    """
    energy = -TARGET.base_log_prob(x) / (2.0 * SIGMA)
    level = (energy + 2.0 * D_SITES) / 4.0
    rounded = level.round()
    assert (level - rounded).abs().max() < 1e-2, \
        "state energies off the exact-level support"
    return rounded.long().clamp(0, D_SITES)


def energy_pmf(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.zeros(D_SITES + 1).index_add_(0, energy_level_index(x), weights)


def composition_pmf(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    up_count = (composition(x) * D_SITES).round().long().clamp(0, D_SITES)
    return torch.zeros(D_SITES + 1).index_add_(0, up_count, weights)


def load_reference(penalty_strength: float, c_target: float) -> torch.Tensor:
    """Pooled post-burn-in VC-SGC frames, order-checked (19's assert)."""
    frames = []
    pattern = f"D{L}_s{SIGMA}_l{penalty_strength:.1f}_c{c_target:.2f}_seed*"
    for run_dir in sorted(VCSGC_RESULTS.glob(pattern)):
        spins = torch.from_numpy(np.load(run_dir / "spins.npy")).float()
        potential = torch.from_numpy(np.load(run_dir / "potential.npy")).float()
        assert torch.allclose(-TARGET.base_log_prob(spins), potential,
                              atol=1e-3), run_dir
        frames.append(spins)
    if not frames:
        raise FileNotFoundError(f"no VC-SGC reference matches {pattern}")
    return torch.cat(frames)


def reference_tv_floor(reference: torch.Tensor, pmf_of, seed: int = 0) -> float:
    """19's block bootstrap, read as TV: N_EVAL-frame replicates of the pool
    scored against the whole pool -- the sampling noise a perfect sampler
    would show at the neural draw count."""
    generator = torch.Generator().manual_seed(seed)
    n_blocks = reference.shape[0] // FLOOR_BLOCK
    by_block = reference[: n_blocks * FLOOR_BLOCK].view(n_blocks, FLOOR_BLOCK, -1)
    uniform_pool = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
    pool_pmf = pmf_of(reference, uniform_pool)
    tvs = []
    for _ in range(N_FLOOR_BOOTSTRAP):
        blocks = torch.randint(0, n_blocks, (N_EVAL // FLOOR_BLOCK,),
                               generator=generator)
        replicate = by_block[blocks].reshape(-1, reference.shape[1])
        uniform = torch.full((replicate.shape[0],), 1.0 / replicate.shape[0])
        tvs.append(marginal_tvd(pmf_of(replicate, uniform), pool_pmf))
    return sum(tvs) / len(tvs)


def load_seed_runs(run_dirs: list[Path]) -> list[dict]:
    runs = []
    for run_dir in run_dirs:
        samples = torch.load(run_dir / "eval" / "samples.pt",
                             weights_only=True).float()
        log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
        runs.append({"samples": samples,
                     "weights": torch.softmax(log_w, dim=0),
                     "name": run_dir.name})
    return runs


def populated_window(support, *pmfs, pad_levels: int = 2):
    populated = torch.zeros_like(pmfs[0], dtype=torch.bool)
    for pmf in pmfs:
        populated |= pmf > 0
    indices = populated.nonzero().flatten()
    lo = max(int(indices.min()) - pad_levels, 0)
    hi = min(int(indices.max()) + pad_levels, len(support) - 1)
    return support[lo].item(), support[hi].item()


def plot_house_panel(ax, support, ref_pmf, seed_pmfs, xlabel, panel_label,
                     n_reference, n_seeds, with_legend):
    ax.step(support, ref_pmf, where="mid", color=REFERENCE_INK, lw=1.6,
            zorder=3, label="VC-SGC")
    seed_band(ax, support, seed_pmfs, SAMPLER_HUE, "DNFS")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability mass")
    ax.set_xlim(*populated_window(support, ref_pmf, *seed_pmfs))
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=FONT_SIZE_LABEL, fontweight="bold", va="top")
    if with_legend:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles,
                  [f"VC-SGC (n={n_reference})",
                   f"soft DNFS IS-weighted\n(min–max, {n_seeds} seeds)"],
                  fontsize=FONT_SIZE_ANNOTATION, frameon=False,
                  loc="upper right")
    style_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, nargs="+", type=Path,
                        help="the headline-window specialist run dirs "
                             "(S2_d10_c05_l50_letf_ne128_anneal seeds 42-45)")
    parser.add_argument("--penalty_strength", type=float, default=50.0)
    parser.add_argument("--c_target", type=float, default=0.50)
    parser.add_argument("--out", type=Path, default=Path("soft_clean_cell.png"))
    args = parser.parse_args()
    use_house_style()

    reference = load_reference(args.penalty_strength, args.c_target)
    seed_runs = load_seed_runs(args.runs)

    support_energy = (torch.arange(D_SITES + 1) * 4.0 - 2.0 * D_SITES) / D_SITES
    support_c = torch.arange(D_SITES + 1) / D_SITES
    uniform = torch.full((reference.shape[0],), 1.0 / reference.shape[0])

    panels = {}
    for key, pmf_of, support in (("energy", energy_pmf, support_energy),
                                 ("composition", composition_pmf, support_c)):
        ref_pmf = pmf_of(reference, uniform)
        seed_pmfs = torch.stack([pmf_of(run["samples"], run["weights"])
                                 for run in seed_runs])
        panels[key] = {
            "support": support, "ref": ref_pmf, "seeds": seed_pmfs,
            "tvds": torch.tensor([marginal_tvd(s, ref_pmf) for s in seed_pmfs]),
            "floor": reference_tv_floor(reference, pmf_of),
        }

    print(f"=== soft K2 cell (lambda={args.penalty_strength:g}, "
          f"c_target={args.c_target:g}, {len(seed_runs)} seeds, "
          f"{reference.shape[0]} reference frames) -- caption numbers ===")
    for label, key in (("(a) energy-level TV", "energy"),
                       ("(b) composition TV ", "composition")):
        tv = panels[key]["tvds"]
        print(f"  {label}: {tv.mean():.3f} +/- {tv.std():.3f} "
              f"(floor {panels[key]['floor']:.3f})")

    fig, (ax_energy, ax_c) = plt.subplots(1, 2, figsize=FIGSIZE_FULL_1X2)
    plot_house_panel(ax_energy, panels["energy"]["support"],
                     panels["energy"]["ref"], panels["energy"]["seeds"],
                     r"$E/d$", "(a)", reference.shape[0], len(seed_runs),
                     with_legend=True)
    plot_house_panel(ax_c, panels["composition"]["support"],
                     panels["composition"]["ref"],
                     panels["composition"]["seeds"],
                     r"composition $c$", "(b)", reference.shape[0],
                     len(seed_runs), with_legend=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=SAVEFIG_DPI)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
