"""8x8 house results cell for the soft chapter: energy and composition marginals.

The house results cell (baseline fig:unconstrained-clean, hard fig:hard-clean-8x8)
is (a) the marginal of the BARE Ising energy per site on its exact levels and
(b) a Z2 read. For the penalised target the magnetisation is 2c - 1, so the Z2
read IS the composition marginal, and that is the second row here. One column
per trained window at the critical coupling, the matched-base recipe off centre
(the centre has no matched-base twin: Bernoulli(1/2) is already matched there).

Three things are drawn per panel:

  reference -- pooled post-burn-in frames of the 4 mchammer VC-SGC chains at
               kappa = lambda, phi = -2 c_target (results/mchammer_vcsgc); the
               same chains the house table's error floors are built from.
  DNFS      -- importance-weighted marginal per seed, seed mean with a min-max
               band, EVERY seed (the chapter's every-seed rule; at sigma_c on
               the matched base every seed is alive, ESS 0.71-0.81).
  envelope  -- composition row only: the analytic exp(-lambda d (c - c_t)^2) on
               the c = k/d grid, width 1/sqrt(2 lambda d) = 0.0125. A guide, not
               a reference: it drops the canonical density of states Z_can(c),
               and the reference sits visibly off it at c_target = 0.25.

Each panel is annotated with the seed-mean TV distance between the DNFS and
reference marginals and the reference's own sampling floor: the TV a 5000-frame
block-bootstrap replicate of the chains shows against the pooled chains, so "at
the floor" means the same thing as in the unconstrained and hard cells.

Example:
    python -m experiments.constrained_soft_02.analysis.25_composition_marginal_overlay_8x8 \
        --coupling sc --matched-base
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from discrete_flow_sampler.diagnostics.figure_style import (
    ANALYTIC_GUIDE, FONT_SIZE_ANNOTATION, FONT_SIZE_LABEL, FULL_WIDTH_IN,
    REFERENCE_INK, SAMPLER_HUE, SAVEFIG_DPI, seed_band, style_axes,
    use_house_style)
from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up as composition)
from discrete_flow_sampler.diagnostics.metrics import marginal_tvd
from discrete_flow_sampler.targets.ising import SIGMA_C, IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"

D_SIDE, N_SITES, LAM = 8, 64, 50.0
TRAINED_COMPOSITIONS = (0.25, 0.375, 0.50)
SEEDS = (42, 43, 44, 45)
COUPLINGS = {"s010": 0.1, "sc": SIGMA_C}
N_EVAL = 5000            # replicate size = the neural draw count
FLOOR_BLOCK = 10         # frames per bootstrap block
N_FLOOR_BOOTSTRAP = 200
ENERGY_SUPPORT = (torch.arange(N_SITES + 1) * 4.0 - 2.0 * N_SITES) / N_SITES
COMPOSITION_SUPPORT = torch.arange(N_SITES + 1).float() / N_SITES


def energy_level_index(target, x):
    """Exact-level index of the bare Ising energy: E = -2d + 4k on the torus."""
    energy = -target.base_log_prob(x) / (2.0 * target.sigma)
    level = (energy + 2.0 * N_SITES) / 4.0
    assert (level - level.round()).abs().max() < 1e-2, "energies off the exact levels"
    return level.round().long().clamp(0, N_SITES)


def energy_pmf(target, x, weights):
    return torch.zeros(N_SITES + 1).index_add_(0, energy_level_index(target, x), weights)


def composition_pmf(target, x, weights):
    bucket = (composition(x) * N_SITES).round().long().clamp(0, N_SITES)
    return torch.zeros(N_SITES + 1).index_add_(0, bucket, weights)


def load_reference(sigma, c_target, target):
    """Pooled post-burn-in VC-SGC frames, energy-checked against the target."""
    pattern = f"D{D_SIDE}_s{sigma:.6g}_l{LAM:.1f}_c{c_target:.3f}_seed*"
    frames = []
    for run_dir in sorted(VCSGC_RESULTS.glob(pattern)):
        spins = torch.from_numpy(np.load(run_dir / "spins.npy")).float()
        potential = torch.from_numpy(np.load(run_dir / "potential.npy")).float()
        assert torch.allclose(-target.base_log_prob(spins), potential, atol=1e-3), run_dir
        frames.append(spins)
    if not frames:
        raise FileNotFoundError(f"no VC-SGC chains match {pattern}")
    return torch.cat(frames)


def reference_tv_floor(reference, pmf_of, seed=0):
    """TV of an N_EVAL-frame block-bootstrap replicate against the pooled chains."""
    generator = torch.Generator().manual_seed(seed)
    n_blocks = reference.shape[0] // FLOOR_BLOCK
    by_block = reference[: n_blocks * FLOOR_BLOCK].view(n_blocks, FLOOR_BLOCK, -1)
    pool_pmf = pmf_of(reference, torch.full((reference.shape[0],), 1.0 / reference.shape[0]))
    tvs = []
    for _ in range(N_FLOOR_BOOTSTRAP):
        blocks = torch.randint(0, n_blocks, (N_EVAL // FLOOR_BLOCK,), generator=generator)
        replicate = by_block[blocks].reshape(-1, N_SITES)
        uniform = torch.full((replicate.shape[0],), 1.0 / replicate.shape[0])
        tvs.append(marginal_tvd(pmf_of(replicate, uniform), pool_pmf))
    return sum(tvs) / len(tvs)


def load_seed_runs(config, eval_dir):
    runs = []
    for seed in SEEDS:
        matches = sorted(RESULTS.glob(f"{config}_seed{seed}_*"))
        if not matches:
            continue
        run_dir = matches[-1]
        samples = torch.load(run_dir / eval_dir / "samples.pt", weights_only=True).float()
        log_w = torch.load(run_dir / eval_dir / "log_weights.pt", weights_only=True)
        runs.append((samples, torch.softmax(log_w, dim=0)))
    return runs


def envelope_pmf(c_target):
    return torch.softmax(-LAM * N_SITES * (COMPOSITION_SUPPORT - c_target) ** 2, dim=0)


def populated_window(support, *pmfs, pad_levels=2):
    populated = torch.zeros_like(pmfs[0], dtype=torch.bool)
    for pmf in pmfs:
        populated |= pmf > 1e-4
    indices = populated.nonzero().flatten()
    lo = max(int(indices.min()) - pad_levels, 0)
    hi = min(int(indices.max()) + pad_levels, len(support) - 1)
    return support[lo].item(), support[hi].item()


def plot_panel(ax, support, ref_pmf, seed_pmfs, tv, floor, label):
    ax.step(support, ref_pmf, where="mid", color=REFERENCE_INK, lw=1.4, zorder=3,
            label="VC-SGC chains")
    seed_band(ax, support, seed_pmfs, SAMPLER_HUE, "soft DNFS, IS-weighted")
    ax.set_xlim(*populated_window(support, ref_pmf, *seed_pmfs))
    ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=FONT_SIZE_LABEL,
            fontweight="bold", va="top")
    # Right-hand shoulder: the marginals peak at the centre and the label
    # owns the top-left, so the only empty strip is the upper right below
    # the label line.
    ax.text(0.98, 0.80, f"TV {tv:.3f}\nfloor {floor:.3f}", transform=ax.transAxes,
            fontsize=FONT_SIZE_ANNOTATION, ha="right", va="top")
    style_axes(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", choices=list(COUPLINGS), default="sc")
    parser.add_argument("--eval_dir", choices=["eval", "eval_ema"], default="eval")
    parser.add_argument("--matched-base", action="store_true",
                        help="off-centre windows read the *_house_mb families")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sigma = COUPLINGS[args.coupling]
    target = IsingTarget(D=D_SIDE, sigma=sigma, bias=0.0)
    config_suffix = "_sc" if args.coupling == "sc" else ""
    rows = (("energy", energy_pmf, ENERGY_SUPPORT, "energy per site $E/d$"),
            ("composition", composition_pmf, COMPOSITION_SUPPORT, "composition $c$"))

    use_house_style()
    fig, axes = plt.subplots(2, len(TRAINED_COMPOSITIONS), figsize=(FULL_WIDTH_IN, 3.9))
    panel_labels = iter("abcdef")
    for col, c_target in enumerate(TRAINED_COMPOSITIONS):
        family = "house_mb" if args.matched_base and c_target != 0.5 else "house"
        config = f"S2_d8_c{int(round(c_target * 1000)):04d}_l50_letf_ne128_{family}{config_suffix}"
        reference = load_reference(sigma, c_target, target)
        runs = load_seed_runs(config, args.eval_dir)
        uniform = torch.full((reference.shape[0],), 1.0 / reference.shape[0])
        for row, (key, pmf_of, support, xlabel) in enumerate(rows):
            marginal = lambda x, w, _f=pmf_of: _f(target, x, w)
            ref_pmf = marginal(reference, uniform)
            seed_pmfs = [marginal(x, w) for x, w in runs]
            tv = float(np.mean([marginal_tvd(pmf, ref_pmf) for pmf in seed_pmfs]))
            floor = reference_tv_floor(reference, marginal)
            ax = axes[row, col]
            plot_panel(ax, support, ref_pmf, seed_pmfs, tv, floor,
                       f"({next(panel_labels)}) $c_\\mathrm{{target}} = {c_target:g}$")
            if key == "composition":
                ax.plot(COMPOSITION_SUPPORT, envelope_pmf(c_target), color=ANALYTIC_GUIDE,
                        lw=1.0, ls="--", label="analytic envelope", zorder=2)
            if row == 1:
                ax.set_xlabel(xlabel)
            print(f"{args.coupling} c={c_target} {key:11} TV {tv:.4f} floor {floor:.4f} "
                  f"seeds {len(seed_pmfs)} ref frames {reference.shape[0]}")
        axes[0, col].set_xlabel(rows[0][3])
    axes[0, 0].set_ylabel("probability mass")
    axes[1, 0].set_ylabel("probability mass")
    handles, labels = axes[1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               fontsize=FONT_SIZE_ANNOTATION)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = args.out or (RESULTS / f"soft_results_cell_8x8_{args.coupling}.png")
    fig.savefig(out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
