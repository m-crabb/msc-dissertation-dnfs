"""fig:unconstrained-clean -- the unconstrained chapter's results-cell figure.

Two panels at the headline 10x10 size:

  (a) energy marginal drawn on the exact energy levels, E/d axis (the same
      per-site convention as the house table's EW2 column). The periodic
      D x D lattice has 2d bonds and E changes by multiples of 4, so the
      support is E in {-2d, -2d+4, ..., 2d}; binning on that support avoids
      the 40-uniform-bin aliasing the old figure carried.
  (b) magnetisation marginal on its exact 101-point support (2k - d)/d --
      the Z2-odd coverage read: the target is symmetric, so a sampler that
      covers both phases puts ~half its weighted mass in each mode, which
      the hard chapter's Kawasaki chains cannot do.

Reference = the certified Wolff cluster pool (built by
wolff_reference_pool.py, R-hat <= 1.002). The pool file is keyed by the run's
own coupling, so legacy runs meet the legacy pool and sigma_c retrains meet
the 0.220343 pool -- couplings are never mixed.

The caption quotes each panel's total-variation distance beside an iid
sampling floor: the mean TV of 5000 independently resampled frames against
the empirical reference pool. Pool uncertainty is a separate diagnostic;
proximity to this mean does not establish statistical equivalence.

The subcritical appendix figure (log-density marginal, uniform bins on a
continuous axis) stays in its pre-house form.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_FULL_1X2_SHORT,
    FIGSIZE_SINGLE,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    REFERENCE_INK,
    SAMPLER_HUE,
    SAVEFIG_DPI,
    seed_band,
    style_axes,
    use_house_style,
)
from discrete_flow_sampler.diagnostics.metrics import marginal_tvd
from discrete_flow_sampler.targets.ising import IsingTarget

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
N_SITES = 100  # D=10 -> d = 100
N_LOG_DENSITY_BINS = 40  # subcritical appendix panel only (continuous axis)
N_FLOOR_BOOTSTRAP = 200


def magnetisation(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=-1)


def load_seed_runs(run_dirs: list[Path]) -> list[dict]:
    """Per-seed eval artefacts: samples, normalised IS weights, stored metrics."""
    runs = []
    for run_dir in run_dirs:
        samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
        log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        runs.append(
            {
                "samples": samples,
                "weights": torch.softmax(log_w, dim=0),
                "metrics": metrics,
                "name": run_dir.name,
            }
        )
    return runs


def energy_level_index(target: IsingTarget, x: torch.Tensor) -> torch.Tensor:
    """Map states to indices on the exact energy-level support.

    E = -log p~ / (2 sigma) (the chapter's convention, bias = 0), and on the
    periodic lattice E = -2d + 4k, so k = (E + 2d) / 4 indexes the d/2 + 1...
    strictly (4d/4)+1 = d+1 levels. The rounding assert is the aliasing
    guard: if energies ever land off-level the support assumption is wrong
    (e.g. a biased target) and this figure must not silently rebin.
    """
    energy = -target.log_prob(x) / (2.0 * target.sigma)
    level = (energy + 2.0 * target.d) / 4.0
    level_rounded = level.round()
    assert (level - level_rounded).abs().max() < 1e-2, (
        "state energies off the exact-level support"
    )
    return level_rounded.long().clamp(0, target.d)


def energy_level_pmfs(
    target: IsingTarget, ref_samples: torch.Tensor, seed_runs: list[dict]
) -> dict:
    """Reference + per-seed pmfs on the exact energy levels, E/d axis."""
    n_levels = target.d + 1
    support_energy_per_site = (torch.arange(n_levels) * 4.0 - 2.0 * target.d) / target.d

    def pmf(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n_levels).index_add_(
            0, energy_level_index(target, x), weights
        )

    uniform = torch.full((ref_samples.shape[0],), 1.0 / ref_samples.shape[0])
    ref_pmf = pmf(ref_samples, uniform)
    seed_pmfs = torch.stack([pmf(run["samples"], run["weights"]) for run in seed_runs])
    return {
        "support": support_energy_per_site,
        "ref": ref_pmf,
        "seeds": seed_pmfs,
        "tvds": [marginal_tvd(s, ref_pmf) for s in seed_pmfs],
    }


def magnetisation_pmfs(ref_samples: torch.Tensor, seed_runs: list[dict]) -> dict:
    """Magnetisation pmfs on the exact 101-point support (2k - d)/d."""
    support_m = (2.0 * torch.arange(N_SITES + 1) - N_SITES) / N_SITES

    def pmf(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        up_count = (magnetisation(samples) + 1.0) * 0.5 * N_SITES
        bucket = up_count.round().long().clamp(0, N_SITES)
        return torch.zeros(N_SITES + 1).index_add_(0, bucket, weights)

    uniform = torch.full((ref_samples.shape[0],), 1.0 / ref_samples.shape[0])
    ref_pmf = pmf(ref_samples, uniform)
    seed_pmfs = torch.stack([pmf(run["samples"], run["weights"]) for run in seed_runs])
    return {
        "support": support_m,
        "ref": ref_pmf,
        "seeds": seed_pmfs,
        "tvds": [marginal_tvd(s, ref_pmf) for s in seed_pmfs],
    }


def reference_tv_floor(
    ref_samples: torch.Tensor,
    n_chains: int,
    pmf_of: callable,
    seed: int = 0,
    n_draws: int = 5000,
) -> float:
    """Mean TV of n_draws iid reference frames against the empirical pool.

    This conditional ideal-draw benchmark excludes uncertainty in the pool
    itself; whole-chain resampling answers that different question. Retain
    n_chains for compatibility with existing plot callers, but do not use it
    to correlate draws. A distance near this mean is not an equivalence test.
    """
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.full((ref_samples.shape[0],), 1.0 / ref_samples.shape[0])
    pool_pmf = pmf_of(ref_samples, uniform)
    draw_weights = torch.full((n_draws,), 1.0 / n_draws)
    tvs = []
    for _ in range(N_FLOOR_BOOTSTRAP):
        indices = torch.randint(ref_samples.shape[0], (n_draws,), generator=generator)
        tvs.append(marginal_tvd(pmf_of(ref_samples[indices], draw_weights), pool_pmf))
    return sum(tvs) / len(tvs)


def populated_window(
    support: torch.Tensor, *pmfs: torch.Tensor, pad_levels: int = 2
) -> tuple[float, float]:
    """x-limits trimmed to the populated levels: the exact support spans the
    whole spectrum but at any one coupling only a narrow window carries mass,
    and plotting the empty tail flattens the visible structure."""
    populated = torch.zeros_like(pmfs[0], dtype=torch.bool)
    for pmf in pmfs:
        populated |= pmf > 0
    indices = populated.nonzero().flatten()
    lo = max(int(indices.min()) - pad_levels, 0)
    hi = min(int(indices.max()) + pad_levels, len(support) - 1)
    return support[lo].item(), support[hi].item()


def plot_house_panel(
    ax, support, ref_pmf, seed_pmfs, xlabel, panel_label, n_reference, with_legend
):
    """One panel: reference as ink steps on the discrete support, sampler as
    the seed-band grammar, bold corner label, house axes."""
    ax.step(
        support,
        ref_pmf,
        where="mid",
        color=REFERENCE_INK,
        lw=1.6,
        zorder=3,
        label=f"Wolff (n={n_reference})",
    )
    seed_band(ax, support, seed_pmfs, SAMPLER_HUE, "DNFS", step=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability mass")
    ax.set_xlim(*populated_window(support, ref_pmf, *seed_pmfs))
    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_LABEL,
        fontweight="bold",
        va="top",
    )
    if with_legend:
        # seed_band's stock label overflows a half-text-width panel; relabel
        # compactly, keeping n on both entries (the band convention).
        handles, _ = ax.get_legend_handles_labels()
        n_seeds = seed_pmfs.shape[0]
        ax.legend(
            handles,
            [
                f"Wolff (n={n_reference})",
                f"DNFS IS-weighted\n(min–max, {n_seeds} seeds)",
            ],
            fontsize=FONT_SIZE_ANNOTATION,
            frameon=False,
            loc="upper right",
        )
    style_axes(ax)


def log_density_marginal(
    target: IsingTarget, ref_samples: torch.Tensor, seed_runs: list[dict]
) -> dict:
    """Pre-house subcritical appendix panel: log p~ marginal on uniform bins
    (continuous axis; bin count quoted in its caption)."""
    ref_values = target.log_prob(ref_samples)
    seed_values = [target.log_prob(run["samples"]) for run in seed_runs]
    lo = min(ref_values.min(), *(v.min() for v in seed_values)).item()
    hi = max(ref_values.max(), *(v.max() for v in seed_values)).item()
    edges = torch.linspace(lo, hi, N_LOG_DENSITY_BINS + 1)

    def pmf(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.bucketize(values, edges[1:-1], right=False)
        return torch.zeros(N_LOG_DENSITY_BINS).index_add_(0, bin_idx, weights)

    uniform = torch.full((ref_values.numel(),), 1.0 / ref_values.numel())
    seed_pmfs = torch.stack(
        [pmf(values, run["weights"]) for values, run in zip(seed_values, seed_runs)]
    )
    return {
        "centres": 0.5 * (edges[:-1] + edges[1:]),
        "ref": pmf(ref_values, uniform),
        "seeds": seed_pmfs,
    }


def final_sigma(config: dict) -> float:
    """Coupling of the frozen eval: the last curriculum stage, else ising.sigma."""
    curriculum = config.get("curriculum")
    return curriculum["stages"][-1]["sigma"] if curriculum else config["ising"]["sigma"]


def load_point(run_dirs: list[Path]) -> dict:
    """Everything one coupling needs: target, Wolff pool, per-seed evals."""
    config = json.loads((run_dirs[0] / "config.json").read_text())
    ising_cfg = config["ising"]
    # A curriculum run stores its starting coupling under ising.sigma; the
    # frozen eval is at the final stage's coupling, which keys the pool.
    sigma = final_sigma(config)
    target = IsingTarget(D=ising_cfg["D"], sigma=sigma, bias=ising_cfg["bias"])
    pool = torch.load(
        RESULTS / f"wolff_ref_d10_sigma{sigma:g}.pt", weights_only=True
    )
    return {
        "target": target,
        "sigma": sigma,
        "ref_samples": pool["samples"].float(),
        "n_chains": pool["n_chains"],
        "seed_runs": load_seed_runs(run_dirs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--budget_runs",
        required=True,
        nargs="+",
        type=Path,
        help="stage_4_d10_budget run dirs (seeds 42-45)",
    )
    parser.add_argument(
        "--critical_runs",
        required=True,
        nargs="+",
        type=Path,
        help="stage_4_d10_critical run dirs (legacy family "
        "until the _sc retrains land, then those)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("unconstrained_clean_demo.png")
    )
    args = parser.parse_args()
    use_house_style()

    # --- critical cell (body figure) ---------------------------------------
    critical = load_point(args.critical_runs)
    target, ref = critical["target"], critical["ref_samples"]
    energy = energy_level_pmfs(target, ref, critical["seed_runs"])
    magnet = magnetisation_pmfs(ref, critical["seed_runs"])

    def energy_pmf_of(x, w):
        return torch.zeros(target.d + 1).index_add_(0, energy_level_index(target, x), w)

    def magnet_pmf_of(x, w):
        up_count = (magnetisation(x) + 1.0) * 0.5 * N_SITES
        return torch.zeros(N_SITES + 1).index_add_(
            0, up_count.round().long().clamp(0, N_SITES), w
        )

    floor_energy = reference_tv_floor(ref, critical["n_chains"], energy_pmf_of)
    floor_magnet = reference_tv_floor(ref, critical["n_chains"], magnet_pmf_of)

    tvd_e, tvd_m = torch.tensor(energy["tvds"]), torch.tensor(magnet["tvds"])
    print(
        f"=== critical (sigma={critical['sigma']:g}, "
        f"{len(critical['seed_runs'])} seeds) -- caption numbers ==="
    )
    print(
        f"  (a) energy-level TV : {tvd_e.mean():.3f} +/- {tvd_e.std():.3f} "
        f"(floor {floor_energy:.3f})"
    )
    print(
        f"  (b) magnetisation TV: {tvd_m.mean():.3f} +/- {tvd_m.std():.3f} "
        f"(floor {floor_magnet:.3f})"
    )
    for run in critical["seed_runs"]:
        mass_plus = run["weights"][magnetisation(run["samples"]) > 0].sum()
        print(f"  {run['name']}: mass(m>0) = {mass_plus:.3f}")

    fig, (ax_energy, ax_magnet) = plt.subplots(1, 2, figsize=FIGSIZE_FULL_1X2_SHORT)
    plot_house_panel(
        ax_energy,
        energy["support"],
        energy["ref"],
        energy["seeds"],
        r"$E/d$",
        "(a)",
        ref.shape[0],
        with_legend=True,
    )
    plot_house_panel(
        ax_magnet,
        magnet["support"],
        magnet["ref"],
        magnet["seeds"],
        r"magnetisation $m$",
        "(b)",
        ref.shape[0],
        with_legend=False,
    )
    fig.tight_layout()
    critical_out = args.out.with_name(f"{args.out.stem}_critical.png")
    fig.savefig(critical_out, dpi=SAVEFIG_DPI)

    # --- subcritical appendix twin (pre-house panel on the house canvas) ----
    subcritical = load_point(args.budget_runs)
    log_density = log_density_marginal(
        subcritical["target"], subcritical["ref_samples"], subcritical["seed_runs"]
    )
    ess = torch.tensor(
        [run["metrics"]["ess_fraction"] for run in subcritical["seed_runs"]]
    )
    fig_sub, ax_sub = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax_sub.plot(
        log_density["centres"],
        log_density["ref"],
        color=REFERENCE_INK,
        lw=1.8,
        label="Wolff reference",
    )
    ax_sub.fill_between(
        log_density["centres"],
        log_density["seeds"].min(dim=0).values,
        log_density["seeds"].max(dim=0).values,
        color=SAMPLER_HUE,
        alpha=0.18,
        label="DNFS (seed min-max)",
    )
    ax_sub.plot(
        log_density["centres"],
        log_density["seeds"].mean(dim=0),
        color=SAMPLER_HUE,
        lw=1.5,
        label="DNFS IS-weighted (mean)",
    )
    ax_sub.set_xlabel(r"$\log \tilde p(x)$")
    ax_sub.set_ylabel("probability mass")
    ax_sub.set_title(
        f"log-density marginal, $\\sigma={subcritical['sigma']:g}$\n"
        f"ESS fraction ${ess.mean():.3f} \\pm {ess.std():.3f}$ (4 seeds)",
        fontsize=10,
    )
    ax_sub.legend(fontsize=8, framealpha=0.9)
    fig_sub.tight_layout()
    subcritical_out = args.out.with_name(f"{args.out.stem}_subcritical.png")
    fig_sub.savefig(subcritical_out, dpi=SAVEFIG_DPI)
    print(f"\nsaved figures to {critical_out} and {subcritical_out}")


if __name__ == "__main__":
    main()
