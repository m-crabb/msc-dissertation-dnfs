"""§3.3 figure: DNFS samples the UNCONSTRAINED Ising cleanly at both regimes.

Chapter 3's closing claim: the failure modes of §3.1 (Kawasaki critical
slowing-down, mode trapping) and §3.2 (soft-penalty inexactness) are about
constraint handling, not about the sampler family. Evidence: on the
unconstrained version of the same D=10 problem, DNFS produces near-independent
samples (ESS fraction ~0.99 subcritical, ~0.91 at sigma_c) whose marginals sit
on a long Gibbs heat-bath reference, at the very coupling sigma_c where §3.1
shows Kawasaki's normalised ESS collapsing to ~0.05.

Three panels:
  (a) log p~(x) marginal at the subcritical operating point (stage_4_d10_budget)
      -- DNFS IS-weighted (4-seed mean + min-max band) vs Gibbs reference.
  (b) same at sigma_c (stage_4_d10_critical_paper_curriculum).
  (c) magnetisation marginal at sigma_c -- the distribution is strongly
      bimodal (the two Z2 phases), and a single DNFS sampling pass covers BOTH
      modes symmetrically, exactly the ergodicity test the §3.1 Kawasaki
      chains fail (each chain stranded in the mode it started in).

The Gibbs reference is the same heat-bath oracle used for the soft-constraint
fidelity checks (mcmc/gibbs.py), run unconstrained: many parallel chains from
random (hot) inits, long burn-in, thinned records. At sigma_c single-flip
dynamics also slow down (that is the physics), so the reference leans on chain
COUNT for independence: with 100 chains whose inits land in either Z2 mode at
random, the pooled histogram is unbiased even if individual chains tunnel
rarely. Cross-chain R-hat on magnetisation is printed as the mixing check.
References are cached to results/01_baseline/gibbs_ref_*.pt (~minutes to build).
"""
import argparse
import json
from pathlib import Path

from discrete_flow_sampler.diagnostics.figure_style import (
    REFERENCE_INK, SAMPLER_HUE, NEURAL_COMPARATOR_HUE, CLASSICAL_HUE,
    CLASSICAL_ALT_HUE, MUTED, GRID, use_house_style)
import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import gelman_rubin
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.diagnostics.metrics import marginal_tvd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "01_baseline"
N_SITES = 100  # D=10 -> d = 100
N_ENERGY_BINS = 40

GIBBS_N_CHAINS = 100
GIBBS_BURN_IN_SWEEPS = 2_000
GIBBS_THIN_SWEEPS = 100
GIBBS_N_RECORDS = 50  # 100 chains x 50 records = 5000 reference samples


def magnetisation(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=-1)


def gibbs_reference(sigma: float, cache_path: Path, seed: int = 0) -> torch.Tensor:
    """Pooled unconstrained Gibbs samples at coupling sigma, cached to disk."""
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        print(f"  [gibbs ref sigma={sigma}] loaded cache {cache_path.name} "
              f"(R-hat(m) = {cached['gelman_rubin_m']:.3f})")
        return cached["samples"]

    target = IsingTarget(D=10, sigma=sigma, bias=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"  [gibbs ref sigma={sigma}] building: {GIBBS_N_CHAINS} chains, "
          f"{GIBBS_BURN_IN_SWEEPS} burn-in + "
          f"{GIBBS_N_RECORDS}x{GIBBS_THIN_SWEEPS} sweeps")
    spins = gibbs_sample(target, GIBBS_N_CHAINS, GIBBS_BURN_IN_SWEEPS,
                         generator=generator)
    records = []
    for _ in range(GIBBS_N_RECORDS):
        spins = gibbs_sample(target, GIBBS_N_CHAINS, GIBBS_THIN_SWEEPS,
                             x_init=spins, generator=generator)
        records.append(spins.clone())
    record_stack = torch.stack(records)  # (n_records, n_chains, d)

    # Mixing check: R-hat on per-chain magnetisation traces. Near 1 => the
    # pooled histogram is trustworthy; large => chains disagree, distrust it.
    m_per_chain = magnetisation(record_stack).T  # (n_chains, n_records)
    rhat_m = gelman_rubin(m_per_chain.numpy())
    print(f"  [gibbs ref sigma={sigma}] R-hat(m) = {rhat_m:.3f}")

    samples = record_stack.reshape(-1, target.d)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "sigma": sigma, "gelman_rubin_m": rhat_m,
                "n_chains": GIBBS_N_CHAINS, "burn_in": GIBBS_BURN_IN_SWEEPS,
                "thin": GIBBS_THIN_SWEEPS}, cache_path)
    return samples


def load_seed_runs(run_dirs: list[Path]) -> list[dict]:
    """Per-seed eval artefacts: samples, normalised IS weights, stored metrics."""
    runs = []
    for run_dir in run_dirs:
        samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
        log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        runs.append({"samples": samples, "weights": torch.softmax(log_w, dim=0),
                     "metrics": metrics, "name": run_dir.name})
    return runs


def energy_marginals(target: IsingTarget, ref_samples: torch.Tensor,
                     seed_runs: list[dict]) -> dict:
    """log p~ histograms on shared bins: reference pmf + per-seed DNFS pmfs."""
    ref_energy = target.log_prob(ref_samples)
    seed_energies = [target.log_prob(run["samples"]) for run in seed_runs]
    lo = min(ref_energy.min(), *(e.min() for e in seed_energies)).item()
    hi = max(ref_energy.max(), *(e.max() for e in seed_energies)).item()
    edges = torch.linspace(lo, hi, N_ENERGY_BINS + 1)

    def pmf(energies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.bucketize(energies, edges[1:-1], right=False)
        return torch.zeros(N_ENERGY_BINS).index_add_(0, bin_idx, weights)

    uniform = torch.full((ref_energy.numel(),), 1.0 / ref_energy.numel())
    ref_pmf = pmf(ref_energy, uniform)
    seed_pmfs = torch.stack([
        pmf(energy, run["weights"])
        for energy, run in zip(seed_energies, seed_runs)
    ])
    return {"centres": 0.5 * (edges[:-1] + edges[1:]), "ref": ref_pmf,
            "seeds": seed_pmfs,
            "tvds": [marginal_tvd(s, ref_pmf) for s in seed_pmfs]}


def magnetisation_pmfs(ref_samples: torch.Tensor, seed_runs: list[dict]) -> dict:
    """Magnetisation histograms on the exact 101-point support (2k - d)/d."""
    support_m = (2.0 * torch.arange(N_SITES + 1) - N_SITES) / N_SITES

    def pmf(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        up_count = ((magnetisation(samples) + 1.0) * 0.5 * N_SITES)
        bucket = up_count.round().long().clamp(0, N_SITES)
        return torch.zeros(N_SITES + 1).index_add_(0, bucket, weights)

    uniform = torch.full((ref_samples.shape[0],), 1.0 / ref_samples.shape[0])
    ref_pmf = pmf(ref_samples, uniform)
    seed_pmfs = torch.stack([pmf(run["samples"], run["weights"])
                             for run in seed_runs])
    return {"support": support_m, "ref": ref_pmf, "seeds": seed_pmfs,
            "tvds": [marginal_tvd(s, ref_pmf) for s in seed_pmfs]}


def ess_fraction_summary(seed_runs: list[dict]) -> tuple[float, float]:
    fracs = torch.tensor([run["metrics"]["ess_fraction"] for run in seed_runs])
    return fracs.mean().item(), fracs.std().item()


def plot_marginal_panel(ax, centres, ref_pmf, seed_pmfs, xlabel, title):
    ax.plot(centres, ref_pmf, color=REFERENCE_INK, lw=1.8, label="Gibbs reference")
    ax.fill_between(centres, seed_pmfs.min(dim=0).values,
                    seed_pmfs.max(dim=0).values, color=SAMPLER_HUE, alpha=0.18,
                    label="DNFS (seed min-max)")
    ax.plot(centres, seed_pmfs.mean(dim=0), color=SAMPLER_HUE, lw=1.5,
            label="DNFS IS-weighted (mean)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability mass")
    ax.set_title(title, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget_runs", required=True, nargs="+", type=Path,
                        help="stage_4_d10_budget run dirs (seeds 42-45)")
    parser.add_argument("--critical_runs", required=True, nargs="+", type=Path,
                        help="stage_4_d10_critical_paper_curriculum run dirs")
    parser.add_argument("--out", type=Path,
                        default=Path("unconstrained_clean_demo.png"))
    args = parser.parse_args()

    panels = []
    for label, run_dirs in (("subcritical", args.budget_runs),
                            ("critical", args.critical_runs)):
        ising_cfg = json.loads((run_dirs[0] / "config.json").read_text())["ising"]
        sigma = ising_cfg["sigma"]
        target = IsingTarget(D=ising_cfg["D"], sigma=sigma, bias=ising_cfg["bias"])
        ref_samples = gibbs_reference(
            sigma, RESULTS / f"gibbs_ref_d10_sigma{sigma:g}.pt")
        seed_runs = load_seed_runs(run_dirs)
        energy = energy_marginals(target, ref_samples, seed_runs)
        magnet = magnetisation_pmfs(ref_samples, seed_runs)
        ess_mean, ess_std = ess_fraction_summary(seed_runs)
        panels.append({"label": label, "sigma": sigma, "energy": energy,
                       "magnet": magnet, "ess": (ess_mean, ess_std)})

        tvd_e = torch.tensor(energy["tvds"])
        tvd_m = torch.tensor(magnet["tvds"])
        print(f"=== {label} (sigma={sigma:g}, {len(seed_runs)} seeds) ===")
        print(f"  ESS fraction            : {ess_mean:.4f} +/- {ess_std:.4f}")
        print(f"  log p~ marginal TVD     : {tvd_e.mean():.4f} +/- {tvd_e.std():.4f}")
        print(f"  magnetisation TVD       : {tvd_m.mean():.4f} +/- {tvd_m.std():.4f}")
        # Z2 mode coverage: weighted mass on each side of m = 0. The target is
        # symmetric, so ~0.5/0.5 means one sampling pass covers both phases.
        for run in seed_runs:
            mass_plus = run["weights"][magnetisation(run["samples"]) > 0].sum()
            mass_minus = run["weights"][magnetisation(run["samples"]) < 0].sum()
            print(f"  {run['name']}: mass(m>0) = {mass_plus:.3f}, "
                  f"mass(m<0) = {mass_minus:.3f}")

    subcritical, critical = panels

    # Two outputs rather than one three-panel strip: rendered at \textwidth the
    # strip put ~5pt tick labels on the page. The critical panels (the ones the
    # argument leans on) are drawn at close to their printed size for the main
    # text; the subcritical panel becomes its own appendix figure.
    fig, (ax_energy, ax_magnet) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    ess_mean, ess_std = critical["ess"]
    plot_marginal_panel(
        ax_energy, critical["energy"]["centres"], critical["energy"]["ref"],
        critical["energy"]["seeds"], r"$\log \tilde p(x)$",
        f"log-density marginal, $\\sigma_c={critical['sigma']:g}$\n"
        f"ESS fraction ${ess_mean:.3f} \\pm {ess_std:.3f}$ (4 seeds)")
    ax_energy.legend(fontsize=8, framealpha=0.9)
    plot_marginal_panel(
        ax_magnet, critical["magnet"]["support"], critical["magnet"]["ref"],
        critical["magnet"]["seeds"], r"magnetisation $m$",
        f"magnetisation marginal, $\\sigma_c={critical['sigma']:g}$\n"
        "one DNFS pass covers both $Z_2$ modes")
    ax_magnet.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    critical_out = args.out.with_name(f"{args.out.stem}_critical.png")
    fig.savefig(critical_out, dpi=200)

    fig_sub, ax_sub = plt.subplots(figsize=(4.6, 3.2))
    ess_mean, ess_std = subcritical["ess"]
    plot_marginal_panel(
        ax_sub, subcritical["energy"]["centres"], subcritical["energy"]["ref"],
        subcritical["energy"]["seeds"], r"$\log \tilde p(x)$",
        f"log-density marginal, $\\sigma={subcritical['sigma']:g}$\n"
        f"ESS fraction ${ess_mean:.3f} \\pm {ess_std:.3f}$ (4 seeds)")
    ax_sub.legend(fontsize=8, framealpha=0.9)
    fig_sub.tight_layout()
    subcritical_out = args.out.with_name(f"{args.out.stem}_subcritical.png")
    fig_sub.savefig(subcritical_out, dpi=200)
    print(f"\nsaved figures to {critical_out} and {subcritical_out}")


if __name__ == "__main__":
    main()
