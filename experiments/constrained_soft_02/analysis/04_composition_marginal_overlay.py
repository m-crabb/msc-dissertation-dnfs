"""§3.2 figure: the composition marginal under a HARD vs SOFT constraint.

The chapter's claim is that a soft (VCSGC-style) penalty does not actually
enforce c(x) = c_target; it only *prefers* it, leaving a residual spread. This
one-panel figure makes that visible by overlaying three composition marginals
for the d=4, c_target=0.5 binary alloy:

  (a) HARD constraint  -- the exact target puts ALL mass on the c=0.5 slice:
      a single spike at c_target. This is what we actually want.
  (b) SOFT target (exact) -- the penalised target enumerated over all 2^16
      states. It is a spread *around* c_target, not a spike. Its width is set
      by the penalty strength lambda, not the physics: for the quadratic
      penalty lambda*d*(c - c_target)^2 the Gaussian approximation gives
      sigma = 1/sqrt(2*lambda*d), which is **temperature-independent** -- it
      does not depend on sigma (the Ising coupling), so the inexactness this
      figure shows at the subcritical sigma transfers unchanged to sigma_c.
  (c) DNFS (IS-weighted) -- samples from the trained sampler, importance-
      weighted. These should track the SOFT target (b), confirming DNFS
      faithfully samples the distribution it was given; the gap to (a) is the
      constraint formulation's fault, not the sampler's.

Pass one run dir to prototype, or all of seeds 42-45 for the report figure: with
several the DNFS marginal is drawn as the across-seed mean with a min-max band.

Two modes (--mode):
  single      -- the original d=4 single-lambda figure above. Default; output
                 unchanged.
  lambda-pair -- the experiments.tex sibling figure next to tab:soft-lambda-sweep
                 (spec 2026-06-11/12): D=10 composition marginals at a weak and a
                 strong lambda overlaid as curves. Exact enumeration is impossible
                 at 2^100 states, so the trusted reference per lambda is a long
                 penalty-aware Gibbs chain (the report's D=10 convention; cached
                 under results/02_constrained_soft/), with the analytic envelope
                 ~ exp(-lambda*d*(c - c_target)^2) as a thin guide and the
                 D=4 exact-vs-envelope deviation printed to stdout. DNFS
                 IS-weighted marginals are plotted on top (pass only healthy
                 seeds for the strong lambda; stuck seeds give garbage IS
                 estimates). Also computes the strong-lambda-reweighted ESS of
                 the weak-lambda samples (multiply the IS weights by the target
                 ratio exp(-(lam_s - lam_w)*d*(c - c_target)^2)): the ESS those
                 samples are worth AGAINST THE STRONG TARGET, which is the
                 apples-to-apples number the figure caption quotes.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
)
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget
from discrete_flow_sampler.diagnostics.metrics import composition_fraction_up as composition

N_SITES = 16  # D=4 -> d = 16; 2^16 = 65,536 enumerable states


def dnfs_composition_pmf(run_dir: Path, n_sites: int = N_SITES) -> torch.Tensor:
    """IS-weighted composition marginal (n_sites+1 support points) for one run."""
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    w = torch.softmax(log_w, dim=0)
    bucket = (composition(samples) * n_sites).round().long().clamp(0, n_sites)
    return torch.zeros(n_sites + 1).index_add_(0, bucket, w)


def soft_composition_pmf(cfg: dict, lam: float, states_f: torch.Tensor) -> torch.Tensor:
    """Exact composition marginal of the penalised target at penalty strength lam."""
    target = IsingTarget(
        D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"],
        target_composition=cfg["target_composition"],
        composition_penalty_strength=lam,
    )
    pi = exact_log_probs(target, states_f).exp()
    bucket = (composition(states_f) * N_SITES).round().long()
    return torch.zeros(N_SITES + 1).index_add_(0, bucket, pi)


def envelope_pmf(n_sites: int, lam: float, c_target: float) -> torch.Tensor:
    """Analytic Gaussian envelope ~ exp(-lam*d*(c - c_t)^2), normalised over the
    n_sites+1 lattice compositions. The lambda-only part of the exact marginal
    p(c) ~ Z_can(c)*exp(-lam*d*(c - c_t)^2); it drops the entropic Z_can factor,
    which is why it is a guide, not the reference."""
    cs = torch.arange(n_sites + 1).float() / n_sites
    return torch.softmax(-lam * n_sites * (cs - c_target) ** 2, dim=0)


def gibbs_reference_pmf(
    cfg: dict, lam: float, n_sites: int, n_chains: int, n_sweeps: int, cache: Path
) -> torch.Tensor:
    """Composition marginal from a long penalty-aware Gibbs reference.

    Final states of n_chains independent heat-bath chains (the same trusted-
    reference pattern as the section 3.2 witness and the stage-2 d10 gibbs
    reference). Cached: delete the .pt to re-run. Prints the chain-mean log-prob
    at the first/middle/last record as the plateau (mixing) check when run.
    """
    if cache.exists():
        samples = torch.load(cache, weights_only=True)
        print(f"  gibbs reference (lam={lam:g}): loaded {samples.shape[0]} cached samples from {cache}")
    else:
        target = IsingTarget(
            D=cfg["D"], sigma=cfg["sigma"], bias=cfg["bias"],
            target_composition=cfg["target_composition"],
            composition_penalty_strength=lam,
        )
        torch.manual_seed(0)
        samples, trace = gibbs_sample(
            target, n_chains=n_chains, n_sweeps=n_sweeps, record_energy_every=50
        )
        m = trace.mean(dim=-1)
        print(f"  gibbs reference (lam={lam:g}): ran {n_chains} chains x {n_sweeps} sweeps; "
              f"chain-mean log p at sweeps [0, mid, end]: "
              f"{m[0].item():.1f}, {m[len(m) // 2].item():.1f}, {m[-1].item():.1f} (plateau check)")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, cache)
    bucket = (composition(samples) * n_sites).round().long().clamp(0, n_sites)
    ones = torch.full((samples.shape[0],), 1.0 / samples.shape[0])
    return torch.zeros(n_sites + 1).index_add_(0, bucket, ones)


def reweighted_ess_fraction(
    run_dir: Path, lam_from: float, lam_to: float, c_target: float, n_sites: int
) -> tuple[float, float]:
    """(own-target ESS fraction, ESS fraction after reweighting to lam_to).

    The reweighting multiplies the IS weights by the unnormalised target ratio
    pi_to/pi_from = exp(-(lam_to - lam_from)*d*(c - c_t)^2), so the second
    number is what the run's samples are worth as importance samples for the
    STRONG target -- the like-for-like comparison across lambdas.
    """
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    n = log_w.numel()
    own = ess_from_log_weights(log_w).item() / n
    delta = (lam_to - lam_from) * n_sites * (composition(samples) - c_target) ** 2
    rew = ess_from_log_weights(log_w - delta).item() / n
    return own, rew


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("single", "lambda-pair"), default="single",
                   help="single: original d=4 figure (default); lambda-pair: D=10 weak-vs-strong lambda figure")
    p.add_argument("--run_dirs", nargs="+", type=Path,
                   help="[single] one or more d=4 c=0.5 run dirs (same cfg, different seeds)")
    p.add_argument("--weak_run_dirs", nargs="+", type=Path,
                   help="[lambda-pair] D=10 run dirs at the weak lambda (all seeds)")
    p.add_argument("--strong_run_dirs", nargs="+", type=Path,
                   help="[lambda-pair] D=10 run dirs at the strong lambda (healthy seeds only)")
    p.add_argument("--window_sites", type=int, default=10,
                   help="[lambda-pair] half-window around c_target in lattice sites (lam=10 reaches 9 off-slice)")
    p.add_argument("--gibbs_chains", type=int, default=5000,
                   help="[lambda-pair] chains for the Gibbs reference")
    p.add_argument("--gibbs_sweeps", type=int, default=1000,
                   help="[lambda-pair] sweeps per Gibbs reference chain")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default per mode)")
    args = p.parse_args()

    if args.mode == "lambda-pair":
        if not (args.weak_run_dirs and args.strong_run_dirs):
            raise SystemExit("lambda-pair mode needs --weak_run_dirs and --strong_run_dirs")
        run_lambda_pair(args)
        return
    if not args.run_dirs:
        raise SystemExit("single mode needs --run_dirs")
    run_single(args)


def run_single(args: argparse.Namespace) -> None:
    args.out = args.out or Path("soft_composition_marginal.png")
    cfg = json.loads((args.run_dirs[0] / "config.json").read_text())["ising"]
    if cfg["D"] != 4:
        raise SystemExit(
            f"this figure exact-enumerates 2^16 states; needs D=4, got D={cfg['D']}"
        )
    c_target = cfg["target_composition"]
    lam = cfg["composition_penalty_strength"]
    sigma_analytic = 1.0 / math.sqrt(2.0 * lam * N_SITES)

    target_idx = round(c_target * N_SITES)  # support index of the c=c_target slice
    states_f = enumerate_states(N_SITES).float()

    # (b) SOFT target at the operating lambda, exact over all 2^16 states.
    soft_pmf = soft_composition_pmf(cfg, lam, states_f)
    off_slice = 1.0 - soft_pmf[target_idx].item()  # mass that violates c = c_target

    # (c) DNFS IS-weighted, stacked across seeds.
    dnfs_pmfs = torch.stack([dnfs_composition_pmf(d) for d in args.run_dirs])
    dnfs_mean = dnfs_pmfs.mean(dim=0)
    dnfs_lo, dnfs_hi = dnfs_pmfs.min(dim=0).values, dnfs_pmfs.max(dim=0).values

    # Panel (b) data: violating mass vs penalty strength (temperature-independent).
    lam_grid = torch.logspace(0, 2.7, 24)  # ~1 .. ~500
    off_grid = torch.tensor([
        1.0 - soft_composition_pmf(cfg, float(lv), states_f)[target_idx].item()
        for lv in lam_grid
    ])

    print(f"=== composition marginal overlay (d=4, c_target={c_target}, lam={lam}) ===")
    print(f"  seeds aggregated          : {len(args.run_dirs)}")
    print(f"  on-slice (c=c_target) mass: {1 - off_slice:.4f}")
    print(f"  off-slice (violating) mass: {off_slice:.4f}  <-- the inexactness")
    print(f"  analytic 1/sqrt(2*lam*d)  : sigma = {sigma_analytic:.4f} (temp-indep)")

    # --- Figure: (a) grouped bars near c_target (linear-y), (b) violating mass vs lambda ---
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) grouped bars over the discrete compositions around c_target: hard target
    # as a full-height bar, soft-exact and DNFS side by side with seed whiskers.
    ks = torch.arange(target_idx - 2, target_idx + 3)
    xs = ks.float() / N_SITES
    width = 1.0 / N_SITES / 4.2
    hard_pmf = torch.zeros(N_SITES + 1)
    hard_pmf[target_idx] = 1.0
    ax.bar(xs - width, hard_pmf[ks], width, color="C3",
           label="hard constraint")
    ax.bar(xs, soft_pmf[ks], width, color="C0",
           label=f"soft target (exact, $\\lambda={lam:g}$)")
    yerr = torch.stack([dnfs_mean[ks] - dnfs_lo[ks], dnfs_hi[ks] - dnfs_mean[ks]])
    ax.bar(xs + width, dnfs_mean[ks], width, color="C1", yerr=yerr.numpy(),
           error_kw={"lw": 1.0, "capsize": 2.5},
           label="DNFS (seed mean, min-max)")
    ax.text(c_target, 0.48, f"violating compositions:\n{off_slice:.1%} of soft mass in total",
            ha="center", fontsize=9, color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    for k in (target_idx - 1, target_idx + 1):
        ax.annotate("", xy=(k / N_SITES, soft_pmf[k].item() + 0.03),
                    xytext=(c_target, 0.46),
                    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.4"})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}/{N_SITES}" for k in ks.tolist()])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"composition $c_+$")
    ax.set_ylabel("probability mass")
    ax.set_title(f"(a) composition marginal at $\\lambda={lam:g}$")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    # (b) violating mass vs lambda: never reaches 0 at finite, samplable lambda.
    axr.plot(lam_grid, off_grid, "-o", color="C0", ms=4)
    axr.plot([lam], [off_slice], marker="*", color="C3", ms=16, ls="none",
             label=f"operating point $\\lambda={lam:g}$ ({off_slice:.1%})")
    axr.set_xscale("log")
    axr.set_xlabel(r"penalty strength $\lambda$")
    axr.set_ylabel(r"mass violating $c_\mathrm{target}$")
    axr.set_title("(b) the cost: violating mass falls only as $\\lambda$ grows")
    axr.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved figure to {args.out}")


def run_lambda_pair(args: argparse.Namespace) -> None:
    args.out = args.out or Path("soft_lambda_marginal.png")
    cfg_w = json.loads((args.weak_run_dirs[0] / "config.json").read_text())["ising"]
    cfg_s = json.loads((args.strong_run_dirs[0] / "config.json").read_text())["ising"]
    for key in ("D", "sigma", "bias", "target_composition"):
        if cfg_w[key] != cfg_s[key]:
            raise SystemExit(f"weak/strong run dirs disagree on ising.{key}")
    lam_w = cfg_w["composition_penalty_strength"]
    lam_s = cfg_s["composition_penalty_strength"]
    c_target = cfg_w["target_composition"]
    n_sites = cfg_w["D"] ** 2
    target_idx = round(c_target * n_sites)
    cache_dir = Path("results/02_constrained_soft")

    print(f"=== lambda-pair marginal (D={cfg_w['D']}, c_target={c_target}, "
          f"lam {lam_w:g} vs {lam_s:g}) ===")

    # Reference curves: long penalty-aware Gibbs chains (the D=10 convention),
    # with the analytic envelope as the lambda-only guide.
    gibbs_w = gibbs_reference_pmf(cfg_w, lam_w, n_sites, args.gibbs_chains,
                                  args.gibbs_sweeps,
                                  cache_dir / f"gibbs_ref_d{cfg_w['D']}_c{c_target:g}_l{lam_w:g}.pt")
    gibbs_s = gibbs_reference_pmf(cfg_s, lam_s, n_sites, args.gibbs_chains,
                                  args.gibbs_sweeps,
                                  cache_dir / f"gibbs_ref_d{cfg_s['D']}_c{c_target:g}_l{lam_s:g}.pt")
    env_w = envelope_pmf(n_sites, lam_w, c_target)
    env_s = envelope_pmf(n_sites, lam_s, c_target)

    # DNFS IS-weighted marginals: weak as seed mean + min-max, strong as the
    # healthy seed(s) passed in.
    dnfs_w = torch.stack([dnfs_composition_pmf(d, n_sites) for d in args.weak_run_dirs])
    dnfs_s = torch.stack([dnfs_composition_pmf(d, n_sites) for d in args.strong_run_dirs])
    w_mean = dnfs_w.mean(dim=0)
    w_err = torch.stack([w_mean - dnfs_w.min(dim=0).values,
                         dnfs_w.max(dim=0).values - w_mean])
    s_mean = dnfs_s.mean(dim=0)

    # Envelope sanity at d=4, where the exact marginal is enumerable. The
    # envelope drops the entropic Z_can(c) factor, so this records how much
    # that costs; the FIGURE's reference curves are the Gibbs chains.
    states_f = enumerate_states(N_SITES).float()
    cfg4 = dict(cfg_w, D=4)
    for lam in (lam_w, lam_s):
        exact4 = soft_composition_pmf(cfg4, lam, states_f)
        env4 = envelope_pmf(N_SITES, lam, c_target)
        print(f"  envelope check at d=4, lam={lam:g}: max |exact - envelope| = "
              f"{(exact4 - env4).abs().max().item():.4f} "
              f"(on-slice exact {exact4[round(c_target * N_SITES)].item():.3f} "
              f"vs envelope {env4[round(c_target * N_SITES)].item():.3f})")

    # The caption number: what the weak-lambda samples are worth against the
    # strong target (reweight by the target ratio, then ESS).
    own_rew = [reweighted_ess_fraction(d, lam_w, lam_s, c_target, n_sites)
               for d in args.weak_run_dirs]
    rews = torch.tensor([r for _, r in own_rew])
    print(f"  lam={lam_w:g} runs reweighted to lam={lam_s:g}:")
    for d, (own, rew) in zip(args.weak_run_dirs, own_rew):
        print(f"    {d.name}: own-target ESS {own:.4f} -> reweighted {rew:.4f}")
    print(f"    reweighted survival: {rews.mean().item():.3f} +/- {rews.std().item():.3f}")

    # Panel (b) data: violating mass vs lambda, exact at d=4 (unchanged from
    # the single-mode figure), with both operating points starred.
    lam_grid = torch.logspace(0, 2.7, 24)
    off_grid = torch.tensor([
        1.0 - soft_composition_pmf(cfg4, float(lv), states_f)[round(c_target * N_SITES)].item()
        for lv in lam_grid
    ])
    off_w4 = 1.0 - soft_composition_pmf(cfg4, lam_w, states_f)[round(c_target * N_SITES)].item()
    off_s4 = 1.0 - soft_composition_pmf(cfg4, lam_s, states_f)[round(c_target * N_SITES)].item()
    print(f"  violating mass (exact, d=4): {off_w4:.1%} at lam={lam_w:g}, {off_s4:.1%} at lam={lam_s:g}")

    # --- Figure: (a) D=10 marginals, weak vs strong; (b) violating mass vs lambda ---
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11, 4.3))

    ks = torch.arange(target_idx - args.window_sites, target_idx + args.window_sites + 1)
    xs = ks.float() / n_sites
    ax.axvline(c_target, color="C3", lw=1.6, label="hard constraint")
    ax.plot(xs, gibbs_w[ks], "-", color="C0", lw=1.6,
            label=f"soft target, $\\lambda={lam_w:g}$ (Gibbs reference)")
    ax.plot(xs, env_w[ks], "--", color="C0", lw=0.9, alpha=0.7,
            label=r"analytic envelope $\propto e^{-\lambda d (c - c_\mathrm{target})^2}$")
    ax.plot(xs, gibbs_s[ks], "-", color="C2", lw=1.6,
            label=f"soft target, $\\lambda={lam_s:g}$ (Gibbs reference)")
    ax.plot(xs, env_s[ks], "--", color="C2", lw=0.9, alpha=0.7)
    ax.errorbar(xs, w_mean[ks], yerr=w_err[:, ks].numpy(), fmt="o", ms=4.5,
                color="C0", mfc="white", elinewidth=1.0, capsize=2.0,
                label=f"DNFS, $\\lambda={lam_w:g}$ (seed mean, min-max)", zorder=5)
    strong_label = (f"DNFS, $\\lambda={lam_s:g}$ (healthy seed)"
                    if len(args.strong_run_dirs) == 1
                    else f"DNFS, $\\lambda={lam_s:g}$ (healthy seeds)")
    ax.plot(xs, s_mean[ks], "s", ms=4.5, color="C2", mfc="white",
            label=strong_label, zorder=5)
    ax.set_xlabel(r"composition $c_+$")
    ax.set_ylabel("probability mass")
    ax.set_title(f"(a) composition marginal, ${cfg_w['D']}\\times{cfg_w['D']}$")
    ax.legend(fontsize=7.5, framealpha=0.9, loc="upper right")

    axr.plot(lam_grid, off_grid, "-o", color="C0", ms=4)
    axr.plot([lam_w], [off_w4], marker="*", color="C0", ms=16, ls="none",
             label=f"$\\lambda={lam_w:g}$ ({off_w4:.1%})")
    axr.plot([lam_s], [off_s4], marker="*", color="C3", ms=16, ls="none",
             label=f"operating point $\\lambda={lam_s:g}$ ({off_s4:.1%})")
    axr.set_xscale("log")
    axr.set_xlabel(r"penalty strength $\lambda$")
    axr.set_ylabel(r"mass violating $c_\mathrm{target}$")
    axr.set_title("(b) the cost: violating mass falls only as $\\lambda$ grows")
    axr.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved figure to {args.out}")


if __name__ == "__main__":
    main()
