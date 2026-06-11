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

Reuses the d=4 exact-enumeration machinery from `03_c05_d4_exact_fidelity.py`.
Local-only (experiments/constrained_soft_02/analysis/ is git-excluded).
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    exact_log_probs,
)
from discrete_flow_sampler.targets.ising import IsingTarget

N_SITES = 16  # D=4 -> d = 16; 2^16 = 65,536 enumerable states


def composition(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).mean(dim=-1)


def dnfs_composition_pmf(run_dir: Path) -> torch.Tensor:
    """IS-weighted composition marginal (17 support points k/16) for one run."""
    samples = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float()
    log_w = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True)
    w = torch.softmax(log_w, dim=0)
    bucket = (composition(samples) * N_SITES).round().long().clamp(0, N_SITES)
    return torch.zeros(N_SITES + 1).index_add_(0, bucket, w)


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dirs", required=True, nargs="+", type=Path,
                   help="one or more d=4 c=0.5 run dirs (same cfg, different seeds)")
    p.add_argument("--out", type=Path, default=Path("soft_composition_marginal.png"),
                   help="output PNG path")
    args = p.parse_args()

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
    axr.plot([lam], [off_slice], marker="*", color="C3", ms=16,
             label=f"operating point $\\lambda={lam:g}$ ({off_slice:.1%})")
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
