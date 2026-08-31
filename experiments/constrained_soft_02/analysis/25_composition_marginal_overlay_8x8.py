"""8x8 composition-marginal overlay: DNFS vs VC-SGC chains vs the envelope.

The 4x4 overlay (04, `single` mode) makes the soft-constraint spread visible
against an exact enumeration; at d=64 (2^64 states) no enumeration exists, so
the trusted reference becomes the mchammer VC-SGC chains themselves — the
same chains the house table's error floors are built from. One panel per
trained composition, three marginals overlaid:

  reference -- pooled post-burn-in composition frames of the 4 VC-SGC chains
               at kappa=lambda, phi=-2c* (results/mchammer_vcsgc, 3-decimal
               naming; composition.npy is already post-burn-in).
  DNFS      -- IS-weighted composition marginal per seed, drawn as the
               across-seed mean with a min-max band. Seeds below the ESS
               floor are EXCLUDED and counted in the annotation instead:
               a stuck seed's IS marginal is garbage, and at sigma_c
               c=0.25 the whole family is below floor — that panel then
               shows the reference against the envelope alone, which is
               the honest picture (the sampler has nothing to overlay).
  envelope  -- the analytic Gaussian ~ exp(-lambda*d*(c-c*)^2), width
               1/sqrt(2*lambda*d) = 0.0125 at lambda=50, d=64. A guide,
               not a reference: it drops the entropic Z_can(c) factor,
               which is exactly what tilts the true marginal off-centre
               at c* != 0.5.

Coupling selects the wave: --coupling s010 (wave 1) or sc (wave 2, the
print candidate — criticality is where the reference is honest work).

Example:
    python -m experiments.constrained_soft_02.analysis.25_composition_marginal_overlay_8x8 \
        --coupling sc --eval_dir eval_ema
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from discrete_flow_sampler.diagnostics.figure_style import (
    FONT_SIZE_ANNOTATION, MUTED, REFERENCE_FILL, REFERENCE_INK, SAMPLER_HUE,
    style_axes, use_house_style)
from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up as composition)
from discrete_flow_sampler.diagnostics.metrics import ess_from_log_weights
from discrete_flow_sampler.targets.ising import SIGMA_C
from experiments.constrained_soft_02.analysis._common import latest_run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"
VCSGC_RESULTS = REPO_ROOT / "results" / "mchammer_vcsgc"

D_SIDE, N_SITES, LAM = 8, 64, 50.0
TRAINED_COMPOSITIONS = (0.25, 0.375, 0.50)
SEEDS = (42, 43, 44, 45)
COUPLINGS = {"s010": 0.1, "sc": SIGMA_C}


def dnfs_pmf(run_dir: Path, eval_dir: str) -> torch.Tensor:
    """IS-weighted composition marginal over the n_sites+1 support points."""
    samples = torch.load(run_dir / eval_dir / "samples.pt",
                         weights_only=True).float()
    log_w = torch.load(run_dir / eval_dir / "log_weights.pt",
                       weights_only=True)
    w = torch.softmax(log_w, dim=0)
    bucket = (composition(samples) * N_SITES).round().long().clamp(0, N_SITES)
    return torch.zeros(N_SITES + 1).index_add_(0, bucket, w)


def ess_fraction(run_dir: Path, eval_dir: str) -> float:
    log_w = torch.load(run_dir / eval_dir / "log_weights.pt",
                       weights_only=True)
    return ess_from_log_weights(log_w).item() / log_w.numel()


def reference_pmf(sigma: float, c_target: float) -> tuple[torch.Tensor, int]:
    """Pooled post-burn-in composition histogram of the 4 VC-SGC chains."""
    # Chain dirs carry the CLI-typed sigma (0.220343), not the full float.
    pattern = f"D{D_SIDE}_s{sigma:.6g}_l{LAM:.1f}_c{c_target:.3f}_seed*"
    run_dirs = sorted(VCSGC_RESULTS.glob(pattern))
    if not run_dirs:
        raise FileNotFoundError(f"no VC-SGC chains match {pattern}")
    frames = np.concatenate(
        [np.load(d / "composition.npy") for d in run_dirs])
    bucket = np.clip(np.rint(frames * N_SITES).astype(int), 0, N_SITES)
    counts = np.bincount(bucket, minlength=N_SITES + 1).astype(float)
    return torch.from_numpy(counts / counts.sum()), len(frames)


def envelope_pmf(c_target: float) -> torch.Tensor:
    cs = torch.arange(N_SITES + 1).float() / N_SITES
    return torch.softmax(-LAM * N_SITES * (cs - c_target) ** 2, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coupling", choices=list(COUPLINGS), default="sc")
    p.add_argument("--eval_dir", choices=["eval", "eval_ema"], default="eval")
    p.add_argument("--ess_floor", type=float, default=0.30)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--matched-base", action="store_true",
        help="read the *_house_mb families at the off-centre windows (the "
             "centre stays the house cell: Bernoulli(0.5) is already "
             "matched there)")
    args = p.parse_args()

    sigma = COUPLINGS[args.coupling]
    config_suffix = "_sc" if args.coupling == "sc" else ""
    use_house_style()
    fig, axes = plt.subplots(
        1, len(TRAINED_COMPOSITIONS), figsize=(10.5, 3.2), sharey=True)

    cs = torch.arange(N_SITES + 1).float() / N_SITES
    for ax, c_target in zip(axes, TRAINED_COMPOSITIONS):
        family = ("house_mb" if args.matched_base and c_target != 0.5
                  else "house")
        config = (f"S2_d8_c{int(round(c_target * 1000)):04d}"
                  f"_l50_letf_ne128_{family}{config_suffix}")

        ref, n_frames = reference_pmf(sigma, c_target)
        ax.bar(cs, ref, width=1 / N_SITES, color=REFERENCE_FILL,
               edgecolor=REFERENCE_INK, linewidth=0.4,
               label=f"VC-SGC chains ({n_frames} frames)")
        ax.plot(cs, envelope_pmf(c_target), color=MUTED, linewidth=1.0,
                linestyle="--", label="analytic envelope")

        pmfs, excluded = [], []
        for seed in SEEDS:
            run_dir = latest_run_dir(RESULTS, config, seed, args.eval_dir)
            if run_dir is None:
                continue
            ess = ess_fraction(run_dir, args.eval_dir)
            if ess < args.ess_floor:
                excluded.append(f"{seed}:{ess:.2f}")
                continue
            pmfs.append(dnfs_pmf(run_dir, args.eval_dir))
        if pmfs:
            stack = torch.stack(pmfs)
            mean = stack.mean(0)
            ax.plot(cs, mean, color=SAMPLER_HUE, linewidth=1.4,
                    label=f"DNFS IS-weighted ({len(pmfs)}/4 seeds)")
            ax.fill_between(cs, stack.min(0).values, stack.max(0).values,
                            color=SAMPLER_HUE, alpha=0.25, linewidth=0)
        if excluded:
            ax.annotate(f"below ESS floor: {', '.join(excluded)}",
                        xy=(0.03, 0.95), xycoords="axes fraction",
                        va="top", fontsize=FONT_SIZE_ANNOTATION, color=MUTED)

        width = 1.0 / np.sqrt(2 * LAM * N_SITES)
        ax.set_xlim(c_target - 5 * width, c_target + 5 * width)
        ax.set_xlabel("composition $c$")
        ax.set_title(f"$c^* = {c_target:g}$")
        style_axes(ax)

    axes[0].set_ylabel("probability mass")
    # Legend on the last panel: the first panel's corner holds the
    # below-floor annotation whenever the stress window fails.
    axes[-1].legend(fontsize=FONT_SIZE_ANNOTATION, loc="upper right")
    fig.suptitle(
        rf"$8\times 8$, $\sigma = {sigma:g}$, $\lambda = {LAM:g}$", y=1.02)
    fig.tight_layout()

    out = args.out or (
        RESULTS
        / f"composition_marginal_8x8_{args.coupling}_{args.eval_dir}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
