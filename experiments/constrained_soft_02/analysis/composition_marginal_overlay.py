"""§3.2 figure: the composition marginal under a HARD vs SOFT constraint.

The chapter's claim is that a soft (VCSGC-style) penalty does not actually
enforce c(x) = c_target; it only *prefers* it, leaving a residual spread. This
one-panel figure makes that visible by overlaying three composition marginals
for the d=4, c_target=0.5 binary alloy:

  (a) HARD constraint  -- the exact target puts ALL mass on the c=0.5 slice:
      a single spike at c_target. This is the object of interest.
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
  single      -- the d=4 figure above, in two panels: (a) the grouped-bar
                 marginal at the operating lambda, (b) the violating mass swept
                 over lambda with BOTH ends of the trade starred (54.4% at
                 lam=10, 7.9% at lam=50). Panel (b) is the thesis's only copy of
                 that sweep. lambda-pair once carried a byte-identical duplicate
                 of it, since dropped: this copy is cited twice in the body and
                 that one never was, and removing it also left each figure on a
                 single lattice (this one d=4, lambda-pair d=10) instead of
                 mixing the two inside one float.
  lambda-pair -- the companion overlay figure for the lambda-sweep
                 comparison: D=10 composition marginals at a weak and a
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

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_FULL_WIDE_SINGLE,
    FONT_SIZE_ANNOTATION,
    FULL_WIDTH_IN,
    HARD_DELTA_HUE,
    MUTED,
    REFERENCE_FILL,
    REFERENCE_INK,
    SAMPLER_HUE,
    parameter_ramp,
    style_axes,
    uncertainty_band,
    use_house_style,
)
from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up as composition,
)
from discrete_flow_sampler.diagnostics.metrics import (
    enumerate_states,
    ess_from_log_weights,
    exact_log_probs,
)
from discrete_flow_sampler.mcmc.gibbs import gibbs_sample
from discrete_flow_sampler.targets.ising import IsingTarget

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
        D=cfg["D"],
        sigma=cfg["sigma"],
        bias=cfg["bias"],
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
        print(
            f"  gibbs reference (lam={lam:g}): loaded {samples.shape[0]} cached samples from {cache}"
        )
    else:
        target = IsingTarget(
            D=cfg["D"],
            sigma=cfg["sigma"],
            bias=cfg["bias"],
            target_composition=cfg["target_composition"],
            composition_penalty_strength=lam,
        )
        torch.manual_seed(0)
        samples, trace = gibbs_sample(
            target, n_chains=n_chains, n_sweeps=n_sweeps, record_energy_every=50
        )
        m = trace.mean(dim=-1)
        print(
            f"  gibbs reference (lam={lam:g}): ran {n_chains} chains x {n_sweeps} sweeps; "
            f"chain-mean log p at sweeps [0, mid, end]: "
            f"{m[0].item():.1f}, {m[len(m) // 2].item():.1f}, {m[-1].item():.1f} (plateau check)"
        )
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("single", "lambda-pair"),
        default="single",
        help="single: original d=4 figure (default); lambda-pair: D=10 weak-vs-strong lambda figure",
    )
    p.add_argument(
        "--run_dirs",
        nargs="+",
        type=Path,
        help="[single] one or more d=4 c=0.5 run dirs (same cfg, different seeds)",
    )
    p.add_argument(
        "--weak_lambda",
        type=float,
        default=10.0,
        help="[single] the weak end of the lambda trade, starred on panel (b) "
        "alongside the run's own operating point so one panel carries the "
        "whole trade (default 10, the chapter's weak cell)",
    )
    p.add_argument(
        "--weak_run_dirs",
        nargs="+",
        type=Path,
        help="[lambda-pair] D=10 run dirs at the weak lambda (all seeds)",
    )
    p.add_argument(
        "--strong_run_dirs",
        nargs="+",
        type=Path,
        help="[lambda-pair] D=10 run dirs at the strong lambda (healthy seeds only)",
    )
    p.add_argument(
        "--window_sites",
        type=int,
        default=10,
        help="[lambda-pair] half-window around c_target in lattice sites (lam=10 reaches 9 off-slice)",
    )
    p.add_argument(
        "--gibbs_chains",
        type=int,
        default=5000,
        help="[lambda-pair] chains for the Gibbs reference",
    )
    p.add_argument(
        "--gibbs_sweeps",
        type=int,
        default=1000,
        help="[lambda-pair] sweeps per Gibbs reference chain",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="output PNG path (default per mode)"
    )
    args = p.parse_args()

    if args.mode == "lambda-pair":
        if not (args.weak_run_dirs and args.strong_run_dirs):
            raise SystemExit(
                "lambda-pair mode needs --weak_run_dirs and --strong_run_dirs"
            )
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
    # 64 points rather than a coarse grid because the two operating points are
    # marked at their exact values and lambda=10 is not itself a grid point: on
    # a coarse grid the curve chords under the convex true curve and the marker
    # floats visibly above its own line.
    lam_grid = torch.logspace(0, 2.7, 64)  # ~1 .. ~500
    off_grid = torch.tensor(
        [
            1.0 - soft_composition_pmf(cfg, float(lv), states_f)[target_idx].item()
            for lv in lam_grid
        ]
    )
    # The weak end of the trade, starred beside the operating point: at lam=10
    # more than half the mass violates (54.4%) against 7.9% at lam=50, which is
    # the contrast the lambda-sweep discussion makes in words and the lambda ->
    # infinity argument needs on an axis.
    off_slice_weak = (
        1.0 - soft_composition_pmf(cfg, args.weak_lambda, states_f)[target_idx].item()
    )

    print(f"=== composition marginal overlay (d=4, c_target={c_target}, lam={lam}) ===")
    print(f"  seeds aggregated          : {len(args.run_dirs)}")
    print(f"  on-slice (c=c_target) mass: {1 - off_slice:.4f}")
    print(f"  off-slice (violating) mass: {off_slice:.4f}  <-- the inexactness")
    print(f"  analytic 1/sqrt(2*lam*d)  : sigma = {sigma_analytic:.4f} (temp-indep)")
    print(
        f"  violating mass at lam={args.weak_lambda:g}    : {off_slice_weak:.4f} (weak end of the trade)"
    )

    # --- Figure: (a) grouped bars near c_target (linear-y), (b) violating mass vs lambda ---
    # Hue carries the ROLE per the house palette: ink is the exact-enumeration
    # reference, blue is our sampler, red is the hard-constraint limit. Lambda
    # never gets a hue of its own -- on panel (b) the two operating points share
    # the limit hue and separate by marker shape.
    use_house_style()
    # Print size (FULL_WIDTH_IN x 2.4): the old 9.0 x 3.8 canvas printed at
    # \textwidth shrank 9 pt type to ~6 pt and took half a page.
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.4))

    # (a) grouped bars over the discrete compositions around c_target: hard target
    # as a full-height bar, soft-exact and DNFS side by side with seed whiskers.
    ks = torch.arange(target_idx - 2, target_idx + 3)
    xs = ks.float() / N_SITES
    width = 1.0 / N_SITES / 4.2
    hard_pmf = torch.zeros(N_SITES + 1)
    hard_pmf[target_idx] = 1.0
    ax.bar(
        xs - width,
        hard_pmf[ks],
        width,
        color=HARD_DELTA_HUE,
        zorder=3,
        label="hard constraint",
    )
    ax.bar(
        xs,
        soft_pmf[ks],
        width,
        color=REFERENCE_FILL,
        zorder=3,
        label=f"soft target (exact, $\\lambda={lam:g}$)",
    )
    yerr = torch.stack([dnfs_mean[ks] - dnfs_lo[ks], dnfs_hi[ks] - dnfs_mean[ks]])
    ax.bar(
        xs + width,
        dnfs_mean[ks],
        width,
        color=SAMPLER_HUE,
        yerr=yerr.numpy(),
        error_kw={"lw": 1.0, "capsize": 2.5, "ecolor": REFERENCE_INK},
        zorder=3,
        label="DNFS (seed mean, min-max)",
    )
    # Note in the empty upper-right, legend in the empty upper-left: the
    # central bars reach 1.0, so nothing may sit over the centre column.
    ax.text(
        0.98,
        0.70,
        f"violating compositions:\n{off_slice:.1%} of soft mass",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_ANNOTATION,
        color=REFERENCE_INK,
        zorder=4,
    )
    for k in (target_idx - 1, target_idx + 1):
        ax.annotate(
            "",
            xy=(k / N_SITES, soft_pmf[k].item() + 0.03),
            xytext=(0.80, 0.55),
            textcoords="axes fraction",
            zorder=4,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": MUTED},
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}/{N_SITES}" for k in ks.tolist()])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"composition $c_+$")
    ax.set_ylabel("probability mass")
    style_axes(ax)
    # Legend below the panels (house pattern): the central bars reach 1.0, so
    # no in-axes corner is free at print width.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        fontsize=FONT_SIZE_ANNOTATION,
    )

    # (b) violating mass vs lambda: never reaches 0 at finite, samplable lambda.
    # Both ends of the trade are marked, so the panel shows what raising lambda
    # buys as well as that it never buys exactness.
    axr.plot(lam_grid, off_grid, "-", color=REFERENCE_INK, lw=1.6, zorder=3)
    axr.plot(
        [args.weak_lambda],
        [off_slice_weak],
        marker="D",
        color=HARD_DELTA_HUE,
        mfc="white",
        ms=7,
        ls="none",
        zorder=4,
        label=f"weak end $\\lambda={args.weak_lambda:g}$ ({off_slice_weak:.1%})",
    )
    axr.plot(
        [lam],
        [off_slice],
        marker="*",
        color=HARD_DELTA_HUE,
        ms=15,
        ls="none",
        zorder=4,
        label=f"operating point $\\lambda={lam:g}$ ({off_slice:.1%})",
    )
    axr.set_xscale("log")
    axr.set_xlabel(r"penalty strength $\lambda$")
    axr.set_ylabel(r"mass violating $c_\mathrm{target}$")
    style_axes(axr)
    axr.legend(framealpha=0.9)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(args.out)
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

    print(
        f"=== lambda-pair marginal (D={cfg_w['D']}, c_target={c_target}, "
        f"lam {lam_w:g} vs {lam_s:g}) ==="
    )

    # Reference curves: long penalty-aware Gibbs chains (the D=10 convention),
    # with the analytic envelope as the lambda-only guide.
    gibbs_w = gibbs_reference_pmf(
        cfg_w,
        lam_w,
        n_sites,
        args.gibbs_chains,
        args.gibbs_sweeps,
        cache_dir / f"gibbs_ref_d{cfg_w['D']}_c{c_target:g}_l{lam_w:g}.pt",
    )
    gibbs_s = gibbs_reference_pmf(
        cfg_s,
        lam_s,
        n_sites,
        args.gibbs_chains,
        args.gibbs_sweeps,
        cache_dir / f"gibbs_ref_d{cfg_s['D']}_c{c_target:g}_l{lam_s:g}.pt",
    )
    env_w = envelope_pmf(n_sites, lam_w, c_target)
    env_s = envelope_pmf(n_sites, lam_s, c_target)

    # DNFS IS-weighted marginals: weak as seed mean + min-max, strong as the
    # healthy seed(s) passed in.
    dnfs_w = torch.stack([dnfs_composition_pmf(d, n_sites) for d in args.weak_run_dirs])
    dnfs_s = torch.stack(
        [dnfs_composition_pmf(d, n_sites) for d in args.strong_run_dirs]
    )
    w_mean = dnfs_w.mean(dim=0)
    w_lo, w_hi = dnfs_w.min(dim=0).values, dnfs_w.max(dim=0).values
    s_mean = dnfs_s.mean(dim=0)

    # Envelope sanity at d=4, where the exact marginal is enumerable. The
    # envelope drops the entropic Z_can(c) factor, so this records how much
    # that costs; the FIGURE's reference curves are the Gibbs chains.
    states_f = enumerate_states(N_SITES).float()
    cfg4 = dict(cfg_w, D=4)
    for lam in (lam_w, lam_s):
        exact4 = soft_composition_pmf(cfg4, lam, states_f)
        env4 = envelope_pmf(N_SITES, lam, c_target)
        print(
            f"  envelope check at d=4, lam={lam:g}: max |exact - envelope| = "
            f"{(exact4 - env4).abs().max().item():.4f} "
            f"(on-slice exact {exact4[round(c_target * N_SITES)].item():.3f} "
            f"vs envelope {env4[round(c_target * N_SITES)].item():.3f})"
        )

    # The caption number: what the weak-lambda samples are worth against the
    # strong target (reweight by the target ratio, then ESS).
    own_rew = [
        reweighted_ess_fraction(d, lam_w, lam_s, c_target, n_sites)
        for d in args.weak_run_dirs
    ]
    rews = torch.tensor([r for _, r in own_rew])
    print(f"  lam={lam_w:g} runs reweighted to lam={lam_s:g}:")
    for d, (own, rew) in zip(args.weak_run_dirs, own_rew):
        print(f"    {d.name}: own-target ESS {own:.4f} -> reweighted {rew:.4f}")
    print(
        f"    reweighted survival: {rews.mean().item():.3f} +/- {rews.std().item():.3f}"
    )

    # Cross-check against the single-mode figure, which is where the sweep over
    # lambda is drawn: 54.4% of the mass violates at lam=10 against 7.9% at
    # lam=50. This figure used to carry a second copy of that sweep as its own
    # panel (b); it was a d=4 object inside a d=10 float, no body text cited it,
    # and it was computed from the same grid, so it was dropped and only the
    # print survives as the agreement check between the two figures.
    off_w4 = (
        1.0
        - soft_composition_pmf(cfg4, lam_w, states_f)[round(c_target * N_SITES)].item()
    )
    off_s4 = (
        1.0
        - soft_composition_pmf(cfg4, lam_s, states_f)[round(c_target * N_SITES)].item()
    )
    print(
        f"  violating mass (exact, d=4): {off_w4:.1%} at lam={lam_w:g}, {off_s4:.1%} at lam={lam_s:g}"
    )

    # --- Figure: the D=10 composition marginals, weak vs strong lambda ---
    # One panel, one lattice. Hue carries the ROLE per the house palette (ink =
    # the Gibbs reference this figure is scored against, blue = our sampler, red
    # = the hard-constraint limit). Lambda is a parameter level, not a role, so
    # the two references separate by lightness within the ink family rather than
    # by linestyle: two solid black curves at 1.6pt tangle at the peak, which is
    # what dash-dot was hiding badly. Each envelope then takes its own lambda's
    # value, so colour says WHICH lambda and the dotted pattern says analytic
    # rather than measured.
    use_house_style()
    reference_weak_hue, reference_strong_hue = parameter_ramp(REFERENCE_INK, 2)
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL_WIDE_SINGLE)

    ks = torch.arange(
        target_idx - args.window_sites, target_idx + args.window_sites + 1
    )
    xs = ks.float() / n_sites
    hard_marker = ax.axvline(
        c_target, color=HARD_DELTA_HUE, lw=1.6, zorder=4, label="hard constraint"
    )
    (reference_weak,) = ax.plot(
        xs,
        gibbs_w[ks],
        "-",
        color=reference_weak_hue,
        lw=1.6,
        zorder=3,
        label=f"soft target, $\\lambda={lam_w:g}$ (Gibbs)",
    )
    (reference_strong,) = ax.plot(
        xs,
        gibbs_s[ks],
        "-",
        color=reference_strong_hue,
        lw=1.6,
        zorder=3,
        label=f"soft target, $\\lambda={lam_s:g}$ (Gibbs)",
    )
    # The envelope's formula lives in the caption, not the legend entry: spelled
    # out here it set the legend's width, and the legend sits under a 6.3 in
    # panel where width is the binding constraint.
    (envelope,) = ax.plot(
        xs,
        env_w[ks],
        ":",
        color=reference_weak_hue,
        lw=1.1,
        zorder=2,
        label="analytic envelope",
    )
    ax.plot(xs, env_s[ks], ":", color=reference_strong_hue, lw=1.1, zorder=2)
    # Seed spread as a shaded min-max band rather than capped bars: the
    # sampler's marginal is a curve over the composition support, so the
    # spread is an envelope along x, not four independent point estimates.
    uncertainty_band(ax, xs, w_lo[ks], w_hi[ks], SAMPLER_HUE, zorder=4)
    (dnfs_weak,) = ax.plot(
        xs,
        w_mean[ks],
        marker="o",
        ms=4.5,
        color=SAMPLER_HUE,
        mfc="white",
        ls="none",
        zorder=5,
        label=f"DNFS, $\\lambda={lam_w:g}$ (mean, min-max band)",
    )
    strong_label = (
        f"DNFS, $\\lambda={lam_s:g}$ (healthy seed)"
        if len(args.strong_run_dirs) == 1
        else f"DNFS, $\\lambda={lam_s:g}$ (healthy seeds)"
    )
    (dnfs_strong,) = ax.plot(
        xs,
        s_mean[ks],
        marker="s",
        ms=4.5,
        color=SAMPLER_HUE,
        mfc="white",
        ls="none",
        label=strong_label,
        zorder=5,
    )
    ax.set_xlabel(r"composition $c_+$")
    ax.set_ylabel("probability mass")
    # No panel title: the caption already names the lattice, and on a single
    # panel the title was 0.25 in of the figure's height for no information.
    style_axes(ax)
    # Explicit handle order: matplotlib sorts error-bar containers after plain
    # lines, which would otherwise list the sampler at lam=50 above the one at
    # lam=10 while the reference curves above them run the other way.
    # Legend BELOW the figure in three columns. Inside the panel it covered the
    # peak (the one feature the figure exists to show) once the figure came
    # down to house width; column-major fill keeps the two references adjacent
    # and the two DNFS series adjacent. Anchored to the FIGURE, not the axes,
    # so the three columns get the full 6.3 in rather than the axes' ~5.5 in.
    fig.legend(
        handles=[
            hard_marker,
            reference_weak,
            reference_strong,
            envelope,
            dnfs_weak,
            dnfs_strong,
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(args.out)
    print(f"\nsaved figure to {args.out}")


if __name__ == "__main__":
    main()
