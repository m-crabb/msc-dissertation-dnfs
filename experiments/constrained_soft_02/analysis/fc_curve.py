"""Assemble the soft-constraint free-energy curve F(c) across compositions.

Reads a set of DNFS run dirs spanning a composition sweep and assembles the
per-window free energy into a curve, with the two jobs the F(c) campaign needs:

  1. **The curve itself.** Per c_target we already have a paper-convention
     free-energy estimate in each run's `eval/metrics.json`
     (`free_energy_per_site`, = -mean(log w)/(2 sigma d), Eq. 37). We group by
     c_target, ESS-gate the seeds (a low-ESS seed's F is Jensen-biased high, so
     it is excluded from the average and reported separately rather than
     silently folded in), and aggregate to F(c) with a seed spread.

  2. **The integrator adjudicator (D <= 4).** Where the fixed-composition
     sector enumerates, `run.py` also writes `free_energy_per_site_exact` and
     the signed `free_energy_per_site_bias` (estimate - exact). The bias is the
     Jensen bias of the IS estimator and grows as ESS falls, so it is expected
     to be small at the centre and to balloon in the tails, which bends the
     *shape* of F(c). Tabulating bias vs composition is how we decide whether a
     finer Euler grid (more n_euler) is justified for the curve: small and flat
     bias means the coarser grid is fine, bias that grows in the tails and
     shrinks with n_euler is the case for the finer one.

Z_2 symmetry (no field) gives F(c) = F(1-c); where both c and 1-c are present
the script prints the gap as a free sanity check.

This is a metrics aggregator: it reuses the free energies `run.py` already
computes in the paper convention rather than recomputing them, so the numbers
match the per-run eval exactly. Weighted thermodynamics with bootstrap error
bars and the vcSGC-TI reference at D=10 are a later pass.

Example (the 8x8 house family; --eval_dir eval_ema reads the dual
eval's shadow-weight draw, archived pre-EMA cells keep the default):
    python -m experiments.constrained_soft_02.analysis.fc_curve \
        --results_dir results/02_constrained_soft \
        --configs S2_d8_c0250_l50_letf_ne128_house \
                  S2_d8_c0375_l50_letf_ne128_house \
                  S2_d8_c0500_l50_letf_ne128_house \
        --seeds 42 43 44 45 --ess_floor 0.30 --eval_dir eval_ema
"""

import argparse
import json
from pathlib import Path

import numpy as np
from experiments.constrained_soft_02.analysis._common import latest_run_dir


def _load_record(run_dir: Path, eval_dir: str = "eval") -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    ising = cfg["ising"]
    metrics = json.loads((run_dir / eval_dir / "metrics.json").read_text())
    D = ising["D"]
    return {
        "name": run_dir.name,
        "D": D,
        "d": D * D,
        "sigma": ising["sigma"],
        "lambda": ising["composition_penalty_strength"],
        "n_euler": cfg["ctmc"]["n_euler_steps"],
        "c_target": ising["target_composition"],
        "F_per_site": metrics["free_energy_per_site"],
        "F_per_site_exact": metrics.get("free_energy_per_site_exact"),
        "F_per_site_bias": metrics.get("free_energy_per_site_bias"),
        "ess_frac": metrics["ess_fraction"],
        "c_mean": metrics["composition_mean"],
        "c_std": metrics["composition_std"],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results_dir", type=Path, default=Path("results/02_constrained_soft")
    )
    p.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="config-name stems (without _seed..); one per composition window",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    p.add_argument(
        "--ess_floor",
        type=float,
        default=0.30,
        help="per-seed ESS-fraction floor for entering the F(c) average",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="optional output path for the F(c) + bias figure",
    )
    p.add_argument(
        "--eval_dir",
        choices=["eval", "eval_ema"],
        default="eval",
        help="which frozen eval to score: raw weights or the "
        "dual eval's EMA shadow draw",
    )
    args = p.parse_args()

    # --- collect every available run ------------------------------------
    records = []
    missing = []
    for config in args.configs:
        for seed in args.seeds:
            rd = latest_run_dir(args.results_dir, config, seed, args.eval_dir)
            if rd is None:
                missing.append(f"{config} seed{seed}")
                continue
            records.append(_load_record(rd, args.eval_dir))
    if missing:
        print(f"[warn] no eval found for: {', '.join(missing)}")
    if not records:
        raise SystemExit("no runs found; check --results_dir / --configs / --seeds")

    n_euler = sorted({r["n_euler"] for r in records})
    sigmas = sorted({r["sigma"] for r in records})
    lambdas = sorted({r["lambda"] for r in records})
    Ds = sorted({r["D"] for r in records})
    print(
        f"=== F(c) curve : D={Ds} sigma={sigmas} lambda={lambdas} "
        f"n_euler={n_euler} ess_floor={args.ess_floor} ==="
    )
    if len(n_euler) > 1:
        print(
            f"[warn] mixed n_euler {n_euler}: F-estimate discretisation bias "
            f"differs across windows, so the curve mixes integrators."
        )

    # --- group by composition, ESS-gate, aggregate ---------------------
    by_c: dict[float, list[dict]] = {}
    for r in records:
        by_c.setdefault(round(r["c_target"], 4), []).append(r)

    has_exact = any(r["F_per_site_exact"] is not None for r in records)
    header = f"{'c_t':>6} {'gated':>6} {'F/site (mean +/- sd)':>22} {'ESS frac':>16}"
    if has_exact:
        header += f" {'F/site exact':>13} {'bias (mean)':>12}"
    header += "  excluded(seed:ESS)"
    print(header)

    curve = []  # (c_target, F_mean, F_sd, n_gated, F_exact, bias_mean)
    for c_t in sorted(by_c):
        rows = by_c[c_t]
        gated = [r for r in rows if r["ess_frac"] >= args.ess_floor]
        excluded = [r for r in rows if r["ess_frac"] < args.ess_floor]
        if gated:
            Fs = np.array([r["F_per_site"] for r in gated])
            F_mean, F_sd = (
                float(Fs.mean()),
                float(Fs.std(ddof=1) if len(Fs) > 1 else 0.0),
            )
        else:
            F_mean = F_sd = float("nan")
        ess_lo = min(r["ess_frac"] for r in rows)
        ess_hi = max(r["ess_frac"] for r in rows)
        F_exact = next(
            (r["F_per_site_exact"] for r in rows if r["F_per_site_exact"] is not None),
            None,
        )
        bias_mean = (
            float(np.mean([r["F_per_site_bias"] for r in gated]))
            if gated and gated[0]["F_per_site_bias"] is not None
            else None
        )

        line = (
            f"{c_t:>6.3f} {len(gated):>2}/{len(rows):<3} "
            f"{F_mean:>11.4f} +/- {F_sd:<6.4f} {ess_lo:>6.3f}-{ess_hi:<6.3f}"
        )
        if has_exact:
            line += f" {F_exact:>13.4f}" if F_exact is not None else f" {'-':>13}"
            line += f" {bias_mean:>+12.4f}" if bias_mean is not None else f" {'-':>12}"
        seed_of = lambda r: r["name"].split("_seed")[1].split("_")[0]
        line += "  " + ",".join(f"{seed_of(r)}:{r['ess_frac']:.2f}" for r in excluded)
        print(line)
        curve.append((c_t, F_mean, F_sd, len(gated), F_exact, bias_mean))

    # --- Z_2 symmetry check : F(c) vs F(1-c) ----------------------------
    cset = {c for c, *_ in curve}
    pairs = sorted(
        {
            (min(c, 1 - c), max(c, 1 - c))
            for c in cset
            if round(1 - c, 4) in cset and abs(c - 0.5) > 1e-6
        }
    )
    if pairs:
        print("\n--- Z_2 check  F(c) vs F(1-c) (no field => should match) ---")
        Fmap = {c: F for c, F, *_ in curve}
        for lo, hi in pairs:
            print(
                f"  F({lo:.3f})={Fmap[lo]:+.4f}  F({hi:.3f})={Fmap[hi]:+.4f}  "
                f"|gap|={abs(Fmap[lo] - Fmap[hi]):.4f}"
            )

    if args.plot is not None:
        _plot(curve, has_exact, n_euler, args.plot)


def _plot(curve, has_exact, n_euler, out: Path) -> None:
    """Working plot of the assembled curve (house palette and geometry).

    The seed spread is drawn as a shaded band rather than capped bars: this
    panel joins its points into a curve in c, so the uncertainty is an
    envelope along that curve. (The thesis's F(c) figure, fc_compare.py, keeps
    capped bars because it draws its windows as discrete marks -- see the
    uncertainty grammar in figure_style.)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from discrete_flow_sampler.diagnostics.figure_style import (
        FIGSIZE_FULL_1X2,
        HARD_DELTA_HUE,
        MUTED,
        REFERENCE_INK,
        SAMPLER_HUE,
        SAVEFIG_DPI,
        SINGLE_PANEL_WIDTH_IN,
        style_axes,
        uncertainty_band,
        use_house_style,
    )

    use_house_style()
    cs = [c for c, *_ in curve]
    F = np.array([f for _, f, *_ in curve])
    sd = np.array([s for _, _, s, *_ in curve])
    ncol = 2 if has_exact else 1
    figsize = FIGSIZE_FULL_1X2 if has_exact else (SINGLE_PANEL_WIDTH_IN, 2.9)
    fig, axes = plt.subplots(1, ncol, figsize=figsize, squeeze=False)
    ax = axes[0][0]
    uncertainty_band(ax, cs, F - sd, F + sd, SAMPLER_HUE)
    ax.plot(
        cs,
        F,
        marker="o",
        ms=4,
        color=SAMPLER_HUE,
        lw=1.4,
        zorder=3,
        label="DNFS soft (IS est., band = seed sd)",
    )
    if has_exact:
        Fe = [fe for *_, _, fe, _ in curve]
        if all(v is not None for v in Fe):
            ax.plot(
                cs,
                Fe,
                "--",
                marker="s",
                ms=4,
                color=REFERENCE_INK,
                lw=1.4,
                zorder=4,
                label="exact enumeration",
            )
    ax.set_xlabel("composition $c$")
    ax.set_ylabel("$F/d$")
    ax.set_title(f"F(c), n_euler={n_euler}")
    ax.legend(frameon=False)
    style_axes(ax)
    if has_exact:
        axb = axes[0][1]
        bias = [b for *_, b in curve]
        if all(v is not None for v in bias):
            axb.axhline(0, color=MUTED, lw=0.8)
            axb.plot(cs, bias, marker="o", ms=4, color=HARD_DELTA_HUE, lw=1.4)
        axb.set_xlabel("composition $c$")
        axb.set_ylabel("$F/d$ bias (est. $-$ exact)")
        axb.set_title("integrator bias vs composition")
        style_axes(axb)
    fig.tight_layout()
    fig.savefig(out, dpi=SAVEFIG_DPI)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
