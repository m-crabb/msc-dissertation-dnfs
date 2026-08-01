"""Overlay the DNFS soft F(c) against the mchammer canonical reference.

This is the headline F(c) comparison. Everything lives on ONE canonical
free-energy-per-site axis, and the figure carries three curves (no new runs):

  1. **Ground truth** - the canonical (fixed-composition) free energy. At D=10
     this is the native mchammer thermodynamic-integration reference built by
     `07_fc_mchammer_reference.py` (results/.../fc_ref_d10.npz); at D<=4 it is
     the exact enumeration `run.py` already stores as
     `free_energy_per_site_exact`.
  2. **DNFS soft, raw** - the soft/vcSGC ensemble's free energy as DNFS measures
     it, converted from the report's reduced convention to plain nats per site
     (multiply `free_energy_per_site` by 2*sigma, since the stored value is
     -mean(log w)/(2*sigma*d)). This sits *below* canonical near the centre by
     the soft->canonical offset and swings *above* in the tails where the IS
     estimator bias grows. That deviation is the soft inexactness.
  3. **DNFS soft, Laplace-corrected** - the soft estimate mapped to the canonical
     curve by inverting the Gaussian composition convolution
     Z_lambda(c_t) = sum_c Z_can(c) * exp(-lambda*d*(c-c_t)^2). For a sharp
     penalty this gives, per window,
       F_can(c_t) = F_lambda(c_t) + log d + 1/2 log(pi/a) + f'^2/(4a),
       a = lambda*d - 1/2 f'' = lambda*d + 1/2 F_can''(c_t),  f' = -F_can'(c_t),
     with the canonical slope/curvature read off the reference curve (the leading
     log d + 1/2 log(pi/(lambda d)) offset is reference-free; only the small
     curvature/slope refinements use the reference shape). What is left after the
     correction, `corrected - truth`, is the pure importance-sampling (Jensen)
     bias: small at the centre, growing into the tails as ESS falls. That residual
     is the cost signal.

The vcSGC benchmark is deliberately NOT on this plot: mchammer has no native
semigrand free-energy tool, and a literal VCSGCEnsemble run yields a *canonical*
curve (via integrating its logged chemical potential) that simply overlays the
ground truth. The apples-to-apples DNFS-vs-vcSGC check therefore lives in the
weighted-thermodynamics comparison (composition marginal, energy, short-range
order at matched kappa), not here.

DNFS error bars come from bootstrapping the per-window importance weights
(`eval/log_weights.pt`); the correction offset is treated as exact.

The sampled windows are RHS-heavy ({0.30,0.50,0.55,0.60,0.65}), so the figure
also draws the Z_2 reflection of each off-centre point (F(c)=F(1-c) for zero-field
Ising) to fill the left segment and the 0.70 tail. Those mirror points are drawn
open-faced: they are symmetry-implied from the trained windows, not independently
trained compositions.

Caveat: the correction is the sharp-penalty *continuum* Laplace form. At
lambda=50 the penalty width 1/sqrt(2*lambda*d) is about one composition step at
D=10 (and narrower than a step at D=4), so the continuum offset is an
approximation; the exact discrete deconvolution is a later refinement.

Example:
    python -m experiments.constrained_soft_02.analysis.08_fc_compare \
        --results_dir results/02_constrained_soft \
        --reference results/02_constrained_soft/fc_ref_d10.npz \
        --configs S2_d10_c030_l50_letf_ne64_anneal \
                  S2_d10_c05_l50_letf_ne64_anneal \
                  S2_d10_c055_l50_letf_ne64_anneal \
                  S2_d10_c060_l50_letf_ne64_anneal \
                  S2_d10_c065_l50_letf_ne64_anneal \
        --seeds 42 43 44 45 --ess_floor 0.30 \
        --plot results/02_constrained_soft/fc_compare_ne64_anneal.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_soft_02.analysis._common import latest_run_dir, seed_of


def _load_record(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    ising = cfg["ising"]
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    D = ising["D"]
    return {
        "name": run_dir.name,
        "run_dir": run_dir,
        "D": D,
        "d": D * D,
        "sigma": ising["sigma"],
        "lambda": ising["composition_penalty_strength"],
        "n_euler": cfg["ctmc"]["n_euler_steps"],
        "c_target": ising["target_composition"],
        "F_per_site": metrics["free_energy_per_site"],
        "F_per_site_exact": metrics.get("free_energy_per_site_exact"),
        "ess_frac": metrics["ess_fraction"],
        "c_mean": metrics["composition_mean"],
        "c_std": metrics["composition_std"],
    }


def _bootstrap_F(run_dir: Path, d: int, n_boot: int, rng) -> tuple[float, np.ndarray]:
    """Per-site canonical-units free energy (nats) and a bootstrap sample of it.

    F_total = -mean(log w); per site = F_total / d. This reproduces
    `free_energy_per_site * 2*sigma` exactly (the stored value is
    -mean(log w)/(2*sigma*d)) but resamples the weights so we get an error bar.
    """
    logw = torch.load(run_dir / "eval" / "log_weights.pt").double().numpy().ravel()
    point = float(-logw.mean() / d)
    n = logw.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = -logw[idx].mean(axis=1) / d
    return point, boot


def _laplace_offset(lam: float, d: int, Fp_total: float, Fpp_total: float) -> float:
    """Soft -> canonical correction (total, nats): F_can = F_lambda + offset.

    a = lambda*d - 1/2 f'' with f = log Z_can = -F_can_total, so f'' = -F_can'',
    giving a = lambda*d + 1/2 F_can''. Slope term f'^2/(4a), f' = -F_can'.
    """
    a = lam * d + 0.5 * Fpp_total
    return float(np.log(d) + 0.5 * np.log(np.pi / a) + (Fp_total ** 2) / (4.0 * a))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path, default=Path("results/02_constrained_soft"))
    p.add_argument("--reference", type=Path, default=None,
                   help="npz from 07_fc_mchammer_reference (D=10 ground truth); "
                        "omit at D<=4 to use the exact-enumeration column")
    p.add_argument("--configs", nargs="+", required=True,
                   help="config-name stems (without _seed..); one per composition window")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    p.add_argument("--ess_floor", type=float, default=0.30,
                   help="per-seed ESS-fraction floor for entering the F(c) average")
    p.add_argument("--n_boot", type=int, default=2000,
                   help="bootstrap resamples of the per-window importance weights")
    p.add_argument("--plot", type=Path, default=None,
                   help="optional output path for the overlay + residual figure")
    args = p.parse_args()
    rng = np.random.default_rng(0)

    # --- collect every available run ------------------------------------
    records, missing = [], []
    for config in args.configs:
        for seed in args.seeds:
            rd = latest_run_dir(args.results_dir, config, seed)
            if rd is None:
                missing.append(f"{config} seed{seed}")
                continue
            records.append(_load_record(rd))
    if missing:
        print(f"[warn] no eval found for: {', '.join(missing)}")
    if not records:
        raise SystemExit("no runs found; check --results_dir / --configs / --seeds")

    Ds = sorted({r["D"] for r in records})
    sigmas = sorted({r["sigma"] for r in records})
    lambdas = sorted({r["lambda"] for r in records})
    n_eulers = sorted({r["n_euler"] for r in records})
    if len(Ds) > 1 or len(lambdas) > 1:
        raise SystemExit(f"expected one D and one lambda; got D={Ds} lambda={lambdas}")
    D, d, lam = Ds[0], records[0]["d"], lambdas[0]
    sigma = sigmas[0]
    two_sigma = 2.0 * sigma
    print(f"=== F(c) compare : D={D} sigma={sigmas} lambda={lam} "
          f"n_euler={n_eulers} ess_floor={args.ess_floor} n_boot={args.n_boot} ===")
    if len(n_eulers) > 1:
        print(f"[warn] mixed n_euler {n_eulers}: discretisation bias differs across windows.")

    # --- canonical reference curve + its slope/curvature ----------------
    # ref_c -> (F_total, F'_total, F''_total); F_total used as ground truth and
    # as the source of the small slope/curvature terms in the Laplace offset.
    if args.reference is not None:
        ref = np.load(args.reference)
        ref_c = np.asarray(ref["c_eff"], float)
        ref_F_total = np.asarray(ref["F_total"], float)
        ref_F_persite = np.asarray(ref["F_per_site"], float)
        ref_src = str(args.reference)
    else:
        # D<=4: build the canonical curve from the exact-enum column run.py stores
        exact = {round(r["c_target"], 4): r["F_per_site_exact"] * two_sigma
                 for r in records if r["F_per_site_exact"] is not None}
        if not exact:
            raise SystemExit("no --reference and no exact column; cannot place ground truth")
        ref_c = np.array(sorted(exact))
        ref_F_persite = np.array([exact[c] for c in ref_c])
        ref_F_total = ref_F_persite * d
        ref_src = "exact enumeration (metrics.json)"
    order = np.argsort(ref_c)
    ref_c, ref_F_total, ref_F_persite = ref_c[order], ref_F_total[order], ref_F_persite[order]
    ref_Fp = np.gradient(ref_F_total, ref_c)
    ref_Fpp = np.gradient(ref_Fp, ref_c)
    print(f"ground truth: {ref_src}")

    def _ref_at(c: float, arr: np.ndarray) -> float | None:
        j = np.argmin(np.abs(ref_c - c))
        return float(arr[j]) if abs(ref_c[j] - c) < 1e-6 else None

    # --- group by composition, ESS-gate, aggregate raw + corrected ------
    by_c: dict[float, list[dict]] = {}
    for r in records:
        by_c.setdefault(round(r["c_target"], 4), []).append(r)

    header = (f"{'c_t':>6} {'gated':>6} {'ESS frac':>14} "
              f"{'F_raw (nats/site)':>20} {'F_corr':>16} {'F_truth':>9} "
              f"{'raw-tru':>9} {'corr-tru':>9}  excluded(seed:ESS)")
    print(header)

    curve = []
    for c_t in sorted(by_c):
        rows = by_c[c_t]
        gated = [r for r in rows if r["ess_frac"] >= args.ess_floor]
        excluded = [r for r in rows if r["ess_frac"] < args.ess_floor]
        ess_lo = min(r["ess_frac"] for r in rows)
        ess_hi = max(r["ess_frac"] for r in rows)

        F_truth = _ref_at(c_t, ref_F_persite)
        if not gated:
            line = (f"{c_t:>6.3f} {0:>2}/{len(rows):<3} {ess_lo:>6.3f}-{ess_hi:<6.3f} "
                    f"{'(all excluded)':>20}")
            print(line)
            curve.append(dict(c=c_t, raw=np.nan, raw_err=np.nan, corr=np.nan,
                              corr_err=np.nan, truth=F_truth, n=0))
            continue

        # per-seed point + bootstrap (per-site nats)
        seed_pts, seed_boots = [], []
        for r in gated:
            pt, boot = _bootstrap_F(r["run_dir"], d, args.n_boot, rng)
            seed_pts.append(pt)
            seed_boots.append(boot)
        seed_pts = np.array(seed_pts)
        F_raw = float(seed_pts.mean())
        # error of the seed-mean: within-seed MC (bootstrap of the average) plus
        # between-seed training scatter, added in quadrature.
        boot_mean = np.mean(np.stack(seed_boots), axis=0)
        within = float(boot_mean.std())
        between = float(seed_pts.std(ddof=1) / np.sqrt(len(seed_pts))) if len(seed_pts) > 1 else 0.0
        F_raw_err = float(np.hypot(within, between))

        # Laplace correction (continuum, with curvature) from the reference shape
        Fp_t = _ref_at(c_t, ref_Fp)
        Fpp_t = _ref_at(c_t, ref_Fpp)
        if Fp_t is None or Fpp_t is None:
            offset_ps = np.nan
        else:
            offset_ps = _laplace_offset(lam, d, Fp_t, Fpp_t) / d
        F_corr = F_raw + offset_ps
        F_corr_err = F_raw_err  # offset treated as exact

        raw_gap = (F_raw - F_truth) if F_truth is not None else None
        corr_gap = (F_corr - F_truth) if F_truth is not None else None

        def _fmt(v, w=9, prec=4):
            return f"{v:>{w}.{prec}f}" if v is not None and not np.isnan(v) else f"{'-':>{w}}"

        line = (f"{c_t:>6.3f} {len(gated):>2}/{len(rows):<3} {ess_lo:>6.3f}-{ess_hi:<6.3f} "
                f"{F_raw:>11.4f} +/-{F_raw_err:<5.4f} {F_corr:>10.4f}{'':>5} "
                f"{_fmt(F_truth)} {_fmt(raw_gap)} {_fmt(corr_gap)}  "
                + ",".join(f"{seed_of(r['name'])}:{r['ess_frac']:.2f}" for r in excluded))
        print(line)
        curve.append(dict(c=c_t, raw=F_raw, raw_err=F_raw_err, corr=F_corr,
                          corr_err=F_corr_err, truth=F_truth, n=len(gated)))

    # --- Z2 symmetry note on the corrected curve ------------------------
    cmap = {row["c"]: row["corr"] for row in curve if not np.isnan(row["corr"])}
    pairs = sorted({(min(c, 1 - c), max(c, 1 - c)) for c in cmap
                    if round(1 - c, 4) in cmap and abs(c - 0.5) > 1e-6})
    if pairs:
        print("\n--- Z_2 check  F_corr(c) vs F_corr(1-c) (no field => should match) ---")
        for lo, hi in pairs:
            print(f"  F({lo:.3f})={cmap[lo]:+.4f}  F({hi:.3f})={cmap[hi]:+.4f}  "
                  f"|gap|={abs(cmap[lo] - cmap[hi]):.4f}")

    if args.plot is not None:
        _plot(curve, ref_c, ref_F_persite, lam, n_eulers, args.plot)


def _mirror_rows(rows: list[dict]) -> list[dict]:
    """Z_2 reflections of the sampled points: F(c)=F(1-c) for zero-field Ising.

    The sampled windows are RHS-heavy ({0.30,0.50,0.55,0.60,0.65}), so the left
    segment and the 0.70 tail are empty. Reflecting each off-centre point across
    c=0.5 fills them in. These are symmetry-implied, NOT independently trained
    compositions (the canonical reference's own Z_2 check is <=0.00013/site), so
    they reuse the source point's value/error and the same truth; we draw them
    open-faced to keep that distinction visible. Skip c=0.5 and any reflection
    that lands on an already-sampled window.
    """
    sampled = {round(r["c"], 4) for r in rows}
    mirrored = []
    for r in rows:
        cm = round(1.0 - r["c"], 4)
        if abs(r["c"] - 0.5) < 1e-6 or cm in sampled:
            continue
        mirrored.append(dict(c=cm, raw=r["raw"], raw_err=r["raw_err"],
                             corr=r["corr"], corr_err=r["corr_err"],
                             truth=r["truth"], n=r["n"]))
    return mirrored


def _plot(curve, ref_c, ref_F_persite, lam, n_eulers, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in curve if not np.isnan(r["raw"])]
    mirror = _mirror_rows(rows)
    cs = [r["c"] for r in rows]
    raw = [r["raw"] for r in rows]
    raw_e = [r["raw_err"] for r in rows]
    corr = [r["corr"] for r in rows]
    corr_e = [r["corr_err"] for r in rows]
    truth = [r["truth"] for r in rows]
    mcs = [r["c"] for r in mirror]
    mraw = [r["raw"] for r in mirror]
    mcorr = [r["corr"] for r in mirror]

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax.plot(ref_c, ref_F_persite, "k-", lw=1.4, label="canonical ground truth (mchammer TI)")
    ax.errorbar(cs, raw, yerr=raw_e, fmt="o", color="tab:orange", alpha=0.55,
                capsize=3, label="DNFS soft, raw ($\\times 2\\sigma$)")
    ax.errorbar(cs, corr, yerr=corr_e, fmt="s", color="tab:blue",
                capsize=3, label="DNFS soft, Laplace-corrected")
    if mirror:
        ax.plot(mcs, mraw, "o", color="tab:orange", alpha=0.55, markerfacecolor="none")
        ax.plot(mcs, mcorr, "s", color="tab:blue", markerfacecolor="none")
    ax.set_xlabel("composition $c$")
    ax.set_ylabel("$F/d$ (nats per site)")
    ax.set_title(f"F(c): soft vs canonical ($\\lambda={lam:g}$, n_euler={n_eulers})")
    handles, labels = ax.get_legend_handles_labels()
    if mirror:
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor="none", markeredgecolor="grey",
                              label="open: $Z_2$ mirror"))
    ax.legend(handles=handles, fontsize=8)

    if all(t is not None for t in truth):
        axr.axhline(0, color="grey", lw=0.8)
        # connect sampled + mirror as one symmetric curve, mark which is which
        allrows = sorted(rows + mirror, key=lambda r: r["c"])
        ac = [r["c"] for r in allrows]
        araw_res = [r["raw"] - r["truth"] for r in allrows]
        acorr_res = [r["corr"] - r["truth"] for r in allrows]
        axr.plot(ac, araw_res, "-", color="tab:orange", alpha=0.5)
        axr.plot(ac, acorr_res, "-", color="tab:blue", alpha=0.5)
        axr.plot(cs, [r - t for r, t in zip(raw, truth)], "o", color="tab:orange",
                 alpha=0.7, label="raw $-$ truth (offset $+$ IS bias)")
        axr.plot(cs, [c - t for c, t in zip(corr, truth)], "s", color="tab:blue",
                 label="corrected $-$ truth (IS bias)")
        if mirror:
            axr.plot(mcs, [r["raw"] - r["truth"] for r in mirror], "o",
                     color="tab:orange", alpha=0.7, markerfacecolor="none")
            axr.plot(mcs, [r["corr"] - r["truth"] for r in mirror], "s",
                     color="tab:blue", markerfacecolor="none")
        axr.set_xlabel("composition $c$")
        axr.set_ylabel("$F/d$ residual (nats per site)")
        axr.set_title("residual vs canonical truth")
        axr.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
