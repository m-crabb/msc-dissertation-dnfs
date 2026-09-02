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

Example (the s95 8x8 house family against the D=8 TI reference;
--eval_dir eval_ema reads the dual eval's shadow-weight draw, archived
pre-EMA d10 cells keep the default):
    python -m experiments.constrained_soft_02.analysis.08_fc_compare \
        --results_dir results/02_constrained_soft \
        --reference results/02_constrained_soft/fc_ref_d8.npz \
        --configs S2_d8_c0250_l50_letf_ne128_house \
                  S2_d8_c0375_l50_letf_ne128_house \
                  S2_d8_c0500_l50_letf_ne128_house \
        --seeds 42 43 44 45 --ess_floor 0.30 --eval_dir eval_ema \
        --plot results/02_constrained_soft/fc_compare_d8_house.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from experiments.constrained_soft_02.analysis._common import latest_run_dir, seed_of


def _load_record(run_dir: Path, eval_dir: str = "eval") -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    ising = cfg["ising"]
    metrics = json.loads((run_dir / eval_dir / "metrics.json").read_text())
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


def _bootstrap_F(run_dir: Path, d: int, n_boot: int, rng,
                 eval_dir: str = "eval") -> tuple[float, np.ndarray]:
    """Per-site canonical-units free energy (nats) and a bootstrap sample of it.

    F_total = -mean(log w); per site = F_total / d. This reproduces
    `free_energy_per_site * 2*sigma` exactly (the stored value is
    -mean(log w)/(2*sigma*d)) but resamples the weights so we get an error bar.
    `eval_dir` selects the frozen draw ("eval") or a grid redraw ("eval_ne256").
    """
    logw = torch.load(run_dir / eval_dir / "log_weights.pt").double().numpy().ravel()
    point = float(-logw.mean() / d)
    n = logw.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = -logw[idx].mean(axis=1) / d
    return point, boot


def _grids_available(run_dir: Path, native_ne: int,
                     eval_dir: str = "eval") -> list[tuple[int, str]]:
    """Euler grids this checkpoint has been drawn on, coarsest first.

    The frozen `{eval_dir}/` is the run's native grid; `{eval_dir}_ne<k>/`
    side dirs are the redraws written by `run.eval_only(n_euler_override=k)`
    (EMA-side redraws would land as `eval_ema_ne<k>/`; none exist yet, so an
    eval_ema pass falls back to the native draw with the caller's warning).
    """
    grids = [(native_ne, eval_dir)]
    for side in run_dir.glob(f"{eval_dir}_ne*"):
        if (side / "log_weights.pt").exists():
            grids.append(
                (int(side.name.removeprefix(f"{eval_dir}_ne")), side.name))
    return sorted(grids)


def _richardson_F(run_dir: Path, native_ne: int, d: int, n_boot: int,
                  rng, eval_dir: str = "eval",
                  ) -> tuple[float, np.ndarray, tuple[int, int] | None]:
    """First-order Richardson extrapolation of F/site to the continuum grid.

    The Euler-grid error is first order (pre-registered 2026-08-21: step
    ratios 0.44-0.49 across ne64 -> 128 -> 256; reproduced by the s62
    retrain halving the residual vs the TI truth in every window), so
    F(g) = F(inf) + C/g and two grids g1 < g2 give

        F(inf) = (g2 F(g2) - g1 F(g1)) / (g2 - g1)

    (the familiar 2 F(2g) - F(g) when g2 = 2 g1). The two FINEST available
    grids are used; the two draws are independent (fresh eval batches), so
    the bootstrap resamples each grid's weights independently and combines
    replicate-wise. Falls back to the native draw (pair = None) when the
    checkpoint has no side-grid redraws -- the caller warns, so a partially
    redrawn family cannot silently mix extrapolated and raw points.
    """
    grids = _grids_available(run_dir, native_ne, eval_dir)
    if len(grids) < 2:
        point, boot = _bootstrap_F(run_dir, d, n_boot, rng, eval_dir)
        return point, boot, None
    (g1, dir1), (g2, dir2) = grids[-2], grids[-1]
    p1, b1 = _bootstrap_F(run_dir, d, n_boot, rng, dir1)
    p2, b2 = _bootstrap_F(run_dir, d, n_boot, rng, dir2)
    point = (g2 * p2 - g1 * p1) / (g2 - g1)
    boot = (g2 * b2 - g1 * b1) / (g2 - g1)
    return point, boot, (g1, g2)


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
    p.add_argument("--richardson", action="store_true",
                   help="extrapolate each seed's F to the continuum grid from its "
                        "two finest available draws (eval/ + eval_ne<k>/ redraws)")
    p.add_argument("--flag_c", nargs="*", type=float, default=[],
                   help="compositions drawn with a provisional ring (their Z2 "
                        "mirrors inherit it); used while a window awaits retrain "
                        "or prints from a different training grid")
    p.add_argument("--eval_dir", choices=["eval", "eval_ema"], default="eval",
                   help="which frozen eval to score: raw weights or the s95 "
                        "dual eval's EMA shadow draw")
    p.add_argument("--hard_rows", nargs="*", type=Path, default=[],
                   help="zero-shot probe JSONs of the HARD (canonical) sampler, "
                        "one per seed, native Euler grid; adds the direct "
                        "fixed-composition free energy as a third series")
    p.add_argument("--hard_rows_fine", nargs="*", type=Path, default=[],
                   help="the same seeds redrawn on a finer Euler grid, same "
                        "order as --hard_rows; enables Richardson extrapolation "
                        "of the hard series")
    args = p.parse_args()
    rng = np.random.default_rng(0)

    # --- collect every available run ------------------------------------
    records, missing = [], []
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
        seed_pts, seed_boots, grid_pairs = [], [], []
        for r in gated:
            if args.richardson:
                pt, boot, pair = _richardson_F(
                    r["run_dir"], r["n_euler"], d, args.n_boot, rng,
                    args.eval_dir)
                if pair is None:
                    print(f"[warn] {r['name']}: no side-grid redraw, "
                          f"point stays on the native ne{r['n_euler']} draw")
                grid_pairs.append(pair)
            else:
                pt, boot = _bootstrap_F(
                    r["run_dir"], d, args.n_boot, rng, args.eval_dir)
            seed_pts.append(pt)
            seed_boots.append(boot)
        if args.richardson and any(gp is not None for gp in grid_pairs):
            pairs_used = sorted({gp for gp in grid_pairs if gp is not None})
            print(f"       richardson pairs at c_t={c_t}: {pairs_used}")
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
        # The soft-ensemble truth: the canonical curve mapped INTO the
        # penalised ensemble (truth minus the same Laplace offset the
        # correction adds). The raw markers should sit ON this line; drawn
        # so the raw-vs-canonical gap reads as the ensemble mapping, not
        # as sampler error (the question every reader otherwise asks).
        soft_offsets = np.array([
            _laplace_offset(lam, d, fp, fpp) / d
            for fp, fpp in zip(ref_Fp, ref_Fpp)])
        hard = (_hard_series(args.hard_rows, args.hard_rows_fine, d,
                             lambda c: _ref_at(c, ref_F_persite))
                if args.hard_rows else None)
        _plot(curve, ref_c, ref_F_persite, ref_F_persite - soft_offsets,
              args.flag_c, args.plot, hard)


def _hard_series(files: list[Path], fine_files: list[Path], d: int,
                 ref_at) -> list[dict]:
    """The hard sampler's F(c) read straight off its slice weights.

    Each probe JSON holds one seed's rows; at stop_time 1 the row carries
    `free_energy_nats_per_site` = -E[log w]/d, the slice free energy with no
    ensemble offset (there is no ensemble to map out of: the base is uniform
    on the slice and every move stays on it). Within-seed Monte Carlo error
    of the mean log-weight is sqrt(Var[log w]/n)/d from the stored variance;
    between-seed scatter is added in quadrature as for the soft series. With
    a fine-grid redraw per seed the point is Richardson-extrapolated,
    F(inf) = (g2 F2 - g1 F1)/(g2 - g1), the same first-order rule as
    `_richardson_F` (the Euler bias is first order in the step for both
    samplers). ESS is carried per composition because the estimate is a
    variational bound whose gap grows as the weights degrade.
    """
    def load(path):
        blob = json.loads(Path(path).read_text())
        rows = {round(r["composition"], 4): r for r in blob["rows"]
                if abs(r["stop_time"] - 1.0) < 1e-9}
        return blob["n_euler_steps"], rows

    seeds = [load(f) for f in files]
    fine = [load(f) for f in fine_files] if fine_files else [None] * len(seeds)
    if fine_files and len(fine_files) != len(files):
        raise SystemExit("--hard_rows_fine must pair one-to-one with --hard_rows")
    comps = sorted(set().union(*(rows.keys() for _, rows in seeds)))
    series = []
    for c in comps:
        points, errors, ess = [], [], []
        for (g1, rows), fine_entry in zip(seeds, fine):
            r = rows[c]
            f1 = r["free_energy_nats_per_site"]
            e1 = np.sqrt(r["var_log_w"] / r["n_samples"]) / d
            if fine_entry is not None:
                g2, fine_rows = fine_entry
                rf = fine_rows[c]
                f2 = rf["free_energy_nats_per_site"]
                e2 = np.sqrt(rf["var_log_w"] / rf["n_samples"]) / d
                point = (g2 * f2 - g1 * f1) / (g2 - g1)
                err = np.hypot(g2 * e2, g1 * e1) / (g2 - g1)
                ess.append(min(r["ess_fraction"], rf["ess_fraction"]))
            else:
                point, err = f1, e1
                ess.append(r["ess_fraction"])
            points.append(point)
            errors.append(err)
        points = np.array(points)
        within = float(np.sqrt(np.mean(np.square(errors)) / len(errors)))
        between = (float(points.std(ddof=1) / np.sqrt(len(points)))
                   if len(points) > 1 else 0.0)
        series.append(dict(c=c, F=float(points.mean()),
                           F_err=float(np.hypot(within, between)),
                           truth=ref_at(c), ess_lo=min(ess), ess_hi=max(ess),
                           n=len(points)))
    print(f"\n--- hard (canonical) series: {len(files)} seeds, "
          f"{'Richardson' if fine_files else 'native grid'} ---")
    print(f"{'c':>6} {'ESS frac':>14} {'F_hard':>10} {'F_truth':>9} {'hard-tru':>9}")
    for row in series:
        gap = row["F"] - row["truth"] if row["truth"] is not None else np.nan
        tru = f"{row['truth']:>9.4f}" if row["truth"] is not None else f"{'-':>9}"
        print(f"{row['c']:>6.3f} {row['ess_lo']:>6.3f}-{row['ess_hi']:<6.3f} "
              f"{row['F']:>10.4f} {tru} {gap:>9.4f}")
    return series


def _mirror_rows(rows: list[dict]) -> list[dict]:
    """Z_2 reflections of the sampled points: F(c)=F(1-c) for zero-field Ising.

    The sampled windows are RHS-heavy ({0.30,0.50,0.55,0.60,0.65}), so the left
    segment and the 0.70 tail are empty. Reflecting each off-centre point across
    c=0.5 fills them in. These are symmetry-implied, NOT independently trained
    compositions (the canonical reference's own Z_2 check is <=0.00013/site), so
    they reuse the source point's value/error and the same truth. They are drawn
    identically to the trained windows — the filled/open marker split confused
    more than it informed (reader feedback 2026-08-11), so the reflection is
    stated once in the dissertation body text instead of per-marker. Skip c=0.5
    and any reflection that lands on an already-sampled window.
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


def _plot(curve, ref_c, ref_F_persite, ref_F_soft_persite, flag_c,
          out: Path, hard: list[dict] | None = None) -> None:
    """House-standard overlay + residual pair (approved s62; relaid out s101).

    Roles: TI truth = REFERENCE_INK line; our sampler = SAMPLER_HUE, with the
    corrected estimate as the filled square (the deliverable) and the raw
    soft-ensemble read as the open, lightened circle (the same object before
    the ensemble mapping -- one role, two intensities, never a second hue).
    The soft-ensemble truth (canonical minus the analytic offset) is the
    raw markers' own reference line, in the raw hue, dashed. Compositions
    in `flag_c` (plus their Z2 mirrors) get a muted provisional ring: the
    point prints from a different training grid or awaits retrain, and the
    caption says which.

    Legend is FIGURE-level, below the panels (s101): no in-axes placement
    is shape-robust across couplings -- the "empty top-centre" that held
    the legend on the U-shaped subcritical curve is exactly the peak of
    the inverted critical one, where it occluded the truth line and its
    sample glyphs printed at data height beside real markers.
    """
    import matplotlib.pyplot as plt

    from discrete_flow_sampler.diagnostics.figure_style import (
        FULL_WIDTH_IN, HARD_DELTA_HUE, MUTED, REFERENCE_INK, SAMPLER_HUE,
        SAVEFIG_DPI, parameter_ramp, style_axes, use_house_style)

    use_house_style()
    rows = [r for r in curve if not np.isnan(r["raw"])]
    mirror = _mirror_rows(rows)
    # One uniform curve: sampled + symmetry-implied points drawn identically,
    # sorted by composition (the reflection is stated in the body text).
    allrows = sorted(rows + mirror, key=lambda r: r["c"])
    cs = [r["c"] for r in allrows]
    raw = [r["raw"] for r in allrows]
    raw_e = [r["raw_err"] for r in allrows]
    corr = [r["corr"] for r in allrows]
    corr_e = [r["corr_err"] for r in allrows]
    truth = [r["truth"] for r in allrows]
    flagged = {round(c, 4) for c in flag_c} | {round(1 - c, 4) for c in flag_c}
    raw_hue = parameter_ramp(SAMPLER_HUE, 2)[0]

    # 2.7 in: the SHORT 2.4 in box plus the strip the below-panel figure
    # legend needs (prints 6.9 cm vs 6.1 cm, +0.8 cm).
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.7))
    ax.plot(ref_c, ref_F_persite, color=REFERENCE_INK, lw=1.4,
            label="TI truth")
    ax.plot(ref_c, ref_F_soft_persite, color=raw_hue, lw=1.0,
            linestyle="--", label="soft truth (TI $-$ offset)")
    # Capped bars, not a shaded band. Each abscissa here is a SEPARATELY
    # TRAINED window (eleven of them, six trained plus their Z2 reflections),
    # so there is no curve in c for a ribbon to be the envelope of: the marks
    # are deliberately unjoined for the same reason. Bands are the house
    # default only for uncertainty along a continuous x (figure_style).
    ax.errorbar(cs, raw, yerr=raw_e, fmt="o", color=raw_hue, mfc="none",
                capsize=2, lw=1.0, label="raw")
    ax.errorbar(cs, corr, yerr=corr_e, fmt="s", color=SAMPLER_HUE,
                capsize=2, lw=1.0, label="Laplace-corrected")
    # The hard sampler's own read of the same object: no offset, no
    # correction, the limit the soft route reaches for (HARD_DELTA_HUE).
    if hard:
        ax.errorbar([h["c"] for h in hard], [h["F"] for h in hard],
                    yerr=[h["F_err"] for h in hard], fmt="^",
                    color=HARD_DELTA_HUE, capsize=2, lw=1.0,
                    label="hard, direct")
    ax.set_xlabel("composition $c$")
    ax.set_ylabel("$F/d$ (nats per site)")

    if all(t is not None for t in truth):
        axr.axhline(0, color=MUTED, lw=0.8)
        araw_res = [r - t for r, t in zip(raw, truth)]
        acorr_res = [c - t for c, t in zip(corr, truth)]
        axr.plot(cs, araw_res, "o-", color=raw_hue, mfc="none", lw=1.0)
        axr.plot(cs, acorr_res, "s-", color=SAMPLER_HUE, lw=1.0)
        if hard:
            with_truth = [h for h in hard if h["truth"] is not None]
            axr.plot([h["c"] for h in with_truth],
                     [h["F"] - h["truth"] for h in with_truth], "^-",
                     color=HARD_DELTA_HUE, lw=1.0)
        axr.set_xlabel("composition $c$")
        axr.set_ylabel("$F/d$ residual (nats per site)")

    for axis, ys in ((ax, corr), (axr, acorr_res if all(t is not None for t in truth) else None)):
        if ys is None:
            continue
        ring_c = [c for c in cs if round(c, 4) in flagged]
        ring_y = [y for c, y in zip(cs, ys) if round(c, 4) in flagged]
        if ring_c:
            axis.scatter(ring_c, ring_y, s=140, facecolors="none",
                         edgecolors=MUTED, linewidths=1.1, zorder=4)
    # The ring alone marks the provisional windows; the caption says why
    # (a ne64-trained point, or a retrain still owed). An in-panel word
    # collides with the legend at these tail positions.

    for i, axis in enumerate((ax, axr)):
        style_axes(axis)
        axis.text(0.02, 1.02, f"({chr(97 + i)})", transform=axis.transAxes,
                  fontweight="bold", va="bottom")
    # Panel (b) reuses (a)'s marker/hue identities, so one four-entry row
    # names everything for both panels.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="lower center")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out, dpi=SAVEFIG_DPI)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
