"""DNFS soft vs mchammer vcSGC at matched kappa=lambda: weighted thermodynamics.

This is the apples-to-apples DNFS-vs-vcSGC comparison the F(c) overlay
(`08_fc_compare.py`) deliberately left off its plot. Both samplers target the SAME
semigrand object here: the soft/penalised 2D Ising at penalty strength lambda is
exactly mchammer's variance-constrained semigrand-canonical (vcSGC) ensemble at
kappa = lambda, phi_1 = -2*c_target (validated 2026-06-13; the mapping now
lives in `discrete_flow_sampler.mcmc.mchammer_ising`, pinned by
`tests/test_mchammer_ising.py`). So at each window we can lay DNFS's
importance-weighted observables directly against a literal VCSGCEnsemble chain, no
free-energy reconciliation needed.

Per composition window we compare three observables:

  1. **Composition marginal** - weighted mean and std of c. Both samplers fluctuate
     around c_target with the penalty-set width 1/sqrt(2*lambda*d); this is the
     constraint axis (does the soft sampler reproduce the vcSGC composition spread,
     not just its mean).
  2. **Energy per site** - E(x)/d with E(x) = -log p(x) the cluster-expansion
     energy. On the DNFS side E(x) = -sigma * S_ord(x), where S_ord is the
     nearest-neighbour product summed over the 4-neighbour torus (each undirected
     bond counted twice); this matches icet's `ClusterExpansionCalculator` total
     energy bit-for-bit (verified against `ce.predict` at D=4). On the vcSGC side it
     is the logged `potential` per site.
  3. **Nearest-neighbour short-range order** - the mean NN pair product
     <x_i x_j> over undirected bonds, S_und/(2d) = S_ord/(4d), in [-1, 1].

Note (honest): with a single NN-pair orbit and no field, energy/site and the NN
SRO are affine-related (SRO = -E_persite / (4*sigma)), so they are one comparison
in two units rather than two independent checks. We report both because energy is
the thermodynamic quantity and SRO is the interpretable correlation; agreement on
one is agreement on the other.

DNFS estimates are self-normalised importance-weighted means from
`eval/{samples.pt, log_weights.pt}`, with seeds ESS-gated (floor 0.30, same as the
F(c) curve) and bootstrap error bars; the vcSGC reference is a native
`VCSGCEnsemble` chain (kT = 1) averaged after burn-in, with the seed-to-seed spread
as its bar. Local CPU, no Modal, no new training.

Example (the s95 8x8 house family; --eval_dir eval_ema reads the dual
eval's shadow-weight draw, archived pre-EMA d10 cells keep the default):
    python -m experiments.constrained_soft_02.analysis.09_fc_weighted_thermo \
        --results_dir results/02_constrained_soft \
        --configs S2_d8_c0250_l50_letf_ne128_house \
                  S2_d8_c0375_l50_letf_ne128_house \
                  S2_d8_c0500_l50_letf_ne128_house \
        --seeds 42 43 44 45 --ess_floor 0.30 --eval_dir eval_ema \
        --vcsgc_seeds 0 1 2 --vcsgc_steps 300000 \
        --plot results/02_constrained_soft/fc_weighted_thermo_d8_house.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.mcmc.mchammer_ising import run_vcsgc
from experiments.constrained_soft_02.analysis._common import latest_run_dir, seed_of


def _load_meta(run_dir: Path, eval_dir: str = "eval") -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    ising = cfg["ising"]
    metrics = json.loads((run_dir / eval_dir / "metrics.json").read_text())
    D = ising["D"]
    return dict(name=run_dir.name, run_dir=run_dir, D=D, d=D * D,
                sigma=ising["sigma"], lam=ising["composition_penalty_strength"],
                c_target=ising["target_composition"],
                ess_frac=metrics["ess_fraction"])


# --------------------------------------------------------------------------- #
# observables
# --------------------------------------------------------------------------- #
def _nn_sum_ordered(spins: np.ndarray, D: int) -> np.ndarray:
    """Sum_<ij> x_i x_j over the 4-neighbour torus, each bond counted twice.

    spins: (n, D*D) array of +/-1. Returns (n,). With this convention the
    cluster-expansion total energy is E = -sigma * S_ord (checked vs ce.predict).
    """
    g = spins.reshape(-1, D, D)
    s = (g * np.roll(g, 1, axis=1)).sum(axis=(1, 2))
    s += (g * np.roll(g, -1, axis=1)).sum(axis=(1, 2))
    s += (g * np.roll(g, 1, axis=2)).sum(axis=(1, 2))
    s += (g * np.roll(g, -1, axis=2)).sum(axis=(1, 2))
    return s


def _wmean(vals: np.ndarray, w: np.ndarray) -> float:
    return float((w * vals).sum())


def _wstd(vals: np.ndarray, w: np.ndarray, mean: float) -> float:
    return float(np.sqrt((w * (vals - mean) ** 2).sum()))


def dnfs_observables(run_dir: Path, D: int, sigma: float, n_boot: int, rng,
                     eval_dir: str = "eval"):
    """Self-normalised IS estimates + bootstrap of (c_mean, c_std, E/site, SRO)."""
    d = D * D
    spins = torch.load(run_dir / eval_dir / "samples.pt", weights_only=True).float().numpy()
    spins = spins.reshape(spins.shape[0], -1)
    logw = torch.load(run_dir / eval_dir / "log_weights.pt", weights_only=True).double().numpy().ravel()
    n = spins.shape[0]

    c = ((spins + 1.0) * 0.5).mean(axis=1)            # fraction of +1 spins
    s_ord = _nn_sum_ordered(spins, D).astype(float)
    e_site = -sigma * s_ord / d                        # CE energy per site
    sro = s_ord / (4.0 * d)                            # <x_i x_j> over undirected bonds

    def _est(idx):
        lw = logw[idx]
        w = np.exp(lw - lw.max())
        w /= w.sum()
        cm = _wmean(c[idx], w)
        return dict(c_mean=cm, c_std=_wstd(c[idx], w, cm),
                    e_site=_wmean(e_site[idx], w), sro=_wmean(sro[idx], w))

    point = _est(np.arange(n))
    boots = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e = _est(idx)
        for k in point:
            boots[k][b] = e[k]
    return point, {k: float(boots[k].std()) for k in point}


def vcsgc_observables(D: int, sigma: float, c_target: float, lam: float,
                      n_steps: int, seeds):
    """Native VCSGCEnsemble reference at kappa=lam, phi=-2*c_target, kT=1.

    Chains are delegated to `mchammer_ising.run_vcsgc`, which keeps the same
    on-target init, 1/3 burn-in and write interval this function used when the
    loop was inline. Returns per-observable (mean over seeds, between-seed
    std). c_std is the within-chain composition spread averaged over seeds.
    """
    d = D * D
    per_seed = {k: [] for k in ("c_mean", "c_std", "e_site", "sro")}
    for s in seeds:
        traces = run_vcsgc(D=D, sigma=sigma, penalty_strength=lam,
                           target_composition=c_target, n_steps=n_steps,
                           seed=s)["traces"]
        c = traces["composition"]
        pot = traces["potential"]                       # total CE energy
        per_seed["c_mean"].append(c.mean())
        per_seed["c_std"].append(c.std())
        per_seed["e_site"].append((pot / d).mean())
        per_seed["sro"].append((-pot / sigma / (4.0 * d)).mean())
    point = {k: float(np.mean(v)) for k, v in per_seed.items()}
    err = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
           for k, v in per_seed.items()}
    return point, err


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path, default=Path("results/02_constrained_soft"))
    p.add_argument("--configs", nargs="+", required=True,
                   help="config-name stems (without _seed..); one per window")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    p.add_argument("--ess_floor", type=float, default=0.30)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--vcsgc_seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--vcsgc_steps", type=int, default=300_000)
    p.add_argument("--plot", type=Path, default=None)
    p.add_argument("--flag_c", nargs="*", type=float, default=[],
                   help="compositions drawn with a provisional ring (their Z2 "
                        "mirrors inherit it); used while a window awaits retrain "
                        "or prints from a different training grid")
    p.add_argument("--eval_dir", choices=["eval", "eval_ema"], default="eval",
                   help="which frozen eval to score: raw weights or the s95 "
                        "dual eval's EMA shadow draw")
    args = p.parse_args()
    rng = np.random.default_rng(0)

    # --- collect DNFS runs, group by composition ------------------------
    metas, missing = [], []
    for config in args.configs:
        for seed in args.seeds:
            rd = latest_run_dir(args.results_dir, config, seed, args.eval_dir)
            (metas if rd is not None else missing).append(
                _load_meta(rd, args.eval_dir) if rd is not None
                else f"{config} seed{seed}")
    if missing:
        print(f"[warn] no eval found for: {', '.join(missing)}")
    if not metas:
        raise SystemExit("no runs found; check --results_dir / --configs / --seeds")

    Ds = sorted({m["D"] for m in metas})
    lams = sorted({m["lam"] for m in metas})
    sigmas = sorted({m["sigma"] for m in metas})
    if len(Ds) > 1 or len(lams) > 1 or len(sigmas) > 1:
        raise SystemExit(f"expected one (D, sigma, lambda); got D={Ds} sigma={sigmas} lambda={lams}")
    D, sigma, lam = Ds[0], sigmas[0], lams[0]
    d = D * D
    analytic_cstd = 1.0 / np.sqrt(2.0 * lam * d)
    print(f"=== weighted thermo: DNFS soft vs vcSGC (kappa=lambda={lam:g}) : "
          f"D={D} sigma={sigma} kT=1 ess_floor={args.ess_floor} ===")
    print(f"analytic composition width 1/sqrt(2*lambda*d) = {analytic_cstd:.4f}")
    print(f"vcSGC: {len(args.vcsgc_seeds)} seeds x {args.vcsgc_steps} steps, burn-in 1/3\n")

    by_c: dict[float, list[dict]] = {}
    for m in metas:
        by_c.setdefault(round(m["c_target"], 4), []).append(m)

    hdr = (f"{'c_t':>6} {'src':>6} {'gated':>6} "
           f"{'<c>':>16} {'std(c)':>16} {'E/site':>18} {'SRO':>16}")
    print(hdr)
    print("-" * len(hdr))

    curve = []
    for c_t in sorted(by_c):
        rows = by_c[c_t]
        gated = [m for m in rows if m["ess_frac"] >= args.ess_floor]
        excluded = [m for m in rows if m["ess_frac"] < args.ess_floor]

        # --- vcSGC reference -------------------------------------------
        v_pt, v_err = vcsgc_observables(D, sigma, c_t, lam, args.vcsgc_steps,
                                        args.vcsgc_seeds)
        print(f"{c_t:>6.3f} {'vcSGC':>6} {len(args.vcsgc_seeds):>2} seeds "
              f"{v_pt['c_mean']:>8.4f}+/-{v_err['c_mean']:<6.4f} "
              f"{v_pt['c_std']:>8.4f}+/-{v_err['c_std']:<6.4f} "
              f"{v_pt['e_site']:>9.4f}+/-{v_err['e_site']:<7.4f} "
              f"{v_pt['sro']:>8.4f}+/-{v_err['sro']:<6.4f}")

        # --- DNFS soft (ESS-gated, seed-averaged) ----------------------
        if not gated:
            ess_lo = min(m["ess_frac"] for m in rows)
            ess_hi = max(m["ess_frac"] for m in rows)
            print(f"{'':>6} {'DNFS':>6} {0:>2}/{len(rows):<3} "
                  f"(all seeds below ESS floor {ess_lo:.3f}-{ess_hi:.3f})\n")
            curve.append(dict(c=c_t, vcsgc=v_pt, vcsgc_err=v_err, dnfs=None))
            continue

        pts = {k: [] for k in ("c_mean", "c_std", "e_site", "sro")}
        within = {k: [] for k in pts}
        for m in gated:
            pt, er = dnfs_observables(m["run_dir"], D, sigma, args.n_boot, rng,
                                      args.eval_dir)
            for k in pts:
                pts[k].append(pt[k])
                within[k].append(er[k])
        d_pt = {k: float(np.mean(v)) for k, v in pts.items()}
        d_err = {}
        for k in pts:
            w = float(np.mean(within[k]))
            b = float(np.std(pts[k], ddof=1) / np.sqrt(len(pts[k]))) if len(pts[k]) > 1 else 0.0
            d_err[k] = float(np.hypot(w, b))

        exc = ",".join(f"{seed_of(m['name'])}:{m['ess_frac']:.2f}" for m in excluded)
        print(f"{'':>6} {'DNFS':>6} {len(gated):>2}/{len(rows):<3} "
              f"{d_pt['c_mean']:>8.4f}+/-{d_err['c_mean']:<6.4f} "
              f"{d_pt['c_std']:>8.4f}+/-{d_err['c_std']:<6.4f} "
              f"{d_pt['e_site']:>9.4f}+/-{d_err['e_site']:<7.4f} "
              f"{d_pt['sro']:>8.4f}+/-{d_err['sro']:<6.4f}"
              + (f"  excluded {exc}" if exc else ""))
        # gap line: DNFS - vcSGC on each observable
        print(f"{'':>6} {'Delta':>6} {'':>6} "
              f"{d_pt['c_mean'] - v_pt['c_mean']:>+8.4f}{'':>8} "
              f"{d_pt['c_std'] - v_pt['c_std']:>+8.4f}{'':>8} "
              f"{d_pt['e_site'] - v_pt['e_site']:>+9.4f}{'':>9} "
              f"{d_pt['sro'] - v_pt['sro']:>+8.4f}\n")
        curve.append(dict(c=c_t, vcsgc=v_pt, vcsgc_err=v_err,
                          dnfs=d_pt, dnfs_err=d_err))

    if args.plot is not None:
        _plot(curve, lam, analytic_cstd, args.flag_c, args.plot)


def _zmirror(have: list[dict], key: str) -> list[tuple]:
    """Z_2 reflections of the sampled windows for one observable.

    The global spin flip x->-x maps composition c->1-c. It leaves the zero-field
    energy, the NN short-range order and the composition width invariant, and
    sends the mean composition to 1-<c>. So a window trained at c_t supplies a
    symmetry-implied point at 1-c_t: the same value for std/energy/SRO, and
    1-value for the mean. This fills the RHS-heavy sampled grid
    ({0.30,0.50,0.55,0.60,0.65}) on the left. Skip c=0.5 and any reflection that
    lands on an already-sampled window. Returns (c, vc, vc_err, dn, dn_err)
    tuples reusing the source window's errors; these are symmetry-implied, not
    independently sampled, but they are drawn identically to the sampled windows
    — the filled/open marker split confused more than it informed (reader
    feedback 2026-08-11), so the reflection is stated once in the dissertation
    body text instead of per-marker.
    """
    sampled = {round(r["c"], 4) for r in have}
    flip = (lambda v: 1.0 - v) if key == "c_mean" else (lambda v: v)
    out = []
    for r in have:
        cm = round(1.0 - r["c"], 4)
        if abs(r["c"] - 0.5) < 1e-6 or cm in sampled:
            continue
        out.append((cm, flip(r["vcsgc"][key]), r["vcsgc_err"][key],
                    flip(r["dnfs"][key]), r["dnfs_err"][key]))
    return out


def _plot(curve, lam, analytic_cstd, flag_c, out: Path) -> None:
    """House-standard 1x4 row (was a 2x2; relayout 2026-08-29).

    The 2x2 printed 14.1 cm tall at \\textwidth, about half a page, for four
    panels that each carry eleven marks. The same four panels in a row print
    6.4 cm. An earlier 1x4 attempt failed because it was drawn 18 in wide and
    left to LaTeX to shrink, which put the type at ~3 pt; here the figure
    stays at the house 6.3 in and the PANELS get narrow instead, so 9 pt on
    the page is still 9 pt. Panel order (a)-(d) is unchanged.

    Roles: VC-SGC chains = CLASSICAL_HUE (the classical comparator, not the
    ink truth -- these are matched chains, not TI); our sampler =
    SAMPLER_HUE; analytic guides dashed ANALYTIC_GUIDE. At 1.6 in per panel
    an in-axes legend covers the data, so the two series are named once in a
    figure-level legend under the row and each analytic guide is labelled in
    place beside its own line. Compositions in `flag_c` (plus Z2 mirrors) get
    the provisional ring on the sampler series, explained in the caption.

    Error bars, not bands: each abscissa is a separately trained window laid
    against its own reference chain, so a ribbon would draw a continuum in c
    that neither sampler measures (figure_style's uncertainty grammar).
    """
    import matplotlib.pyplot as plt

    from discrete_flow_sampler.diagnostics.figure_style import (
        ANALYTIC_GUIDE, CLASSICAL_HUE, FIGSIZE_FULL_1X4, FONT_SIZE_ANNOTATION,
        MUTED, SAMPLER_HUE, SAVEFIG_DPI, style_axes, use_house_style)

    use_house_style()
    have = [r for r in curve if r["dnfs"] is not None]
    panels = [("c_mean", r"$\langle c\rangle$"),
              ("c_std", r"std$(c)$"),
              ("e_site", r"$E/d$"),
              ("sro", r"$\langle x_i x_j\rangle_{NN}$")]
    flagged = {round(c, 4) for c in flag_c} | {round(1 - c, 4) for c in flag_c}
    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_FULL_1X4)
    for i, (ax, (key, ylab)) in enumerate(zip(axes.ravel(), panels)):
        # sampled + Z_2-reflected points merged into one uniformly-drawn series,
        # sorted by composition (the reflection is stated in the body text)
        mir = _zmirror(have, key)
        pts = sorted(
            [(r["c"], r["vcsgc"][key], r["vcsgc_err"][key],
              r["dnfs"][key], r["dnfs_err"][key]) for r in have] + mir)
        pc = [p[0] for p in pts]
        vc, vce = [p[1] for p in pts], [p[2] for p in pts]
        dn, dne = [p[3] for p in pts], [p[4] for p in pts]
        # both series as discrete markers (no connecting line): the comparison is
        # per-composition agreement at matched windows, not a trend, so a
        # joining line would imply interpolation neither sampler measures.
        # ms 3.5, not the default 6: at 1.6 in per panel a default marker is
        # wider than the gap between neighbouring windows, and the DNFS square
        # then hides the vcSGC circle it is supposed to be compared with.
        ax.errorbar(pc, vc, yerr=vce, fmt="o", ms=3.5, color=CLASSICAL_HUE,
                    capsize=1.5, lw=0.8,
                    label="vcSGC (mchammer)" if i == 0 else None)
        ax.errorbar(pc, dn, yerr=dne, fmt="s", ms=3.5, color=SAMPLER_HUE,
                    capsize=1.5, lw=0.8,
                    label="DNFS soft (IS)" if i == 0 else None)
        ring = [(c, y) for c, y in zip(pc, dn) if round(c, 4) in flagged]
        if ring:
            ax.scatter([c for c, _ in ring], [y for _, y in ring], s=50,
                       facecolors="none", edgecolors=MUTED, linewidths=1.0,
                       zorder=4)
        # Guides are labelled in place: at 1.6 in wide a legend box would sit
        # on top of the eleven marks it is explaining.
        if key == "c_mean":
            ax.plot(pc, pc, ls="--", color=ANALYTIC_GUIDE, lw=0.8)
            ax.text(0.96, 0.06, "$c=c_t$", transform=ax.transAxes, ha="right",
                    color=ANALYTIC_GUIDE, fontsize=FONT_SIZE_ANNOTATION)
        # Three x ticks and at most four y ticks: the axis spans 0.2-0.8 in
        # every panel and a narrow panel cannot carry the default five labels
        # without overprinting.
        ax.set_xticks([0.2, 0.5, 0.8])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if key == "c_std":
            ax.axhline(analytic_cstd, ls="--", color=ANALYTIC_GUIDE, lw=0.8)
            # std(c) is near-constant ~0.010, so autoscale zooms into the noise;
            # pin a +/-0.001 window around the analytic value so the tiny (and
            # expected) IS-vs-chain differences don't dominate the panel. Three
            # explicit ticks (set AFTER the locator, which would otherwise
            # replace them): 0.0090/0.0100/0.0110 are the widest labels in the
            # figure and five of them will not fit a 1.6 in panel.
            ax.set_ylim(analytic_cstd - 0.001, analytic_cstd + 0.001)
            ax.set_yticks([analytic_cstd - 0.001, analytic_cstd,
                           analytic_cstd + 0.001])
            ax.text(0.04, 0.90, r"$1/\sqrt{2\lambda d}$", transform=ax.transAxes,
                    color=ANALYTIC_GUIDE, fontsize=FONT_SIZE_ANNOTATION)
        ax.set_xlabel("composition $c$")
        ax.set_ylabel(ylab)
        style_axes(ax)
        ax.text(0.02, 1.03, f"({chr(97 + i)})", transform=ax.transAxes,
                fontweight="bold", va="bottom")
    # One figure-level legend for the two series, which are shared by all four
    # panels; naming them once was already the 2x2's convention.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    fig.savefig(out, dpi=SAVEFIG_DPI, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
