"""DNFS soft vs mchammer vcSGC at matched kappa=lambda: weighted thermodynamics.

This is the apples-to-apples DNFS-vs-vcSGC comparison the F(c) overlay
(`08_fc_compare.py`) deliberately left off its plot. Both samplers target the SAME
semigrand object here: the soft/penalised 2D Ising at penalty strength lambda is
exactly mchammer's variance-constrained semigrand-canonical (vcSGC) ensemble at
kappa = lambda, phi_1 = -2*c_target (pinned in the 2026-06-13 spike,
scripts/icet_vcsgc_spike.py). So at each window we can lay DNFS's
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

Example:
    python -m experiments.constrained_soft_02.analysis.09_fc_weighted_thermo \
        --results_dir results/02_constrained_soft \
        --configs S2_d10_c030_l50_letf_ne64_anneal \
                  S2_d10_c05_l50_letf_ne64_anneal \
                  S2_d10_c055_l50_letf_ne64_anneal \
                  S2_d10_c060_l50_letf_ne64_anneal \
                  S2_d10_c065_l50_letf_ne64_anneal \
        --seeds 42 43 44 45 --ess_floor 0.30 \
        --vcsgc_seeds 0 1 2 --vcsgc_steps 300000 \
        --plot results/02_constrained_soft/fc_weighted_thermo_ne64_anneal.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.units import kB
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble

from scripts.icet_vcsgc_spike import ising_cluster_expansion


# --------------------------------------------------------------------------- #
# run discovery (cloned from 08_fc_compare.py; helpers aren't shared by house
# convention in this analysis dir)
# --------------------------------------------------------------------------- #
def _latest_run_dir(results_dir: Path, config: str, seed: int) -> Path | None:
    matches = set(results_dir.glob(f"{config}_seed{seed}_*"))
    bare = results_dir / f"{config}_seed{seed}"
    if bare.exists():
        matches.add(bare)
    matches = sorted(m for m in matches if (m / "eval" / "metrics.json").exists())
    return matches[-1] if matches else None


def _seed_of(name: str) -> str:
    return name.split("_seed")[1].split("_")[0]


def _load_meta(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    ising = cfg["ising"]
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
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


def dnfs_observables(run_dir: Path, D: int, sigma: float, n_boot: int, rng):
    """Self-normalised IS estimates + bootstrap of (c_mean, c_std, E/site, SRO)."""
    d = D * D
    spins = torch.load(run_dir / "eval" / "samples.pt", weights_only=True).float().numpy()
    spins = spins.reshape(spins.shape[0], -1)
    logw = torch.load(run_dir / "eval" / "log_weights.pt", weights_only=True).double().numpy().ravel()
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
                      n_steps: int, seeds, burn_frac: float = 1.0 / 3.0):
    """Native VCSGCEnsemble reference at kappa=lam, phi=-2*c_target, kT=1.

    Returns per-observable (mean over seeds, between-seed std). c_std is the
    within-chain composition spread averaged over seeds.
    """
    prim, _, ce = ising_cluster_expansion(sigma)
    d = D * D
    per_seed = {k: [] for k in ("c_mean", "c_std", "e_site", "sro")}
    for s in seeds:
        sc = prim.repeat((D, D, 1))
        N = len(sc)
        n_up = int(round(c_target * N))
        syms = ["Au"] * n_up + ["Ag"] * (N - n_up)
        np.random.default_rng(s).shuffle(syms)
        sc.set_chemical_symbols(syms)
        calc = ClusterExpansionCalculator(sc, ce)
        ens = VCSGCEnsemble(sc, calc, temperature=1.0 / kB, kappa=lam,
                            phis={"Au": -2.0 * c_target}, random_seed=s,
                            ensemble_data_write_interval=100)
        ens.run(n_steps)
        df = ens.data_container.data
        keep = df.iloc[int(len(df) * burn_frac):]
        c = keep["Au_count"].values / N
        pot = keep["potential"].values                  # total CE energy
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
    args = p.parse_args()
    rng = np.random.default_rng(0)

    # --- collect DNFS runs, group by composition ------------------------
    metas, missing = [], []
    for config in args.configs:
        for seed in args.seeds:
            rd = _latest_run_dir(args.results_dir, config, seed)
            (metas if rd is not None else missing).append(
                _load_meta(rd) if rd is not None else f"{config} seed{seed}")
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
            pt, er = dnfs_observables(m["run_dir"], D, sigma, args.n_boot, rng)
            for k in pts:
                pts[k].append(pt[k])
                within[k].append(er[k])
        d_pt = {k: float(np.mean(v)) for k, v in pts.items()}
        d_err = {}
        for k in pts:
            w = float(np.mean(within[k]))
            b = float(np.std(pts[k], ddof=1) / np.sqrt(len(pts[k]))) if len(pts[k]) > 1 else 0.0
            d_err[k] = float(np.hypot(w, b))

        exc = ",".join(f"{_seed_of(m['name'])}:{m['ess_frac']:.2f}" for m in excluded)
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
        _plot(curve, lam, analytic_cstd, args.plot)


def _plot(curve, lam, analytic_cstd, out: Path) -> None:
    import matplotlib.pyplot as plt

    have = [r for r in curve if r["dnfs"] is not None]
    cs = [r["c"] for r in have]
    panels = [("c_mean", r"$\langle c\rangle$", "mean composition"),
              ("c_std", r"std$(c)$", "composition width"),
              ("e_site", r"$E/d$", "energy per site"),
              ("sro", r"$\langle x_i x_j\rangle_{NN}$", "NN short-range order")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, ylab, title) in zip(axes, panels):
        vc = [r["vcsgc"][key] for r in have]
        vce = [r["vcsgc_err"][key] for r in have]
        dn = [r["dnfs"][key] for r in have]
        dne = [r["dnfs_err"][key] for r in have]
        ax.errorbar(cs, vc, yerr=vce, fmt="k-o", capsize=3, lw=1.2,
                    label="vcSGC (mchammer)")
        ax.errorbar(cs, dn, yerr=dne, fmt="s", color="tab:blue", capsize=3,
                    label="DNFS soft (IS)")
        if key == "c_mean":
            ax.plot(cs, cs, ":", color="grey", lw=0.8, label="$c=c_t$")
        if key == "c_std":
            ax.axhline(analytic_cstd, ls=":", color="grey", lw=0.8,
                       label=r"$1/\sqrt{2\lambda d}$")
        ax.set_xlabel("composition $c$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"DNFS soft vs vcSGC at matched $\\kappa=\\lambda={lam:g}$", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
