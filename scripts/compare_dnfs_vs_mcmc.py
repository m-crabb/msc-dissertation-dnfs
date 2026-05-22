"""Overlay DNFS-sampled marginals against MCMC ground truth.

Loads MCMC traces from `vcsgc_mcmc_validation.py` and DNFS eval samples from
one or more training runs at the matching (D, sigma, lam, c_target) cell.
Computes c(x), m(x), and the single-bond E_Ising on the DNFS samples, then
produces a 2x2 comparison plot and a JSON summary.

Default arguments target `S2_d10_c03_l50_letf_ne128_relaunch_seed{42..45}`.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_adjacency(D):
    """Symmetric undirected adjacency for the DxD torus (each edge once in
    both [i, j] and [j, i]). Matches the DNFS IsingTarget construction."""
    d = D * D
    A = np.zeros((d, d), dtype=np.float64)
    for r in range(D):
        for c in range(D):
            i = r * D + c
            A[i, r * D + (c + 1) % D] = 1.0
            A[i, ((r + 1) % D) * D + c] = 1.0
    return A + A.T


def load_dnfs_samples(run_dir):
    x = torch.load(run_dir / "eval/samples.pt", map_location="cpu",
                   weights_only=False).numpy().astype(np.float64)
    lw = torch.load(run_dir / "eval/log_weights.pt", map_location="cpu",
                    weights_only=False).numpy().astype(np.float64)
    return x, lw


def importance_weights(log_w):
    """Self-normalised IS weights (numerically stable)."""
    lw = log_w - log_w.max()
    w = np.exp(lw)
    return w / w.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcmc_dir", default="results/vcsgc_mcmc/D10_s0.1_l50.0_c0.3_seed42")
    p.add_argument("--dnfs_root", default="results/02_constrained_soft")
    p.add_argument("--dnfs_run_pattern", default="S2_d10_c03_l50_letf_ne128_relaunch_seed{seed}")
    p.add_argument("--dnfs_seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=50.0)
    p.add_argument("--c_target", type=float, default=0.3)
    p.add_argument("--out", default="results/vcsgc_mcmc/comparison_l50_relaunch")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcmc_dir = Path(args.mcmc_dir)
    c_mcmc = np.load(mcmc_dir / "composition.npy").ravel()
    m_mcmc = np.load(mcmc_dir / "magnetisation.npy").ravel()
    E_mcmc = np.load(mcmc_dir / "E_ising.npy").ravel()

    A = build_adjacency(args.D)

    dnfs = {}
    for seed in args.dnfs_seeds:
        run_dir = Path(args.dnfs_root) / args.dnfs_run_pattern.format(seed=seed)
        x, lw = load_dnfs_samples(run_dir)
        c = ((x + 1.0) * 0.5).mean(axis=1)
        m = x.mean(axis=1)
        # E_Ising single-bond = -sigma * sum_<i,j> x_i x_j = -(sigma/2) * x^T A x.
        E = -0.5 * args.sigma * np.einsum("ni,ij,nj->n", x, A, x)
        w = importance_weights(lw)
        ess = 1.0 / (w * w).sum()
        dnfs[seed] = dict(
            c=c, m=m, E=E, w=w, ess=ess,
            c_mean=c.mean(), c_std=c.std(),
            m_mean=m.mean(), m_std=m.std(),
            E_mean=E.mean(), E_std=E.std(),
            c_mean_w=float((w * c).sum()),
            c_std_w=float(np.sqrt((w * (c - (w * c).sum()) ** 2).sum())),
            E_mean_w=float((w * E).sum()),
            E_std_w=float(np.sqrt((w * (E - (w * E).sum()) ** 2).sum())),
        )

    print("\n=== Marginal stats: DNFS vs MCMC ===")
    print(f"{'source':<22} {'c mean':>9} {'c std':>9} "
          f"{'m mean':>9} {'m std':>9} {'E mean':>9} {'E std':>9} {'ESS':>7}")
    print(f"{'MCMC (pool of 4)':<22} {c_mcmc.mean():>9.4f} {c_mcmc.std():>9.4f} "
          f"{m_mcmc.mean():>9.4f} {m_mcmc.std():>9.4f} "
          f"{E_mcmc.mean():>9.4f} {E_mcmc.std():>9.4f} {len(c_mcmc):>7d}")
    for seed, d_ in dnfs.items():
        print(f"{'DNFS s' + str(seed) + ' (raw)':<22} "
              f"{d_['c_mean']:>9.4f} {d_['c_std']:>9.4f} "
              f"{d_['m_mean']:>9.4f} {d_['m_std']:>9.4f} "
              f"{d_['E_mean']:>9.4f} {d_['E_std']:>9.4f} "
              f"{int(d_['ess']):>7d}")
    print()
    print(f"{'source':<22} {'c mean_w':>9} {'c std_w':>9} "
          f"{'E mean_w':>9} {'E std_w':>9}     (IS-corrected)")
    for seed, d_ in dnfs.items():
        print(f"{'DNFS s' + str(seed) + ' (IS)':<22} "
              f"{d_['c_mean_w']:>9.4f} {d_['c_std_w']:>9.4f} "
              f"{d_['E_mean_w']:>9.4f} {d_['E_std_w']:>9.4f}")

    summary = {
        "config": vars(args),
        "mcmc": {
            "c_mean": float(c_mcmc.mean()), "c_std": float(c_mcmc.std()),
            "m_mean": float(m_mcmc.mean()), "m_std": float(m_mcmc.std()),
            "E_mean": float(E_mcmc.mean()), "E_std": float(E_mcmc.std()),
            "n_samples": int(len(c_mcmc)),
        },
        "dnfs": {
            str(seed): {
                k: float(v) for k, v in d_.items()
                if k not in ("c", "m", "E", "w")
            }
            for seed, d_ in dnfs.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    colors = plt.cm.tab10(np.arange(len(args.dnfs_seeds)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    bins_c = np.linspace(0.24, 0.38, 30)
    ax.hist(c_mcmc, bins=bins_c, density=True, alpha=0.35,
            color="black", label=f"MCMC (n={len(c_mcmc)})")
    for (seed, d_), col in zip(dnfs.items(), colors):
        ax.hist(d_["c"], bins=bins_c, density=True, histtype="step",
                linewidth=1.6, color=col, label=f"DNFS s{seed}")
    ax.axvline(args.c_target, color="red", ls="--", lw=1, alpha=0.6,
               label=f"c_target={args.c_target}")
    ax.set_xlabel("composition c(x)")
    ax.set_ylabel("density")
    ax.set_title("composition marginal")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    E_lo = min(E_mcmc.min(), *(d_["E"].min() for d_ in dnfs.values()))
    E_hi = max(E_mcmc.max(), *(d_["E"].max() for d_ in dnfs.values()))
    bins_E = np.linspace(E_lo, E_hi, 40)
    ax.hist(E_mcmc, bins=bins_E, density=True, alpha=0.35,
            color="black", label="MCMC")
    for (seed, d_), col in zip(dnfs.items(), colors):
        ax.hist(d_["E"], bins=bins_E, density=True, histtype="step",
                linewidth=1.6, color=col, label=f"DNFS s{seed}")
    ax.set_xlabel("E_Ising (single-bond convention)")
    ax.set_ylabel("density")
    ax.set_title("Ising energy marginal")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins_m = np.linspace(-0.52, -0.30, 30)
    ax.hist(m_mcmc, bins=bins_m, density=True, alpha=0.35,
            color="black", label="MCMC")
    for (seed, d_), col in zip(dnfs.items(), colors):
        ax.hist(d_["m"], bins=bins_m, density=True, histtype="step",
                linewidth=1.6, color=col, label=f"DNFS s{seed}")
    ax.axvline(2 * args.c_target - 1, color="red", ls="--", lw=1, alpha=0.6,
               label=f"m@c_target={2*args.c_target-1:+.2f}")
    ax.set_xlabel("magnetisation m(x)")
    ax.set_ylabel("density")
    ax.set_title("magnetisation marginal")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.errorbar(c_mcmc.mean(), E_mcmc.mean(),
                xerr=c_mcmc.std(), yerr=E_mcmc.std(),
                fmt="s", color="black", markersize=9, capsize=5,
                label="MCMC", zorder=5)
    for (seed, d_), col in zip(dnfs.items(), colors):
        ax.errorbar(d_["c_mean"], d_["E_mean"],
                    xerr=d_["c_std"], yerr=d_["E_std"],
                    fmt="o", color=col, markersize=7, capsize=4,
                    label=f"DNFS s{seed}")
    ax.axvline(args.c_target, color="red", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("composition mean ± 1σ")
    ax.set_ylabel("E_Ising mean ± 1σ")
    ax.set_title("(c, E_Ising) joint summary")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"DNFS vs MCMC  |  {args.dnfs_run_pattern.format(seed='[42..45]')}  "
        f"|  D={args.D}, σ={args.sigma}, λ={args.lam}, c_target={args.c_target}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote: {out_dir}/comparison.png  and  summary.json")


if __name__ == "__main__":
    main()
