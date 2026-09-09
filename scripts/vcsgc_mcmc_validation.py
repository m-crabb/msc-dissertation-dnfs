"""VCSGC-Metropolis MCMC for soft-constrained 2D Ising on a torus.

Direct comparison reference for the DNFS soft-constraint sampler. The target
matches `src/discrete_flow_sampler/targets/ising.py` exactly:

    log p(x) = sigma * x^T A x  -  lambda * d * (c(x) - c_target)^2

where A is the symmetric undirected adjacency with A[i,j] = A[j,i] = 1 for
each nearest-neighbour pair (so x^T A x = 2 * sum_<i,j> x_i x_j). With this
convention sigma_critical for the 2D Ising at unit beta is ~0.22, not 0.44.

The MCMC accept ratio is computed in log_prob space to match the DNFS target
bit-for-bit. The Ising energy reported in traces uses the single-bond
physical convention (E_Ising = -sigma * sum_<i,j> x_i x_j) regardless.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def neighbour_sum(x, i, D):
    """Sum of x over the 4 nearest neighbours of site i on a DxD torus."""
    r = i // D
    c = i % D
    up = ((r - 1) % D) * D + c
    down = ((r + 1) % D) * D + c
    left = r * D + (c - 1) % D
    right = r * D + (c + 1) % D
    return x[up] + x[down] + x[left] + x[right]


@njit(cache=True)
def initial_ising_energy(x, D, sigma):
    """E_Ising in single-bond convention: -sigma * sum_<i,j> x_i x_j."""
    E = 0.0
    for r in range(D):
        for c in range(D):
            i = r * D + c
            right = r * D + (c + 1) % D
            down = ((r + 1) % D) * D + c
            E -= sigma * x[i] * x[right]
            E -= sigma * x[i] * x[down]
    return E


@njit(cache=True)
def run_chain(x, D, sigma, lam, c_target, n_steps, seed):
    """Single-spin-flip Metropolis. Accept ratio in DNFS log_prob convention.

    Δ log_prob_ising = -4 * sigma * x_i * sum_{j∈N(i)} x_j
                       (because log_prob_ising = 2*sigma*sum_<i,j> x_i x_j,
                        and flipping x_i changes 4 of its incident bonds.)
    Δ log_prob_pen   = -lambda * d * [ (c_new - c_target)^2 - (c - c_target)^2 ]
    Accept with prob min(1, exp(Δ log_prob)).
    """
    d = D * D
    np.random.seed(seed)

    n_plus = 0
    for i in range(d):
        if x[i] == 1:
            n_plus += 1
    c = n_plus / d
    E_ising = initial_ising_energy(x, D, sigma)

    E_trace = np.empty(n_steps, dtype=np.float64)
    m_trace = np.empty(n_steps, dtype=np.float64)
    c_trace = np.empty(n_steps, dtype=np.float64)
    n_accept = 0

    for step in range(n_steps):
        i = np.random.randint(d)
        nsum = neighbour_sum(x, i, D)

        # Change in single-bond E_Ising from flipping site i.
        delta_E_ising = 2.0 * sigma * x[i] * nsum

        # Composition: flipping x_i changes n_plus by -x_i, so Δc = -x_i / d.
        delta_c = -x[i] / d
        c_new = c + delta_c
        diff_old = c - c_target
        diff_new = c_new - c_target
        delta_pen = lam * d * (diff_new * diff_new - diff_old * diff_old)

        # Δ log_prob in DNFS convention. Note log_prob_ising change is
        # -2 * delta_E_ising (because DNFS doubles bonds and is +sign-flipped).
        delta_log_prob = -2.0 * delta_E_ising - delta_pen

        if delta_log_prob >= 0.0 or np.random.random() < np.exp(delta_log_prob):
            x[i] = -x[i]
            E_ising += delta_E_ising
            c = c_new
            n_accept += 1

        E_trace[step] = E_ising
        c_trace[step] = c
        m_trace[step] = 2.0 * c - 1.0

    return E_trace, m_trace, c_trace, n_accept


def integrated_autocorr(x, c_window=5.0):
    """Sokal automatic-windowing estimator.

    τ_int = 1 + 2 Σ_{k=1}^{W} ρ(k), with W the smallest integer such that
    W >= c_window * τ_int(W). Uses FFT for the autocovariance.
    """
    n = len(x)
    x = x - x.mean()
    var0 = np.dot(x, x) / n
    if var0 == 0:
        return 1.0
    n2 = 1
    while n2 < 2 * n:
        n2 *= 2
    f = np.fft.fft(x, n=n2)
    acov = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    acf = acov / (var0 * np.arange(n, 0, -1))

    tau = 1.0
    for W in range(1, n):
        tau += 2.0 * acf[W]
        if W >= c_window * tau:
            return tau
    return tau


def init_random_at_composition(d, c_target, rng):
    """±1 array of length d with exactly round(c_target * d) +1 sites."""
    n_plus = max(0, min(d, int(round(c_target * d))))
    x = -np.ones(d, dtype=np.int64)
    idx = rng.choice(d, size=n_plus, replace=False)
    x[idx] = 1
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=10, help="grid side length; d = D*D")
    p.add_argument(
        "--sigma", type=float, default=0.1, help="Ising coupling (DNFS convention)"
    )
    p.add_argument("--lam", type=float, default=50.0, help="soft-constraint strength")
    p.add_argument("--c_target", type=float, default=0.3)
    p.add_argument("--n_steps", type=int, default=1_000_000)
    p.add_argument("--n_burn", type=int, default=100_000)
    p.add_argument("--thin", type=int, default=100)
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/vcsgc_mcmc")
    p.add_argument("--skip_sanity", action="store_true")
    args = p.parse_args()

    d = args.D * args.D
    out_dir = Path(args.out) / (
        f"D{args.D}_s{args.sigma}_l{args.lam}_c{args.c_target}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"VCSGC-Metropolis: D={args.D} (d={d}), sigma={args.sigma}, "
        f"lambda={args.lam}, c_target={args.c_target}"
    )
    print(
        f"  n_steps={args.n_steps}, n_burn={args.n_burn}, "
        f"thin={args.thin}, n_chains={args.n_chains}, seed={args.seed}"
    )
    print(f"  numba: {'on' if HAS_NUMBA else 'OFF (pure-python fallback, slow)'}")

    if not args.skip_sanity:
        print(
            "\n[sanity] lambda=0 short run (n=50_000) — expect |m| ≈ 0 at "
            f"sigma={args.sigma} < sigma_c≈0.22 (paramagnetic)."
        )
        rng_s = np.random.default_rng(args.seed + 999)
        x_san = init_random_at_composition(d, 0.5, rng_s)
        t0 = time.time()
        _, m_san, _, _ = run_chain(
            x_san, args.D, args.sigma, 0.0, 0.5, 50_000, args.seed + 999
        )
        m_sanity = m_san[10_000:].mean()
        msg = f"[sanity] mean m = {m_sanity:+.4f}  (elapsed {time.time() - t0:.1f}s)"
        if abs(m_sanity) > 0.3:
            print(msg + "  WARNING: |m| > 0.3, suspect sign-convention bug.")
        else:
            print(msg + "  OK.")

    accept_rates, E_chains, m_chains, c_chains = [], [], [], []
    for chain in range(args.n_chains):
        seed_c = args.seed + chain
        rng_c = np.random.default_rng(seed_c)
        x = init_random_at_composition(d, args.c_target, rng_c)
        t0 = time.time()
        E_tr, m_tr, c_tr, n_acc = run_chain(
            x, args.D, args.sigma, args.lam, args.c_target, args.n_steps, seed_c
        )
        elapsed = time.time() - t0
        acc = n_acc / args.n_steps
        accept_rates.append(acc)
        post = slice(args.n_burn, args.n_steps, args.thin)
        E_chains.append(E_tr[post])
        m_chains.append(m_tr[post])
        c_chains.append(c_tr[post])
        print(f"[chain {chain}] seed={seed_c}  acc={acc:.3f}  elapsed={elapsed:.1f}s")

    E_arr = np.stack(E_chains)
    m_arr = np.stack(m_chains)
    c_arr = np.stack(c_chains)
    np.save(out_dir / "E_ising.npy", E_arr)
    np.save(out_dir / "magnetisation.npy", m_arr)
    np.save(out_dir / "composition.npy", c_arr)

    # τ_int on the thinned trace, then multiply by `thin` to get raw-step units.
    tau_E = (
        np.array([integrated_autocorr(E_arr[i]) for i in range(args.n_chains)])
        * args.thin
    )
    tau_m = (
        np.array([integrated_autocorr(m_arr[i]) for i in range(args.n_chains)])
        * args.thin
    )
    n_post = args.n_steps - args.n_burn
    ess_E = n_post / np.where(tau_E > 0, tau_E, np.inf)
    ess_m = n_post / np.where(tau_m > 0, tau_m, np.inf)

    E_flat, m_flat, c_flat = E_arr.ravel(), m_arr.ravel(), c_arr.ravel()
    summary = {
        "config": vars(args),
        "acceptance_rate_mean": float(np.mean(accept_rates)),
        "acceptance_per_chain": [float(a) for a in accept_rates],
        "E_ising": {
            "mean": float(E_flat.mean()),
            "std": float(E_flat.std()),
            "tau_int_raw_steps": tau_E.tolist(),
            "ess_per_chain": ess_E.tolist(),
        },
        "magnetisation": {
            "mean": float(m_flat.mean()),
            "std": float(m_flat.std()),
            "tau_int_raw_steps": tau_m.tolist(),
            "ess_per_chain": ess_m.tolist(),
        },
        "composition": {
            "mean": float(c_flat.mean()),
            "std": float(c_flat.std()),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (pooled across chains, post-burn-in, thinned) ===")
    print(f"acceptance rate : {summary['acceptance_rate_mean']:.3f}")
    print(
        f"E_ising         : mean={summary['E_ising']['mean']:+.4f}  "
        f"std={summary['E_ising']['std']:.4f}"
    )
    print(f"                  τ_int (raw steps): {[f'{t:.0f}' for t in tau_E]}")
    print(f"                  ESS              : {[f'{e:.0f}' for e in ess_E]}")
    print(
        f"magnetisation   : mean={summary['magnetisation']['mean']:+.4f}  "
        f"std={summary['magnetisation']['std']:.4f}"
    )
    print(f"                  τ_int (raw steps): {[f'{t:.0f}' for t in tau_m]}")
    print(f"                  ESS              : {[f'{e:.0f}' for e in ess_m]}")
    print(
        f"composition     : mean={summary['composition']['mean']:.4f}  "
        f"std={summary['composition']['std']:.4f}  "
        f"(c_target={args.c_target})"
    )

    n_kept = c_arr.shape[1]
    stride = max(1, n_kept // 5000)
    xs = np.arange(0, n_kept, stride) * args.thin + args.n_burn

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for i in range(args.n_chains):
        axes[0, 0].plot(xs, c_arr[i, ::stride], alpha=0.7, lw=0.8, label=f"chain {i}")
    axes[0, 0].axhline(
        args.c_target, color="k", ls="--", lw=1, label=f"c_target={args.c_target}"
    )
    axes[0, 0].set_xlabel("MCMC step")
    axes[0, 0].set_ylabel("composition c(x)")
    axes[0, 0].set_title("composition trace")
    axes[0, 0].legend(fontsize=8, loc="best")

    for i in range(args.n_chains):
        axes[0, 1].plot(xs, E_arr[i, ::stride], alpha=0.7, lw=0.8)
    axes[0, 1].set_xlabel("MCMC step")
    axes[0, 1].set_ylabel("E_Ising (single-bond convention)")
    axes[0, 1].set_title("Ising energy trace")

    axes[1, 0].hist(m_flat, bins=60, density=True, alpha=0.85)
    axes[1, 0].axvline(
        2 * args.c_target - 1,
        color="k",
        ls="--",
        lw=1,
        label=f"m at c_target = {2 * args.c_target - 1:+.2f}",
    )
    axes[1, 0].set_xlabel("magnetisation m(x)")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("magnetisation (pooled, post-burn-in)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(c_flat, bins=60, density=True, alpha=0.85)
    axes[1, 1].axvline(
        args.c_target, color="k", ls="--", lw=1, label=f"c_target = {args.c_target}"
    )
    axes[1, 1].set_xlabel("composition c(x)")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("composition (pooled, post-burn-in)")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"VCSGC-Metropolis  |  D={args.D}, σ={args.sigma}, "
        f"λ={args.lam}, c_target={args.c_target}, n_chains={args.n_chains}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=120)
    plt.close(fig)

    print(f"\nwrote: {out_dir}")


if __name__ == "__main__":
    main()
