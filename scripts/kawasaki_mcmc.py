"""Single-(D, σ) Kawasaki driver for the §3.1 hard-constraint failure demo.

Runs multiple composition-preserving chains, reports τ_int / ESS / R̂, and
writes a per-run summary + diagnostics plot. The composition is fixed by
construction (swap moves), so unlike the soft VCSGC driver there is no
composition trace to monitor — instead a hard assert verifies conservation.

Example:
    pixi run python scripts/kawasaki_mcmc.py --D 16 --sigma 0.3 --n_chains 4
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discrete_flow_sampler.diagnostics.metrics import (
    gelman_rubin,
    integrated_autocorr,
)
from discrete_flow_sampler.mcmc.kawasaki import (
    init_random_at_composition,
    run_chain,
)


def run_config(D, sigma, c_target, n_steps, n_burn, thin, n_chains, seed):
    """Run n_chains Kawasaki chains. Returns (E, finals, accepts, tau, ess, rhat).

    E:       (n_chains, n_kept) post-burn-in thinned energy traces
    finals:  (n_chains, d) final lattice configs
    accepts: (n_chains,) acceptance rates
    tau:     (n_chains,) τ_int in raw swap steps
    ess:     (n_chains,) effective sample size
    rhat:    scalar Gelman-Rubin R̂ on the energy traces
    """
    d = D * D
    energy_chains, finals, accepts = [], [], []
    for chain in range(n_chains):
        s = seed + chain
        rng = np.random.default_rng(s)
        x = init_random_at_composition(d, c_target, rng)
        n_plus0 = int((x == 1).sum())
        t0 = time.time()
        energy_trace, x_final, n_accept = run_chain(x, D, sigma, n_steps, s)
        assert int((x_final == 1).sum()) == n_plus0, "composition not conserved!"
        energy_chains.append(energy_trace[n_burn::thin])
        finals.append(x_final.copy())
        accepts.append(n_accept / n_steps)
        print(f"[chain {chain}] acc={accepts[-1]:.3f} elapsed={time.time() - t0:.1f}s")
    E = np.stack(energy_chains)
    tau = np.array([integrated_autocorr(E[i]) for i in range(n_chains)]) * thin
    ess = (n_steps - n_burn) / np.where(tau > 0, tau, np.inf)
    rhat = gelman_rubin(E)
    return E, np.stack(finals), np.array(accepts), tau, ess, rhat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--c_target", type=float, default=0.5)
    p.add_argument("--n_steps", type=int, default=2_000_000)
    p.add_argument("--n_burn", type=int, default=200_000)
    p.add_argument("--thin", type=int, default=50)
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/kawasaki")
    args = p.parse_args()

    out_dir = (
        Path(args.out) / f"D{args.D}_s{args.sigma}_c{args.c_target}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    E, finals, accepts, tau, ess, rhat = run_config(
        args.D,
        args.sigma,
        args.c_target,
        args.n_steps,
        args.n_burn,
        args.thin,
        args.n_chains,
        args.seed,
    )
    summary = {
        "config": vars(args),
        "acceptance_mean": float(accepts.mean()),
        "tau_int_raw_steps": tau.tolist(),
        "ess_per_chain": ess.tolist(),
        "rhat_energy": float(rhat),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(E.shape[1]) * args.thin + args.n_burn
    for i in range(args.n_chains):
        ax[0].plot(xs, E[i], lw=0.7, alpha=0.8, label=f"chain {i}")
    ax[0].set_xlabel("swap step")
    ax[0].set_ylabel("log_prob_ising")
    ax[0].set_title(f"energy traces  (R̂={rhat:.2f})")
    ax[0].legend(fontsize=8)
    ax[1].hist(E.ravel(), bins=60, density=True, alpha=0.85)
    ax[1].set_xlabel("log_prob_ising")
    ax[1].set_ylabel("density")
    ax[1].set_title("pooled energy (post-burn-in)")
    fig.suptitle(
        f"Kawasaki  D={args.D}, σ={args.sigma}, c={args.c_target}  "
        f"τ_int≈{tau.mean():.0f}  ESS≈{ess.mean():.0f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=120)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
