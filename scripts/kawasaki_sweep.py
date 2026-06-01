"""Kawasaki failure-mode sweep → dissertation §3.1 figures.

Produces:
  (1) failure_curves.png — τ_int and ESS vs σ for D∈{10,16,24}, with σ=0.1 (the
      DNFS operating point) and σ_c≈0.223 marked. Critical slowing-down: mixing
      is fine at the subcritical operating point and degrades sharply as σ enters
      the ordered, constrained low-T regime — worsening with system size. The σ
      grid stops at 0.26: past there acceptance → 0 (the chain freezes) and the
      τ_int estimator is no longer reliable; that deep-frozen regime is the
      subject of the ergodicity figure instead.
  (2) mode_coverage.png — DIFFERENT-INITIALISATION ergodicity test. Chains are
      seeded in distinct modes (+domain left vs right) and we track a mode-
      sensitive order parameter φ = left − right sublattice magnetisation. If the
      sampler is ergodic the chains forget their init and φ's between-chain R̂→1;
      if mode coverage fails they stay stuck (R̂≫1). High-σ is the failure, low-σ
      (σ=0.1, the operating point) is the built-in positive control.

Run:  pixi run python -m scripts.kawasaki_sweep
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discrete_flow_sampler.diagnostics.metrics import gelman_rubin
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    run_chain_order_param,
)
from scripts.kawasaki_mcmc import run_config

OUT = Path("results/kawasaki/figures")
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_OPERATING = 0.10       # DNFS paper operating point (subcritical)
SIGMA_CRITICAL = 0.22305
CURVE_SIGMAS = [0.05, 0.10, 0.16, 0.20, 0.22305, 0.26]   # monotonic τ_int regime
ERGO_SIGMAS = [0.05, 0.10, 0.16, 0.20, 0.22305, 0.26, 0.32, 0.40]
DEMO_D = [10, 16, 24]
ERGO_D = 24


def failure_curves():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for D in DEMO_D:
        taus, esss = [], []
        for sigma in CURVE_SIGMAS:
            _, _, _, tau, ess, _ = run_config(
                D, sigma, 0.5, n_steps=1_500_000, n_burn=200_000,
                thin=50, n_chains=4, seed=100,
            )
            taus.append(tau.mean())
            esss.append(ess.mean())
            print(f"D={D} σ={sigma}: τ_int≈{tau.mean():.0f} ESS≈{ess.mean():.0f}")
        ax[0].plot(CURVE_SIGMAS, taus, "o-", label=f"D={D}")
        ax[1].plot(CURVE_SIGMAS, esss, "o-", label=f"D={D}")
    for a in ax:
        a.axvline(SIGMA_OPERATING, color="green", ls=":", lw=1.2,
                  label="σ=0.1 (DNFS operating pt)")
        a.axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label="σ_c")
        a.set_xlabel("σ (coupling; larger = lower T)")
        a.set_yscale("log")
    ax[0].set_ylabel("τ_int (swap steps)")
    ax[0].set_title("critical slowing-down")
    ax[1].set_ylabel("ESS")
    ax[1].set_title("effective sample size collapse")
    ax[0].legend(fontsize=8)
    fig.suptitle("Kawasaki on the hard-composition canonical Ising (c=0.5): "
                 "mixing degrades past σ_c, worsening with lattice size")
    fig.tight_layout()
    fig.savefig(OUT / "failure_curves.png", dpi=140)
    plt.close(fig)


def _ergodicity_ensemble(D, sigma, n_steps, n_burn, thin, seed):
    """4 chains, 2 seeded +domain-left and 2 +domain-right. Returns
    (rhat, phi_traces) where rhat is the between-chain R̂ of the order
    parameter φ on the post-burn-in samples."""
    burn_rec = n_burn // thin
    phi_traces, post = [], []
    for chain in range(4):
        side = 0 if chain < 2 else 1                     # left vs right init
        x = init_phase_separated(D, side)
        phi, _, _ = run_chain_order_param(x, D, sigma, n_steps, seed + chain, thin)
        phi_traces.append(phi)
        post.append(phi[burn_rec:])
    rhat = gelman_rubin(np.stack(post))
    return rhat, phi_traces


def mode_coverage():
    n_steps, n_burn, thin = 1_500_000, 200_000, 200
    rhats = []
    traces_by_sigma = {}
    for sigma in ERGO_SIGMAS:
        rhat, phi_traces = _ergodicity_ensemble(
            ERGO_D, sigma, n_steps, n_burn, thin, seed=300)
        rhats.append(rhat)
        traces_by_sigma[sigma] = phi_traces
        print(f"ergodicity D={ERGO_D} σ={sigma}: R̂(φ)={rhat:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(ERGO_SIGMAS, rhats, "o-", color="purple")
    ax[0].axhline(1.1, color="grey", ls=":", lw=1, label="R̂=1.1 (mixed)")
    ax[0].axvline(SIGMA_OPERATING, color="green", ls=":", lw=1.2,
                  label="σ=0.1 (operating pt)")
    ax[0].axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label="σ_c")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("σ")
    ax[0].set_ylabel("R̂ of order parameter φ")
    ax[0].set_title("ergodicity breaking")
    ax[0].legend(fontsize=8)

    xs_scale = thin
    for label, sigma, axi in [("low T: σ=0.40 (trapped)", 0.40, ax[1]),
                              ("operating pt: σ=0.10 (mixed)", 0.10, ax[2])]:
        for c, phi in enumerate(traces_by_sigma[sigma]):
            colour = "C0" if c < 2 else "C3"
            axi.plot(np.arange(len(phi)) * xs_scale, phi, lw=0.8, alpha=0.8,
                     color=colour)
        axi.axhline(0, color="k", lw=0.6)
        axi.set_ylim(-2.2, 2.2)
        axi.set_xlabel("swap step")
        axi.set_ylabel("φ = left − right magnetisation")
        axi.set_title(label + "\n(blue = +domain-left init, red = right init)")
    fig.suptitle("Mode coverage via different-init ergodicity test: at the "
                 "operating point chains forget their init (φ→0, R̂≈1); at low T "
                 "they stay trapped on their starting side (R̂≫1)")
    fig.tight_layout()
    fig.savefig(OUT / "mode_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    failure_curves()
    mode_coverage()
    print(f"wrote figures to {OUT}")
