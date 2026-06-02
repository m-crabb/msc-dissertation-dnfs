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
    left_minus_right,
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
            # Normalised chain ESS = ESS/N = 1/τ_int ∈ (0,1] (fraction of the
            # chain that is effectively independent). Comparable across chain
            # lengths and lattice sizes; distinct from the IS-ESS used elsewhere.
            esss.append((1.0 / tau).mean())
            print(f"D={D} σ={sigma}: τ_int≈{tau.mean():.0f} "
                  f"normESS≈{(1.0/tau).mean():.2e}")
        ax[0].plot(CURVE_SIGMAS, taus, "o-", label=f"D={D}")
        ax[1].plot(CURVE_SIGMAS, esss, "o-", label=f"D={D}")
    for a in ax:
        a.axvline(SIGMA_OPERATING, color="green", ls=":", lw=1.2,
                  label=r"$\sigma=0.1$ (DNFS operating pt)")
        a.axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label=r"$\sigma_c$")
        a.set_xlabel(r"$\sigma$ (coupling; larger = lower $T$)")
        a.set_yscale("log")
    ax[0].set_ylabel(r"$\tau_{\mathrm{int}}$ (swap steps)")
    ax[0].set_title("critical slowing-down")
    ax[1].set_ylabel(r"normalised ESS  $= 1/\tau_{\mathrm{int}}$")
    ax[1].set_title("sampling efficiency collapse")
    ax[0].legend(fontsize=8)
    fig.suptitle(r"Kawasaki on the hard-composition canonical Ising ($c=0.5$): "
                 r"mixing degrades past $\sigma_c$, worsening with lattice size")
    fig.tight_layout()
    fig.savefig(OUT / "failure_curves.png", dpi=140)
    plt.close(fig)


def _ergodicity_ensemble(D, sigma, n_steps, n_burn, thin, seed):
    """4 chains, 2 seeded +domain-left and 2 +domain-right. Returns
    (rhat, phi_traces, finals): rhat is the between-chain R̂ of the order
    parameter φ on the post-burn-in samples; finals are the end configs."""
    burn_rec = n_burn // thin
    phi_traces, finals, post = [], [], []
    for chain in range(4):
        side = 0 if chain < 2 else 1                     # left vs right init
        x = init_phase_separated(D, side)
        phi, x_final, _ = run_chain_order_param(x, D, sigma, n_steps, seed + chain, thin)
        phi_traces.append(phi)
        finals.append(x_final)
        post.append(phi[burn_rec:])
    rhat = gelman_rubin(np.stack(post))
    return rhat, phi_traces, finals


def mode_coverage():
    n_steps, n_burn, thin = 1_500_000, 200_000, 200
    burn_rec = n_burn // thin
    rhats = []
    traces_by_sigma, finals_by_sigma = {}, {}
    for sigma in ERGO_SIGMAS:
        rhat, phi_traces, finals = _ergodicity_ensemble(
            ERGO_D, sigma, n_steps, n_burn, thin, seed=300)
        rhats.append(rhat)
        traces_by_sigma[sigma] = phi_traces
        finals_by_sigma[sigma] = finals
        print(f"ergodicity D={ERGO_D} σ={sigma}: R̂(φ)={rhat:.2f}")

    fig = plt.figure(figsize=(15, 8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    top = outer[0].subgridspec(1, 3, wspace=0.3)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)

    # top-left: R̂(φ) vs σ — the rigorous ergodicity-breaking curve
    axr = fig.add_subplot(top[0])
    axr.plot(ERGO_SIGMAS, rhats, "o-", color="purple")
    axr.axhline(1.1, color="grey", ls=":", lw=1, label=r"$\hat{R}=1.1$ (mixed)")
    axr.axvline(SIGMA_OPERATING, color="green", ls=":", lw=1.2,
                label=r"$\sigma=0.1$ (operating pt)")
    axr.axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label=r"$\sigma_c$")
    axr.set_yscale("log")
    axr.set_xlabel(r"$\sigma$")
    axr.set_ylabel(r"$\hat{R}$ of order parameter $\phi$")
    axr.set_title("ergodicity breaking")
    axr.legend(fontsize=8)

    # top-centre/right: φ traces at low T (trapped) and operating point (mixed)
    for label, sigma, col in [(r"low $T$: $\sigma=0.40$ (trapped)", 0.40, 1),
                              (r"operating pt: $\sigma=0.10$ (mixed)", 0.10, 2)]:
        axi = fig.add_subplot(top[col])
        for c, phi in enumerate(traces_by_sigma[sigma]):
            axi.plot(np.arange(len(phi)) * thin, phi, lw=0.8, alpha=0.8,
                     color="C0" if c < 2 else "C3")
        axi.axhline(0, color="k", lw=0.6)
        axi.set_ylim(-2.2, 2.2)
        axi.set_xlabel("swap step")
        axi.set_ylabel(r"$\phi$")
        axi.set_title(label)

    # bottom-left: snapshot grid of the trapped configs at σ=0.40 (the modes)
    cont = fig.add_subplot(bot[0])
    cont.axis("off")
    cont.set_title(r"trapped configurations at $\sigma=0.40$ "
                   r"(blue = left init, red = right init)", fontsize=9)
    snaps = bot[0].subgridspec(1, 4, wspace=0.15)
    for c, x_final in enumerate(finals_by_sigma[0.40]):
        axs = fig.add_subplot(snaps[c])
        axs.imshow(x_final.reshape(ERGO_D, ERGO_D), cmap="binary", vmin=-1, vmax=1)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title(rf"$\phi={left_minus_right(x_final, ERGO_D):+.1f}$",
                      fontsize=8, color="C0" if c < 2 else "C3")

    # bottom-right: φ distribution — bimodal (trapped) vs unimodal (mixed)
    axh = fig.add_subplot(bot[1])
    for sigma, colour, lab in [(0.40, "C3", r"$\sigma=0.40$ (low $T$)"),
                               (0.10, "C0", r"$\sigma=0.10$ (operating pt)")]:
        pooled = np.concatenate([phi[burn_rec:] for phi in traces_by_sigma[sigma]])
        axh.hist(pooled, bins=60, range=(-2.2, 2.2), density=True, alpha=0.6,
                 color=colour, label=lab)
    axh.set_xlabel(r"$\phi$ = left $-$ right magnetisation")
    axh.set_ylabel("density")
    axh.set_title(r"$\phi$ distribution (the modes)")
    axh.legend(fontsize=8)

    fig.suptitle(r"Mode coverage via different-init ergodicity test: at low $T$ the "
                 r"chains stay trapped in distinct modes ($\hat{R}\gg1$, bimodal "
                 r"$\phi$); at the operating point they mix ($\hat{R}\approx1$, "
                 r"unimodal $\phi$)")
    fig.savefig(OUT / "mode_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    failure_curves()
    mode_coverage()
    print(f"wrote figures to {OUT}")
