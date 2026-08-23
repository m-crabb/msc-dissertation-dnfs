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
from discrete_flow_sampler.diagnostics.figure_style import SPIN_CMAP
from discrete_flow_sampler.diagnostics.metrics import gelman_rubin
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    left_minus_right,
    run_chain_order_param,
)
from scripts.kawasaki_mcmc import run_config

OUT = Path("results/kawasaki/figures")
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_OPERATING = 0.10  # DNFS paper operating point (subcritical)
SIGMA_CRITICAL = 0.22305
CURVE_SIGMAS = [0.05, 0.10, 0.16, 0.20, 0.22305, 0.26]  # monotonic τ_int regime
ERGO_SIGMAS = [0.05, 0.10, 0.16, 0.20, 0.22305, 0.26, 0.32, 0.40]
DEMO_D = [8, 16]  # capped at the largest lattice the neural sampler reaches
ERGO_D = 24
# R̂ is a between-chain statistic, so a band on it means repeating the whole
# 4-chain ensemble under independent base seeds. The illustrative panels
# (φ traces, trapped snapshots, φ histogram) are taken from the first seed only.
ERGO_SEEDS = [300, 400, 500, 600]

# Each (D, σ) point runs N_CHAINS independent chains (seeds seed..seed+N_CHAINS-1),
# so τ_int is estimated N_CHAINS times over. We plot the across-chain mean with a
# ±1 std band rather than a single bare point, so the reported degradation carries
# its own error bar. Bumped 4 → 8 for a tighter band; the numba inner loop makes
# the extra chains cheap (~seconds each).
N_CHAINS = 8


def _annealed_tau_rows():
    """Annealed-start tau_int rows for the overlay, if the check has been run.

    Produced by `scripts/kawasaki_annealed_check.py full_curve` (simulated-
    annealing initialisation walked up a sigma ladder, then the identical
    measurement protocol). Drawn as hollow markers on the tau panel so the
    figure itself answers the cold-start objection: if the slowing-down curve
    were an initialisation artefact, the annealed markers would fall below the
    cold bands. Returns {(D, sigma): mean} or empty if the check hasn't run.
    """
    path = Path("results/kawasaki/annealed_check/tau_full_curve.json")
    if not path.exists():
        return {}
    import json
    return {(row["D"], round(row["sigma"], 5)): row["tau_int_sweeps_mean"]
            for row in json.loads(path.read_text())["rows"]}


def failure_curves():
    annealed_tau = _annealed_tau_rows()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for D in DEMO_D:
        d_sites = D * D
        tau_mean, tau_std, ess_mean, ess_std = [], [], [], []
        for sigma in CURVE_SIGMAS:
            _, _, _, tau, ess, _ = run_config(
                D,
                sigma,
                0.5,
                n_steps=1_500_000,
                n_burn=200_000,
                thin=50,
                n_chains=N_CHAINS,
                seed=100,
            )
            # Work in SWEEPS (one sweep = d swap attempts). A local move touches
            # only 2 of d sites, so any local sampler needs ~1 sweep per
            # independent configuration; measuring τ_int in single-swap units
            # would make every large lattice look slow purely from the step
            # definition, and unfairly penalise larger d at the operating point.
            tau_sweeps = tau / d_sites  # per-chain, (N_CHAINS,)
            # Normalised ESS = independent samples per sweep, capped at 1
            # (≈1 means as good as i.i.d. at sweep resolution), matching the
            # (·, 1] scale of the IS-ESS reported elsewhere.
            ess_sweeps = np.minimum(1.0, 1.0 / tau_sweeps)
            tau_mean.append(tau_sweeps.mean())
            tau_std.append(tau_sweeps.std(ddof=1))
            ess_mean.append(ess_sweeps.mean())
            ess_std.append(ess_sweeps.std(ddof=1))
            print(
                f"D={D} σ={sigma}: "
                f"τ_int≈{tau_sweeps.mean():.2f}±{tau_sweeps.std(ddof=1):.2f} sweeps  "
                f"normESS≈{ess_sweeps.mean():.2f}±{ess_sweeps.std(ddof=1):.2f}"
            )
        tau_mean, tau_std = np.array(tau_mean), np.array(tau_std)
        ess_mean, ess_std = np.array(ess_mean), np.array(ess_std)
        (line0,) = ax[0].plot(CURVE_SIGMAS, tau_mean, "o-", label=f"D={D}")
        ax[0].fill_between(
            CURVE_SIGMAS,
            np.maximum(tau_mean - tau_std, 1e-3),
            tau_mean + tau_std,
            color=line0.get_color(),
            alpha=0.2,
        )
        annealed = [annealed_tau.get((D, round(s, 5))) for s in CURVE_SIGMAS]
        if any(a is not None for a in annealed):
            have = [(s, a) for s, a in zip(CURVE_SIGMAS, annealed) if a is not None]
            ax[0].plot([s for s, _ in have], [a for _, a in have], "o",
                       color=line0.get_color(), markerfacecolor="none", ms=9)
        (line1,) = ax[1].plot(CURVE_SIGMAS, ess_mean, "o-", label=f"D={D}")
        ax[1].fill_between(
            CURVE_SIGMAS,
            np.clip(ess_mean - ess_std, 0, 1),
            np.clip(ess_mean + ess_std, 0, 1),
            color=line1.get_color(),
            alpha=0.2,
        )
    for a in ax:
        a.axvline(
            SIGMA_OPERATING,
            color="green",
            ls=":",
            lw=1.2,
            label=r"$\sigma=0.1$ (DNFS operating pt)",
        )
        a.axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label=r"$\sigma_c$")
        a.set_xlabel(r"$\sigma$ (coupling; larger = lower $T$)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\tau_{\mathrm{int}}$ (sweeps)")
    ax[0].set_title("critical slowing-down")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("normalised ESS (indep. samples / sweep)")
    ax[1].set_title("sampling efficiency collapse")
    handles, _ = ax[0].get_legend_handles_labels()
    if annealed_tau:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor="none", markeredgecolor="grey",
                              label="open: annealed start"))
    ax[0].legend(handles=handles, fontsize=8)
    fig.suptitle(
        r"Kawasaki on the hard-composition canonical Ising ($c=0.5$): "
        r"mixing degrades past $\sigma_c$, worsening with lattice size"
    )
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
        side = 0 if chain < 2 else 1  # left vs right init
        x = init_phase_separated(D, side)
        phi, x_final, _ = run_chain_order_param(
            x, D, sigma, n_steps, seed + chain, thin
        )
        phi_traces.append(phi)
        finals.append(x_final)
        post.append(phi[burn_rec:])
    rhat = gelman_rubin(np.stack(post))
    return rhat, phi_traces, finals


def mode_coverage():
    n_steps, n_burn, thin = 1_500_000, 200_000, 200
    burn_rec = n_burn // thin
    rhats_by_seed = []
    traces_by_sigma, finals_by_sigma = {}, {}
    for si, base_seed in enumerate(ERGO_SEEDS):
        rhats_this = []
        for sigma in ERGO_SIGMAS:
            rhat, phi_traces, finals = _ergodicity_ensemble(
                ERGO_D, sigma, n_steps, n_burn, thin, seed=base_seed
            )
            rhats_this.append(rhat)
            if si == 0:  # illustrative panels from one ensemble
                traces_by_sigma[sigma] = phi_traces
                finals_by_sigma[sigma] = finals
            print(f"ergodicity D={ERGO_D} σ={sigma} seed={base_seed}: R̂(φ)={rhat:.2f}")
        rhats_by_seed.append(rhats_this)
    rhats_arr = np.array(rhats_by_seed)  # (n_seed, n_sigma)
    rhat_mean = rhats_arr.mean(axis=0)
    rhat_std = rhats_arr.std(axis=0, ddof=1)

    fig = plt.figure(figsize=(15, 8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    top = outer[0].subgridspec(1, 3, wspace=0.3)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)

    # top-left: R̂(φ) vs σ — the rigorous ergodicity-breaking curve
    axr = fig.add_subplot(top[0])
    axr.plot(ERGO_SIGMAS, rhat_mean, "o-", color="purple")
    axr.fill_between(
        ERGO_SIGMAS,
        np.maximum(rhat_mean - rhat_std, 0.9),
        rhat_mean + rhat_std,
        color="purple",
        alpha=0.2,
    )
    axr.axhline(1.1, color="grey", ls=":", lw=1, label=r"$\hat{R}=1.1$ (mixed)")
    axr.axvline(
        SIGMA_OPERATING,
        color="green",
        ls=":",
        lw=1.2,
        label=r"$\sigma=0.1$ (operating pt)",
    )
    axr.axvline(SIGMA_CRITICAL, color="k", ls="--", lw=1, label=r"$\sigma_c$")
    axr.set_yscale("log")
    axr.set_xlabel(r"$\sigma$")
    axr.set_ylabel(r"$\hat{R}$ of order parameter $\phi$")
    axr.set_title("ergodicity breaking")
    axr.legend(fontsize=8)

    # top-centre/right: φ traces at low T (trapped) and operating point (mixed)
    for label, sigma, col in [
        (r"low $T$: $\sigma=0.40$ (trapped)", 0.40, 1),
        (r"operating pt: $\sigma=0.10$ (mixed)", 0.10, 2),
    ]:
        axi = fig.add_subplot(top[col])
        for c, phi in enumerate(traces_by_sigma[sigma]):
            axi.plot(
                np.arange(len(phi)) * thin,
                phi,
                lw=0.8,
                alpha=0.8,
                color="C0" if c < 2 else "C3",
            )
        axi.axhline(0, color="k", lw=0.6)
        axi.set_ylim(-2.2, 2.2)
        axi.set_xlabel("swap step")
        axi.set_ylabel(r"$\phi$")
        axi.set_title(label)

    # bottom-left: snapshot grid of the trapped configs at σ=0.40 (the modes)
    cont = fig.add_subplot(bot[0])
    cont.axis("off")
    cont.set_title(
        r"trapped configurations at $\sigma=0.40$ "
        r"(blue = left init, red = right init)",
        fontsize=9,
    )
    snaps = bot[0].subgridspec(1, 4, wspace=0.15)
    for c, x_final in enumerate(finals_by_sigma[0.40]):
        axs = fig.add_subplot(snaps[c])
        axs.imshow(
            (x_final.reshape(ERGO_D, ERGO_D) + 1) * 0.5,
            cmap=SPIN_CMAP,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title(
            rf"$\phi={left_minus_right(x_final, ERGO_D):+.1f}$",
            fontsize=8,
            color="C0" if c < 2 else "C3",
        )

    # bottom-right: φ distribution — bimodal (trapped) vs unimodal (mixed)
    axh = fig.add_subplot(bot[1])
    for sigma, colour, lab in [
        (0.40, "C3", r"$\sigma=0.40$ (low $T$)"),
        (0.10, "C0", r"$\sigma=0.10$ (operating pt)"),
    ]:
        pooled = np.concatenate([phi[burn_rec:] for phi in traces_by_sigma[sigma]])
        axh.hist(
            pooled,
            bins=60,
            range=(-2.2, 2.2),
            density=True,
            alpha=0.6,
            color=colour,
            label=lab,
        )
    axh.set_xlabel(r"$\phi$ = left $-$ right magnetisation")
    axh.set_ylabel("density")
    axh.set_title(r"$\phi$ distribution (the modes)")
    axh.legend(fontsize=8)

    fig.suptitle(
        r"Mode coverage via different-init ergodicity test: at low $T$ the "
        r"chains stay trapped in distinct modes ($\hat{R}\gg1$, bimodal "
        r"$\phi$); at the operating point they mix ($\hat{R}\approx1$, "
        r"unimodal $\phi$)"
    )
    fig.savefig(OUT / "mode_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    failure_curves()
    mode_coverage()
    print(f"wrote figures to {OUT}")
