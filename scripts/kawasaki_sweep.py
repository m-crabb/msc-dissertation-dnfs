"""Kawasaki failure-mode sweep → dissertation §5.1 figures.

Produces:
  (1) failure_curves.png — τ_int and ESS vs σ for D∈{8,16,24} (trained at,
      solid) and D=32 (never trained at, dotted), with σ=0.1 (the DNFS
      operating point) and σ_c marked. Linestyle is decided at plot time from
      TRAINED_D, so the cached `trained_at` flags (2026-08-26) are stale
      metadata; 24 moved from dotted to solid on 2026-09-07 once the 24x24 rung
      was in print. Mixing is fine at the subcritical operating point and
      degrades sharply as σ enters the ordered low-T regime, worsening with
      system size. The σ grid stops at 0.26: past there acceptance → 0 and the
      τ_int estimator is no longer reliable.

      Protocol is per-site: every (D, σ) cell spends MEASURE_SWEEPS sweeps
      after BURN_SWEEPS, one sweep = d = D² swap attempts, so the four sizes
      share one budget per site. A fixed raw-step budget (pre-2026-08-26) gave
      the 32×32 chain 1/16 of the 8×8 chain's sweeps; thinning is d-scaled
      (RECORDS_PER_SWEEP per sweep) because a fixed raw thin of 50 floored the
      8×8 τ_int at 50/64 ≈ 0.78 sweeps by the sampling interval alone.

      Annealing runs for every (D, σ) cell of both panels: each annealed chain
      walks the training curriculum's σ-ladder
      (scripts.kawasaki_annealed_check.anneal_ladder, BURN_SWEEPS sweeps per
      rung) before the identical measurement. Annealed markers inside the
      cold-start band say the slowing-down is the dynamics, not an unrelaxed
      start.
  (2) mode_coverage.png — different-initialisation ergodicity test. Chains are
      seeded in distinct modes (+domain left vs right) and tracked through the
      order parameter φ = left − right sublattice magnetisation. Ergodic chains
      forget their init and R̂(φ) → 1; failed mode coverage stays stuck (R̂≫1).
      High-σ is the failure, σ=0.1 (the operating point) the positive control.

Run:  pixi run python -m scripts.kawasaki_sweep            # both figures
      pixi run python -m scripts.kawasaki_sweep curves     # failure curves only
      pixi run python -m scripts.kawasaki_sweep replot     # re-plot from cache
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from discrete_flow_sampler.diagnostics import figure_style as fs
from discrete_flow_sampler.diagnostics.figure_style import SPIN_CMAP
from discrete_flow_sampler.diagnostics.metrics import (
    gelman_rubin,
    integrated_autocorr,
)
from discrete_flow_sampler.mcmc.kawasaki import (
    init_phase_separated,
    init_random_at_composition,
    left_minus_right,
    run_chain,
    run_chain_order_param,
)

OUT = Path("results/kawasaki/figures")
OUT.mkdir(parents=True, exist_ok=True)
CURVE_CACHE = OUT / "failure_curves_data.json"

SIGMA_OPERATING = 0.10  # DNFS paper operating point (subcritical)
from discrete_flow_sampler.targets.ising import SIGMA_C

SIGMA_CRITICAL = SIGMA_C  # exact = log(1+sqrt(2))/4 = 0.220343
CURVE_SIGMAS = [0.05, 0.10, 0.16, 0.20, SIGMA_CRITICAL, 0.26]  # monotonic τ_int regime
ERGO_SIGMAS = [0.05, 0.10, 0.16, 0.20, SIGMA_CRITICAL, 0.26, 0.32, 0.40]
TRAINED_D = [8, 16, 24]  # the lattices this thesis trains a neural sampler at
EXTRAPOLATED_D = [32]    # never trained at: the classical chain run on alone
DEMO_D = TRAINED_D + EXTRAPOLATED_D
ERGO_D = 24
# R̂ is a between-chain statistic, so a band on it repeats the whole 4-chain
# ensemble under independent base seeds; illustrative panels use the first seed.
ERGO_SEEDS = [300, 400, 500, 600]

# Each (D, σ, init) point runs N_CHAINS independent chains (seeds
# CHAIN_SEED..CHAIN_SEED+N_CHAINS-1), plotted as the across-chain mean with a
# min–max band so the reported degradation carries its own spread.
N_CHAINS = 8
CHAIN_SEED = 100

# Per-site budget: one sweep = d = D² swap attempts. 30k sweeps is ≳400 τ_int
# even at the slowest cell measured (32×32 at σ=0.26, τ_int ≈ 70 sweeps).
MEASURE_SWEEPS = 30_000
BURN_SWEEPS = 3_000       # also the dwell per rung of the annealing ladder
RECORDS_PER_SWEEP = 5     # thinning: d // 5 raw steps between recorded energies


def _measure_cell(D, sigma, init_kind):
    """One (D, sigma, init) cell: N_CHAINS chains, tau_int per chain in SWEEPS.

    tau_int is Sokal-windowed on the post-burn-in thinned energy trace and
    converted to sweeps by (tau_records * thin) / d, so the unit is "swap
    attempts per site": a local move touches 2 of d sites, so quoting tau_int
    in raw swaps would make every large lattice look slow purely from the step
    definition.

    init_kind "cold" starts from a uniform random configuration at c = 0.5;
    "annealed" walks that configuration up the training curriculum's
    sigma-ladder, BURN_SWEEPS sweeps per rung, first. Both then burn
    BURN_SWEEPS more sweeps.
    """
    from scripts.kawasaki_annealed_check import annealed_init

    d = D * D
    thin = max(1, d // RECORDS_PER_SWEEP)
    n_steps, n_burn = MEASURE_SWEEPS * d + BURN_SWEEPS * d, BURN_SWEEPS * d
    taus, acceptances = [], []
    for chain in range(N_CHAINS):
        seed = CHAIN_SEED + chain
        if init_kind == "cold":
            x = init_random_at_composition(d, 0.5, np.random.default_rng(seed))
        else:
            x = annealed_init(D, sigma, seed, dwell_steps=BURN_SWEEPS * d)
        energy, x_final, n_accept = run_chain(x, D, sigma, n_steps, seed)
        assert int((x_final == 1).sum()) == d // 2, "composition not conserved!"
        kept = energy[n_burn::thin]
        taus.append(integrated_autocorr(kept) * thin / d)
        acceptances.append(n_accept / n_steps)
    taus = np.array(taus)
    print(f"  D={D:2d} sigma={sigma:.6f} {init_kind:8s}: "
          f"tau_int={taus.mean():7.3f} sweeps "
          f"[{taus.min():.3f}, {taus.max():.3f}]  "
          f"acc={np.mean(acceptances):.4f}")
    return taus, float(np.mean(acceptances))


def curve_data():
    """Run every (D, sigma) cell of both init arms and cache the result.

    Split from the plotting so a restyle never recomputes chains. Normalised
    ESS = independent configurations per sweep, capped at 1, matching the
    (0, 1] scale of the IS-ESS reported for the neural samplers elsewhere.
    """
    rows = []
    for D in DEMO_D:
        for sigma in CURVE_SIGMAS:
            cell = {"D": D, "sigma": sigma,
                    "trained_at": D in TRAINED_D}
            for init_kind in ("cold", "annealed"):
                taus, acceptance = _measure_cell(D, sigma, init_kind)
                cell[init_kind] = {
                    "tau_int_sweeps": taus.tolist(),
                    "ess_per_sweep": np.minimum(1.0, 1.0 / taus).tolist(),
                    "acceptance": acceptance,
                }
            rows.append(cell)
    payload = {
        "protocol": {
            "measure_sweeps": MEASURE_SWEEPS,
            "burn_sweeps": BURN_SWEEPS,
            "records_per_sweep": RECORDS_PER_SWEEP,
            "n_chains": N_CHAINS,
            "chain_seed": CHAIN_SEED,
            "c_target": 0.5,
            "sigma_c": SIGMA_CRITICAL,
            "anneal_dwell_sweeps_per_rung": BURN_SWEEPS,
            "trained_D": TRAINED_D,
            "extrapolated_D": EXTRAPOLATED_D,
        },
        "rows": rows,
    }
    CURVE_CACHE.write_text(json.dumps(payload, indent=2))
    print(f"wrote {CURVE_CACHE}")
    return payload


def _sigma_guides(ax, label=True):
    """The two vertical reference couplings, labelled in-axes rather than in
    the legend, which already carries four sizes plus the marker grammar.
    Labels are the bare symbol and sit along the bottom of the tau panel, the
    one strip both the curves and the two legends leave empty."""
    for sigma, text in [(SIGMA_OPERATING, r"$\sigma_\mathrm{op}$"),
                        (SIGMA_CRITICAL, r"$\sigma_c$")]:
        ax.axvline(sigma, color=fs.ANALYTIC_GUIDE, ls=(0, (4, 3)), lw=0.8,
                   zorder=1)
        if label:
            ax.text(sigma, 0.015, text, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=fs.FONT_SIZE_ANNOTATION, color=fs.ANALYTIC_GUIDE)


def failure_curves(payload=None):
    """Plot the two failure panels from the cache (computing it if absent).

    Colour: four distinct hues, one per lattice size, from the house palette's
    CVD-validated five-hue set; a lightness ramp within one role was not
    separable at four levels on the printed page. Linestyle carries the
    orthogonal distinction: solid for the sizes a neural sampler is trained at,
    dotted for the sizes only the classical chain reaches.

    Identity is carried by a legend rather than by labels at the curve ends:
    the figure is ~2.4 in tall, so the four right-hand ends fall within a few
    points of each other. The legend sits in the tau panel's empty upper-left
    corner.
    """
    payload = payload or json.loads(CURVE_CACHE.read_text())
    rows = {(row["D"], round(row["sigma"], 6)): row for row in payload["rows"]}
    n_chains = payload["protocol"]["n_chains"]
    # Blue and gold keep 8 and 16 on the hues the previous version of this
    # figure used for them; purple and red extend the set for 24 and 32.
    colours = dict(zip(DEMO_D, [fs.SAMPLER_HUE, fs.CLASSICAL_HUE,
                                fs.CLASSICAL_ALT_HUE, fs.HARD_DELTA_HUE]))

    fs.use_house_style()
    # Aspect 12:4.5 = 2.667 at the house print width (6.3 in = \textwidth),
    # so the text is set at a true 9 pt.
    fig, ax = plt.subplots(1, 2, figsize=(fs.FULL_WIDTH_IN,
                                          fs.FULL_WIDTH_IN * 4.5 / 12.0))
    size_entries = []
    for D in DEMO_D:
        cells = [rows[(D, round(s, 6))] for s in CURVE_SIGMAS]
        colour = colours[D]
        trained = D in TRAINED_D
        style = "-" if trained else (0, (1.6, 1.6))
        for index, key in ((0, "tau_int_sweeps"), (1, "ess_per_sweep")):
            panel = ax[index]
            cold = np.array([c["cold"][key] for c in cells])   # (n_sigma, n_chain)
            annealed = np.array([c["annealed"][key] for c in cells]).mean(axis=1)
            panel.fill_between(CURVE_SIGMAS, cold.min(axis=1), cold.max(axis=1),
                               color=colour, alpha=0.18, linewidth=0, zorder=2)
            panel.plot(CURVE_SIGMAS, cold.mean(axis=1), linestyle=style,
                       color=colour, linewidth=1.5, marker="o", markersize=2.6,
                       zorder=3)
            panel.plot(CURVE_SIGMAS, annealed, linestyle="none", marker="o",
                       markersize=5, markerfacecolor="none",
                       markeredgecolor=colour, markeredgewidth=0.8, zorder=4)
        size_entries.append(
            Line2D([0], [0], color=colour, lw=1.5, ls=style,
                   label=rf"$D={D}$" if trained else rf"$D={D}$ (untrained)"))

    for panel in ax:
        _sigma_guides(panel, label=panel is ax[0])
        fs.style_axes(panel)
        panel.set_xlabel(r"$\sigma$ (coupling; larger = lower $T$)")
        panel.set_xlim(0.04, 0.275)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\tau_{\mathrm{int}}$ (sweeps)")
    ax[0].set_title("critical slowing-down")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("norm. ESS (samples / sweep)")
    ax[1].set_title("sampling efficiency collapse")

    grammar_entries = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor="none", markeredgecolor=fs.ANALYTIC_GUIDE,
               label="annealed start"),
        Patch(facecolor=fs.ANALYTIC_GUIDE, alpha=0.18,
              label=f"min-max, {n_chains} chains"),
    ]
    # Two small legends, each in a corner its own panel leaves empty: the sizes
    # in the tau panel's upper left, the marker grammar in the ESS panel's
    # lower left.
    size_legend = ax[0].legend(handles=size_entries, loc="upper left",
                               handlelength=1.9, labelspacing=0.25,
                               borderpad=0.3, borderaxespad=0.3,
                               handletextpad=0.5, facecolor="white",
                               edgecolor="none", framealpha=0.92)
    ax[0].add_artist(size_legend)
    ax[1].legend(handles=grammar_entries, loc="lower left", handlelength=1.4,
                 labelspacing=0.25, borderpad=0.3, borderaxespad=0.3,
                 handletextpad=0.5, facecolor="white", edgecolor="none",
                 framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "failure_curves.png", dpi=fs.SAVEFIG_DPI)
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "replot":          # restyle without recomputing any chain
        failure_curves()
    elif mode == "curves":
        failure_curves(curve_data())
    else:
        failure_curves(curve_data())
        mode_coverage()
    print(f"wrote figures to {OUT}")
