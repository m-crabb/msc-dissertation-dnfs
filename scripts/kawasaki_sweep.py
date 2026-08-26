"""Kawasaki failure-mode sweep → dissertation §5.1 figures.

Produces:
  (1) failure_curves.png — τ_int and ESS vs σ for D∈{8,16} (the sizes the thesis
      trains a neural sampler at, drawn solid) and D∈{24,32} (sizes it never
      trains at, drawn dotted: the classical chain extrapolated past the reach
      of the learned one), with σ=0.1 (the DNFS operating point) and σ_c marked.
      Critical slowing-down: mixing is fine at the subcritical operating point
      and degrades sharply as σ enters the ordered, constrained low-T regime —
      worsening with system size, which is the scaling claim the figure exists
      to make. The σ grid stops at 0.26: past there acceptance → 0 (the chain
      freezes) and the τ_int estimator is no longer reliable; that deep-frozen
      regime is the subject of the ergodicity figure instead.

      PROTOCOL (per-site, not per-step). Every (D, σ) cell spends the same
      number of SWEEPS — MEASURE_SWEEPS after BURN_SWEEPS, one sweep = d = D²
      swap attempts — so the four lattice sizes are compared under one budget
      per site rather than one budget per raw swap. A fixed raw-step budget
      (the pre-2026-08-26 protocol) gave the 32×32 chain 1/16 of the 8×8
      chain's sweeps and would have manufactured part of the size trend the
      figure reports. Thinning is likewise d-scaled (RECORDS_PER_SWEEP records
      per sweep), so the τ_int estimator has the same resolution in sweeps at
      every size; with a fixed raw thin of 50 the 8×8 τ_int was floored at
      50/64 ≈ 0.78 sweeps by the sampling interval alone.

      ANNEALING is run for EVERY (D, σ) cell of both panels, not as a spot
      check: each annealed chain is initialised by walking the training
      curriculum's σ-ladder (scripts.kawasaki_annealed_check.anneal_ladder,
      dwelling BURN_SWEEPS sweeps per rung) and then measured under the
      identical protocol. Cold-start curves that slowed down only because a
      random start had not relaxed would show annealed markers sitting below
      the band; markers inside the band say the slowing-down is the dynamics.
  (2) mode_coverage.png — DIFFERENT-INITIALISATION ergodicity test. Chains are
      seeded in distinct modes (+domain left vs right) and we track a mode-
      sensitive order parameter φ = left − right sublattice magnetisation. If the
      sampler is ergodic the chains forget their init and φ's between-chain R̂→1;
      if mode coverage fails they stay stuck (R̂≫1). High-σ is the failure, low-σ
      (σ=0.1, the operating point) is the built-in positive control.

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

SIGMA_CRITICAL = SIGMA_C  # exact = log(1+sqrt(2))/4 = 0.220343 (s58 migration)
CURVE_SIGMAS = [0.05, 0.10, 0.16, 0.20, SIGMA_CRITICAL, 0.26]  # monotonic τ_int regime
ERGO_SIGMAS = [0.05, 0.10, 0.16, 0.20, SIGMA_CRITICAL, 0.26, 0.32, 0.40]
TRAINED_D = [8, 16]     # the lattices this thesis trains a neural sampler at
EXTRAPOLATED_D = [24, 32]  # never trained at: the classical chain run on alone
DEMO_D = TRAINED_D + EXTRAPOLATED_D
ERGO_D = 24
# R̂ is a between-chain statistic, so a band on it means repeating the whole
# 4-chain ensemble under independent base seeds. The illustrative panels
# (φ traces, trapped snapshots, φ histogram) are taken from the first seed only.
ERGO_SEEDS = [300, 400, 500, 600]

# Each (D, σ, init) point runs N_CHAINS independent chains (seeds
# CHAIN_SEED..CHAIN_SEED+N_CHAINS-1), so τ_int is estimated N_CHAINS times over.
# We plot the across-chain mean with a min–max band (the house uncertainty
# grammar) rather than a single bare point, so the reported degradation carries
# its own spread. The numba inner loop makes the extra chains cheap.
N_CHAINS = 8
CHAIN_SEED = 100

# Per-site budget: one sweep = d = D² swap attempts. See the module docstring
# for why these are sweeps and not raw steps. 30k sweeps is ≳400 τ_int even at
# the slowest cell measured (32×32 at σ=0.26, τ_int ≈ 70 sweeps).
MEASURE_SWEEPS = 30_000
BURN_SWEEPS = 3_000       # also the dwell per rung of the annealing ladder
RECORDS_PER_SWEEP = 5     # thinning: d // 5 raw steps between recorded energies


def _measure_cell(D, sigma, init_kind):
    """One (D, sigma, init) cell: N_CHAINS chains, tau_int per chain in SWEEPS.

    tau_int is Sokal-windowed on the post-burn-in thinned energy trace and
    converted to sweeps by (tau_records * thin) / d, so the unit is "swap
    attempts per site" and lattice sizes are directly comparable: a local move
    touches 2 of d sites, so any such sampler needs ~1 sweep per independent
    configuration and quoting tau_int in raw swaps would make every large
    lattice look slow purely from the step definition.

    init_kind "cold" starts from a uniform random configuration at c = 0.5;
    "annealed" starts from the same random configuration walked up the training
    curriculum's sigma-ladder, BURN_SWEEPS sweeps per rung, before the
    identical measurement begins. Both then burn BURN_SWEEPS more sweeps, so
    the annealed arm is strictly the more generous of the two.
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

    Split from the plotting so a restyle never recomputes chains (the figure
    was previously flagged "NOT regenerable from archive" for exactly that
    reason). Normalised ESS = independent configurations per sweep, capped at
    1, matching the (0, 1] scale of the IS-ESS reported for the neural
    samplers elsewhere in the thesis.
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
    the legend: the legend has to carry the solid/dotted and open-marker
    grammar, and four more entries would crowd it out. Only the left panel is
    labelled -- the guides are at the same two couplings in both, and the
    rotated text collides with the ESS curves where they sit at 1."""
    for sigma, text in [(SIGMA_OPERATING, r"$\sigma=0.1$ (DNFS op. pt)"),
                        (SIGMA_CRITICAL, r"$\sigma_c$")]:
        ax.axvline(sigma, color=fs.ANALYTIC_GUIDE, ls=(0, (4, 3)), lw=0.8,
                   zorder=1)
        if label:
            ax.text(sigma - 0.004, 0.97, text,
                    transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top",
                    fontsize=fs.FONT_SIZE_ANNOTATION, color=fs.ANALYTIC_GUIDE)


def _direct_labels(ax, anchors, min_gap_pt=9.0):
    """Label each curve at its right-hand end, nudged apart where two ends sit
    on top of each other (D=24 and D=32 differ by a factor 1.2 in tau_int at
    the last coupling, which is a few points on a log axis).

    anchors is [(y_data, text, colour)] at the shared right-edge x. Positions
    are separated greedily in DISPLAY points and the shift is applied as an
    offset, so the label still points at the curve it names -- the house rule
    that colour is never the only identity channel is what forces a label per
    curve here rather than a four-entry legend.
    """
    x_right = CURVE_SIGMAS[-1]
    # transData is in pixels, annotate offsets are in points: one conversion,
    # or the nudge silently comes out 1.5x too small at figure.dpi = 110.
    pixels_per_point = ax.figure.dpi / 72.0
    order = sorted(range(len(anchors)), key=lambda i: anchors[i][0])
    placed, previous = {}, -np.inf
    for i in order:
        y_pixels = ax.transData.transform((x_right, anchors[i][0]))[1]
        placed[i] = max(y_pixels, previous + min_gap_pt * pixels_per_point)
        previous = placed[i]
    for i, (y_data, text, colour) in enumerate(anchors):
        shift = placed[i] - ax.transData.transform((x_right, y_data))[1]
        ax.annotate(text, xy=(x_right, y_data),
                    xytext=(4, shift / pixels_per_point),
                    textcoords="offset points", va="center", color=colour,
                    fontsize=fs.FONT_SIZE_ANNOTATION)


def failure_curves(payload=None):
    """Plot the two failure panels from the cache (computing it if absent).

    Colour follows the house rule's stated exception: every curve is the same
    ROLE (the classical Kawasaki baseline), and the contrast between them IS a
    parameter level (lattice size), so the role's hue gets a lightness ramp,
    light = small lattice. Linestyle carries the second, orthogonal
    distinction the chapter needs: solid for the sizes a neural sampler is
    trained at here, dotted for the sizes only the classical chain reaches.
    """
    payload = payload or json.loads(CURVE_CACHE.read_text())
    rows = {(row["D"], round(row["sigma"], 6)): row for row in payload["rows"]}
    n_chains = payload["protocol"]["n_chains"]
    colours = dict(zip(DEMO_D,
                      fs.parameter_ramp(fs.CLASSICAL_HUE, len(DEMO_D),
                                        lightest=0.45)))

    fs.use_house_style()
    fig, ax = plt.subplots(1, 2, figsize=(fs.FULL_WIDTH_IN, 3.1))
    label_anchors = {0: [], 1: []}
    for D in DEMO_D:
        cells = [rows[(D, round(s, 6))] for s in CURVE_SIGMAS]
        colour = colours[D]
        style = "-" if D in TRAINED_D else (0, (1.6, 1.6))
        for index, key in ((0, "tau_int_sweeps"), (1, "ess_per_sweep")):
            panel = ax[index]
            cold = np.array([c["cold"][key] for c in cells])       # (n_sigma, n_chain)
            annealed = np.array([c["annealed"][key] for c in cells]).mean(axis=1)
            panel.fill_between(CURVE_SIGMAS, cold.min(axis=1), cold.max(axis=1),
                               color=colour, alpha=0.18, linewidth=0, zorder=2)
            panel.plot(CURVE_SIGMAS, cold.mean(axis=1), linestyle=style,
                       color=colour, linewidth=1.6, marker="o", markersize=3,
                       zorder=3)
            panel.plot(CURVE_SIGMAS, annealed, linestyle="none", marker="o",
                       markersize=6, markerfacecolor="none",
                       markeredgecolor=colour, markeredgewidth=0.9, zorder=4)
            label_anchors[index].append((cold.mean(axis=1)[-1], f"$D={D}$", colour))

    for panel in ax:
        _sigma_guides(panel, label=panel is ax[0])
        fs.style_axes(panel)
        panel.set_xlabel(r"$\sigma$ (coupling; larger = lower $T$)")
        panel.set_xlim(0.04, 0.30)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\tau_{\mathrm{int}}$ (sweeps)")
    ax[0].set_title("critical slowing-down")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("norm. ESS (indep. samples / sweep)")
    ax[1].set_title("sampling efficiency collapse")

    legend_entries = [
        Line2D([0], [0], color=fs.ANALYTIC_GUIDE, lw=1.6,
               label="solid: trained at here"),
        Line2D([0], [0], color=fs.ANALYTIC_GUIDE, lw=1.6, ls=(0, (1.6, 1.6)),
               label="dotted: not trained at"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="none", markeredgecolor=fs.ANALYTIC_GUIDE,
               label="open: annealed start"),
        Patch(facecolor=fs.ANALYTIC_GUIDE, alpha=0.18,
              label=f"band: min-max, {n_chains} chains"),
    ]
    ax[1].legend(handles=legend_entries, loc="lower left", frameon=False,
                 handlelength=1.8, labelspacing=0.3, borderpad=0.2)
    fig.tight_layout()
    # Labels are placed in display coordinates, so the layout must be settled
    # first; tight_layout after this would move the axes out from under them.
    for index in (0, 1):
        _direct_labels(ax[index], label_anchors[index])
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
