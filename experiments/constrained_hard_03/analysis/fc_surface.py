"""F(c, sigma) surface from ONE amortised 8x8 sampler, no new training.

The hard sampler's annealing path on the fixed-composition slice is a coupling
anneal (probe_zero_shot_transfer, axis 1): the base density is uniform on the
slice, so p~_t is the Ising target at coupling t * sigma, and the running
importance weight at grid time t is the path estimator of log Z_t. One draw per
composition therefore prices the free energy at EVERY coupling between zero and
the trained one:

    F(c, t sigma) / d = -[ E_q log w_t + (1 - t) log C(d, N_A) ] / d      (Jensen: a bound from above)

The classical toolchain needs one thermodynamic-integration ladder per (c, T)
point (icet ThermodynamicIntegrationEnsemble) or one chain per (phi, T) in
VC-SGC; here the whole temperature ray at fixed composition falls out of the
weights the sampler already computes. That asymmetry is the exhibit.

Inputs (already on disk, tag 20260831-camort-d64, checkpoint final_ema.pt,
ne128): each seed's `zero_shot_fc.json`, seven stop times k/127 for
k = 16, 32, 58, 76, 95, 111, 127 by seven compositions. Reference: mchammer TI at
the same couplings (`fc_ref_d8_k{K}.npz` from 07_fc_mchammer_reference.py, run
per stop time; `fc_ref_d8_sc.npz` is the printed t = 1 truth on five
compositions). Compositions 0.3125 and 0.4375 are mirrored to 0.6875 and
0.5625 under the target family's exact Z2 symmetry (the printed F(c) figure
does the same), and the caption says so.

Panel (a): F/d vs c, one curve per coupling on the sampler hue's lightness
ramp (light = weak coupling, dark = sigma_c; figure_style.parameter_ramp),
band = min-max over three seeds, plus the analytic sigma = 0 limit
-log C(d, N_A)/d as a guide. Panel (b): residual against the TI truth wherever
a reference exists, same ramp, with the sign the bound requires (>= 0 up to
the Euler-grid bias, which Richardson removed in the printed t = 1 series and
is NOT removed here: this is the native ne128 read). Panel (c): the
central curvature, the second difference [F(0.4375) - 2F(0.5) + F(0.5625)]/d
over the composition step 1/16, against sigma/sigma_c. It changes sign between
0.75 and 0.87 sigma_c: above that the fixed-composition ensemble on this torus
lowers its free energy by demixing, the finite-size signature of the ordering
transition (the infinite-volume F is convex everywhere and flat inside the
binodal; on an 8x8 torus at sigma_c the correlation length exceeds the box).
Read from ONE model's running weights, at every coupling, with the TI truth
beside it where it exists.

Companion figure (`--sro`): the Warren-Cowley short-range-order parameter
from the same rows' weighted nearest-neighbour spin product g = <x_i x_j>.
With occupations n = (1 + x)/2 the ordered-pair fraction P_AB = (1 - g)/4, so

    alpha_1(c, sigma) = 1 - P_{A|B} / c_A = 1 - (1 - g) / (4 c (1 - c)),

zero at random mixing (g = (2c - 1)^2), positive for like-neighbour
clustering (the ferromagnet), negative for ordering. Composition is exact on
every draw, so this is the canonical SRO at fixed c with no reweighting --
the panel the CE tutorials draw from a VC-SGC chain per (phi, T).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SIGMA_C = 0.22034339675488573
STOP_GRID = 127  # ne128: ts = k / 127
# The two amortised rungs. The 16x16 twin (one seed, tag 20260831-camort-d256,
# probed 3-Sep on Modal with the same stop-time grid) has no TI reference yet;
# its SRO cross comes from the pooled certified Kawasaki draws instead of the
# 8x8's per-chain npz files.
RUNGS = {
    8: dict(D=64, seeds=(42, 43, 44), ti_prefix="fc_ref_d8",
            template="H2_d64_camort_s220_letf_thp_50k_curr_seed{seed}_20260831-camort-d64"),
    16: dict(D=256, seeds=(42,), ti_prefix="fc_ref_d16",
             template="H2_d256_camort_s220_letf_thp2_100k_curr_seed{seed}_20260831-camort-d256"),
}


def load_sampler_surface(results_dir: Path, seeds, template: str) -> dict:
    """{(composition, k): {"F": [per seed], "ess": [per seed]}} with k the
    stop-time grid index, so couplings match the reference files exactly."""
    surface: dict = {}
    for seed in seeds:
        payload = json.loads(
            (results_dir / template.format(seed=seed) / "zero_shot_fc.json").read_text())
        assert payload["n_euler_steps"] == STOP_GRID + 1
        for row in payload["rows"]:
            k = round(row["stop_time"] * STOP_GRID)
            cell = surface.setdefault((row["composition"], k), {"F": [], "ess": [], "nn": []})
            cell["F"].append(row["free_energy_nats_per_site"])
            cell["ess"].append(row["ess_fraction"])
            cell["nn"].append(row["nn_correlation"])
    return surface


def mirror_missing(table: dict) -> dict:
    """Add c -> 1 - c for compositions not present: the slice family is
    exactly Z2 symmetric, so F(c) = F(1 - c) by construction. Applied to the
    sampler surface and to the TI reference alike (the curvature stencil
    needs 0.5625, which neither ran)."""
    present = {c for c, _ in table}
    mirrored = dict(table)
    for (c, k), cell in table.items():
        if round(1 - c, 6) not in present:
            mirrored[(round(1 - c, 6), k)] = cell
    return mirrored


def load_reference(reference_dir: Path, ti_prefix: str | None) -> dict:
    """{(composition, k): F_per_site} from every TI file present."""
    reference: dict = {}
    if ti_prefix is None:
        return reference
    # The printed t = 1 truth takes precedence where it overlaps the k = 127
    # replicate, so the surface's t = 1 residuals are the printed ones. The two
    # independent TI runs differ by up to 0.0024 nats/site at c = 0.375
    # (-0.8469 vs -0.8493), above the ~0.001/site hysteresis bracket each
    # reports: the TI's own replicate spread is the reference floor here.
    printed = reference_dir / f"{ti_prefix}_sc.npz"
    if printed.is_file():
        z = np.load(printed)
        for c, F in zip(z["c_target"], z["F_per_site"]):
            reference[(round(float(c), 6), STOP_GRID)] = float(F)
    for path in sorted(reference_dir.glob(f"{ti_prefix}_k*.npz")):
        k = int(path.stem.split("_k")[-1])
        z = np.load(path)
        for c, F in zip(z["c_target"], z["F_per_site"]):
            reference.setdefault((round(float(c), 6), k), float(F))
    return reference


def ideal_mixing_per_site(c: float, D: int) -> float:
    n = round(c * D)
    return -math.lgamma(D + 1) / D + (math.lgamma(n + 1) + math.lgamma(D - n + 1)) / D


def warren_cowley(nn_correlation: float, c: float) -> float:
    """alpha_1 from the mean bond spin product at composition c (docstring)."""
    return 1.0 - (1.0 - nn_correlation) / (4.0 * c * (1.0 - c))


# The certified 8x8 Kawasaki chains (house_table_8x8's reference) at c = 0.5:
# s220 is the exact sigma_c (k = 127); s100 is sigma = 0.1 against the grid's
# k = 58 at 0.1006 (a 0.6% coupling offset, stated in the caption).
KAWASAKI_TAG_TO_K = {"s100": 58, "s220": STOP_GRID}


def reference_sro_at_half(results_root: Path, edge: int, burn_in_fraction=0.2) -> dict:
    """{k: alpha_1 at c = 0.5} from the certified Kawasaki reference: the 8x8
    per-chain npz files (post burn-in) or the 16x16 pooled thinned draws."""
    import torch

    from discrete_flow_sampler.diagnostics.metrics import nn_correlation
    from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

    adjacency = FixedCompositionIsingTarget(D=edge, sigma=SIGMA_C, target_composition=0.5).A
    out = {}
    for tag, k in KAWASAKI_TAG_TO_K.items():
        if edge == 8:
            chains = sorted((results_root / "03_hard" / "kawasaki_w2").glob(f"kawasaki_D8_{tag}_seed*.npz"))
            draws = [torch.from_numpy(np.load(c)["spins"]).float() for c in chains]
            draws = [d[int(len(d) * burn_in_fraction):] for d in draws]
        else:
            pooled = results_root / f"kawasaki_ref_d256_{tag.replace('s100', 's010')}" / "samples.pt"
            draws = [torch.load(pooled, weights_only=True).float()] if pooled.is_file() else []
        if not draws:
            continue
        g = [float(nn_correlation(d, adjacency).mean()) for d in draws]
        out[k] = warren_cowley(float(np.mean(g)), 0.5)
    return out


def plot_sro(surface: dict, out: Path, reference_half: dict) -> None:
    import matplotlib.pyplot as plt

    from discrete_flow_sampler.diagnostics.figure_style import (
        FIGSIZE_SINGLE, MUTED, REFERENCE_INK, SAMPLER_HUE, SAVEFIG_DPI,
        parameter_ramp, style_axes, uncertainty_band, use_house_style)

    use_house_style()
    ks = sorted({k for _, k in surface})
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for hue, k in zip(parameter_ramp(SAMPLER_HUE, len(ks)), ks):
        cs = sorted(c for c, kk in surface if kk == k)
        per_seed = np.array([[warren_cowley(g, c) for g in surface[(c, k)]["nn"]]
                             for c in cs]).T
        uncertainty_band(ax, cs, per_seed.min(0), per_seed.max(0), hue)
        ax.plot(cs, per_seed.mean(0), color=hue, linewidth=1.4, marker="o",
                markersize=3, zorder=3, label=rf"$\sigma/\sigma_c = {k / STOP_GRID:.2f}$")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    if reference_half:
        ax.plot([0.5] * len(reference_half), list(reference_half.values()), color=REFERENCE_INK,
                linestyle="none", marker="x", markersize=6, zorder=4,
                label="Kawasaki chains, $c = 0.5$")
    ax.set_xlabel("composition $c$")
    ax.set_ylabel(r"Warren--Cowley $\alpha_1$")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(out, dpi=SAVEFIG_DPI)


CENTRE_STENCIL = (0.4375, 0.5, 0.5625)


def central_curvature(values_by_c: dict) -> float:
    """Second difference of F/d at c = 0.5 over the 1/16 composition step."""
    left, centre, right = (values_by_c[c] for c in CENTRE_STENCIL)
    return (left - 2 * centre + right) / (1 / 16) ** 2


def plot(surface: dict, reference: dict, out: Path, D: int) -> None:
    import matplotlib.pyplot as plt

    from discrete_flow_sampler.diagnostics.figure_style import (
        ANALYTIC_GUIDE, FULL_WIDTH_IN, MUTED, REFERENCE_INK, SAMPLER_HUE, SAVEFIG_DPI,
        parameter_ramp, style_axes, uncertainty_band, use_house_style)

    use_house_style()
    ks = sorted({k for _, k in surface})
    ramp = parameter_ramp(SAMPLER_HUE, len(ks))
    # No residual panel until a TI reference exists at this rung.
    if reference:
        fig, (ax, axr, axc) = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.7))
    else:
        fig, (ax, axc) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.7))
        axr = None

    cs_guide = np.linspace(0.2, 0.8, 200)
    ax.plot(cs_guide, [ideal_mixing_per_site(c, D) for c in cs_guide], color=ANALYTIC_GUIDE,
            linestyle=":", linewidth=1.0, label=r"$\sigma = 0$ (ideal mixing)")
    for hue, k in zip(ramp, ks):
        cs = sorted(c for c, kk in surface if kk == k)
        per_seed = np.array([surface[(c, k)]["F"] for c in cs]).T  # (seeds, c)
        uncertainty_band(ax, cs, per_seed.min(0), per_seed.max(0), hue)
        ax.plot(cs, per_seed.mean(0), color=hue, linewidth=1.4, marker="o",
                markersize=3, zorder=3, label=rf"$\sigma/\sigma_c = {k / STOP_GRID:.2f}$")
        with_ref = [c for c in cs if (c, k) in reference]
        if with_ref:
            residual = [surface[(c, k)]["F"] for c in with_ref]
            residual = np.array(residual).T - np.array([reference[(c, k)] for c in with_ref])
            uncertainty_band(axr, with_ref, residual.min(0), residual.max(0), hue)
            axr.plot(with_ref, residual.mean(0), color=hue, linewidth=1.4, marker="o",
                     markersize=3, zorder=3)
    if axr is not None:
        axr.axhline(0, color=MUTED, linewidth=0.8)
        axr.set_xlabel("composition $c$")
        axr.set_ylabel("$F/d$ residual vs TI truth")

    couplings = [k / STOP_GRID for k in ks]
    per_seed_curv = np.array([
        [central_curvature({c: surface[(c, k)]["F"][seed] for c in CENTRE_STENCIL})
         for k in ks]
        for seed in range(len(next(iter(surface.values()))["F"]))])
    uncertainty_band(axc, couplings, per_seed_curv.min(0), per_seed_curv.max(0), SAMPLER_HUE)
    axc.plot(couplings, per_seed_curv.mean(0), color=SAMPLER_HUE, linewidth=1.4,
             marker="o", markersize=3, zorder=3)
    ref_k = [k for k in ks if all((c, k) in reference for c in CENTRE_STENCIL)]
    if ref_k:
        axc.plot([k / STOP_GRID for k in ref_k],
                 [central_curvature({c: reference[(c, k)] for c in CENTRE_STENCIL}) for k in ref_k],
                 color=REFERENCE_INK, linestyle="none", marker="x", markersize=5, zorder=4,
                 label="TI truth")
    axc.axhline(0, color=MUTED, linewidth=0.8)
    axc.set_xlabel(r"$\sigma/\sigma_c$")
    axc.set_ylabel(r"$\partial_c^2 (F/d)$ at $c=0.5$")
    ax.set_xlabel("composition $c$")
    ax.set_ylabel("$F/d$ (nats per site)")
    for i, axis in enumerate([a for a in (ax, axr, axc) if a is not None]):
        style_axes(axis)
        axis.text(0.02, 1.02, f"({chr(97 + i)})", transform=axis.transAxes,
                  fontweight="bold", va="bottom")
    handles, labels = ax.get_legend_handles_labels()
    ref_handles, ref_labels = axc.get_legend_handles_labels()
    fig.legend(handles + ref_handles, labels + ref_labels, frameon=False, ncol=5,
               loc="lower center")
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(out, dpi=SAVEFIG_DPI)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "03_hard")
    parser.add_argument("--reference-dir", type=Path,
                        default=REPO_ROOT / "results" / "02_constrained_soft")
    parser.add_argument("--rung", type=int, default=8, choices=sorted(RUNGS))
    parser.add_argument("--seeds", default=None, help="default: the rung's own seeds")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/03_hard/fc_surface_<rung>x<rung>.png")
    args = parser.parse_args(argv)

    rung = RUNGS[args.rung]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(rung["seeds"])
    out = args.out or REPO_ROOT / "results" / "03_hard" / f"fc_surface_{args.rung}x{args.rung}.png"
    surface = mirror_missing(load_sampler_surface(args.results_dir, seeds, rung["template"]))
    reference = mirror_missing(load_reference(args.reference_dir, rung["ti_prefix"]))

    print(f"{'c':>7} {'sig/sig_c':>9} {'F/d mean':>10} {'seed sd':>8} {'ESS':>6} {'TI truth':>10} {'resid':>8}")
    table = []
    for (c, k) in sorted(surface, key=lambda ck: (ck[1], ck[0])):
        cell = surface[(c, k)]
        F = float(np.mean(cell["F"])); sd = float(np.std(cell["F"])); ess = float(np.mean(cell["ess"]))
        truth = reference.get((c, k))
        resid = F - truth if truth is not None else None
        table.append({"composition": c, "k": k, "sigma": k / STOP_GRID * SIGMA_C, "F_per_site": F,
                      "F_seed_sd": sd, "ess_fraction": ess, "ti_truth": truth, "residual": resid,
                      "warren_cowley": float(np.mean([warren_cowley(g, c) for g in cell["nn"]]))})
        print(f"{c:7.4f} {k / STOP_GRID:9.3f} {F:10.4f} {sd:8.4f} {ess:6.2f} "
              f"{'--' if truth is None else f'{truth:10.4f}':>10} "
              f"{'--' if resid is None else f'{resid:+8.4f}':>8}")
    out.with_suffix(".json").write_text(json.dumps(table, indent=2))
    plot(surface, reference, out, rung["D"])
    sro_out = out.with_name(out.stem + "_sro" + out.suffix)
    plot_sro(surface, sro_out, reference_sro_at_half(args.results_dir.parent, args.rung))
    print(f"wrote {out} and {sro_out}")


if __name__ == "__main__":
    main()
