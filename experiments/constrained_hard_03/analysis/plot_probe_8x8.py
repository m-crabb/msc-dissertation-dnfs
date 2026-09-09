"""Figures for the (sigma_c, 8x8) mixing-probe results section.

Reads the probe analysis output (probe_8x8_headline.json), the raw
replicate/chain artefacts, and the competitor chains' meta.json timing
records, and renders three PNGs:

  cost_quality_crossover.png -- cumulative N_eff(energy) against cumulative
      wall-clock seconds (log-log), one panel per operating point. Chain
      curves use the Sokal accumulation N_eff(t) = post-burn-in sweeps /
      (2 tau_int) with the per-chain tau_int (batch means), so the curve
      starts where the burn-in rule starts counting; endpoints agree with
      the headline table's Var/MSE N_eff up to the two constructions'
      usual gap (the Var/MSE form credits cross-chain averaging; printed
      at build time). The neural sampler appears twice: sampling cost
      alone, and sampling plus one-off training wall-clock.
  fidelity_coverage_sc.png -- two panels at the headline cell: the energy
      marginal (IS-weighted neural histogram, 8 replicates pooled, over
      the certified reference chains' post-discard histogram -- the
      DNFS-paper Figure-5 format) and the mode order parameter phi. Each
      panel annotates total variation against the reference and the 95%
      finite-sample noise floors of both the neural replicate pool and the
      competitor chains, so agreement is read against what perfect
      sampling would show at these effective sizes, not against zero.
  fidelity_coverage_s010.png -- the same figure at the subcritical floor.

Energy histogram support: the sigma-free slice energy x^T A x is integer-
valued on the +/-1 lattice (steps of 8 under swap moves), so both sides
are binned on the union of exact observed levels; no continuous binning
choice enters. TV floors reuse tv_noise_floor with the published
effective sizes: Kish ESS for the weighted neural pool and n/tau_int for
the chains, tau_int(energy) for the energy panel and the published
tau_int(phi)-based count for the phi panel.

Wall-clock provenance (each method on its own best hardware, disclosed):
  * neural eval: 92.4 s per 5,000-draw replicate on one NVIDIA A30
    (median of successive replicate-artefact mtime deltas, jobs 273209
    [sigma_c] and 273275 [floor]; deltas are metronome-regular, +/-1 s).
  * neural training (one-off): sigma_c record = ma_100k at the published
    end-to-end rate for this head and recipe (1.3 h per 50k steps,
    tab:head-ess-d8) -> 2.6 h; floor record = 50k run, 1.8 h continuous
    artefact span (config.json -> final checkpoint). The sigma_c record's
    own artefact span (44.6 h) crosses queue gaps and a resume, so it is
    unusable.
  * kawasaki chains: wall_seconds_run from each chain's meta.json
    (MC loop only, setup excluded), run on the Mac (Apple silicon CPU);
    local = numba nearest-neighbour (~40M proposals/s), nonlocal =
    literal mchammer (~130k proposals/s) -- the tool as shipped; a native
    all-pair implementation would sit near the numba rate, which the
    caption discloses.

Hues fixed per sampler entity (validated colourblind-safe, worst adjacent
CVD dE 22.4, light surface): neural #2a78d6 (house masked-attention hue),
kawasaki nonlocal #eda100 (house kawasaki hue), kawasaki local #8e63c5.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.constrained_hard_03.analysis.probe_analysis_8x8 import (
    LATTICE_SIDE,
    OPERATING_POINTS,
    kawasaki_chain_rows,
    observable_values,
    total_variation,
    tv_noise_floor,
)

from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_ALT_HUE as KAWASAKI_LOCAL_HUE,
)
from discrete_flow_sampler.diagnostics.figure_style import (
    CLASSICAL_HUE as KAWASAKI_NONLOCAL_HUE,
)
from discrete_flow_sampler.diagnostics.figure_style import MUTED
from discrete_flow_sampler.diagnostics.figure_style import REFERENCE_INK as INK
from discrete_flow_sampler.diagnostics.figure_style import SAMPLER_HUE as NEURAL_HUE
from discrete_flow_sampler.targets.ising import FixedCompositionIsingTarget

matplotlib.use("Agg")


torch.set_num_threads(2)  # the 16x16 reference chain owns the Mac's cores


SWEEPS_PER_CHAIN = 1_000_000
NEURAL_SECONDS_PER_REPLICATE = 92.4  # A30; mtime-delta method, see module docstring
NEURAL_TRAINING_SECONDS = {"sc": 2.6 * 3600, "s010": 1.8 * 3600}
POINT_TITLES = {"sc": r"$\sigma_c = 0.223$", "s010": r"$\sigma = 0.10$"}
RUN_DIR_DEFAULTS = {
    "sc": "results/03_hard/H2_d64_c50_s223_letf_ma_100k_curr_seed42_20260722-124315",
    "s010": "results/03_hard/H2_d64_c50_s010_letf_ma_50k_seed42_20260812-floor",
}
NOISE_FLOOR_SEED = 20260813


def _style_axis(ax):
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


# ---------------------------------------------------------------------------
# Cost-vs-quality crossover
# ---------------------------------------------------------------------------


def chain_wall_seconds(probe_root, point, variant):
    """Per-chain MC-loop wall seconds from meta.json, in chain order."""
    chain_dirs = sorted((probe_root / "competitor" / point / variant).glob("chain_*"))
    return [
        json.loads((d / "meta.json").read_text())["wall_seconds_run"]
        for d in chain_dirs
    ]


def chain_accumulation(chain_rows, wall_seconds, points_per_chain=64):
    """Ensemble-serial accumulation of the Sokal chain N_eff.

    Chains are charged serially (total cost = sum of chain wall-clocks;
    the 6-worker parallelism of the run is a throughput convenience, not
    a cost reduction). Within chain c the post-burn-in effective count is
    (sweeps(t) - burn_in_c) / (2 tau_c), zero before burn-in ends -- the
    scallop at each chain start is the burn-in being paid again.
    """
    xs, ys = [], []
    x_offset, y_offset = 0.0, 0.0
    for row, wall in zip(chain_rows, wall_seconds):
        tau = row["tau_int_batch_means_sweeps"]
        burn_in = row["burn_in_sweeps"]
        t = np.linspace(0.0, wall, points_per_chain)
        sweeps = t / wall * SWEEPS_PER_CHAIN
        n_eff = np.maximum(0.0, sweeps - burn_in) / (2.0 * tau)
        xs.append(x_offset + t)
        ys.append(y_offset + n_eff)
        x_offset += wall
        y_offset += n_eff[-1]
    return np.concatenate(xs), np.concatenate(ys)


def plot_cost_quality(analysis, probe_root, out_path, summary_rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, point in zip(axes, ("sc", "s010")):
        result = analysis["results"][point]
        variant_specs = [
            ("local", KAWASAKI_LOCAL_HUE, "kawasaki local (numba)"),
            ("nonlocal", KAWASAKI_NONLOCAL_HUE, "kawasaki non-local (mchammer)"),
        ]
        neural_n_eff = result["neural"]["n_eff_per_pass"]["observables"]["energy"][
            "n_eff"
        ]
        neural_rate = neural_n_eff / NEURAL_SECONDS_PER_REPLICATE
        gaps = []
        for variant, hue, label in variant_specs:
            rows = result["kawasaki"][variant]["chains"]
            walls = chain_wall_seconds(probe_root, point, variant)
            x, y = chain_accumulation(rows, walls)
            positive = y > 0
            ax.plot(
                x[positive],
                y[positive],
                color=hue,
                linewidth=1.6,
                zorder=3,
                label=label,
            )
            gaps.append((variant, (y[-1] / x[-1]) / neural_rate))
            summary_rows.append(
                {
                    "point": point,
                    "sampler": f"kawasaki_{variant}",
                    "wall_seconds_total": float(x[-1]),
                    "n_eff_energy_total": float(y[-1]),
                    "seconds_per_effective_sample": float(x[-1] / y[-1]),
                    "n_eff_per_second": float(y[-1] / x[-1]),
                }
            )

        replicate_index = np.arange(1, 9)
        sampling_x = replicate_index * NEURAL_SECONDS_PER_REPLICATE
        sampling_y = replicate_index * neural_n_eff
        ax.plot(
            sampling_x,
            sampling_y,
            color=NEURAL_HUE,
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            zorder=4,
            label="neural swap-CTMC (sampling)",
        )
        training = NEURAL_TRAINING_SECONDS[point]
        ax.plot(
            sampling_x + training,
            sampling_y,
            color=NEURAL_HUE,
            linewidth=1.4,
            linestyle=":",
            marker="o",
            markersize=3,
            zorder=4,
            label="neural incl. one-off training",
        )
        summary_rows.append(
            {
                "point": point,
                "sampler": "neural_ma",
                "wall_seconds_total": float(sampling_x[-1]),
                "n_eff_energy_total": float(sampling_y[-1]),
                "seconds_per_effective_sample": float(
                    NEURAL_SECONDS_PER_REPLICATE / neural_n_eff
                ),
                "n_eff_per_second": float(neural_rate),
                "training_seconds_one_off": float(training),
            }
        )

        # Reported when no break-even crossing occurs in range: post-burn-in
        # rates are constant, so the ratio holds at every later wall-clock.
        gap_text = "chain lead at equal wall-clock:\n" + "\n".join(
            f"  {variant}: {gap:,.0f}x" if gap >= 100 else f"  {variant}: {gap:.1f}x"
            for variant, gap in gaps
        )
        ax.text(
            0.97,
            0.05,
            gap_text,
            transform=ax.transAxes,
            fontsize=7,
            color=MUTED,
            ha="right",
            va="bottom",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(POINT_TITLES[point], fontsize=9, color=INK)
        ax.set_xlabel("cumulative wall-clock (s)", fontsize=8, color=MUTED)
        _style_axis(ax)
    axes[0].set_ylabel(r"cumulative $N_{\rm eff}$(energy)", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    # The report caption supplies the figure description.
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fidelity/coverage two-panel overlays (energy + phi)
# ---------------------------------------------------------------------------


def pooled_neural_energy(run_dir, target):
    """Energies + pooled IS weights across the 8 archived replicates.

    Weights: each replicate's log-weights are self-normalised in its own
    5,000-draw batch (the estimator the analysis scores), then the eight
    batches are averaged with equal replicate weight -- the same pooling
    the analysis's coverage block applies to the phi histogram.
    """
    energies, weights = [], []
    replicate_dirs = sorted(Path(run_dir).glob("eval_replicate_s*"))
    for replicate_dir in replicate_dirs:
        samples = torch.load(
            replicate_dir / "samples.pt", map_location="cpu", weights_only=True
        )
        log_w = torch.load(
            replicate_dir / "log_weights.pt", map_location="cpu", weights_only=True
        )
        energies.append(observable_values("energy", samples.float(), target).numpy())
        weights.append(torch.softmax(log_w, dim=0).numpy())
    n_replicates = len(replicate_dirs)
    return (np.concatenate(energies), np.concatenate(weights) / n_replicates)


def reference_energies(probe_root, point, target):
    """Pooled post-discard reference energies (second half of each chain,
    the same discard reference_block applies before trusting moments)."""
    chain_dirs = sorted((probe_root / "reference" / point).glob("chain_*"))
    pooled = []
    for chain_dir in chain_dirs:
        spins = np.load(chain_dir / "snapshots.npz")["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        trace = observable_values("energy", states, target).numpy()
        pooled.append(trace[trace.size // 2 :])
    return np.concatenate(pooled)


def mass_on_levels(values, weights, levels):
    """Weighted mass on exact integer energy levels (no binning choice:
    the slice energy is integer-valued, so levels are matched exactly)."""
    index = np.searchsorted(levels, np.rint(values))
    index = np.clip(index, 0, len(levels) - 1)
    mass = np.zeros(len(levels))
    np.add.at(mass, index, weights)
    return mass


def energy_panel_data(point, run_dir, probe_root, coverage):
    """Histograms + TV/floor numbers for one operating point's energy
    panel. Competitor TV uses the nonlocal chains (the carrier variant)
    with n/tau_int(energy) effective size; floors are seeded for
    reproducible figures."""
    target = FixedCompositionIsingTarget(
        D=LATTICE_SIDE,
        sigma=OPERATING_POINTS[point],
        target_composition=0.5,
        bias=0.0,
        device="cpu",
    )
    neural_vals, neural_w = pooled_neural_energy(run_dir, target)
    ref_vals = reference_energies(probe_root, point, target)
    kaw_rows = kawasaki_chain_rows(probe_root, point, "nonlocal", target)
    kaw_traces, kaw_n_eff = [], 0.0
    for chain_dir, row in zip(
        sorted((probe_root / "competitor" / point / "nonlocal").glob("chain_*")),
        kaw_rows,
    ):
        spins = np.load(chain_dir / "snapshots.npz")["spins"]
        states = torch.from_numpy(spins.astype(np.float32))
        trace = observable_values("energy", states, target).numpy()
        kept = (
            np.arange(1, trace.size + 1) * (SWEEPS_PER_CHAIN // trace.size)
            > row["burn_in_sweeps"]
        )
        kaw_traces.append(trace[kept])
        kaw_n_eff += (
            kept.sum()
            * (SWEEPS_PER_CHAIN // trace.size)
            / (2.0 * row["tau_int_batch_means_sweeps"])
        )
    kaw_vals = np.concatenate(kaw_traces)

    levels = np.unique(np.rint(np.concatenate([ref_vals, neural_vals, kaw_vals])))
    ref_hist = mass_on_levels(
        ref_vals, np.full(ref_vals.size, 1.0 / ref_vals.size), levels
    )
    neural_hist = mass_on_levels(neural_vals, neural_w, levels)
    kaw_hist = mass_on_levels(
        kaw_vals, np.full(kaw_vals.size, 1.0 / kaw_vals.size), levels
    )

    rng = np.random.default_rng(NOISE_FLOOR_SEED)
    neural_tv = total_variation(neural_hist, ref_hist)
    kaw_tv = total_variation(kaw_hist, ref_hist)
    neural_floor = tv_noise_floor(ref_hist, coverage["neural_effective_draws"], rng)
    kaw_floor = tv_noise_floor(ref_hist, kaw_n_eff, rng)
    return {
        "levels": levels,
        "ref_hist": ref_hist,
        "neural_hist": neural_hist,
        "neural_tv": neural_tv,
        "neural_floor": neural_floor,
        "kaw_tv": kaw_tv,
        "kaw_floor": kaw_floor,
        "kaw_n_eff_energy": float(kaw_n_eff),
    }


def _tv_annotation(ax, neural_tv, neural_floor, kaw_tv, kaw_floor, y):
    """Compact TV box in the panel's empty left region (the marginals'
    left tails carry no mass, so ha='left' text there never collides
    with the histograms; y clears the legend where one is present)."""
    neural_excess = max(0.0, neural_tv - neural_floor)
    kaw_excess = max(0.0, kaw_tv - kaw_floor)
    ax.text(
        0.03,
        y,
        f"TV(neural) = {neural_tv:.4f} (floor {neural_floor:.4f})\n"
        f"TV(kawasaki) = {kaw_tv:.4f} (floor {kaw_floor:.4f})\n"
        f"excess: {neural_excess:.4f} / {kaw_excess:.4f}",
        transform=ax.transAxes,
        fontsize=7,
        color=MUTED,
        ha="left",
        va="top",
    )


def plot_fidelity_coverage(
    analysis, point, run_dir, probe_root, out_path, summary_rows
):
    coverage = analysis["results"][point]["neural"]["coverage"]
    energy = energy_panel_data(point, run_dir, probe_root, coverage)

    fig, (ax_energy, ax_phi) = plt.subplots(1, 2, figsize=(9.5, 3.9))

    ax_energy.step(
        energy["levels"],
        energy["ref_hist"],
        where="mid",
        color=INK,
        linewidth=1.8,
        zorder=3,
        label=r"reference chains ($\hat R \leq 1.01$)",
    )
    ax_energy.step(
        energy["levels"],
        energy["neural_hist"],
        where="mid",
        color=NEURAL_HUE,
        linewidth=1.2,
        alpha=0.9,
        zorder=4,
        label="neural, IS-weighted (8 replicates pooled)",
    )
    _tv_annotation(
        ax_energy,
        energy["neural_tv"],
        energy["neural_floor"],
        energy["kaw_tv"],
        energy["kaw_floor"],
        y=0.74,
    )
    ax_energy.set_xlabel(r"slice energy $x^\top A x$", fontsize=8, color=MUTED)
    ax_energy.set_ylabel("probability mass", fontsize=8, color=MUTED)
    ax_energy.set_title(
        f"energy marginal  ({POINT_TITLES[point]})", fontsize=9, color=INK
    )
    _style_axis(ax_energy)
    ax_energy.legend(fontsize=7, frameon=False, loc="upper left")

    support = np.asarray(coverage["phi_support"], dtype=float)
    reference = np.asarray(coverage["reference_phi_hist"], dtype=float)
    neural = np.asarray(coverage["neural_phi_hist"], dtype=float)
    ax_phi.plot(
        support,
        reference / reference.sum(),
        color=INK,
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        zorder=3,
    )
    ax_phi.plot(
        support,
        neural / neural.sum(),
        color=NEURAL_HUE,
        linewidth=1.4,
        marker="o",
        markersize=3,
        alpha=0.9,
        zorder=4,
    )
    _tv_annotation(
        ax_phi,
        coverage["neural_phi_tv_vs_reference"],
        coverage["neural_tv_noise_floor_95"],
        coverage["kawasaki_phi_tv_vs_reference"],
        coverage["kawasaki_tv_noise_floor_95"],
        y=0.96,
    )
    ax_phi.set_xlabel(r"mode order parameter $\phi$", fontsize=8, color=MUTED)
    ax_phi.set_title(f"mode coverage  ({POINT_TITLES[point]})", fontsize=9, color=INK)
    _style_axis(ax_phi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    summary_rows.append(
        {
            "point": point,
            "figure": out_path.name,
            "energy_tv_neural": energy["neural_tv"],
            "energy_tv_floor_neural": energy["neural_floor"],
            "energy_tv_kawasaki": energy["kaw_tv"],
            "energy_tv_floor_kawasaki": energy["kaw_floor"],
            "energy_levels": int(energy["levels"].size),
        }
    )


def endpoint_sanity(analysis):
    """Print the tau-form endpoint against the analysis's Var/MSE table; the
    two differ only in credit for cross-chain averaging, so a large gap means
    a broken input."""
    for point in ("sc", "s010"):
        for variant in ("local", "nonlocal"):
            block = analysis["results"][point]["kawasaki"][variant]
            table_n_eff = block["n_eff"]["observables"]["energy"]["n_eff"]
            rows = block["chains"]
            tau_form = np.mean(
                [
                    (SWEEPS_PER_CHAIN - r["burn_in_sweeps"])
                    / (2.0 * r["tau_int_batch_means_sweeps"])
                    for r in rows
                ]
            )
            print(
                f"[plot] {point}/{variant}: tau-form N_eff/chain "
                f"{tau_form:,.0f} vs frozen Var/MSE {table_n_eff:,.0f} "
                f"(ratio {table_n_eff / tau_form:.2f})",
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-dir", default="results/03_hard/probe_8x8_headline")
    parser.add_argument("--probe-root", default="results/kawasaki_probe")
    parser.add_argument("--sc-run-dir", default=RUN_DIR_DEFAULTS["sc"])
    parser.add_argument("--s010-run-dir", default=RUN_DIR_DEFAULTS["s010"])
    args = parser.parse_args(argv)
    headline_dir = Path(args.headline_dir)
    probe_root = Path(args.probe_root)
    figures_dir = headline_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((headline_dir / "probe_8x8_headline.json").read_text())
    endpoint_sanity(analysis)
    summary_rows = []
    plot_cost_quality(
        analysis, probe_root, figures_dir / "cost_quality_crossover.png", summary_rows
    )
    fidelity_rows = []
    for point, run_dir in (("sc", args.sc_run_dir), ("s010", args.s010_run_dir)):
        plot_fidelity_coverage(
            analysis,
            point,
            run_dir,
            probe_root,
            figures_dir / f"fidelity_coverage_{point}.png",
            fidelity_rows,
        )
    summary_path = headline_dir / "cost_quality_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "neural_seconds_per_replicate": NEURAL_SECONDS_PER_REPLICATE,
                "neural_training_seconds": NEURAL_TRAINING_SECONDS,
                "wall_clock_provenance": "see plot_probe_8x8.py module docstring",
                "cost_quality_rows": summary_rows,
                "fidelity_rows": fidelity_rows,
            },
            indent=2,
        )
    )
    print(
        f"[plot] wrote 3 figures to {figures_dir} and {summary_path.name}", flush=True
    )


if __name__ == "__main__":
    main()
