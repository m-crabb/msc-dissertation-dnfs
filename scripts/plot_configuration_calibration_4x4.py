"""Six state-frequency panels with finite-sample reference and seed-wise metrics.

Each point is a configuration in one training seed, never a pooled seed mean.
The grey region is a pointwise 95% binomial count interval under q=pi, since
each cell of a multinomial histogram has that marginal. It is not a
simultaneous confidence band or a model-error interval. Zero observations
are placed on a labelled display row, not assigned a pseudocount.

The x-axis cut is fixed before inspecting generated counts. Full-support TV
and unvisited target mass include the small-probability states off the plot.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from discrete_flow_sampler.diagnostics.figure_style import (
    ANALYTIC_GUIDE,
    REFERENCE_INK,
    SAMPLER_HUE,
    style_axes,
    use_house_style,
)
from scripts.configuration_calibration_4x4 import OUTPUT, calibration_metrics


def load_panels(output):
    tasks = json.loads((output / "manifest.json").read_text())
    panels = {}
    for task in tasks:
        prefix = output / "counts" / task["run"]
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        data = np.load(prefix.with_suffix(".npz"))
        counts, probabilities = data["counts"], data["probabilities"]
        for key in ("panel", "training_seed", "n_samples", "redraw_seed"):
            if metadata[key] != task[key]:
                raise ValueError(f"mismatched {key} in {prefix}")
        if counts.sum() != task["n_samples"]:
            raise ValueError(f"incomplete draw: {prefix}")
        config = output / "inputs" / task["run"] / "config.json"
        if hashlib.sha256(config.read_bytes()).hexdigest() != metadata["config_sha256"]:
            raise ValueError(f"config provenance mismatch: {prefix}")
        for key, value in calibration_metrics(counts, probabilities).items():
            if not np.isclose(value, metadata[key], atol=1e-12, rtol=1e-12):
                raise ValueError(f"metric mismatch: {key} in {prefix}")
        entries = panels.setdefault(task["panel"], [])
        if entries:
            np.testing.assert_allclose(probabilities, entries[0][2], rtol=0, atol=1e-12)
        entries.append((metadata, counts, probabilities))
    return panels


def percent_range(values, digits=2):
    low, high = np.min(values), np.max(values)
    return f"{100 * low:.{digits}f}–{100 * high:.{digits}f}%"


def plot(output=OUTPUT):
    output = Path(output)
    panels = load_panels(output)
    use_house_style()
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.0), sharex=True, sharey=True)
    rows = [
        ("unconstrained", "Unconstrained"),
        ("hard", "Hard · patch head"),
        ("soft", r"Soft · $c=0.5$"),
    ]
    cutoff = 1e-7
    n = next(iter(panels.values()))[0][0]["n_samples"]
    zero_row = 0.25 / n
    guide = np.geomspace(cutoff, 0.999, 600)
    lower, upper = binom.ppf(0.025, n, guide) / n, binom.ppf(0.975, n, guide) / n
    report = {
        "n_per_checkpoint": n,
        "display_probability_cutoff": cutoff,
        "zero_display_position": zero_row,
        "panels": {},
    }
    for row, (key, label) in enumerate(rows):
        for col, suffix in enumerate(("s010", "sc")):
            entries = panels[f"{key}_{suffix}"]
            ax = axes[row, col]
            style_axes(ax, grid_axis="both")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.fill_between(
                guide,
                np.maximum(lower, zero_row),
                np.maximum(upper, zero_row),
                color=ANALYTIC_GUIDE,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
            ax.plot(guide, guide, color=REFERENCE_INK, linewidth=0.8, zorder=2)
            ax.axhline(zero_row, color=ANALYTIC_GUIDE, linestyle=":", linewidth=0.7)
            for metadata, counts, probabilities in entries:
                visible = probabilities >= cutoff
                visited = visible & (counts > 0)
                ax.scatter(
                    probabilities[visited],
                    counts[visited] / n,
                    s=2.0,
                    alpha=0.12,
                    linewidths=0,
                    color=SAMPLER_HUE,
                    rasterized=True,
                    zorder=3,
                )
                unvisited = visible & (counts == 0)
                ax.scatter(
                    probabilities[unvisited],
                    np.full(unvisited.sum(), zero_row),
                    s=3,
                    alpha=0.06,
                    marker="v",
                    linewidths=0,
                    color=SAMPLER_HUE,
                    rasterized=True,
                    zorder=3,
                )
            tv = [entry[0]["tv"] for entry in entries]
            unseen = [entry[0]["unvisited_target_mass"] for entry in entries]
            reference = entries[0][0]["reference"]
            note = (
                f"Empirical TV {percent_range(tv)}\n"
                f"Exact draws {100 * reference['tv_low']:.2f}–"
                f"{100 * reference['tv_high']:.2f}%\n"
                f"Unvisited mass {percent_range(unseen)}"
            )
            ax.text(
                0.025,
                0.965,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.6,
                color=REFERENCE_INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.87,
                    "pad": 1.5,
                },
            )
            ax.set_xlim(cutoff, 1.0)
            ax.set_ylim(min(cutoff, zero_row * 0.65), 1.0)
            ax.set_xticks([1e-6, 1e-4, 1e-2, 1.0])
            ticks = [zero_row, 1e-5, 1e-3, 1e-1]
            ax.set_yticks(ticks, ["0", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(r"$\sigma=0.1$" if col == 0 else r"$\sigma=\sigma_c$")
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            probabilities = entries[0][2]
            report["panels"][f"{key}_{suffix}"] = {
                "seeds": [entry[0]["training_seed"] for entry in entries],
                "below_display_target_mass": float(
                    probabilities[probabilities < cutoff].sum()
                ),
                "reference": reference,
                "runs": [entry[0] for entry in entries],
            }
    fig.supxlabel(r"Exact target probability $\pi(x)$", fontsize=9, y=0.035)
    fig.supylabel(
        r"Unweighted configuration frequency $\widehat q(x)=n_x/N$", fontsize=9, x=0.01
    )
    fig.text(
        0.53,
        0.007,
        "Blue: each training seed  ·  Grey: 95% pointwise exact-count range",
        ha="center",
        fontsize=7,
        color=ANALYTIC_GUIDE,
    )
    fig.tight_layout(rect=(0.035, 0.055, 1, 1), h_pad=1.1, w_pad=1)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"configuration_calibration_4x4.{ext}", dpi=300)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "configuration_calibration_4x4.pdf")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    plot(parser.parse_args().output)
