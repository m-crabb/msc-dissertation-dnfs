"""Figure `hard_composition_transfer`: ESS across composition slices for the amortised
and centre-trained hard samplers at 8x8 and 16x16, both on the dense 63-slice
probe (c = k/64: every slice at 8x8, every fourth at 16x16) over three training
seeds. Existing evaluations only; no sampling or training. Run templates come
from `zero_shot_tables`, so both exhibits move together when a run tag changes.

The centre-trained curves are the sparse six-composition probes of
`tab:zero-shot-composition`. Show measured compositions directly, without
mirroring the learned sampler: the target's spin-flip symmetry does not
establish model equivariance.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from experiments.constrained_hard_03.analysis.zero_shot_tables import (
    CAMORT_TEMPLATE,
    D64_TEMPLATE,
    D256_CAMORT_TEMPLATE,
    D256_TEMPLATE,
)

from discrete_flow_sampler.diagnostics.figure_style import (
    MUTED,
    NEURAL_COMPARATOR_HUE,
    SAMPLER_HUE,
    style_axes,
    use_house_style,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

REPO = Path(__file__).resolve().parents[3]

OUT = REPO / "results" / "03_hard"
RESULTS = REPO / "results" / "03_hard"
TRAINED = [0.3125, 0.375, 0.4375, 0.46875, 0.5]


def read_series(template, seeds, filename):
    values = {}
    for seed in seeds:
        path = RESULTS / template.format(seed=seed) / filename
        payload = json.loads(path.read_text())
        assert payload["checkpoint"] == "final_ema.pt"
        for row in payload["rows"]:
            if abs(row["stop_time"] - 1) < 1e-6:
                assert row["n_samples"] == 5000
                values.setdefault(row["composition"], []).append(row["ess_fraction"])
    assert all(len(cell) == len(seeds) for cell in values.values())
    compositions = np.array(sorted(values))
    return compositions, np.array([values[c] for c in compositions])


def curve(ax, template, seeds, filename, colour, label, linestyle="-"):
    compositions, values = read_series(template, seeds, filename)
    ax.plot(
        compositions,
        values.mean(axis=1),
        linestyle=linestyle,
        marker="o",
        markersize=2.5,
        color=colour,
        label=label,
    )
    if len(seeds) > 1:
        ax.fill_between(
            compositions,
            values.min(axis=1),
            values.max(axis=1),
            color=colour,
            alpha=0.16,
            linewidth=0,
        )
    return {str(c): list(v) for c, v in zip(compositions, values)}


use_house_style()
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True)
record = {}
for ax in axes:
    style_axes(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1.06), xlabel="Composition $c$")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.plot(
        TRAINED,
        [1.035] * len(TRAINED),
        "|",
        color=MUTED,
        markersize=7,
        markeredgewidth=1.3,
        clip_on=False,
    )
axes[0].set_ylabel("ESS fraction")
axes[0].set_title(r"(a) $8\times8$: three training seeds", loc="left", fontsize=9)
axes[1].set_title(r"(b) $16\times16$: three training seeds", loc="left", fontsize=9)
record["d64_amortised"] = curve(
    axes[0],
    CAMORT_TEMPLATE,
    (42, 43, 44),
    "zero_shot_fc_grid.json",
    SAMPLER_HUE,
    "Amortised",
)
record["d64_specialist"] = curve(
    axes[0],
    D64_TEMPLATE,
    (42, 43, 44),
    "zero_shot_transfer.json",
    NEURAL_COMPARATOR_HUE,
    "Centre-trained, zero-shot",
    "--",
)
record["d256_amortised"] = curve(
    axes[1],
    D256_CAMORT_TEMPLATE,
    (42, 43, 44),
    "zero_shot_fc_grid.json",
    SAMPLER_HUE,
    "Amortised",
)
record["d256_specialist"] = curve(
    axes[1],
    D256_TEMPLATE,
    (42, 43, 44),
    "zero_shot_transfer.json",
    NEURAL_COMPARATOR_HUE,
    "Centre-trained, zero-shot",
    "--",
)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.045),
    ncol=2,
    frameon=False,
    fontsize=8,
)
fig.text(
    0.5,
    0.015,
    "Critical coupling; EMA; 5000 draws per slice. "
    "Top ticks: five training compositions.",
    ha="center",
    fontsize=7.5,
    color=MUTED,
)
fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.26, wspace=0.16)
for extension in ("pdf", "png"):
    fig.savefig(OUT / f"hard_composition_transfer.{extension}", dpi=220)
(OUT / "hard_composition_transfer_values.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print("wrote", OUT / "hard_composition_transfer.pdf")
