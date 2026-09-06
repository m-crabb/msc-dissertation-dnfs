"""fig:penalty-variance: the penalty-variance shock and its three fates.

Drawn at the 8x8 sigma_c centre window:
estimator-integrand variance over training for three house arms, one
declared lever apart --

  no channel -- the house recipe minus the exact-field channel: no seed
                discharges the shock (final ESS 0.005 +/- 0.006).
  anneal     -- no channel + the lambda 10 -> 25 -> 50 schedule at
                0/10k/20k: the shock is deferred and repaid at each
                boundary (raw finals 0.40 +/- 0.21, 3/4 over 0.30).
  channel    -- the house recipe: the shock discharges early and stays
                down (finals 0.71 +/- 0.03).

The y-axis is var_estimator_integrand from training_log.csv: the
POST-control-variate variance, i.e. the noise the optimiser actually
sees, which is why the no-channel plateau is the failure mechanism and
not something more variance reduction could fix (the naive/post ratio at
the end of training is 1.7-2.7x for nochan vs 27-28x for the channel).
Log y; rolling-median smoothing (window 51 logged steps) so per-seed
lines stay readable without hiding the boundary spikes.
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_FULL_WIDE_SINGLE,
    FONT_SIZE_ANNOTATION,
    MUTED,
    NEURAL_COMPARATOR_HUE,
    SAMPLER_HUE,
    style_axes,
    use_house_style,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"

ARMS = (
    ("no channel", "S2_d8_c0500_l50_letf_ne128_house_sc_nochan_seed4?_*", MUTED),
    (
        "$\\lambda$-annealed, no channel",
        "S2_d8_c0500_l50_letf_ne128_house_sc_anneal_seed4?_*",
        NEURAL_COMPARATOR_HUE,
    ),
    (
        "exact-field channel",
        "S2_d8_c0500_l50_letf_ne128_house_sc_seed4?_*",
        SAMPLER_HUE,
    ),
)
ANNEAL_BOUNDARIES = (10_000, 20_000)
SMOOTH_WINDOW = 51


def main() -> None:
    use_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL_WIDE_SINGLE)

    for label, pattern, colour in ARMS:
        run_dirs = sorted(glob.glob(str(RESULTS / pattern)))
        assert len(run_dirs) == 4, (pattern, run_dirs)
        for index, run_dir in enumerate(run_dirs):
            log = pd.read_csv(
                Path(run_dir) / "training_log.csv",
                usecols=["step", "var_estimator_integrand"],
            )
            smoothed = log.var_estimator_integrand.rolling(
                SMOOTH_WINDOW, center=True, min_periods=1
            ).median()
            ax.plot(
                log.step,
                smoothed,
                color=colour,
                linewidth=0.8,
                alpha=0.8,
                label=label if index == 0 else None,
            )

    for boundary in ANNEAL_BOUNDARIES:
        ax.axvline(boundary, color=MUTED, linewidth=0.6, linestyle=":")
    ax.annotate(
        "$\\lambda$ boundaries",
        xy=(ANNEAL_BOUNDARIES[0], 3e3),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=FONT_SIZE_ANNOTATION,
        color=MUTED,
    )

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("estimator-integrand variance")
    ax.legend(fontsize=FONT_SIZE_ANNOTATION, loc="upper right")
    style_axes(ax)
    fig.tight_layout()

    out = RESULTS / "soft_shock_discharge_8x8_sc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
