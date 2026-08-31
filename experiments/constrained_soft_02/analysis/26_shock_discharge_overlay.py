"""fig:penalty-variance: the penalty-variance shock and its three fates.

The chapter's one 10x10 float (revamp plan, s95): estimator-integrand
variance over training for the three arms at the size where fixed
lambda=50 fails --

  parent  -- fixed lambda, no channel: three of four seeds never
             discharge the shock (final ESS 0.02/0.06/0.78/0.05).
  anneal  -- the lambda 10 -> 25 -> 50 schedule: variance starts low
             because the target starts easy, then the shock is REPAID at
             every boundary (10k, 20k; the batch-ESS collapse 3000 ->
             16-540 at the first boundary is the caption's number).
  channel -- exact-field channel, fixed lambda: the shock discharges
             within ~500 steps and stays down (finals 0.95/0.93/0.94/0.96).

All three arms are the matched ne64 families (parent 20260609, anneal
20260611, efc twins 20260829 -- the efc twin differs from the parent by
the channel alone, test-pinned). The y-axis is var_estimator_integrand
from training_log.csv: the POST-control-variate variance, i.e. the noise
the optimiser actually sees, which is why the parent's plateau is the
failure mechanism and not something more variance reduction could fix.
Log y; rolling-median smoothing (window 51 logged steps) so per-seed
lines stay readable without hiding the boundary spikes.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_FULL_WIDE_SINGLE, FONT_SIZE_ANNOTATION, MUTED,
    NEURAL_COMPARATOR_HUE, SAMPLER_HUE, style_axes, use_house_style)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "02_constrained_soft"

ARMS = (
    ("fixed $\\lambda$ (parent)", "S2_d10_c05_l50_letf_ne64_seed4?_*", MUTED),
    ("$\\lambda$-annealed", "S2_d10_c05_l50_letf_ne64_anneal_seed4?_*",
     NEURAL_COMPARATOR_HUE),
    ("exact-field channel", "S2_d10_c05_l50_letf_ne64_efc_seed4?_*",
     SAMPLER_HUE),
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
            log = pd.read_csv(Path(run_dir) / "training_log.csv",
                              usecols=["step", "var_estimator_integrand"])
            smoothed = (log.var_estimator_integrand
                        .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
                        .median())
            ax.plot(log.step, smoothed, color=colour, linewidth=0.8,
                    alpha=0.8, label=label if index == 0 else None)

    for boundary in ANNEAL_BOUNDARIES:
        ax.axvline(boundary, color=MUTED, linewidth=0.6, linestyle=":")
    ax.annotate("$\\lambda$ boundaries", xy=(ANNEAL_BOUNDARIES[0], 3e3),
                xytext=(4, 0), textcoords="offset points",
                fontsize=FONT_SIZE_ANNOTATION, color=MUTED)

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("estimator-integrand variance")
    ax.legend(fontsize=FONT_SIZE_ANNOTATION, loc="upper right")
    style_axes(ax)
    fig.tight_layout()

    out = RESULTS / "soft_shock_discharge.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
