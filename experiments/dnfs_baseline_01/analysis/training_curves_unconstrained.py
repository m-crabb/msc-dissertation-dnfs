"""App I figure: unconstrained ESS-over-training, house standard (K4).

The appendix kit's training-curve figure for the unconstrained chapter: the
in-training evaluation ESS fraction over the 200k-step budget, both 10x10
families (subcritical sigma=0.1 and critical-curriculum), four seeds each as
the house seed-band grammar. The two families share one role (our sampler),
so they take a ``parameter_ramp`` on the sampler hue -- light = subcritical,
dark = critical -- rather than two roles.

EGM-style early-phase shading: the critical family
trains under a sigma curriculum, so until the final stage begins its ESS is
measured against the *current stage's* target, not the final sigma_c -- a
read against a moving goalpost. That span is shaded as unreliable rather
than cropped, so the reader sees the whole trajectory and knows which part
supports conclusions. The subcritical family has no curriculum; the shading
belongs to the critical trace only, which the caption states.

The y values are the IN-TRAINING diagnostic (n_eval_samples draws at each
eval step), not the frozen end-of-run evaluation the house table prints --
figure titles and captions carry no house-table number by the settled rule,
and the two reads genuinely differ (the frozen eval redraws under the final
parameters).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from discrete_flow_sampler.diagnostics.figure_style import (
    FIGSIZE_SINGLE,
    FONT_SIZE_ANNOTATION,
    GRID,
    MUTED,
    SAMPLER_HUE,
    SAVEFIG_DPI,
    parameter_ramp,
    seed_band,
    style_axes,
    use_house_style,
)


def load_family(run_dirs: list[Path]) -> dict:
    """Eval-step ESS fractions for one config family, aligned across seeds."""
    per_seed, steps = [], None
    config = json.loads((run_dirs[0] / "config.json").read_text())
    n_eval = config["eval"]["n_eval_samples"]
    for run_dir in run_dirs:
        log = pd.read_csv(run_dir / "training_log.csv", usecols=["step", "ess"]).dropna(
            subset=["ess"]
        )
        if steps is None:
            steps = log["step"].to_numpy()
        else:
            log = log[log["step"].isin(steps)]
            assert len(log) == len(steps), (
                f"{run_dir.name}: eval steps misaligned across seeds"
            )
        per_seed.append(log["ess"].to_numpy() / n_eval)
    curriculum = config.get("curriculum")
    final_stage_start = curriculum["stages"][-1]["start_step"] if curriculum else None
    return {
        "steps": steps,
        "ess_fractions": per_seed,
        "sigma": config["ising"]["sigma"],
        "final_stage_start": final_stage_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--budget_runs",
        required=True,
        nargs="+",
        type=Path,
        help="stage_4_d10_budget run dirs (sigma=0.1)",
    )
    parser.add_argument(
        "--critical_runs",
        required=True,
        nargs="+",
        type=Path,
        help="stage_4_d10_critical run dirs (legacy family "
        "until the _sc retrains land, then those)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("training_curves_unconstrained.png")
    )
    args = parser.parse_args()
    use_house_style()

    subcritical = load_family(args.budget_runs)
    critical = load_family(args.critical_runs)
    light_hue, dark_hue = parameter_ramp(SAMPLER_HUE, 2)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ramp_end = critical["final_stage_start"]
    ax.axvspan(0, ramp_end / 1e3, color=GRID, zorder=0)
    ax.text(
        ramp_end / 1e3 * 0.5,
        1.035,
        "$\\sigma$ curriculum ramp",
        fontsize=FONT_SIZE_ANNOTATION,
        color=MUTED,
        ha="center",
    )
    seed_band(
        ax,
        subcritical["steps"] / 1e3,
        subcritical["ess_fractions"],
        light_hue,
        f"$\\sigma = {subcritical['sigma']:g}$",
    )
    seed_band(
        ax,
        critical["steps"] / 1e3,
        critical["ess_fractions"],
        dark_hue,
        "$\\sigma_c$ (curriculum)",
    )
    ax.set_xlabel(r"training step ($10^3$)")
    ax.set_ylabel("ESS fraction (in-training eval)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, critical["steps"].max() / 1e3)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [
            f"$\\sigma = {subcritical['sigma']:g}$ (4 seeds)",
            "$\\sigma_c$ curriculum (4 seeds)",
        ],
        fontsize=FONT_SIZE_ANNOTATION,
        frameon=False,
        loc="lower right",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=SAVEFIG_DPI)

    for name, family in (("subcritical", subcritical), ("critical", critical)):
        final = torch.tensor([f[-1] for f in family["ess_fractions"]])
        print(
            f"{name}: last in-training eval ESS fraction "
            f"{final.mean():.3f} +/- {final.std():.3f} "
            f"(diagnostic read, NOT the frozen eval)"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
