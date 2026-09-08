"""Appendix figure: hard-chapter ESS-over-training at 16x16, house standard.

The hard twin of the unconstrained and soft training-curve panels: the in-training
evaluation ESS fraction over the 100k-step sigma_c curriculum at 16x16 for the two heads
that decide the family -- the two-hole patch head (R=2, the record) and the masked-
attention band with one sweep (dead 3/3 at sigma_c). Three seeds each as the house
seed-band grammar; the two families share one role (our sampler) so they take a
``parameter_ramp`` on the sampler hue -- dark = the head the chapter carries, light = the
head it retires at this size.

The curriculum ramp (stages before the final sigma_c stage) is shaded as in the sibling
panels: until the final stage begins the diagnostic reads against the current stage's
target, not the final coupling.

The y values are the in-training diagnostic (`n_eval_samples_training` draws, reduced
precision, every 500 steps), not the frozen fp32 evaluation the house table prints.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    n_eval = (
        config["eval"].get("n_eval_samples_training")
        or config["eval"]["n_eval_samples"]
    )
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
    stages = config["curriculum"]["stages"]
    return {
        "steps": steps,
        "ess_fractions": per_seed,
        "final_stage_start": stages[-1]["start_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patch_runs",
        required=True,
        nargs="+",
        type=Path,
        help="two-hole patch head R=2 sigma_c run dirs (3 seeds)",
    )
    parser.add_argument(
        "--band_runs",
        required=True,
        nargs="+",
        type=Path,
        help="masked-attention band, one sweep, sigma_c run dirs (3 seeds)",
    )
    parser.add_argument("--out", type=Path, default=Path("training_curves_hard.png"))
    args = parser.parse_args()
    use_house_style()

    patch = load_family(args.patch_runs)
    band = load_family(args.band_runs)
    light_hue, dark_hue = parameter_ramp(SAMPLER_HUE, 2)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ramp_end = patch["final_stage_start"]
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
        band["steps"] / 1e3,
        band["ess_fractions"],
        light_hue,
        "masked-attention band, one sweep",
    )
    seed_band(
        ax,
        patch["steps"] / 1e3,
        patch["ess_fractions"],
        dark_hue,
        "two-hole patch head, $R = 2$",
    )
    ax.set_xlabel(r"training step ($10^3$)")
    ax.set_ylabel("ESS fraction (in-training eval)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, patch["steps"].max() / 1e3)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [
            f"masked-attention band, one sweep ({len(band['ess_fractions'])} seeds)",
            f"two-hole patch head, $R = 2$ ({len(patch['ess_fractions'])} seeds)",
        ],
        fontsize=FONT_SIZE_ANNOTATION,
        frameon=False,
        loc="center right",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=SAVEFIG_DPI)

    for name, family in (("patch R=2", patch), ("masked-attention", band)):
        final = [f[-1] for f in family["ess_fractions"]]
        print(
            f"{name}: last in-training eval ESS fraction per seed "
            f"{[round(v, 3) for v in final]} (diagnostic read, NOT the frozen eval)"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
