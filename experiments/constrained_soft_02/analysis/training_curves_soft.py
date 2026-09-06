"""App I figure: soft-chapter ESS-over-training, house standard (K4).

The appendix kit's training-curve figure for the soft chapter, mirroring
training_curves_unconstrained: in-training evaluation ESS fraction over
the 50k-step budget at the headline window (10x10, lambda=50, c=0.50),
four seeds each as the house seed-band grammar. Two families, one role
(our sampler), so a ``parameter_ramp`` on the sampler hue rather than two
roles: light = fixed lambda=50 (the instability the chapter reports:
penalty variance ~ lambda^2 kills most seeds early), dark = the
lambda-annealed rescue (10 -> 25 -> 50), which is the recipe every
printed F(c) window trains under.

EGM-style early-phase shading on the annealed family's ramp span: until
the final lambda stage begins, its ESS is measured against the *current
stage's* soft target (a moving goalpost), so that span is shaded as
unreliable rather than cropped. The fixed family has no schedule; its
whole trace reads against the final target.

The fixed-lambda family predates the ne128 recipe (n_e = 64); the caption
carries that clause. The in-training read is the diagnostic, not the
frozen eval the house table prints -- no table number appears here
(settled rule).
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
    lambda_curriculum = config.get("lambda_curriculum")
    final_stage_start = (
        lambda_curriculum["stages"][-1]["start_step"] if lambda_curriculum else None
    )
    return {
        "steps": steps,
        "ess_fractions": per_seed,
        "final_stage_start": final_stage_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixed_runs",
        required=True,
        nargs="+",
        type=Path,
        help="fixed lambda=50 run dirs (c=0.50, seeds 42-45)",
    )
    parser.add_argument(
        "--annealed_runs",
        required=True,
        nargs="+",
        type=Path,
        help="lambda-annealed run dirs (c=0.50, seeds 42-45)",
    )
    parser.add_argument("--out", type=Path, default=Path("training_curves_soft.png"))
    args = parser.parse_args()
    use_house_style()

    fixed = load_family(args.fixed_runs)
    annealed = load_family(args.annealed_runs)
    light_hue, dark_hue = parameter_ramp(SAMPLER_HUE, 2)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ramp_end = annealed["final_stage_start"]
    ax.axvspan(0, ramp_end / 1e3, color=GRID, zorder=0)
    ax.text(
        ramp_end / 1e3 * 0.5,
        1.035,
        "$\\lambda$ anneal ramp",
        fontsize=FONT_SIZE_ANNOTATION,
        color=MUTED,
        ha="center",
    )
    seed_band(
        ax,
        fixed["steps"] / 1e3,
        fixed["ess_fractions"],
        light_hue,
        "fixed $\\lambda=50$",
    )
    seed_band(
        ax, annealed["steps"] / 1e3, annealed["ess_fractions"], dark_hue, "annealed"
    )
    ax.set_xlabel(r"training step ($10^3$)")
    ax.set_ylabel("ESS fraction (in-training eval)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, annealed["steps"].max() / 1e3)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [
            "fixed $\\lambda = 50$ (4 seeds)",
            "$\\lambda$ annealed $10{\\to}25{\\to}50$ (4 seeds)",
        ],
        fontsize=FONT_SIZE_ANNOTATION,
        frameon=False,
        loc="center right",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=SAVEFIG_DPI)

    for name, family in (("fixed", fixed), ("annealed", annealed)):
        final = torch.tensor([f[-1] for f in family["ess_fractions"]])
        print(
            f"{name}: last in-training eval ESS fraction "
            f"{final.mean():.3f} +/- {final.std():.3f} "
            f"(diagnostic read, NOT the frozen eval)"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
