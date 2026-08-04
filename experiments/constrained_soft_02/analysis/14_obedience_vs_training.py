"""Does composition obedience improve with training, or is it stuck?

The D=10 amortised run reached the end of training with a healthy sampler but
a broken controller: eval ESS at c=0.5 was 0.392 (fine), while the obedience
slope d(delivered)/d(requested) over the claim band [0.3, 0.7] was 0.206
(the pre-registered gate wanted >= 0.9). A single end-of-run number cannot
distinguish the two explanations that matter:

  * UNDERTRAINED — the slope is climbing and simply ran out of steps. The
    optimisation was throttled by grad-clip 50 (it learns ~20x slower than the
    clip-500 twin), so more steps or a faster spine would land the claim.
  * STUCK — the slope is flat across the whole final coverage stage. Then no
    amount of the same training fixes it, and the limit is capacity, the
    conditioning pathway, or the estimator's signal-to-noise at this D.

This script separates them by re-drawing the sweep from step-tagged
checkpoints and fitting the slope at each. Only checkpoints from the FINAL
coverage stage (half-width 0.20, from step 36k) are directly comparable to one
another: earlier ones were trained on a narrower window, so their behaviour at
c = 0.3 is extrapolation and a rising slope across a widening would be an
artefact of coverage rather than of learning. The two pre-36k checkpoints are
drawn anyway, reported separately, as the coverage-era context.

A confound this design has to survive: the run ends in an excursion/recovery
limit cycle, so individual checkpoints land in trough or peak states almost at
random (grad median 36.7 at step 50k versus 2,147 at 37.5k). Reading a trend
off two checkpoints would measure that luck. Sampling every 2.5k across the
final stage means the cycle averages out and a real trend has to show through
it -- which is why the trend is fitted over all final-stage points rather than
taken as an endpoint difference.

Rows use common random numbers (`composition_sweep` reseeds per composition),
so differences down a column are the model's response to c, not draw noise.
`save=False` throughout: the archived `eval/composition_sweep.json` is the
run's recorded result and must not be overwritten by this diagnostic.

Usage (needs the GPU env; ~90 s per composition per checkpoint):
    python -m experiments.constrained_soft_02.analysis.14_obedience_vs_training \
        --run-dir results/02_constrained_soft/<run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.dnfs_baseline_01.run import composition_sweep

# The claim band exactly: half-width 0.20 about 0.5 is what the run was
# finally trained to cover, so this is where obedience is being asserted.
CLAIM_BAND = (0.30, 0.40, 0.50, 0.60, 0.70)

# Final-coverage-stage checkpoints (half-width 0.20 begins at step 36k) plus
# two earlier ones for context. `final.pt` is the step-50k state; there is no
# step_050000.pt.
FINAL_STAGE = ("step_037500.pt", "step_040000.pt", "step_042500.pt",
               "step_045000.pt", "step_047500.pt", "final.pt")
EARLIER = ("step_027500.pt", "step_032500.pt")


def obedience_slope(rows: list[dict]) -> tuple[float, float]:
    """Least-squares slope and intercept of delivered vs requested composition.

    Slope 1.0 is perfect obedience; 0.0 means the model ignores the request
    entirely and emits one distribution whatever it is asked for.
    """
    requested = np.array([r["composition"] for r in rows])
    delivered = np.array([r["composition_mean"] for r in rows])
    slope, intercept = np.polyfit(requested, delivered, 1)
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None,
                        help="JSON path; defaults to <run-dir>/eval/"
                             "obedience_vs_training.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else (
        run_dir / "eval" / "obedience_vs_training.json"
    )

    results = []
    for checkpoint in EARLIER + FINAL_STAGE:
        if not (run_dir / "checkpoints" / checkpoint).exists():
            print(f"skip {checkpoint}: absent")
            continue
        rows = composition_sweep(
            run_dir, CLAIM_BAND, checkpoint=checkpoint, save=False,
        )
        slope, intercept = obedience_slope(rows)
        # Delivered span is the honest companion to the slope: a model pinned
        # to a single distribution has span ~0 regardless of how the fit lands.
        delivered = [r["composition_mean"] for r in rows]
        record = {
            "checkpoint": checkpoint,
            "final_stage": checkpoint in FINAL_STAGE,
            "slope": slope,
            "intercept": intercept,
            "delivered_span": max(delivered) - min(delivered),
            "delivered_min": min(delivered),
            "delivered_max": max(delivered),
            "ess_at_c05": next(r["ess_fraction"] for r in rows
                               if abs(r["composition"] - 0.5) < 1e-9),
            "rows": rows,
        }
        results.append(record)
        print(f"{checkpoint:>18}  slope {slope:6.3f}  span "
              f"{record['delivered_span']:.4f}  "
              f"[{min(delivered):.3f}, {max(delivered):.3f}]  "
              f"ESS@0.5 {record['ess_at_c05']:.4f}")

    final_stage = [r for r in results if r["final_stage"]]
    summary = {"n_final_stage": len(final_stage)}
    if len(final_stage) >= 3:
        steps = np.array([
            50000 if r["checkpoint"] == "final.pt"
            else int(r["checkpoint"].split("_")[1].split(".")[0])
            for r in final_stage
        ])
        for key in ("slope", "delivered_span"):
            values = np.array([r[key] for r in final_stage])
            trend = float(np.polyfit(steps, values, 1)[0]) * 10000
            summary[f"{key}_per_10k"] = trend
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_range"] = [float(values.min()), float(values.max())]
        print(f"\nfinal-stage trend: slope {summary['slope_per_10k']:+.4f} "
              f"per 10k steps (mean {summary['slope_mean']:.3f}, "
              f"range {summary['slope_range'][0]:.3f}-"
              f"{summary['slope_range'][1]:.3f})")
        print(f"final-stage trend: span  {summary['delivered_span_per_10k']:+.4f}"
              f" per 10k steps (mean {summary['delivered_span_mean']:.4f})")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(
        {"run_dir": str(run_dir), "summary": summary, "checkpoints": results},
        indent=2,
    ))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
