"""Archive table: obedience slope and Z2 mirror mismatch for every swept run.

Two numbers per run, both read off artefacts already on disk
(`eval/composition_sweep.json`), so this is a pure re-derivation with no GPU
and no training:

  * **obedience slope** -- least-squares fit of delivered onto requested
    composition over the claim band [0.3, 0.7]. 1.0 is perfect obedience;
    0.0 means the model emits one distribution whatever it is asked for.
  * **Z2 mirror mismatch** -- the constrained target family is symmetric under
    a global spin flip, which maps composition to 1-comp and the penalty
    (comp - c)^2 to (comp - (1-c))^2. The target at c is therefore the exact
    mirror of the target at 1-c, so a correct model must satisfy
    `delivered(c) = 1 - delivered(1-c)`. Nothing in the LeT architecture
    enforces this (it reads tokens in {0,1} with no spin-flip equivariance),
    so the residual measures learning rather than restating an identity. It
    needs no exact enumeration, which is what makes it usable at D=10 where
    `enumerate_states` is impossible.

Why this script exists: the per-family slope comparison carries a
correction that matters --
grouping by family shows the "conditioning is attenuated, slope 0.39" reading
came from a pre-fix family that averaged dead seeds together with healthy
ones. Runs are therefore grouped by config family (name minus the `_seedNN`
suffix) and BOTH the per-seed values and the healthy count are emitted: a
family mean alone is exactly what produced the wrong conclusion.

Health threshold is a band fixed in advance (slope in [0.9, 1.1]) so the
count is not tuned to the data it summarises.

Usage:
    python -m experiments.constrained_soft_02.analysis.obedience_slope_table
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.constrained_soft_02.analysis._common import REVAMP_GRID

CLAIM_BAND = (0.30, 0.70)
HEALTHY_BAND = (0.9, 1.1)


def slope_and_mirror(rows: list[dict]) -> tuple[float, float, float, str]:
    """Return (slope, mean Z2 mismatch, delivered span, fit grid name).

    Grid convention: a sweep on the revamp grid is fitted over ALL nine
    points, because the exact reference it scores against (0.9950 at
    lambda=50, obedience_reference_revamp_grid.json) was fitted that way;
    restricting to the claim band here would compare slopes fitted on
    different point sets and call the difference "the model". Legacy
    sweeps keep the claim-band fit and its 0.976-family references.
    """
    delivered = {r["composition"]: r["composition_mean"] for r in rows}
    if set(delivered) == set(REVAMP_GRID):
        band, grid_name = delivered, "revamp"
    else:
        band = {c: m for c, m in delivered.items()
                if CLAIM_BAND[0] - 1e-9 <= c <= CLAIM_BAND[1] + 1e-9}
        grid_name = "claim_band"
    requested = np.array(sorted(band))
    realised = np.array([band[c] for c in requested])
    slope = float(np.polyfit(requested, realised, 1)[0]) if len(band) >= 3 else np.nan

    mismatches = [
        delivered[c] - (1.0 - delivered[round(1.0 - c, 4)])
        for c in delivered if round(1.0 - c, 4) in delivered and c <= 0.5
    ]
    mirror = float(np.mean(mismatches)) if mismatches else np.nan
    return slope, mirror, float(realised.max() - realised.min()), grid_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/02_constrained_soft")
    parser.add_argument("--eval_dir", choices=["eval", "eval_ema"],
                        default="eval")
    parser.add_argument("--out", default="results/02_constrained_soft/"
                                         "obedience_slope_table.csv")
    args = parser.parse_args()

    records = []
    for sweep_path in sorted(
            Path(args.results).glob(f"*/{args.eval_dir}/composition_sweep.json")):
        run_dir = sweep_path.parent.parent
        cfg = json.loads((run_dir / "config.json").read_text())
        rows = json.loads(sweep_path.read_text())
        if len(rows) < 3:
            continue
        slope, mirror, span, grid_name = slope_and_mirror(rows)
        name = run_dir.name
        records.append({
            "run": name,
            "family": name.rsplit("_seed", 1)[0],
            "D": cfg["ising"]["D"],
            "conditioned": cfg["model"].get("condition_on_composition", False),
            "fit_grid": grid_name,
            "slope": round(slope, 4),
            "z2_mirror_mismatch": round(mirror, 4),
            "delivered_span": round(span, 4),
            "healthy": HEALTHY_BAND[0] <= slope <= HEALTHY_BAND[1],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    families = defaultdict(list)
    for record in records:
        families[(record["D"], record["family"])].append(record)
    print(f"{'D':>3}  {'family':<50}{'n':>3}{'mean':>8}{'healthy':>9}   slopes")
    for (dim, family), group in sorted(
        families.items(), key=lambda kv: (kv[0][0], -np.mean([r["slope"] for r in kv[1]]))
    ):
        slopes = [r["slope"] for r in group]
        healthy = sum(r["healthy"] for r in group)
        print(f"{dim:>3}  {family[:50]:<50}{len(group):>3}{np.mean(slopes):>8.3f}"
              f"{healthy:>6}/{len(group):<2}   "
              f"{', '.join(f'{s:.2f}' for s in sorted(slopes))}")
    print(f"\nwrote {out_path} ({len(records)} runs)")


if __name__ == "__main__":
    main()
