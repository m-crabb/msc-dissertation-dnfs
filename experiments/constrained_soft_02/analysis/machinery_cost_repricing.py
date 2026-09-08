"""Archive table: the conditioning-machinery cost, re-priced at the final recipe.

The dissertation's original machinery price (0.155 of ESS fraction) came from
a null/specialist pair run at the *pre-fix* recipe -- 10k steps at the
inherited gradient clip of 500 -- so it confounds the conditioning path with
short training under a saturating clip. The re-pricing pair
(`S2_d4_cnull_50k_l50_letf_anneal_offset_clip50` and its `_c05_` twin,
jobs 271495/271496) holds everything at the delivered recipe and differs only
in the conditioning path; the two admissible outcomes are recorded in
`configs.py`.

Three numbers are archived, all seed means over seeds 42-45:

  * machinery cost = specialist - null, both read from `eval/metrics.json`
    `ess_fraction` at c = 0.5. The null is the conditioned cell with a
    zero-width draw window: the same target reached through the whole
    conditioning path, so the difference isolates the machinery.
  * range-drawing gap = null - amortised-at-centre, the amortised family's
    c = 0.5 row from `eval/composition_sweep.json`. With the machinery priced
    separately, this is the part of the amortised family's centre deficit
    attributable to training over a range of compositions.
  * the per-seed values behind both, so the prose's "zero to within seed
    noise" is reproducible (the specialist's spread is 4x the null's, driven
    by seed 43).

No machinery cost may be quoted if a null seed fails to train (ESS fraction
< 0.1); the script asserts that veto rather than reporting around it.

Usage:
    python -m experiments.constrained_soft_02.analysis.machinery_cost_repricing
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("results/02_constrained_soft")
NULL_CELL = "S2_d4_cnull_50k_l50_letf_anneal_offset_clip50"
SPECIALIST_CELL = "S2_d4_c05_50k_l50_letf_anneal_offset_clip50"
AMORTISED_CELL = "S2_d4_camort_50k_l50_letf_anneal_offset_clip50"
CENTRE_COMPOSITION = 0.5
NULL_TRAINING_VETO = 0.1


def seed_of(run_dir: Path) -> int:
    return int(run_dir.name.split("seed")[1].split("_")[0])


def point_ess_by_seed(cell: str) -> dict[int, float]:
    """ESS fraction from the single-composition eval of each seed's run."""
    runs = sorted(RESULTS_DIR.glob(f"{cell}_seed*"))
    if not runs:
        raise FileNotFoundError(f"no runs for {cell} under {RESULTS_DIR}")
    return {
        seed_of(r): json.loads((r / "eval" / "metrics.json").read_text())[
            "ess_fraction"
        ]
        for r in runs
    }


def centre_ess_by_seed(cell: str) -> dict[int, float]:
    """ESS fraction at the centre composition from each seed's sweep eval."""
    out: dict[int, float] = {}
    for r in sorted(RESULTS_DIR.glob(f"{cell}_seed*")):
        rows = json.loads((r / "eval" / "composition_sweep.json").read_text())
        (centre_row,) = [
            row for row in rows if row["composition"] == CENTRE_COMPOSITION
        ]
        out[seed_of(r)] = centre_row["ess_fraction"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "machinery_cost_repricing.json",
        help="where to write the archive JSON",
    )
    args = parser.parse_args()

    null = point_ess_by_seed(NULL_CELL)
    specialist = point_ess_by_seed(SPECIALIST_CELL)
    amortised = centre_ess_by_seed(AMORTISED_CELL)

    failed_null_seeds = {s: v for s, v in null.items() if v < NULL_TRAINING_VETO}
    assert not failed_null_seeds, (
        f"veto fires: null seeds {failed_null_seeds} failed to "
        "train, so no machinery cost may be quoted"
    )

    null_mean = statistics.mean(null.values())
    specialist_mean = statistics.mean(specialist.values())
    amortised_mean = statistics.mean(amortised.values())

    archive = {
        "cells": {
            "null": NULL_CELL,
            "specialist": SPECIALIST_CELL,
            "amortised": AMORTISED_CELL,
        },
        "ess_fraction_by_seed": {
            "null": null,
            "specialist": specialist,
            "amortised_at_centre": amortised,
        },
        "seed_means": {
            "null": null_mean,
            "specialist": specialist_mean,
            "amortised_at_centre": amortised_mean,
        },
        "seed_stdevs": {
            "null": statistics.stdev(null.values()),
            "specialist": statistics.stdev(specialist.values()),
            "amortised_at_centre": statistics.stdev(amortised.values()),
        },
        "machinery_cost_specialist_minus_null": specialist_mean - null_mean,
        "range_drawing_gap_null_minus_amortised": null_mean - amortised_mean,
        "superseded_prefix_recipe_cost": 0.155,
        "preregistered_branch": (
            "A: the 0.155 was recipe-confounded; this pair prices the "
            "machinery of the system as delivered"
        ),
    }
    args.out.write_text(json.dumps(archive, indent=2) + "\n")

    print(f"null        {sorted(null.items())}  mean {null_mean:.4f}")
    print(f"specialist  {sorted(specialist.items())}  mean {specialist_mean:.4f}")
    print(f"amortised   {sorted(amortised.items())}  mean {amortised_mean:.4f}")
    print(f"machinery cost   = {specialist_mean - null_mean:+.4f}")
    print(f"range-drawing gap = {null_mean - amortised_mean:+.4f}")
    print(f"archived to {args.out}")


if __name__ == "__main__":
    main()
