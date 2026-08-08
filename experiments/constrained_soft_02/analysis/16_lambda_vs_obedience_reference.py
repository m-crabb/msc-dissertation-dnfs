"""Archive table: the soft TARGET's own obedience slope versus lambda.

At finite lambda a request for c_target is a request for a *tilt*, not a
value, so even a perfect sampler of the soft target under-delivers: the
exactly-enumerated target at lambda = 50 obeys its own requests at a slope
below 1. Every model slope in the dissertation is scored against this
target-level reference, never against 1 -- a sampler at slope 1.000 would be
unfaithful, not perfect. This script regenerates that reference as a function
of lambda, from two independent constructions:

  * **exact** -- enumerate all 2^(D*D) states, weight by
    p(x) proportional to exp(base_log_prob(x) - lambda*d*(c(x) - c_req)^2)
    (the t = 1 soft target), and read off E[c] at each requested c_req.
  * **mchammer** -- literal VC-SGC chains via `mcmc.mchammer_ising.run_vcsgc`
    (kappa = lambda and phi = -2*c_target inside `vcsgc_parameters`), so the
    reference is what a materials practitioner's tool actually delivers, not
    our own arithmetic twice.

Agreement between the two columns is what licenses using mchammer as the
target-level yardstick at sizes where enumeration is impossible.

The slope is the least-squares fit of delivered on requested composition.
Two composition grids are emitted because the fit depends (mildly) on the
grid: the model sweeps' band grid (the ten swept points inside [0.30, 0.70])
and a regular five-point grid {0.30, 0.40, 0.50, 0.60, 0.70}. Whichever grid
a prose number came from, it should be reproducible from this file's output.

Usage:
    python -m experiments.constrained_soft_02.analysis.16_lambda_vs_obedience_reference
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from discrete_flow_sampler.diagnostics.metrics import (
    composition_fraction_up,
    enumerate_states,
)
from discrete_flow_sampler.mcmc.mchammer_ising import run_vcsgc
from discrete_flow_sampler.targets.ising import IsingTarget

PENALTY_STRENGTHS = (10.0, 25.0, 50.0)
# The ten swept compositions inside the claim band, as used by every model
# composition_sweep.json, and the regular five-point grid.
SWEEP_BAND_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.575, 0.60, 0.65, 0.70)
REGULAR_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


def exact_delivered(D: int, sigma: float, strength: float, c_req: float) -> float:
    """E[c] under the exactly-enumerated t=1 soft target."""
    n_sites = D * D
    states = enumerate_states(n_sites).float()
    target = IsingTarget(
        D=D,
        sigma=sigma,
        target_composition=c_req,
        composition_penalty_strength=strength,
    )
    log_p = target.log_prob(states)
    probabilities = torch.softmax(log_p, dim=0)
    return float((probabilities * composition_fraction_up(states)).sum())


def mchammer_delivered(
    D: int, sigma: float, strength: float, c_req: float,
    n_steps: int, seeds: tuple[int, ...],
) -> float:
    """Mean delivered composition over VC-SGC chains (seed-averaged)."""
    means = []
    for seed in seeds:
        summary = run_vcsgc(
            D=D, sigma=sigma, penalty_strength=strength,
            target_composition=c_req, n_steps=n_steps, seed=seed,
        )
        means.append(summary["observables"]["composition"]["mean"])
    return float(np.mean(means))


def fitted_slope(requested: list[float], delivered: list[float]) -> float:
    return float(np.polyfit(np.array(requested), np.array(delivered), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/02_constrained_soft/lambda_vs_obedience_reference"),
    )
    args = parser.parse_args()

    rows = []
    for strength in PENALTY_STRENGTHS:
        exact_by_c = {
            c: exact_delivered(args.D, args.sigma, strength, c)
            for c in SWEEP_BAND_GRID
        }
        mchammer_by_c = {
            c: mchammer_delivered(
                args.D, args.sigma, strength, c, args.n_steps, tuple(args.seeds)
            )
            for c in SWEEP_BAND_GRID
        }
        for grid_name, grid in (
            ("sweep_band", SWEEP_BAND_GRID),
            ("regular", REGULAR_GRID),
        ):
            rows.append({
                "lambda": strength,
                "grid": grid_name,
                "exact_slope": round(
                    fitted_slope(list(grid), [exact_by_c[c] for c in grid]), 4
                ),
                "mchammer_slope": round(
                    fitted_slope(list(grid), [mchammer_by_c[c] for c in grid]), 4
                ),
            })
        print(
            f"lambda={strength:5.1f}  "
            + "  ".join(
                f"{r['grid']}: exact {r['exact_slope']:.4f} "
                f"mchammer {r['mchammer_slope']:.4f}"
                for r in rows[-2:]
            )
        )
        rows[-2]["delivered_exact"] = {str(c): round(exact_by_c[c], 5) for c in SWEEP_BAND_GRID}
        rows[-2]["delivered_mchammer"] = {
            str(c): round(mchammer_by_c[c], 5) for c in SWEEP_BAND_GRID
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "D": args.D, "sigma": args.sigma, "n_steps": args.n_steps,
        "seeds": args.seeds, "rows": rows,
    }, indent=2))
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["lambda", "grid", "exact_slope", "mchammer_slope"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
