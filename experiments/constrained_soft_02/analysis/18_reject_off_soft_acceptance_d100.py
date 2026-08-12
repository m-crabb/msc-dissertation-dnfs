"""Measure the reject-off-soft acceptance at d=100 from the archived
lambda-sweep eval draws (zero GPU; judge-synthesis ordered action 4).

The hard chapter's route (iii) prices exact fixed-composition sampling by
the probability mass the trained soft sampler puts on the exact slice
c(x) = c_t: rejection keeps only on-slice draws, and conditioning is exact
at every lambda (p_soft(x | c=c_t) = p_hard(x)). At 4x4 that mass is
enumerable (92% at lambda=50); at 10x10 the printed text could only argue
the sqrt(d/lambda) scaling — "reasoning, not measurement". The archived
S2_d10_c05_l{5,10,50,100} eval draws (5000 samples + IS log-weights per
seed) make it measurable:

- raw acceptance  = fraction of draws with c(x) = c_t exactly — what a
  filter on the sampler's raw stream pays;
- weighted slice mass = sum(w * 1[on slice]) / sum(w) — the soft TARGET's
  slice probability (the IS estimate of the enumerated 92% analogue),
  trustworthy only where the seed clears the 0.30 ESS floor the F(c)
  windows use;
- analytic prediction = mass of the width-(1/100) composition bin under
  the Gaussian composition marginal with std(c) = 1/sqrt(2*lambda*d),
  the width the soft chapter verifies the sampler sits on.

Output: one row per (lambda, seed) + a per-lambda summary, written to
results/02_constrained_soft/analysis/reject_off_soft_acceptance_d100.csv.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import torch

RESULTS_ROOT = Path("results/02_constrained_soft")
OUT_PATH = RESULTS_ROOT / "analysis" / "reject_off_soft_acceptance_d100.csv"
LAMBDAS = [5, 10, 50, 100]
SEEDS = [42, 43, 44, 45]
N_SITES = 100
TARGET_COMPOSITION = 0.5
ESS_FLOOR = 0.30  # the F(c) windows' gating convention


def gaussian_bin_mass(penalty_strength: float) -> float:
    """Mass of the central width-(1/d) bin under N(c_t, 1/(2*lambda*d)).

    The composition takes values k/d, so the exact-slice event is the
    central bin of width 1/d; erf gives its mass under the analytic
    marginal the soft chapter verifies."""
    std = 1.0 / math.sqrt(2.0 * penalty_strength * N_SITES)
    half_bin = 0.5 / N_SITES
    return math.erf(half_bin / (std * math.sqrt(2.0)))


def one_cell(penalty_strength: int, seed: int) -> dict | None:
    pattern = (
        f"S2_d10_c05_l{penalty_strength}_letf_ne64_seed{seed}_*/eval"
    )
    matches = glob.glob(str(RESULTS_ROOT / pattern))
    if not matches:
        return None
    eval_dir = Path(matches[0])
    samples = torch.load(
        eval_dir / "samples.pt", map_location="cpu", weights_only=True
    )
    log_weights = torch.load(
        eval_dir / "log_weights.pt", map_location="cpu", weights_only=True
    )
    composition = (samples == 1).float().mean(dim=1)
    on_slice = torch.isclose(
        composition, torch.tensor(TARGET_COMPOSITION), atol=1e-6
    )
    weights = torch.softmax(log_weights, dim=0)
    ess_fraction = float(
        1.0 / (weights.pow(2).sum() * len(weights))
    )
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        recorded = json.loads(metrics_path.read_text()).get("ess_fraction")
    else:
        recorded = None
    return {
        "lambda": penalty_strength,
        "seed": seed,
        "raw_acceptance": float(on_slice.float().mean()),
        "weighted_slice_mass": float(weights[on_slice].sum()),
        "ess_fraction": ess_fraction,
        "ess_fraction_recorded": recorded,
        "clears_ess_floor": ess_fraction >= ESS_FLOOR,
        "analytic_bin_mass": gaussian_bin_mass(penalty_strength),
    }


def main() -> None:
    rows = [
        cell
        for penalty_strength in LAMBDAS
        for seed in SEEDS
        if (cell := one_cell(penalty_strength, seed)) is not None
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_PATH} ({len(rows)} cells)\n")
    print("lambda | analytic | raw acceptance (all seeds) | "
          "weighted slice mass (ESS-gated seeds)")
    for penalty_strength in LAMBDAS:
        cell_rows = [r for r in rows if r["lambda"] == penalty_strength]
        if not cell_rows:
            continue
        raw = [r["raw_acceptance"] for r in cell_rows]
        gated = [
            r["weighted_slice_mass"] for r in cell_rows
            if r["clears_ess_floor"]
        ]
        gated_txt = (
            f"{min(gated):.3f}-{max(gated):.3f} (n={len(gated)})"
            if gated else "no seed clears the floor"
        )
        print(
            f"l={penalty_strength:>3} | "
            f"{cell_rows[0]['analytic_bin_mass']:.3f} | "
            f"{min(raw):.3f}-{max(raw):.3f} | {gated_txt}"
        )


if __name__ == "__main__":
    main()
