"""Rows for tab:rate-field from the trainers' own telemetry (no checkpoint forward needed).

Every swap run logs, at each in-training evaluation, the mean one-way rate over all
(state, pair) entries of the rollout (`rate_pair_mean`), the 99th percentile of the
one-event product Lambda dt (`lambda_dt_p99`), the fraction of rollout states whose
Lambda dt exceeds one (`lambda_dt_clipped_frac`, the one-event under-fire probability)
and, under the matching step, the fired swaps per site per Euler step
(`events_per_site_per_step`). The total escape rate is
    Lambda = rate_pair_mean * d (d - 1) / 2,
because the logged mean runs over all d(d-1)/2 unordered pairs (same-spin pairs carry
rate exactly zero by antisymmetry, so they dilute the mean but not the sum).

The row is the mean of the last `--tail` evaluations of each seed (the run's final
coupling), then mean +- SD over seeds. The adjacent/non-adjacent split is NOT logged
and needs a head forward; it is left blank here on purpose.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["rate_pair_mean", "lambda_dt_p99", "lambda_dt_clipped_frac",
           "events_per_site_per_step"]


def row_for(run_dirs, tail):
    per_seed = []
    for run_dir in run_dirs:
        log = pd.read_csv(run_dir / "training_log.csv")
        log = log.dropna(subset=["rate_pair_mean"]).tail(tail)
        per_seed.append(log[COLUMNS].mean(numeric_only=True))
    frame = pd.DataFrame(per_seed)
    return frame.mean(), frame.std(ddof=0), len(per_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", nargs="+", metavar="LABEL DIR",
                        help="a label followed by the run dirs of one cell", required=True)
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()
    print(f"{'cell':<28}{'d':>5}{'n':>3}{'Lambda':>14}{'p99 Ldt':>10}{'clip':>8}{'ev/site':>10}")
    for label, *dirs in args.cell:
        run_dirs = [Path(d) for d in dirs]
        cfg = json.loads((run_dirs[0] / "config.json").read_text())
        d = int(cfg["ising"]["D"]) ** 2
        mean, sd, n = row_for(run_dirs, args.tail)
        pairs = d * (d - 1) / 2
        lam, lam_sd = mean["rate_pair_mean"] * pairs, sd["rate_pair_mean"] * pairs
        print(f"{label:<28}{d:>5}{n:>3}{lam:>8.1f} +- {lam_sd:<4.1f}"
              f"{mean['lambda_dt_p99']:>9.2f}{mean['lambda_dt_clipped_frac']:>8.3f}"
              f"{mean['events_per_site_per_step']:>10.4f}")


if __name__ == "__main__":
    main()
