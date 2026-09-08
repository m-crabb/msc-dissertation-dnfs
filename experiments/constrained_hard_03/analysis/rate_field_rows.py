"""Rows for tab:rate-field from the trainers' own telemetry (no checkpoint forward needed).

Every swap run logs, at each in-training evaluation, the mean one-way rate over all
(state, pair) entries of the training minibatch (`rate_pair_mean`), the 99th
percentile of the one-event product Lambda dt (`lambda_dt_p99`), the fraction of states whose
Lambda dt exceeds one (`lambda_dt_clipped_frac`, a hypothetical one-event diagnostic
when matching is used)
and, under the matching step, the fired swaps per site per Euler step
(`events_per_site_per_step`). The total escape rate is
    Lambda = rate_pair_mean * d (d - 1) / 2,
because the logged mean runs over all d(d-1)/2 unordered pairs (same-spin pairs carry
rate exactly zero by antisymmetry, so they dilute the mean but not the sum).

The row is the mean of the last `--tail` evaluations of each seed (the run's final
coupling), then mean +- SD over seeds. Rate diagnostics use raw parameters and
training minibatches sampled across path times; matching load comes from the
corresponding rollout batch. Runs must be completed, use one common configuration,
and have distinct seeds. The adjacent/non-adjacent split is not logged and
needs a head forward; it is left blank here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "rate_pair_mean",
    "lambda_dt_p99",
    "lambda_dt_clipped_frac",
    "events_per_site_per_step",
]


def row_for(run_dirs, tail):
    """Aggregate full runs with a common recipe and distinct seeds."""
    if tail < 1 or not run_dirs:
        raise ValueError("provide at least one run and a positive tail length")
    per_seed = []
    seeds = set()
    common_config = None
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        seed = cfg["train"]["seed"]
        if seed in seeds:
            raise ValueError(
                f"duplicate seed {seed}: select one completed run per seed"
            )
        seeds.add(seed)
        log = pd.read_csv(run_dir / "training_log.csv")
        # A four-step smoke run alongside the full seed-42 run reproduces the
        # erroneous 4x4 row (Lambda=1.8 rather than 2.4); its eight-step grid
        # matches the old caption, while the completed runs used 100 steps.
        if log.empty or int(log["step"].iloc[-1]) != cfg["train"]["n_steps"] - 1:
            raise ValueError(f"incomplete training log: {run_dir}")
        log = log.dropna(subset=["rate_pair_mean"]).tail(tail)
        if len(log) != tail:
            raise ValueError(
                f"need {tail} rate diagnostics in {run_dir}, found {len(log)}"
            )
        diagnostics = log[COLUMNS[:3]].to_numpy(dtype=float)
        if not np.isfinite(diagnostics).all():
            raise ValueError(f"non-finite rate diagnostics in {run_dir}")
        if not np.allclose(
            log["sigma_current"], cfg["ising"]["sigma"], rtol=0, atol=1e-10
        ):
            raise ValueError(f"tail spans a different target coupling: {run_dir}")
        if (
            cfg["ctmc"]["use_matching_step"]
            and not np.isfinite(
                log["events_per_site_per_step"].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(f"missing matching-step load in {run_dir}")
        del cfg["train"]["seed"]
        # Older saved configs predate these inert options; absence means
        # the original behaviour, matching the defaults in configs.py.
        cfg.setdefault("pair_position_mode", "absolute")
        cfg.setdefault("separable_band_scores", False)
        cfg.setdefault("global_bond_features", False)
        cfg["model"].setdefault("exact_field_channel", False)
        if common_config is None:
            common_config = cfg
        elif cfg != common_config:
            raise ValueError(f"mixed run configurations: {run_dir}")
        per_seed.append(log[COLUMNS].mean(numeric_only=True))
    frame = pd.DataFrame(per_seed)
    return frame.mean(), frame.std(ddof=0), len(per_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--cell",
        action="append",
        nargs="+",
        metavar="LABEL DIR",
        help="a label followed by the run dirs of one cell",
    )
    selection.add_argument(
        "--manifest",
        type=Path,
        help="JSON with tail and cells; run paths relative to the repository",
    )
    parser.add_argument(
        "--tail", type=int, help="evaluations per seed (default: manifest tail or 10)"
    )
    args = parser.parse_args()
    cells = args.cell
    tail = args.tail if args.tail is not None else 10
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        repo = Path(__file__).resolve().parents[3]
        cells = [
            [label, *(repo / path for path in paths)]
            for label, paths in manifest["cells"].items()
        ]
        if args.tail is None:
            tail = manifest["tail"]
    print(
        f"{'cell':<28}{'d':>5}{'n':>3}{'Lambda':>14}{'p99 Ldt':>10}{'clip':>8}{'ev/site':>10}"
    )
    for label, *dirs in cells:
        run_dirs = [Path(d) for d in dirs]
        if not run_dirs:
            parser.error(f"no run directories for {label}")
        cfg = json.loads((run_dirs[0] / "config.json").read_text())
        d = int(cfg["ising"]["D"]) ** 2
        mean, sd, n = row_for(run_dirs, tail)
        pairs = d * (d - 1) / 2
        lam, lam_sd = mean["rate_pair_mean"] * pairs, sd["rate_pair_mean"] * pairs
        print(
            f"{label:<28}{d:>5}{n:>3}{lam:>8.1f} +- {lam_sd:<4.1f}"
            f"{mean['lambda_dt_p99']:>9.2f}{mean['lambda_dt_clipped_frac']:>8.3f}"
            f"{mean['events_per_site_per_step']:>10.4f}"
        )


if __name__ == "__main__":
    main()
